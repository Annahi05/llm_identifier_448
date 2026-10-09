import argparse, re, random, os
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["cnn", "lstm"], required=True)
p.add_argument("--input", choices=["output", "input", "both"], default="output")
p.add_argument("--label", choices=["family", "llm_name"], default="family")
p.add_argument("--epochs", type=int, default=6)
p.add_argument("--max_rows", type=int, default=0, help="use only N training rows (quick test)")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--device", default="auto")
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
if args.device == "auto":
    args.device = "mps" if torch.backends.mps.is_available() else "cpu"
dev = torch.device(args.device)
print("Device:", dev)

# ---------- data ----------
train = pd.read_csv("train.csv"); val = pd.read_csv("val.csv"); test = pd.read_csv("test.csv")
if args.max_rows:
    train = train.sample(args.max_rows, random_state=args.seed)
    val = val.sample(min(len(val), args.max_rows // 4), random_state=args.seed)
    test = test.sample(min(len(test), args.max_rows // 4), random_state=args.seed)

classes = sorted(train[args.label].unique())
c2i = {c: i for i, c in enumerate(classes)}
print("Classes:", classes)

# ---------- tokenizer ----------
TOK = re.compile(r"\w+|[^\w\s]")
def tokenize(text):
    text = str(text).replace("\n", " NEWLINE ")
    return TOK.findall(text.lower())

IN_LEN = 128 if args.input == "input" else 64   # max tokens kept from the prompt
OUT_LEN = 256                                    # max tokens kept from the response

def tokens_for(row_in, row_out):
    if args.input == "output":
        return tokenize(row_out)[:OUT_LEN]
    if args.input == "input":
        return tokenize(row_in)[:IN_LEN]
    return tokenize(row_in)[:IN_LEN] + ["SEPTOKEN"] + tokenize(row_out)[:OUT_LEN]

def build(df):
    return [tokens_for(a, b) for a, b in zip(df["llm_input"], df["llm_output"])]

print("Tokenizing...")
tr_tok, va_tok, te_tok = build(train), build(val), build(test)

# vocabulary from TRAIN only (index 0 = padding, 1 = unknown)
counts = Counter(t for toks in tr_tok for t in toks)
vocab = {"<pad>": 0, "<unk>": 1}
for w, c in counts.most_common(30000):
    if c >= 3:
        vocab[w] = len(vocab)
print("Vocab size:", len(vocab))

max_len = {"output": OUT_LEN, "input": IN_LEN, "both": IN_LEN + 1 + OUT_LEN}[args.input]
def to_tensor(tok_lists):
    X = np.zeros((len(tok_lists), max_len), dtype=np.int64)
    for i, toks in enumerate(tok_lists):
        ids = [vocab.get(t, 1) for t in toks][:max_len]
        X[i, :len(ids)] = ids
    return torch.from_numpy(X)

Xtr, Xva, Xte = to_tensor(tr_tok), to_tensor(va_tok), to_tensor(te_tok)
ytr = torch.tensor([c2i[c] for c in train[args.label]])
yva = torch.tensor([c2i[c] for c in val[args.label]])
yte = torch.tensor([c2i[c] for c in test[args.label]])

# ---------- models ----------
class CNN(nn.Module):
    def __init__(self, V, n_cls, emb=128, filters=128, kernels=(3, 4, 5)):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, filters, k) for k in kernels])
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(filters * len(kernels), n_cls)
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)                       # (batch, emb, len)
        pooled = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(pooled, dim=1)))

class LSTM(nn.Module):
    def __init__(self, V, n_cls, emb=128, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.rnn = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden * 2, n_cls)
    def forward(self, x):
        out, _ = self.rnn(self.emb(x))                        # (batch, len, 2*hidden)
        mask = (x != 0).unsqueeze(2)
        out = out.masked_fill(~mask, -1e9)                    # ignore padding
        return self.fc(self.drop(out.max(dim=1).values))      # max over time

n_cls = len(classes)
model = (CNN if args.model == "cnn" else LSTM)(len(vocab), n_cls).to(dev)
print("Parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()
BS = 64

def predict(X):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            preds.append(model(X[i:i + 256].to(dev)).argmax(1).cpu())
    return torch.cat(preds)

# ---------- train with early stopping on validation accuracy ----------
best_acc, best_state, bad = -1, None, 0
for epoch in range(1, args.epochs + 1):
    model.train()
    perm = torch.randperm(len(Xtr)); total = 0
    for i in range(0, len(Xtr), BS):
        idx = perm[i:i + BS]
        opt.zero_grad()
        loss = lossf(model(Xtr[idx].to(dev)), ytr[idx].to(dev))
        loss.backward(); opt.step()
        total += loss.item() * len(idx)
    va_acc = accuracy_score(yva, predict(Xva))
    print(f"epoch {epoch}: train loss {total/len(Xtr):.4f}  val acc {va_acc:.4f}", flush=True)
    if va_acc > best_acc:
        best_acc, bad = va_acc, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= 2:
            print("Early stopping"); break

# ---------- final test evaluation (best epoch) ----------
model.load_state_dict(best_state)
pred = predict(Xte)
acc = accuracy_score(yte, pred); f1 = f1_score(yte, pred, average="macro")
name = f"{args.model}_{args.input}_{args.label}"
print(f"\nTEST {name}: accuracy = {acc:.4f}, macro-F1 = {f1:.4f}  (best val acc {best_acc:.4f})")

os.makedirs("results", exist_ok=True)
with open("results/results.csv", "a") as f:
    f.write(f"{name},{acc:.4f},{f1:.4f},{best_acc:.4f},rows={len(train)}\n")
cm = confusion_matrix(yte, pred)
pd.DataFrame(cm, index=classes, columns=classes).to_csv(f"results/cm_{name}.csv")
print("Confusion matrix (rows = true, cols = predicted):")
print(pd.DataFrame(cm, index=classes, columns=classes))
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_cls)); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(n_cls)); ax.set_yticklabels(classes)
    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(name)
    fig.tight_layout(); fig.savefig(f"results/cm_{name}.png", dpi=150)
except Exception as e:
    print("Could not draw plot:", e)
