import re, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

train = pd.read_csv("train.csv"); test = pd.read_csv("test.csv")

def feats(t):
    t = str(t)
    words = re.findall(r"[A-Za-z']+", t)
    nw = max(len(words), 1)
    lower = [w.lower() for w in words]
    lines = t.split("\n")
    sents = [s for s in re.split(r"[.!?]+\s", t) if s.strip()]
    return {
        # length
        "n_words": len(words), "n_chars": len(t), "n_sentences": len(sents),
        "avg_sentence_len": len(words) / max(len(sents), 1),
        # formatting
        "n_lines": len(lines), "n_blank_lines": sum(1 for l in lines if not l.strip()),
        "n_bullets": sum(1 for l in lines if re.match(r"\s*[-*\u2022]\s", l)),
        "n_numbered": sum(1 for l in lines if re.match(r"\s*\d+[.)]\s", l)),
        "n_bold": t.count("**"),
        "n_headers": sum(1 for l in lines if l.lstrip().startswith("#")),
        "n_code_fences": t.count("```"),
        # vocabulary / lexical diversity
        "type_token_ratio": len(set(lower)) / nw,
        "avg_word_len": sum(len(w) for w in words) / nw,
        "n_unique_words": len(set(lower)),
        # punctuation habits
        "commas_per_word": t.count(",") / nw, "excl_per_word": t.count("!") / nw,
        "quest_per_word": t.count("?") / nw, "colons_per_word": t.count(":") / nw,
        "parens_per_word": t.count("(") / nw, "quotes_per_word": t.count('"') / nw,
        "dashes_per_word": (t.count("\u2014") + t.count(" - ")) / nw,
    }

GROUPS = {
    "length": ["n_words", "n_chars", "n_sentences", "avg_sentence_len"],
    "formatting": ["n_lines", "n_blank_lines", "n_bullets", "n_numbered", "n_bold",
                   "n_headers", "n_code_fences"],
    "vocabulary": ["type_token_ratio", "avg_word_len", "n_unique_words"],
    "punctuation": ["commas_per_word", "excl_per_word", "quest_per_word", "colons_per_word",
                    "parens_per_word", "quotes_per_word", "dashes_per_word"],
}

print("Computing features (about a minute)...")
Ftr = pd.DataFrame([feats(t) for t in train["llm_output"]])
Fte = pd.DataFrame([feats(t) for t in test["llm_output"]])

# ---------- A. what does each AI look like on average? ----------
os.makedirs("results", exist_ok=True)
means = Ftr.groupby(train["llm_name"].values).mean().round(3)
means.T.to_csv("results/rq4_feature_means.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
short = {c: c.split("/")[-1][:14] for c in means.index}
print("\nAverage of each feature per model (train set):")
print(means.rename(index=short).T.to_string())

# ---------- B. can hand-made features alone identify the family? ----------
ytr, yte = train["family"].values, test["family"].values

def run_lr(cols):
    Xtr = np.log1p(Ftr[cols].values); Xte = np.log1p(Fte[cols].values)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    return accuracy_score(yte, pred), f1_score(yte, pred, average="macro"), clf

all_cols = [c for g in GROUPS.values() for c in g]
acc, f1, clf = run_lr(all_cols)
print(f"\nLogistic regression, ALL {len(all_cols)} features: accuracy = {acc:.3f}, macro-F1 = {f1:.3f}")

hgb = HistGradientBoostingClassifier(random_state=0).fit(np.log1p(Ftr[all_cols].values), ytr)
p2 = hgb.predict(np.log1p(Fte[all_cols].values))
print(f"Gradient boosting, ALL features:      accuracy = {accuracy_score(yte, p2):.3f}, "
      f"macro-F1 = {f1_score(yte, p2, average='macro'):.3f}")

print("\nFeature groups ALONE (logistic regression):")
rows = [("all features", acc, f1)]
for g, cols in GROUPS.items():
    a, f, _ = run_lr(cols)
    rows.append((f"only {g}", a, f)); print(f"  only {g:12s}: accuracy = {a:.3f}, macro-F1 = {f:.3f}")
print("\nDropping one group (logistic regression):")
for g in GROUPS:
    cols = [c for c in all_cols if c not in GROUPS[g]]
    a, f, _ = run_lr(cols)
    rows.append((f"without {g}", a, f)); print(f"  without {g:12s}: accuracy = {a:.3f}, macro-F1 = {f:.3f}")
pd.DataFrame(rows, columns=["setting", "accuracy", "macro_f1"]).round(4).to_csv(
    "results/rq4_feature_groups.csv", index=False)

# which single features matter most (average size of the weights across the 6 classes)
imp = pd.Series(np.abs(clf.coef_).mean(axis=0), index=all_cols).sort_values(ascending=False)
print("\nTop 10 features by weight size (all-feature logistic regression):")
print(imp.head(10).round(3).to_string())
