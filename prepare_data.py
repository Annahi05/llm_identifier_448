import re
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
MIN_WORDS = 5
BALANCE = True

# ---------- 1. load ----------
ds_dict = load_dataset("namkoong-lab/PersonalLLM")
print("Splits:", {k: len(v) for k, v in ds_dict.items()})
df = concatenate_datasets(list(ds_dict.values())).to_pandas()
print("Columns:", list(df.columns)[:30], "...")

# ---------- 2. flatten wide -> long ----------
resp_cols = sorted(c for c in df.columns if re.fullmatch(r"response_\d+", c))
print("Response columns found:", resp_cols)

if "prompt_id" not in df.columns:
    df["prompt_id"] = range(len(df))

rows = []
for c in resp_cols:
    model_col = f"{c}_model"
    sub = pd.DataFrame({
        "prompt_id": df["prompt_id"],
        "llm_input": df["prompt"],
        "llm_output": df[c],
        "llm_name": df[model_col] if model_col in df.columns else c,
    })
    if "subset" in df.columns:
        sub["subset"] = df["subset"]
    rows.append(sub)
long = pd.concat(rows, ignore_index=True)

# ---------- 3. clean ----------
long = long.dropna(subset=["llm_input", "llm_output", "llm_name"])
long = long[long["llm_output"].str.split().str.len() >= MIN_WORDS]
long = long.drop_duplicates(subset=["llm_name", "llm_input", "llm_output"])

long["family"] = long["llm_name"].str.split("/").str[0]
print("\nRows per model:\n", long["llm_name"].value_counts())
print("\nRows per family:\n", long["family"].value_counts())

# ---------- 4. balance (per model) ----------
if BALANCE:
    n = long["llm_name"].value_counts().min()
    long = (long.groupby("llm_name", group_keys=False)
                .apply(lambda g: g.sample(n, random_state=SEED))
                .reset_index(drop=True))
    print(f"\nBalanced to {n} rows per model")

# ---------- 5. split by prompt (80/10/10) ----------
groups = long["prompt_id"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, rest_idx = next(gss.split(long, groups=groups))
train, rest = long.iloc[train_idx], long.iloc[rest_idx]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
val_idx, test_idx = next(gss2.split(rest, groups=rest["prompt_id"]))
val, test = rest.iloc[val_idx], rest.iloc[test_idx]

assert not set(train.prompt_id) & set(val.prompt_id)
assert not set(train.prompt_id) & set(test.prompt_id)
assert not set(val.prompt_id) & set(test.prompt_id)

for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_csv(f"{name}.csv", index=False)
    print(f"{name}: {len(part)} rows, {part.prompt_id.nunique()} prompts")

# ---------- 6. length stats ----------
long["n_words"] = long["llm_output"].str.split().str.len()
print("\nMean output length (words) per model:\n",
      long.groupby("llm_name")["n_words"].mean().round(1))
