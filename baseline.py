import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Baseline 1: always guess the most common family
majority = train["family"].value_counts().idxmax()
maj_acc = (test["family"] == majority).mean()
print(f"Always guess '{majority}': accuracy = {maj_acc:.3f}")

# Baseline 2: count words/punctuation (TF-IDF), then logistic regression
vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=5,
                      sublinear_tf=True, lowercase=False,
                      token_pattern=r"(?u)\b\w+\b|[^\w\s]")
X_train = vec.fit_transform(train["llm_output"])
X_test = vec.transform(test["llm_output"])

clf = LogisticRegression(max_iter=300)
clf.fit(X_train, train["family"])
pred = clf.predict(X_test)

acc = accuracy_score(test["family"], pred)
f1 = f1_score(test["family"], pred, average="macro")
print(f"TF-IDF + LogReg (output only, 6 families): accuracy = {acc:.3f}, macro-F1 = {f1:.3f}")
print(classification_report(test["family"], pred))
