"""
Project: SMS Spam Detection - Milestone 3
Description: 
    This script extends the reproduction of Bishi et al. (2024) by introducing:
    1. Bigram TF-IDF representations (Experiment 1).
    2. Feature Engineering (message length, digit count).
    3. An additional classifier (Logistic Regression).
    4. A comparative ablation study across baseline and experimental settings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

from scipy.sparse import hstack, csr_matrix

RANDOM_STATE = 42

# 1. Load dataset (same as Milestone 2)
rows = []
with open("data/SMSSpamCollection", "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        rows.append((parts[0], parts[1]))

df = pd.DataFrame(rows, columns=["label", "text"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df["text"] = df["text"].astype(str).str.strip()

print(f"(shape, counts) after load: {df.shape} "
      f"{df['label'].map({0: 'ham', 1: 'spam'}).value_counts().to_dict()}")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"].values,
    df["label"].values,
    test_size=0.20,
    stratify=df["label"].values,
    random_state=RANDOM_STATE,
)

# 2. NLTK tokenizer + stemming
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


class StemmingTokenizer:
    """Custom tokenizer that also stems words."""
    def __init__(self):
        self.stemmer = PorterStemmer()

    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in nltk.word_tokenize(doc)]


# 3. Helper: feature engineering
def engineer_features(text_array):
    """
    Simple structural / numeric features:
      - message length (characters)
      - digit count
    Returns a dense 2D numpy array of shape (n_samples, 2)
    """
    lengths = np.array([len(t) for t in text_array], dtype=float).reshape(-1, 1)
    digit_counts = np.array(
        [sum(ch.isdigit() for ch in t) for t in text_array], dtype=float
    ).reshape(-1, 1)

    feats = np.hstack([lengths, digit_counts])
    return feats


def evaluate(name, clf, X_train, X_test, y_train, y_test):
    """
    Fit the classifier, compute metrics, print them, and return a dict.
    """
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, pred)
    print(f"\n{name}\nAcc:{acc:.6f}  Prec:{prec:.6f}  Rec:{rec:.6f}  F1:{f1:.6f}")
    print("Confusion matrix:\n", cm)
    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


# 4. Baseline: Unigram TF-IDF (same as Milestone 2)
print("\n====================")
print("BASELINE: Unigram TF-IDF (Milestone 2 reproduction)")
print("====================")

tfidf_uni = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    stop_words="english",
    min_df=2,
    tokenizer=StemmingTokenizer(),
)
X_train_uni = tfidf_uni.fit_transform(X_train_text)
X_test_uni = tfidf_uni.transform(X_test_text)

# Baseline models (same as Milestone 2)
nb = MultinomialNB()
svm = LinearSVC(random_state=RANDOM_STATE, max_iter=5000)
rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
et = ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)

voter_hard_uni = VotingClassifier(
    estimators=[("nb", nb), ("svm", svm), ("rf", rf), ("et", et)],
    voting="hard",
)

baseline_results = []
for name, clf in [
    ("Naive Bayes (UNI)", nb),
    ("Linear SVM (UNI)", svm),
    ("Random Forest (UNI)", rf),
    ("Extra Trees (UNI)", et),
    ("Voting (NB+SVM+RF+ET, UNI)", voter_hard_uni),
]:
    baseline_results.append(
        evaluate(name, clf, X_train_uni, X_test_uni, y_train, y_test)
    )

baseline_df = pd.DataFrame(baseline_results).sort_values("f1", ascending=False)
print("\nBaseline Summary (Unigram TF-IDF):\n", baseline_df.to_string(index=False))

# 5. Experiment 1: Unigram + Bigram TF-IDF
print("\n====================")
print("EXPERIMENT 1: Unigram + Bigram TF-IDF")
print("====================")

tfidf_uni_bi = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    stop_words="english",
    min_df=2,
    ngram_range=(1, 2),           # NEW: bigrams
    tokenizer=StemmingTokenizer(),
)
X_train_uni_bi = tfidf_uni_bi.fit_transform(X_train_text)
X_test_uni_bi = tfidf_uni_bi.transform(X_test_text)

# Reuse same base models, but they will be re-fit on new features
nb_bi = MultinomialNB()
svm_bi = LinearSVC(random_state=RANDOM_STATE)
rf_bi = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
et_bi = ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)

voter_hard_uni_bi = VotingClassifier(
    estimators=[("nb", nb_bi), ("svm", svm_bi), ("rf", rf_bi), ("et", et_bi)],
    voting="hard",
)

uni_bi_results = []
for name, clf in [
    ("Naive Bayes (UNI+BI)", nb_bi),
    ("Linear SVM (UNI+BI)", svm_bi),
    ("Random Forest (UNI+BI)", rf_bi),
    ("Extra Trees (UNI+BI)", et_bi),
    ("Voting (NB+SVM+RF+ET, UNI+BI)", voter_hard_uni_bi),
]:
    uni_bi_results.append(
        evaluate(name, clf, X_train_uni_bi, X_test_uni_bi, y_train, y_test)
    )

uni_bi_df = pd.DataFrame(uni_bi_results).sort_values("f1", ascending=False)
print("\nExperiment 1 Summary (Unigram + Bigram TF-IDF):\n", uni_bi_df.to_string(index=False))

# 6. Experiment 2: UNI+BI TF-IDF + Engineered Features + Logistic Regression
print("\n====================")
print("EXPERIMENT 2: UNI+BI TF-IDF + Engineered Features + Logistic Regression")
print("====================")

# Numeric features on raw text
train_feats = engineer_features(X_train_text)
test_feats = engineer_features(X_test_text)

# Convert dense numeric feats to sparse and hstack with TF-IDF
train_feats_sp = csr_matrix(train_feats)
test_feats_sp = csr_matrix(test_feats)

X_train_extended = hstack([X_train_uni_bi, train_feats_sp])
X_test_extended = hstack([X_test_uni_bi, test_feats_sp])

# Add Logistic Regression
lr = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=1000,
    n_jobs=-1
)

# Rebuild models on extended feature space
nb_ext = MultinomialNB()
svm_ext = LinearSVC(random_state=RANDOM_STATE)
rf_ext = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
et_ext = ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
lr_ext = lr

voter_hard_ext = VotingClassifier(
    estimators=[
        ("nb", nb_ext),
        ("svm", svm_ext),
        ("rf", rf_ext),
        ("et", et_ext),
        ("lr", lr_ext),
    ],
    voting="hard",
)

extended_results = []
for name, clf in [
    ("Naive Bayes (EXT)", nb_ext),
    ("Linear SVM (EXT)", svm_ext),
    ("Random Forest (EXT)", rf_ext),
    ("Extra Trees (EXT)", et_ext),
    ("Logistic Regression (EXT)", lr_ext),
    ("Voting (NB+SVM+RF+ET+LR, EXT)", voter_hard_ext),
]:
    extended_results.append(
        evaluate(name, clf, X_train_extended, X_test_extended, y_train, y_test)
    )

extended_df = pd.DataFrame(extended_results).sort_values("f1", ascending=False)
print("\nExperiment 2 Summary (UNI+BI TF-IDF + Engineered Features):\n",
      extended_df.to_string(index=False))

# 7. 5-fold CV F1 for best models per setting
print("\n====================")
print("CROSS-VALIDATION (5-fold F1) for selected models")
print("====================")

# Pick representative strong models: SVM + Voting for each setting, and LR on extended
models_for_cv = [
    ("SVM (UNI)", svm, X_train_uni),
    ("Voting (UNI)", voter_hard_uni, X_train_uni),
    ("SVM (UNI+BI)", svm_bi, X_train_uni_bi),
    ("Voting (UNI+BI)", voter_hard_uni_bi, X_train_uni_bi),
    ("LR (EXT)", lr_ext, X_train_extended),
    ("Voting (EXT)", voter_hard_ext, X_train_extended),
]

for name, clf, Xtr in models_for_cv:
    scores_f1 = cross_val_score(
        clf, Xtr, y_train, cv=5, scoring="f1_weighted", n_jobs=-1
    )
    print(f"{name}: CV F1 mean={scores_f1.mean():.6f}  std={scores_f1.std():.6f}")

# 8. AUROC & Precision–Recall for best new ensemble (EXT with soft voting)
print("\n====================")
print("ROC & Precision–Recall (Soft Voting on extended features)")
print("====================")

svm_cal_ext = CalibratedClassifierCV(
    LinearSVC(random_state=RANDOM_STATE),
    cv=5
)

soft_voter_ext = VotingClassifier(
    estimators=[
        ("nb", MultinomialNB()),
        ("svm_cal", svm_cal_ext),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ("et", ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ("lr", LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            n_jobs=-1
        )),
    ],
    voting="soft",
)

soft_voter_ext.fit(X_train_extended, y_train)
y_scores_ext = soft_voter_ext.predict_proba(X_test_extended)[:, 1]

# ROC
fpr_ext, tpr_ext, _ = roc_curve(y_test, y_scores_ext)
roc_auc_ext = auc(fpr_ext, tpr_ext)

# Precision–Recall
precisions_ext, recalls_ext, _ = precision_recall_curve(y_test, y_scores_ext)
pr_auc_ext = auc(recalls_ext, precisions_ext)

print(f"\nAUROC (Soft Voting, EXT): {roc_auc_ext:.6f}")
print(f"PR  AUC (Soft Voting, EXT): {pr_auc_ext:.6f}")

# Save plots to reports/
plt.figure()
plt.plot(fpr_ext, tpr_ext, label=f"ROC (AUC={roc_auc_ext:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Soft Voting (UNI+BI TF-IDF + Engineered Features)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("reports/roc_curve_m3.png", dpi=200)
print("Saved: reports/roc_curve_m3.png")

plt.figure()
plt.plot(recalls_ext, precisions_ext, label=f"PR (AUC={pr_auc_ext:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – Soft Voting (UNI+BI TF-IDF + Engineered Features)")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("reports/pr_curve_m3.png", dpi=200)
print("Saved: reports/pr_curve_m3.png")
