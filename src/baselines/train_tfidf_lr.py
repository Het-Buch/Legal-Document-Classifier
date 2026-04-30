import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from src.data.cuad_loader import load_cuad_dataset
from sklearn.preprocessing import MultiLabelBinarizer

from src.data.dataset_reader import load_split
from src.baselines.utils_baseline import fit_binarizer, transform_labels, evaluate
from src.utils.file_io import save_json

def filter_empty_text(X, y):
    X_f, y_f = [], []
    for x, labels in zip(X, y):
        if len(x.strip()) > 0:
            X_f.append(x)
            y_f.append(labels)
    return X_f, y_f

def train_lr():
    X_train, y_train_labels, _ = load_split("train")
    X_val, y_val_labels, _ = load_split("val")

    X_train, y_train_labels = filter_empty_text(X_train, y_train_labels)
    X_val, y_val_labels = filter_empty_text(X_val, y_val_labels)

    print("Non-empty train docs:", len(X_train))

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=1,
        stop_words=None,
        token_pattern=r"(?u)\b\w+\b"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # 🔑 Correct binarization
    # 🔑 FIX: force global label space
    _, _, meta = load_cuad_dataset()
    all_labels = meta["label_set"]

    mlb = MultiLabelBinarizer(classes=all_labels)
    y_train = mlb.fit_transform(y_train_labels)
    y_val = mlb.transform(y_val_labels)

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=300, n_jobs=-1)
    )
    clf.fit(X_train_vec, y_train)

    y_val_pred = clf.predict(X_val_vec)
    metrics = evaluate(y_val, y_val_pred)

    print("LR Validation Metrics:", metrics)

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": clf,
            "mlb": mlb
        },
        "models/tfidf_lr_cuad.joblib"
    )

    save_json(metrics, "artifacts/eval/baseline_lr_eval.json")

if __name__ == "__main__":
    train_lr()
