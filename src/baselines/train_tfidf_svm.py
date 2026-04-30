import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

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


def train_svm():
    # Load splits
    X_train, y_train_labels, _ = load_split("train")
    X_val, y_val_labels, _ = load_split("val")

    # 🔑 Drop empty-label docs
    X_train, y_train_labels = filter_empty_text(X_train, y_train_labels)
    X_val, y_val_labels = filter_empty_text(X_val, y_val_labels)


    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=3
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # Labels
    y_train, mlb = fit_binarizer(y_train_labels)
    y_val = transform_labels(mlb, y_val_labels)

    # Model
    clf = OneVsRestClassifier(LinearSVC())
    clf.fit(X_train_vec, y_train)

    # Validation
    y_val_pred = clf.predict(X_val_vec)
    metrics = evaluate(y_val, y_val_pred)

    print("SVM Validation Metrics:", metrics)

    # Save
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": clf,
            "mlb": mlb
        },
        "models/tfidf_svm_cuad.joblib"
    )

    save_json(metrics, "artifacts/eval/baseline_svm_eval.json")


if __name__ == "__main__":
    train_svm()
