import joblib
import numpy as np
from src.data.dataset_reader import load_split
from src.baselines.utils_baseline import transform_labels
from src.utils.file_io import save_json

MODEL_PATH = "models/tfidf_svm_cuad.joblib"
OUT_PATH = "artifacts/diagnostics/svm_collapse_analysis.json"


def expose_svm_collapse():
    # Load model bundle
    bundle = joblib.load(MODEL_PATH)
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]
    mlb = bundle["mlb"]

    # Load TEST split
    X_test, y_test_labels, meta = load_split("test")

    X_test_vec = vectorizer.transform(X_test)
    y_true = transform_labels(mlb, y_test_labels)
    y_pred = model.predict(X_test_vec)

    learned_labels = mlb.classes_.tolist()

    diagnostics = {}

    # 1️⃣ Label-space collapse
    diagnostics["num_cuad_labels"] = len(meta["label_set"])
    diagnostics["num_svm_learned_labels"] = len(learned_labels)
    diagnostics["learned_label_names"] = learned_labels

    # 2️⃣ Average labels per document
    diagnostics["avg_true_labels_per_doc"] = float(
        np.mean(y_true.sum(axis=1))
    )
    diagnostics["avg_predicted_labels_per_doc"] = float(
        np.mean(y_pred.sum(axis=1))
    )

    # 3️⃣ Label-wise positive rate (ONLY learned labels)
    label_positive_rates = {}
    for i, label in enumerate(learned_labels):
        label_positive_rates[label] = float(np.mean(y_pred[:, i]))

    diagnostics["label_positive_prediction_rate"] = label_positive_rates

    # 4️⃣ Degenerate prediction behavior
    diagnostics["fraction_all_positive_docs"] = float(
        np.mean(y_pred.sum(axis=1) == y_pred.shape[1])
    )

    diagnostics["fraction_zero_positive_docs"] = float(
        np.mean(y_pred.sum(axis=1) == 0)
    )

    save_json(diagnostics, OUT_PATH)

    print("✅ SVM collapse analysis saved to:", OUT_PATH)
    print("Learned labels:", learned_labels)
    print(
        "Avg predicted labels / doc:",
        diagnostics["avg_predicted_labels_per_doc"]
    )


if __name__ == "__main__":
    expose_svm_collapse()
