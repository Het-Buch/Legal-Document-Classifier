from sklearn.preprocessing import MultiLabelBinarizer
from src.utils.metrics import multilabel_metrics

def fit_binarizer(train_labels):
    """
    Fit MultiLabelBinarizer ONLY on training labels
    """
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(train_labels)
    return y, mlb

def transform_labels(mlb, labels):
    """
    Transform val/test labels using the SAME binarizer
    """
    return mlb.transform(labels)

def evaluate(y_true, y_pred):
    return multilabel_metrics(y_true, y_pred)
