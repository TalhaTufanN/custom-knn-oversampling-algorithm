import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
# ==========================================
# PART 3: TRAINING AND METRICS
# ==========================================
def train_and_get_metrics(X, y, model):
    """Trains model, returns Prec, Rec, F1, AUC and ROC data."""
    # Stratify is important to keep class ratio in splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Scaling is crucial for KNN and some datasets
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
    else:
        auc = 0.5
        fpr, tpr = [0, 1], [0, 1]

    return prec, rec, f1, auc, fpr, tpr