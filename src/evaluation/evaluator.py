from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.models.trainers import train_and_get_metrics
from src.sampling.custom_oversample import custom_oversample
from src.visualization.plots import plot_results

def evaluate_dataset(dataset_name, X, y, k=5):
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*60}")

    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)

    # Before oversampling
    dt_m_bef = train_and_get_metrics(X, y, dt)
    knn_m_bef = train_and_get_metrics(X, y, knn)

    # Oversampling
    X_res, y_res = custom_oversample(X, y, k)

    # After oversampling
    dt_m_aft = train_and_get_metrics(X_res, y_res, dt)
    knn_m_aft = train_and_get_metrics(X_res, y_res, knn)

    # Visualization
    plot_results(
        dataset_name,
        X, y,
        X_res, y_res,
        dt_m_bef, dt_m_aft,
        knn_m_bef, knn_m_aft
    )
        # --- PHASE 4: SCORE TABLES ---
    print(f"\n   [TABLE 1] DECISION TREE PERFORMANCE")
    print(f"   {'-'*60}")
    print(f"   {'METRIC':<15} | {'BEFORE':<15} | {'AFTER':<15}")
    print(f"   {'-'*60}")
    print(f"   {'Precision':<15} | {dt_m_bef[0]:.4f}          | {dt_m_aft[0]:.4f}")
    print(f"   {'Recall':<15} | {dt_m_bef[1]:.4f}          | {dt_m_aft[1]:.4f}")
    print(f"   {'F1-Score':<15} | {dt_m_bef[2]:.4f}          | {dt_m_aft[2]:.4f}")
    print(f"   {'AUC Score':<15} | {dt_m_bef[3]:.4f}          | {dt_m_aft[3]:.4f}")
    print(f"   {'-'*60}")

    print(f"\n   [TABLE 2] KNN CLASSIFIER PERFORMANCE")
    print(f"   {'-'*60}")
    print(f"   {'METRIC':<15} | {'BEFORE':<15} | {'AFTER':<15}")
    print(f"   {'-'*60}")
    print(f"   {'Precision':<15} | {knn_m_bef[0]:.4f}          | {knn_m_aft[0]:.4f}")
    print(f"   {'Recall':<15} | {knn_m_bef[1]:.4f}          | {knn_m_aft[1]:.4f}")
    print(f"   {'F1-Score':<15} | {knn_m_bef[2]:.4f}          | {knn_m_aft[2]:.4f}")
    print(f"   {'AUC Score':<15} | {knn_m_bef[3]:.4f}          | {knn_m_aft[3]:.4f}")
    print(f"   {'-'*60}")
