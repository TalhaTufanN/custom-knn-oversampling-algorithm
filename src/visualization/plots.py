import matplotlib.pyplot as plt
def plot_results(
    dataset_name,
    X, y,
    X_res, y_res,
    dt_m_bef, dt_m_aft,
    knn_m_bef, knn_m_aft
):
# --- PHASE 5: VISUALIZATION ---
    plt.figure(figsize=(18, 6))

    # Plot 1: Scatter (Before)
    plt.subplot(1, 3, 1)
    plt.title(f"1. Before Oversampling\n({dataset_name})")
    plt.scatter(X[y==0][:,0], X[y==0][:,1], c='#1f77b4', label="Majority", alpha=0.5, s=20)
    plt.scatter(X[y==1][:,0], X[y==1][:,1], c='#d62728', label="Minority", alpha=0.9, s=40)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)

    # Plot 2: Scatter (After)
    plt.subplot(1, 3, 2)
    plt.title(f"2. After Oversampling\n(Balanced)")
    plt.scatter(X_res[y_res==0][:,0], X_res[y_res==0][:,1], c='#1f77b4', label="Majority", alpha=0.5, s=20)
    plt.scatter(X_res[y_res==1][:,0], X_res[y_res==1][:,1], c='#d62728', label="Minority+Syn", alpha=0.9, s=40)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)

    # Plot 3: ROC Curve Comparison
    plt.subplot(1, 3, 3)
    plt.title(f"3. ROC Analysis (Before vs After)")

    # DT Lines (Blue)
    plt.plot(dt_m_bef[4], dt_m_bef[5], color='blue', linestyle=':', alpha=0.6, label=f'DT Before')
    plt.plot(dt_m_aft[4], dt_m_aft[5], color='blue', linewidth=2, label=f'DT After')

    # KNN Lines (Green)
    plt.plot(knn_m_bef[4], knn_m_bef[5], color='green', linestyle=':', alpha=0.6, label=f'KNN Before')
    plt.plot(knn_m_aft[4], knn_m_aft[5], color='green', linewidth=2, label=f'KNN After')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", fontsize='x-small')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

