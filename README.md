# Custom KNN-Based Oversampling Algorithm

This project implements a custom oversampling technique designed to handle class imbalance in machine learning datasets. Developed as part of the **COME401 Data Mining** course at Uskudar University, this tool uses K-Nearest Neighbors (KNN) and linear interpolation to generate high-quality synthetic samples for minority classes.

## 🚀 Overview

Class imbalance is a common challenge in data mining where one class significantly outperforms another in frequency, leading to biased models. This project provides a custom implementation of an oversampling strategy, similar to SMOTE, to balance datasets effectively before training.

### How it Works
[cite_start]The algorithm follows a 4-step process for each synthetic sample generated [cite: 76-84]:
1. **Random Selection:** Picks a point $p$ from the minority class.
2. **Neighbor Identification:** Finds the $k$ nearest neighbors of $p$ within the minority class.
3. **Random Neighbor Selection:** Randomly selects one neighbor from the identified group.
4. **Linear Interpolation:** Creates a new synthetic sample $s$ using the formula:
   $$s = p + (neighbor - p) \times gap$$
   *Where $gap$ is a random value between 0 and 1.*



## ✨ Features

* [cite_start]**Custom Algorithm:** Independent implementation using `numpy` and `sklearn.neighbors`[cite: 16, 19].
* [cite_start]**Synthetic Data Generation:** Includes a utility to generate 5 types of geometric distributions: Circles, Moons, Blobs, Mixed Classification, and Gaussian Quantiles [cite: 18, 216-221].
* [cite_start]**Performance Benchmarking:** Compares **Decision Tree** and **KNN** classifiers before and after resampling [cite: 20-21, 134-137].
* [cite_start]**Visualization:** Generates 3-panel plots showing data distribution (Before/After) and ROC curve analysis [cite: 17, 154-174].

## 📊 Results

The algorithm consistently improves the **Recall** and **F1-Score** for imbalanced datasets. [cite_start]For example, in the Gaussian Quantiles test, the KNN Recall improved from **0.82 to 0.97** after applying the custom oversampling[cite: 233].

| Metric | Before Resampling | After Resampling |
| :--- | :---: | :---: |
| Precision | 0.9756 | 0.9632 |
| Recall | 0.8163 | 0.9704 |
| F1-Score | 0.8889 | 0.9668 |

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/username/custom-knn-oversampling-algorithm.git](https://github.com/username/custom-knn-oversampling-algorithm.git)
