from src.config import DATASETS
from src.datasets.generator import create_dataset
from src.evaluation.evaluator import evaluate_dataset

def main():
    for cfg in DATASETS:
        X, y = create_dataset(
            cfg["type"],
            cfg["samples"],
            cfg["imbalance"]
        )
        evaluate_dataset(cfg["name"], X, y)

if __name__ == "__main__":
    main()
