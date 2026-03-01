from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

def main():
    # Load dataset
    data = load_breast_cancer(as_frame=True)
    df = data.frame

    X = df.drop("target", axis=1)
    y = df["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build pipeline
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000)
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    test_accuracy = model.score(X_test, y_test)
    print("Test Accuracy:", test_accuracy)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Accuracy:", np.mean(cv_scores))


if __name__ == "__main__":
    main()