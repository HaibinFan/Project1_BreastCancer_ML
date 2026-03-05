from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os

def main():
    # load dataset
    data = load_breast_cancer(as_frame=True)
    df = data.frame

    X = df.drop("target", axis=1)
    y = df["target"]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # build pipeline with best hyperparameters from tuning
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            penalty='l2',
            C=0.1,
            class_weight='balanced',
            solver='liblinear',
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    print("Test Accuracy:", model.score(X_test, y_test))

    # save model to models folder
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'log_reg_breast_cancer.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print("Model saved as models/log_reg_breast_cancer.pkl")

if __name__ == "__main__":
    main()