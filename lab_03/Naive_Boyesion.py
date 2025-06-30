import pandas as pd
import numpy as np
from collections import defaultdict

# ---------- Load and Clean Data ----------
df = pd.read_csv("Data/laptop_buy_data.csv")
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces

# ---------- Naive Bayes Training ----------
class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.features = X.columns
        self.class_probs = {}
        self.feature_probs = {}

        # Prior probabilities
        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_probs[cls] = len(X_cls) / len(X)

            self.feature_probs[cls] = {}
            for feature in self.features:
                feature_vals = X[feature].unique()
                self.feature_probs[cls][feature] = {}

                for val in feature_vals:
                    count = len(X_cls[X_cls[feature] == val])
                    # Laplace smoothing
                    prob = (count + 1) / (len(X_cls) + len(feature_vals))
                    self.feature_probs[cls][feature][val] = prob

    # ---------- Prediction ----------
    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            class_scores = {}

            for cls in self.classes:
                score = np.log(self.class_probs[cls])
                for feature in self.features:
                    val = row[feature]
                    prob = self.feature_probs[cls][feature].get(val, 1e-6)
                    score += np.log(prob)
                class_scores[cls] = score

            best_class = max(class_scores, key=class_scores.get)
            predictions.append(best_class)
        return predictions

# ---------- Main ----------
if __name__ == "__main__":
    # Split features and target
    target = 'Class'
    X = df.drop(columns=[target])
    y = df[target]

    model = NaiveBayesClassifier()
    model.fit(X, y)

    print("=== Testing on Training Data ===")
    predictions = model.predict(X)

    for i, (features, actual, pred) in enumerate(zip(X.values, y, predictions)):
        print(f"Example {i+1}: Predicted = {pred}, Actual = {actual}, Features = {features}")

    accuracy = np.mean(predictions == y)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
