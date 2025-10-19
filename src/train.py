import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import argparse
import os

# -----------------------------
# Parse command-line arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Train Decision Tree on IRIS dataset")
parser.add_argument('--data', type=str, required=True, help='Path to input CSV dataset')
parser.add_argument('--model', type=str, default='models/model_default.pkl', help='Path to save/load trained model')
parser.add_argument('--log', type=str, default='logs/training_default.txt', help='Path to save training log')
parser.add_argument('--retrain', action='store_true', help='If set, load existing model and continue training')
args = parser.parse_args()

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv(args.data)
X = data[['sepal_length','sepal_width','petal_length','petal_width']]
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)

# -----------------------------
# Load or train model
# -----------------------------
os.makedirs(os.path.dirname(args.model), exist_ok=True)
os.makedirs(os.path.dirname(args.log), exist_ok=True)

if args.retrain and os.path.exists(args.model):
    # Load existing model and continue training (fit again)
    model = joblib.load(args.model)
    print(f"Loaded existing model from {args.model}")
else:
    # Create a new model
    model = DecisionTreeClassifier(max_depth=3, random_state=1)
    print("Training new model from scratch")

# Fit the model
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"The accuracy of the Decision Tree is {accuracy:.3f}")

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, args.model)
print(f"Model saved to {args.model}")

# -----------------------------
# Save training log
# -----------------------------
with open(args.log, 'w') as f:
    f.write(f"Accuracy: {accuracy:.3f}\n")
print(f"Training log saved to {args.log}")

