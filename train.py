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
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='Path to input CSV dataset')
parser.add_argument('--model', type=str, default=None, help='Path to save trained model')
parser.add_argument('--log', type=str, default=None, help='Path to save training log (optional)')
args = parser.parse_args()

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv(args.data)
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train['species']
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test['species']

# -----------------------------
# Train model
# -----------------------------
model = DecisionTreeClassifier(max_depth=3, random_state=1)
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print(f"The accuracy of the Decision Tree is {accuracy:.3f}")

# -----------------------------
# Save model
# -----------------------------
os.makedirs(os.path.dirname(args.model), exist_ok=True)
joblib.dump(model, args.model)
print(f"Model saved to {args.model}")

# -----------------------------
# Save training log
# -----------------------------
if args.log:
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, 'w') as f:
        f.write(f"Accuracy: {accuracy:.3f}\n")
    print(f"Training log saved to {args.log}")
