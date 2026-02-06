import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "students.csv")

data = pd.read_csv(data_path)

# Split features and target
X = data.drop("placed", axis=1)
y = data["placed"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# ============================
# Feature Importance
# ============================
import pandas as pd

feature_names = X.columns
importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)

print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save trained model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/placement_model.pkl")

print("\nModel saved successfully in model/placement_model.pkl")