import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv(r'C:\Users\sriya\OneDrive\Desktop\traffic_pothoe_project\traffic\Banglore_traffic_Dataset.csv')

# Prepare the data
X = data[['Date', 'Road/Intersection Name', 'Traffic Volume']]
y = data['Congestion Level']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a decision tree model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100}%')

# Save the model for later use
import joblib
joblib.dump(model, 'traffic_model.pkl')