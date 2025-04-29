import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import sys

# Load dataset
with open('asl_data.pickle', 'rb') as f:
    dataset = pickle.load(f)

data = dataset['data']
labels = dataset['labels']

# Count label frequency
label_counts = Counter(labels)
print("📊 Label counts before filtering:", label_counts)

# Filter out labels with <2 samples
filtered_data = []
filtered_labels = []

for d, l in zip(data, labels):
    if label_counts[l] >= 2:
        filtered_data.append(d)
        filtered_labels.append(l)

# Check if we have enough data
if len(filtered_data) == 0:
    print("❌ Not enough data to train. Collect at least 2 samples per letter.")
    sys.exit()

filtered_data = np.array(filtered_data)
filtered_labels = np.array(filtered_labels)

# Recheck label counts
filtered_counts = Counter(filtered_labels)
print("✅ Label counts after filtering:", filtered_counts)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    filtered_data, filtered_labels, test_size=0.2, stratify=filtered_labels, random_state=42
)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and evaluate
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {acc * 100:.2f}%")

# Save the trained model
with open('asl_model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("💾 Model saved to 'asl_model.p'")
