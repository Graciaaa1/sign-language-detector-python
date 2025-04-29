import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize MediaPipe hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Capture hand data and labels
data = []
labels = []
actions = ['hello', 'thanks', 'yes', 'no']  # Add more signs as needed

cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    print("Press 'a' to collect data for 'hello', 's' for 'thanks', 'd' for 'yes', 'f' for 'no'")
    print("Press 'q' to quit data collection")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])  # 3D features
                key = cv2.waitKey(10)
                if key == ord('a'):
                    data.append(landmarks)
                    labels.append('hello')
                elif key == ord('s'):
                    data.append(landmarks)
                    labels.append('thanks')
                elif key == ord('d'):
                    data.append(landmarks)
                    labels.append('yes')
                elif key == ord('f'):
                    data.append(landmarks)
                    labels.append('no')

        cv2.imshow('Data Collection - Press q to quit', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save the data
print("Saving collected data...")
data_np = np.array(data)
labels_np = np.array(labels)
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data_np, 'labels': labels_np}, f)
print("Data saved!")

# Train the classifier
print("Training classifier...")
x_train, x_test, y_train, y_test = train_test_split(data_np, labels_np, test_size=0.2, stratify=labels_np)
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print("Model saved!")

