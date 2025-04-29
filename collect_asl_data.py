import cv2
import mediapipe as mp
import numpy as np
import pickle
import string

# Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

data = []
labels = []

alphabet = list(string.ascii_lowercase)  # ['a', 'b', 'c', ..., 'z']

print("📷 Data Collection Mode")
print("Press a-z keys to collect landmarks for that letter.")
print("Press 'q' to quit and save.")

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])  # 3D landmark points

                key = cv2.waitKey(10) & 0xFF
                if key in range(ord('a'), ord('z') + 1):
                    label = chr(key)
                    data.append(landmarks)
                    labels.append(label)
                    print(f"✅ Saved sample for: {label}")

        cv2.imshow('Collecting ASL Data - Press a-z to record, q to quit', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save data
with open('asl_data.pickle', 'wb') as f:
    pickle.dump({'data': np.array(data), 'labels': np.array(labels)}, f)

print("📦 Dataset saved as 'asl_data.pickle'")
