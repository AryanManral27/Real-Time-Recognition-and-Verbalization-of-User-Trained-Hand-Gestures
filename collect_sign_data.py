import cv2
import mediapipe as mp
import pandas as pd
import os

label = input("Enter the label for the gesture (e.g., Hello): ").strip()
samples = int(input("How many samples do you want to collect?: "))
output_csv = "isl_custom_landmarks.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
collected = 0
data = []

print(f"[INFO] Starting collection for '{label}'... Press Q to quit early.")

while collected < samples:
    ret, frame = cap.read()
    if not ret:
        continue
    image = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks.append(label)
            data.append(landmarks)
            collected += 1
            print(f"[INFO] Collected sample {collected}/{samples}")
            break

    cv2.putText(image, f"Label: {label} | Count: {collected}/{samples}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Collecting Gesture", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
columns = [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]] + ["label"]
df_new = pd.DataFrame(data, columns=columns)

if os.path.exists(output_csv):
    df_existing = pd.read_csv(output_csv)
    df_new = pd.concat([df_existing, df_new], ignore_index=True)

df_new.to_csv(output_csv, index=False)
print(f"✅ Saved {collected} samples to {output_csv}")
