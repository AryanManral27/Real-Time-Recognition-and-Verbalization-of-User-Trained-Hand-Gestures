import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
from gtts import gTTS
import pygame
import tempfile
import time
import os
import threading

# Load trained model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev_prediction = ""
last_speak_time = time.time()
tts_lock = threading.Lock()  # ensure one TTS at a time

# Speak using pygame (threaded but controlled)
def speak(text):
    def _play():
        with tts_lock:
            try:
                path = os.path.join(tempfile.gettempdir(), f"speech_{int(time.time())}.mp3")
                tts = gTTS(text)
                tts.save(path)
                pygame.mixer.init()
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    continue
                os.remove(path)
            except Exception as e:
                print("[TTS ERROR]", e)
    threading.Thread(target=_play, daemon=True).start()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            break

    if len(landmarks) == 63:
        try:
            columns = [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
            prediction = model.predict(pd.DataFrame([landmarks], columns=columns))[0]

            # Show on screen
            cv2.putText(image, f"{prediction}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # Speak only if new prediction and 2 seconds passed
            if prediction != prev_prediction and time.time() - last_speak_time > 2:
                prev_prediction = prediction
                last_speak_time = time.time()
                speak(prediction)

        except Exception as e:
            print("[PREDICTION ERROR]", e)

    cv2.imshow("Real-Time Gesture Recognition", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
