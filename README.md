# ✌️ Real-Time Recognition and Verbalization of User-Trained Hand Gestures

This project is a real-time ISL hand gesture recognition system using MediaPipe and a Random Forest model trained on custom data to predict gestures like Hello, Yes, and I Love You. It provides both text and voice feedback via Pygame, enabling accessible communication for people with hearing or speech impairments

---

## 🎯 Core Features
- Custom Gesture Collection – Capture hand landmark data for any ISL sign (e.g., Hello, Thank You)
- User-Trained Model – Train a Random Forest classifier on your own collected gesture dataset
- Real-Time Gesture Prediction – Recognize gestures instantly using MediaPipe for 3D hand landmark extraction
- Audio Feedback – Convert recognized gestures into speech using gTTS or Pygame
- On-Screen Text Output – Display the predicted gesture text in real time
- Supports Multiple Gestures – Works with gestures like Yes, No, Victory, OK, and I Love You
- Accessible Communication – Designed to assist individuals with hearing or speech impairments

---

## 🛠 How to Set Up and Run

1. **Install Required Dependencies**
- Run the following command to install all necessary packages:
```bash
pip install -r requirements.txt
```

2. **Collect Gesture Data**
- Capture approximately 30 samples per gesture by running:
```bash
python collect_sign_data.py
```

3. **Train the Model**
- Use the collected dataset to train your custom model:
```bash
python train_model.py
```

4. **Run Real-Time Recognition**
- Start the gesture recognition system with:
```bash
python real_time_recognizer.py
```

---

## 📁 Files
- `README.md` → Contains instructions and documentation for the project
- `collect_sign_data.py` → Collects hand gesture landmarks from the webcam and saves them with labels in a CSV file
- `gesture_model.pkl` → The saved user-trained machine learning model used for predicting gestures
- `isl_custom_landmarks.csv` → Stores collected hand landmark data with labels; used for training the model
- `real_time_recognizer.py` → Runs the real-time gesture recognition system and converts predictions into speech 
- `requirements.txt` → Lists all Python libraries required to run the project (e.g., OpenCV, MediaPipe)
- `train_model.py` → Loads the CSV data, trains the Random Forest model, and saves it as gesture_model.pkl  

---

## 💡 Future Scope
- Add more gestures to expand the vocabulary
- Support multiple languages in audio feedback
- Develop mobile and web-based app versions
- Integrate into gesture-based communication systems

---

## 🚀 Best Practices for Accuracy
- Gather a minimum of 30–50 samples per gesture for better prediction accuracy
- Ensure your hand is clearly visible to the webcam
- Perform each gesture in a consistent position and manner
- Use good lighting to help the landmark detector work effectively
