# ISL Word-Level Hand Sign Recognition (Custom Data)

This project allows you to collect custom Indian Sign Language (ISL) gestures using your webcam and recognize them in real time using a machine learning model.

---

## ✨ Features
- Collect hand landmark data for any ISL sign (e.g., Hello, Thank You)
- Train a model on your collected data
- Predict in real-time with audio feedback using gTTS

---

## 🔧 Setup Instructions

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the collector script** (collect ~30 samples per sign):
```bash
python collect_sign_data.py
```

3. **Train your model**:
```bash
python train_model.py
```

4. **Run real-time recognition**:
```bash
python real_time_recognizer.py
```

---

## 🎯 Tips
- Collect at least 30 samples per gesture for better results
- Perform the gesture clearly with one hand in front of webcam
