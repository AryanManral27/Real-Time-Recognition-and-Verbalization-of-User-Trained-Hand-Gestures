import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("isl_custom_landmarks.csv")
df = df.dropna()

X = df.iloc[:, :-1]
y = df.iloc[:, -1].astype(str)

print("[INFO] Classes:", sorted(y.unique()))

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as gesture_model.pkl")
