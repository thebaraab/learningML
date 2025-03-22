import streamlit as st
from sklearn.linear_model import LogisticRegression
import numpy as np

# 🎓 Sample training data
X = [
    [1, 4, 2],   # [study hours, sleep hours, classes attended]
    [2, 5, 2],
    [3, 5, 3],
    [4, 6, 4],
    [5, 6, 4],
    [6, 7, 5]
]
y = [0, 0, 0, 1, 1, 1]

# 🚀 Train the model
model = LogisticRegression()
model.fit(X, y)

# 🌐 Streamlit app 
st.title("📘 Will You Pass? - Logistic Regression Predictor")

st.markdown("Enter your study habits below:")

# 🎛️ User inputs
study = st.slider("📚 Hours Studied", 0.0, 10.0, 3.0)
sleep = st.slider("😴 Hours Slept", 0.0, 10.0, 6.0)
classes = st.slider("🏫 Classes Attended", 0, 10, 3)

import matplotlib.pyplot as plt

# 🔮 Prediction
user_input = np.array([[study, sleep, classes]])
prediction = model.predict(user_input)[0]
confidence = model.predict_proba(user_input)[0][1]

# 📊 Visualizations
study_range = np.linspace(0, 10, 100)
sleep_range = np.linspace(0, 10, 100)
class_range = np.arange(0, 11)

study_probs = [model.predict_proba([[s, sleep, classes]])[0][1] for s in study_range]
sleep_probs = [model.predict_proba([[study, s, classes]])[0][1] for s in sleep_range]
class_probs = [model.predict_proba([[study, sleep, c]])[0][1] for c in class_range]

# 📈 Plotting all three curves
fig, ax = plt.subplots(figsize=(10, 6))

# Study curve
ax.plot(study_range, study_probs, label="📚 Varying Study Hours")
ax.scatter([study], [confidence], color="blue", s=80)

# Sleep curve
ax.plot(sleep_range, sleep_probs, label="😴 Varying Sleep Hours")

# Classes curve
ax.plot(class_range, class_probs, label="🏫 Varying Classes Attended", marker='o')

# Styling
ax.axhline(0.5, color="gray", linestyle="--", label="Threshold = 50%")
ax.set_xlabel("Feature Value")
ax.set_ylabel("Probability of Passing")
ax.set_ylim(-0.1, 1.1)
ax.set_title("📊 Logistic Regression: Feature Impact on Pass Probability")
ax.legend()
st.pyplot(fig)

st.subheader("📂 Upload CSV to Predict Multiple Students")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    import pandas as pd
    df = pd.read_csv(uploaded_file)

    # Expect columns: 'study', 'sleep', 'classes'
    inputs = df[['study', 'sleep', 'classes']]
    df['Pass_Prob'] = model.predict_proba(inputs)[:, 1]
    df['Prediction'] = model.predict(inputs)

    st.dataframe(df)

    st.download_button(
        "Download Predictions as CSV",
        df.to_csv(index=False).encode('utf-8'),
        "predictions.csv",
        "text/csv"
    )
