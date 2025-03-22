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

# 📊 Visual curve based on "study hours" (hold sleep/classes constant)
study_range = np.linspace(0, 10, 100) # 0 to 10, 100 evenly spaced values.
probs = []

# Plot
fig, ax = plt.subplots()
ax.plot(study_range, probs, label="Pass Probability (varying study)")
ax.axhline(0.5, color="gray", linestyle="--", label="Threshold = 50%")
ax.scatter([study], [confidence], color="blue", s=100, label="Your Prediction")

ax.set_xlabel("Study Hours")
ax.set_ylabel("Probability of Passing")
ax.set_ylim(-0.1, 1.1)
ax.set_title("📈 Sigmoid Curve: Logistic Regression")
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
