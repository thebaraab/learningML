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

# 🔮 Prediction
user_input = np.array([[study, sleep, classes]])
prediction = model.predict(user_input)[0]
confidence = model.predict_proba(user_input)[0][1]

# 🧾 Output
st.subheader("Result:")
if prediction == 1:
    st.success(f"✅ You will probably PASS! (Confidence: {round(confidence*100)}%)")
else:
    st.error(f"❌ You might FAIL. (Confidence: {round(confidence*100)}%)")
