import streamlit as st
import pickle
import pandas as pd

# Load model and scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter the patient's details below to predict heart disease.")

# ---------- Input Fields ----------

age = st.number_input("Age", min_value=1, max_value=120, value=45)

sex = st.selectbox("Sex", ["Male", "Female"])

cp = st.selectbox("Chest Pain Type", ["typical angina",
                                      "atypical angina",
                                      "non-anginal",
                                      "asymptomatic"])

trestbps = st.number_input("Resting Blood Pressure")

chol = st.number_input("Cholesterol")

fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", ["False", "True"])

restecg = st.selectbox("Rest ECG",
                       ["normal",
                        "st-t abnormality",
                        "lv hypertrophy"])

thalch = st.number_input("Maximum Heart Rate")

exang = st.selectbox("Exercise Induced Angina",
                     ["No","Yes"])

oldpeak = st.number_input("Old Peak")

slope = st.selectbox("Slope",
                     ["upsloping",
                      "flat",
                      "downsloping"])

ca = st.number_input("Major Vessels", min_value=0, max_value=4)

thal = st.selectbox("Thal",
                    ["normal",
                     "fixed defect",
                     "reversible defect"])

sex = 1 if sex=="Male" else 0

cp_dict = {
    "typical angina":0,
    "atypical angina":1,
    "non-anginal":2,
    "asymptomatic":3
}
cp = cp_dict[cp]

fbs = 1 if fbs=="True" else 0

restecg_dict = {
    "normal":0,
    "st-t abnormality":1,
    "lv hypertrophy":2
}
restecg = restecg_dict[restecg]

exang = 1 if exang=="Yes" else 0

slope_dict = {
    "upsloping":0,
    "flat":1,
    "downsloping":2
}
slope = slope_dict[slope]

thal_dict = {
    "normal":0,
    "fixed defect":1,
    "reversible defect":2
}
thal = thal_dict[thal]


if st.button("Predict"):

    data = [[age,
             sex,
             cp,
             trestbps,
             chol,
             fbs,
             restecg,
             thalch,
             exang,
             oldpeak,
             slope,
             ca,
             thal]]

    df = pd.DataFrame(data)

    scaled = scaler.transform(df)

    prediction = model.predict(scaled)[0]

    probability = model.predict_proba(scaled)[0][1]

    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")

    st.write(f"Prediction Probability: **{probability:.2%}**")