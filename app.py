import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Muat model dan scaler
model = joblib.load('diabetes_model.sav')
scaler = joblib.load('scaler.pkl')

# Judul aplikasi
st.title("Aplikasi Prediksi Diabetes")

# Gambar untuk tampilan lebih menarik
image = Image.open('diabetes.jpg')  # Pastikan gambar ini ada di direktori yang sama
st.image(image, caption='Diabetes Prediction', use_column_width=True)

# Input data pengguna
pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, step=1)
glucose = st.number_input("Kadar Glukosa", min_value=0, step=1)
blood_pressure = st.number_input("Tekanan Darah", min_value=0, step=1)
skin_thickness = st.number_input("Ketebalan Kulit", min_value=0, step=1)
insulin = st.number_input("Kadar Insulin", min_value=0, step=1)
bmi = st.number_input("BMI", min_value=0.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
age = st.number_input("Umur", min_value=0, step=1)

# Tombol untuk memprediksi
if st.button("Prediksi"):
    # Buat array numpy dari data yang diinput
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

    # Standarisasi data
    std_data = scaler.transform(input_data)

    # Prediksi menggunakan model
    prediction = model.predict(std_data)

    # Hasil prediksi dengan tampilan yang lebih menarik
    if prediction[0] == 1:
        st.error('Pasien terkena diabetes', icon="⚠️")
    else:
        st.success('Pasien tidak terkena diabetes', icon="✅")
