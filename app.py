# app.py — Aplikasi Streamlit Prediksi Gempa
import os
import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
import gdown

# =============================
# 1️⃣ Konfigurasi Halaman
# =============================
st.set_page_config(page_title="Prediksi Potensi Gempa Kuat", layout="wide")
st.title("🌋 Prediksi Potensi Gempa Kuat di Indonesia")
st.markdown("""
Aplikasi ini memprediksi apakah sebuah kejadian gempa termasuk **gempa kuat (magnitudo ≥ 6)**
berdasarkan parameter spasial dan temporal.
Model dilatih menggunakan data historis gempa dari **BMKG & USGS** dengan pendekatan **CRISP-DM**.
""")

# =============================
# 2️⃣ Load / Download Model
# =============================
MODEL_FILE = "best_model_gempa.pkl"
MODEL_ID = "1kY3dqUueLood8WNPU5vK39GZI49D0FpH"  # ID Google Drive model milikmu
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

if not os.path.exists(MODEL_FILE):
    st.info("📥 Mengunduh model dari Google Drive...")
    try:
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False)
        st.success("✅ Model berhasil diunduh!")
    except Exception as e:
        st.error(f"Gagal mengunduh model: {e}")
        st.stop()

try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    st.error("❌ File model tidak ditemukan. Pastikan `best_model_gempa.pkl` tersedia.")
    st.stop()

# =============================
# 3️⃣ Input Pengguna
# =============================
st.sidebar.header("Masukkan Parameter Gempa")

latitude = st.sidebar.number_input("Latitude (Lintang)", min_value=-11.0, max_value=6.0, value=-2.5, step=0.1)
longitude = st.sidebar.number_input("Longitude (Bujur)", min_value=94.0, max_value=142.0, value=118.0, step=0.1)
depth = st.sidebar.number_input("Kedalaman (km)", min_value=0.0, max_value=700.0, value=50.0, step=1.0)
mag = st.sidebar.number_input("Magnitudo", min_value=1.0, max_value=9.5, value=5.0, step=0.1)
year = st.sidebar.number_input("Tahun", min_value=2000, max_value=2030, value=2023, step=1)
month = st.sidebar.slider("Bulan", 1, 12, 6)
hour = st.sidebar.slider("Jam", 0, 23, 12)

# =============================
# 4️⃣ Prediksi
# =============================
input_data = pd.DataFrame({
    "latitude": [latitude],
    "longitude": [longitude],
    "depth": [depth],
    "mag": [mag],
    "year": [year],
    "month": [month],
    "hour": [hour]
})

if st.button("🔍 Prediksi"):
    try:
        proba = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]
        kategori = "Gempa Kuat (≥6)" if pred == 1 else "Bukan Gempa Kuat (<6)"

        # Hasil Prediksi
        st.subheader("📊 Hasil Prediksi")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Kategori", value=kategori)
        with col2:
            st.metric(label="Probabilitas Gempa Kuat", value=f"{proba*100:.2f}%")

        # Visualisasi lokasi gempa
        st.subheader("🗺️ Peta Lokasi Gempa")
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=input_data,
            get_position='[longitude, latitude]',
            get_color='[255, 0, 0]' if pred == 1 else '[0, 128, 255]',
            get_radius=30000,
        )
        view_state = pdk.ViewState(latitude=latitude, longitude=longitude, zoom=5)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# =============================
# 5️⃣ Footer
# =============================
st.markdown("---")
st.markdown("Dikembangkan oleh **M. Hibban Ramadhan** • Data: BMKG & USGS • Model: Random Forest/XGBoost • Pipeline: CRISP-DM")
