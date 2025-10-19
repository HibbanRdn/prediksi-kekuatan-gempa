# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import gdown
import os
import pydeck as pdk

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Kategori Gempa Indonesia",
    page_icon="🌋",
    layout="wide"
)

# --- Judul Aplikasi ---
st.markdown(
    "<h1 style='text-align:center;'>🌋 Prediksi Kategori Gempa di Indonesia</h1>",
    unsafe_allow_html=True
)
st.write("""
Aplikasi ini memprediksi **kategori kekuatan gempa** berdasarkan **Magnitudo (Skala Richter)** dan **Kedalaman (km)**.  
Model ini dilatih menggunakan data gempa Indonesia periode **2008–2023** dengan pendekatan **CRISP-DM**.
""")

# --- URL Google Drive untuk Model & Encoder ---
MODEL_URL = "https://drive.google.com/uc?id=1tkqKxH3YQ9wNxMXpbchCF9wcZase-d_Z"
ENCODER_URL = "https://drive.google.com/uc?id=1pIYTRtB-i2LWXkJu-pqubornaGebSqP4"

MODEL_PATH = "best_model_kategori_gempa.pkl"
ENCODER_PATH = "label_encoder_kategori_gempa.pkl"

# --- Fungsi untuk download dan load model ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Mengunduh model dari Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    if not os.path.exists(ENCODER_PATH):
        with st.spinner("📥 Mengunduh label encoder dari Google Drive..."):
            gdown.download(ENCODER_URL, ENCODER_PATH, quiet=False)
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

try:
    model, le = load_model()
    st.success("✅ Model dan encoder berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- Input Parameter ---
st.header("🧾 Masukkan Parameter Gempa")

col1, col2 = st.columns(2)
with col1:
    mag = st.number_input("Magnitudo (Skala Richter)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
with col2:
    depth = st.number_input("Kedalaman (km)", min_value=0.0, max_value=700.0, value=10.0, step=1.0)

st.markdown("### 🌍 (Opsional) Koordinat Lokasi Gempa")
col3, col4 = st.columns(2)
with col3:
    lat = st.number_input("Latitude (Lintang)", min_value=-11.0, max_value=6.0, value=-2.0, step=0.1)
with col4:
    lon = st.number_input("Longitude (Bujur)", min_value=95.0, max_value=142.0, value=118.0, step=0.1)

# --- Tombol Prediksi ---
if st.button("🔍 Prediksi"):
    input_data = np.array([[depth, mag]])
    try:
        pred = model.predict(input_data)
        kategori = None

        # --- Deteksi tipe output model ---
        try:
            if pred.dtype.kind in ("i", "u", "f"):
                kategori = le.inverse_transform(pred.astype(int))[0]
            else:
                first = pred[0]
                kategori = first if first in getattr(le, "classes_", []) else first
        except Exception:
            kategori = pred[0]

        # --- Probabilitas / Confidence ---
        try:
            probs = model.predict_proba(input_data)
            prob_max = np.max(probs) * 100
            confidence = f"{prob_max:.2f}%"
        except Exception:
            confidence = "N/A"

        # --- Tampilkan hasil ---
        st.subheader("🌏 Hasil Prediksi:")
        st.success(f"Kategori Gempa: **{kategori}**")
        st.info(f"Tingkat keyakinan model: **{confidence}**")

        st.caption("Estimasi berdasarkan model pembelajaran mesin CRISP-DM dengan data gempa Indonesia 2008–2023.")

        # --- PETA INTERAKTIF ---
        st.markdown("### 🗺️ Visualisasi Lokasi Gempa")
        df_map = pd.DataFrame({
            "latitude": [lat],
            "longitude": [lon],
            "kategori": [kategori],
            "mag": [mag]
        })

        color_map = {
            "Gempa Mikro": [173, 216, 230],
            "Gempa Minor": [100, 181, 246],
            "Gempa Ringan": [72, 201, 176],
            "Gempa Sedang": [255, 204, 128],
            "Gempa Kuat": [255, 167, 38],
            "Gempa Dahsyat": [244, 67, 54],
        }
        color = color_map.get(kategori, [255, 255, 255])

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position='[longitude, latitude]',
            get_fill_color=color,
            get_radius=50000,
            pickable=True,
        )

        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=5)
        tooltip = {"text": "Kategori: {kategori}\nMagnitudo: {mag}"}

        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

        # --- Legenda Warna ---
        st.markdown("#### 🟢 Legenda Kategori Gempa")
        legend_html = """
        <div style='display: flex; flex-wrap: wrap; gap: 8px;'>
            <div style='background-color: rgb(173,216,230); padding:4px 8px; border-radius:6px;'>Gempa Mikro</div>
            <div style='background-color: rgb(100,181,246); padding:4px 8px; border-radius:6px;'>Gempa Minor</div>
            <div style='background-color: rgb(72,201,176); padding:4px 8px; border-radius:6px;'>Gempa Ringan</div>
            <div style='background-color: rgb(255,204,128); padding:4px 8px; border-radius:6px;'>Gempa Sedang</div>
            <div style='background-color: rgb(255,167,38); padding:4px 8px; border-radius:6px;'>Gempa Kuat</div>
            <div style='background-color: rgb(244,67,54); padding:4px 8px; border-radius:6px;'>Gempa Dahsyat</div>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

# --- Info Tambahan ---
with st.expander("ℹ️ Tentang Model"):
    st.markdown("""
    Model dikembangkan menggunakan metodologi **CRISP-DM**:
    1. **Business Understanding** — Prediksi tingkat kekuatan gempa.
    2. **Data Understanding** — Dataset katalog gempa Indonesia (2008–2023).
    3. **Data Preparation** — Fitur utama: `depth` dan `mag`.
    4. **Modeling** — Algoritma *RandomForest* dan *XGBoost*.
    5. **Evaluation** — Akurasi > 90% pada data uji.
    6. **Deployment** — Aplikasi Streamlit Cloud ini.
    """)

st.caption("Dibuat oleh: [Nama Kamu] — Proyek Prediksi Kategori Gempa Indonesia (CRISP-DM)")
