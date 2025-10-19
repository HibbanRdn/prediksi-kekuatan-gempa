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
    "<h1 style='text-align:left; margin-bottom:0;'>🌋 Prediksi Kategori Gempa di Indonesia</h1>",
    unsafe_allow_html=True
)
st.write("""
Apps ini memprediksi kategori gempa berdasarkan **kedalaman (km)** dan **magnitudo (Skala Richter)**.  
Model dilatih menggunakan algoritma *Random Forest/XGBoost* dengan data gempa Indonesia tahun **2008–2023**.
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

# --- Spacer kecil untuk jarak visual ---
st.markdown("<br>", unsafe_allow_html=True)

# --- Input Parameter ---
st.header("🧾 Masukkan Parameter Gempa")

col1, col2 = st.columns(2)
with col1:
    mag = st.number_input("Magnitudo (Skala Richter)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
with col2:
    depth = st.number_input("Kedalaman (km)", min_value=0.0, max_value=700.0, value=10.0, step=1.0)

st.markdown("### 🌍 Koordinat Lokasi Gempa")
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

        # --- Hasil Prediksi ---
        st.subheader("🌏 Hasil Prediksi:")
        st.success(f"Kategori Gempa: **{kategori}**")
        st.info(f"Tingkat keyakinan model: **{confidence}**")

        st.caption("Estimasi berdasarkan model pembelajaran mesin CRISP-DM dengan data gempa Indonesia 2008–2023.")

        # --- Peta Interaktif ---
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

# --- Spacer sebelum bagian info ---
st.markdown("<br><hr><br>", unsafe_allow_html=True)

# --- Info Tambahan (tanpa emoji) ---
with st.expander("Info Tentang Model"):
    st.markdown("""
    Metodologi pengembangan model mengikuti tahapan **CRISP-DM (Cross Industry Standard Process for Data Mining)**:
    
    **1. Business Understanding** — Menentukan tujuan prediksi tingkat kekuatan gempa di Indonesia.  
    **2. Data Understanding** — Mengumpulkan dan menganalisis data gempa BMKG & USGS (2008–2023).  
    **3. Data Preparation** — Menyaring dan menormalkan fitur utama: `kedalaman` dan `magnitudo`.  
    **4. Modeling** — Melatih model menggunakan algoritma *RandomForest* dan *XGBoost*.  
    **5. Evaluation** — Mengukur akurasi model (>90%) pada data uji.  
    **6. Deployment** — Implementasi ke aplikasi Streamlit untuk prediksi interaktif.
    """)

# --- Footer Dinamis ---
import datetime
year = datetime.datetime.now().year

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style='text-align: center; color: gray; font-size: 0.9rem; margin-top: 10px;'>
        © {year} <b>M. Hibban Ramadhan</b> — Proyek <i>Prediksi Kategori Gempa Indonesia</i><br>
        Dibangun menggunakan <a href='https://streamlit.io' target='_blank' style='color: #4b9cd3; text-decoration: none;'>Streamlit</a>
    </div>
    """,
    unsafe_allow_html=True
)
