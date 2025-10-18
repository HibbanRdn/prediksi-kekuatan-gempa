import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import gdown
import pydeck as pdk  # Untuk visualisasi peta

# =============================
# DOWNLOAD MODEL DARI GOOGLE DRIVE
# =============================
MODEL_FILE = "model_klasifikasi_gempa.pkl"
MODEL_ID = "15ZNnOsumDX3iuOtu5eSdpKrksfd7bccu"  # ganti dengan ID model kamu di Google Drive
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

if not os.path.exists(MODEL_FILE):
    st.info("📥 Menyiapkan model...")
    gdown.download(MODEL_URL, MODEL_FILE, quiet=False)
    st.success("✅ Model berhasil diunduh!")

# =============================
# CONFIG & HEADER
# =============================
st.set_page_config(
    page_title="🌋 Prediksi Kategori Gempa",
    page_icon="🌋",
    layout="wide"
)

st.title("🌋 Prediksi Kategori Gempa Berdasarkan Kedalaman dan Magnitudo")
st.markdown(
    """
    Aplikasi ini menggunakan model **Random Forest Classifier** untuk memprediksi **kategori tingkat kekuatan gempa bumi** 
    berdasarkan **kedalaman (km)** dan **magnitudo (Skala Richter)**.  
    Kamu bisa:
    - 📤 Upload file CSV, atau  
    - ✍️ Masukkan data secara manual.
    """
)
st.divider()

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# =============================
# MODE INPUT
# =============================
tab1, tab2 = st.tabs(["📂 Upload CSV", "🧮 Input Manual"])

# =============================
# MODE 1: UPLOAD CSV
# =============================
with tab1:
    uploaded_file = st.file_uploader("Upload file CSV (harus berisi kolom 'depth' dan 'mag')", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("📄 Data yang Diunggah")
        st.dataframe(data.head())

        if model is not None:
            try:
                # Pastikan kolom sesuai dengan model
                expected_features = ["depth", "mag"]
                data = data.reindex(columns=expected_features, fill_value=0)

                pred = model.predict(data)
                data["Prediksi Kategori"] = pred

                st.subheader("🔮 Hasil Prediksi")
                st.dataframe(data.head())

                st.subheader("📊 Ringkasan Prediksi")
                counts = data["Prediksi Kategori"].value_counts()
                st.bar_chart(counts)

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
                ax.axis("equal")
                st.pyplot(fig)

                csv = data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 Download Hasil Prediksi (CSV)",
                    data=csv,
                    file_name="hasil_prediksi_gempa.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data: {e}")
    else:
        st.info("Silakan upload file CSV terlebih dahulu untuk memulai prediksi.")

# =============================
# MODE 2: INPUT MANUAL
# =============================
with tab2:
    st.subheader("🧮 Input Manual Gempa")

    if model is not None:
        depth = st.slider("Kedalaman Gempa (km)", 0, 700, 10, 1)
        mag = st.slider("Magnitudo Gempa (Skala Richter)", 0.0, 10.0, 5.0, 0.1)
        lat = st.slider("Lintang (Latitude)", -90.0, 90.0, 0.0, 0.1)
        lon = st.slider("Bujur (Longitude)", -180.0, 180.0, 0.0, 0.1)

        if st.button("Prediksi Sekarang 🔮"):
            try:
                input_df = pd.DataFrame([{"depth": depth, "mag": mag}])
                prediction = model.predict(input_df)[0]

                st.success(f"🌍 Kategori Gempa: {prediction}")
                st.metric(label="Prediksi Akhir", value=prediction)

                # Preview lokasi di peta
                st.subheader("🗺️ Lokasi Gempa")
                st.pydeck_chart(pdk.Deck(
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    initial_view_state=pdk.ViewState(
                        latitude=lat,
                        longitude=lon,
                        zoom=4,
                        pitch=0,
                    ),
                    layers=[
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=pd.DataFrame([{"lat": lat, "lon": lon}]),
                            get_position='[lon, lat]',
                            get_color='[255, 0, 0]',
                            get_radius=50000,
                            pickable=True
                        )
                    ]
                ))

            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {e}")
