import streamlit as st
import numpy as np
import pandas as pd
import joblib
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
    "<h1 style='text-align: center;'>🌋 Prediksi Kategori Gempa Indonesia</h1>",
    unsafe_allow_html=True
)

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = "model_gempa.pkl"
    if not os.path.exists(model_path):
        st.error("File model_gempa.pkl tidak ditemukan.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# --- Fungsi Prediksi ---
def hitung_indeks_gempa(depth, mag):
    return mag * (100 - depth) / 100

def prediksi_kategori(depth, mag):
    IG = hitung_indeks_gempa(depth, mag)
    pred = model.predict([[depth, mag, IG]])[0]
    return pred, IG

# --- Tab Menu ---
tab1, tab2 = st.tabs(["📊 Input Manual", "📂 Upload CSV"])

# ================= TAB 1: INPUT MANUAL =================
with tab1:
    st.subheader("Masukkan Data Gempa Secara Manual")

    col1, col2 = st.columns(2)
    with col1:
        depth = st.number_input("Kedalaman (km)", min_value=0.0, max_value=700.0, value=10.0, step=0.1)
    with col2:
        mag = st.number_input("Magnitudo (Skala Richter)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

    if st.button("Prediksi Gempa"):
        kategori, IG = prediksi_kategori(depth, mag)
        st.success(f"**Kategori Gempa:** {kategori}")
        st.info(f"**Indeks Gempa (IG):** {IG:.2f}")

# ================= TAB 2: UPLOAD CSV =================
with tab2:
    st.subheader("Unggah File CSV Data Gempa")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded_file:
        try:
            # Coba baca dengan delimiter koma, jika gagal coba tab
            try:
                df = pd.read_csv(uploaded_file)
            except:
                df = pd.read_csv(uploaded_file, delimiter="\t")

            df.columns = [c.strip().lower() for c in df.columns]  # normalisasi nama kolom

            required_cols = {"depth", "mag", "lat", "lon"}
            if not required_cols.issubset(df.columns):
                st.error("❌ Kolom wajib tidak ditemukan. Pastikan ada: depth, mag, lat, lon")
                st.stop()

            # Hitung Indeks Gempa & Prediksi
            df["indeks_gempa"] = df.apply(lambda x: hitung_indeks_gempa(x["depth"], x["mag"]), axis=1)
            X = df[["depth", "mag", "indeks_gempa"]]
            df["prediksi_kategori"] = model.predict(X)

            st.success("✅ File berhasil diproses!")
            st.dataframe(df[["depth", "mag", "lat", "lon", "indeks_gempa", "prediksi_kategori"]])

            # --- Visualisasi di Peta ---
            st.subheader("🗺️ Visualisasi Lokasi Gempa")
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position=["lon", "lat"],
                get_color="[255, 0, 0, 160]",
                get_radius=50000,
                pickable=True
            )
            view_state = pdk.ViewState(latitude=df["lat"].mean(), longitude=df["lon"].mean(), zoom=4)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")

# --- Footer ---
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:14px;'>
    Dibuat dengan ❤️ menggunakan Streamlit | Model: Random Forest<br>
    Data Gempa Indonesia 2008–2023
    </p>
    """,
    unsafe_allow_html=True
)
