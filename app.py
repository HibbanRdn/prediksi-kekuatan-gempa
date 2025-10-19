# app.py — Mobile-friendly Streamlit Prediksi Gempa (Manual & CSV)

import os
import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Kategori Gempa", layout="wide")
st.title("🌋 Prediksi Kategori Gempa Indonesia")
st.markdown("""
Aplikasi memprediksi kategori gempa berdasarkan **magnitudo & kedalaman**.  
Bisa input **manual** atau **upload CSV** dengan kolom berbeda-beda.  
Jika kolom hilang, digunakan nilai default.
""")

# =============================
# Load Model & LabelEncoder
# =============================
MODEL_FILE = "best_model_kategori_gempa.pkl"
LABEL_FILE = "label_encoder_kategori_gempa.pkl"

if not os.path.exists(MODEL_FILE) or not os.path.exists(LABEL_FILE):
    st.error("❌ Model atau LabelEncoder tidak ditemukan.")
    st.stop()

model = joblib.load(MODEL_FILE)
le = joblib.load(LABEL_FILE)

# =============================
# Input Data
# =============================
mode = st.radio("Pilih cara input:", ["Manual", "Upload CSV"])
input_data = None

# Default values
DEFAULTS = {"depth":50.0, "mag":5.0, "latitude":-2.5, "longitude":118.0}

# Manual input
if mode == "Manual":
    depth = st.number_input("Kedalaman (km)", 0.0, 700.0, DEFAULTS["depth"])
    mag = st.number_input("Magnitudo", 1.0, 9.5, DEFAULTS["mag"])
    latitude = st.number_input("Latitude", -11.0, 6.0, DEFAULTS["latitude"])
    longitude = st.number_input("Longitude", 94.0, 142.0, DEFAULTS["longitude"])
    input_data = pd.DataFrame({
        "depth":[depth],
        "mag":[mag],
        "latitude":[latitude],
        "longitude":[longitude]
    })

# CSV upload
elif mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df_csv = pd.read_csv(uploaded_file)

        # Auto-detect columns
        col_map = {}
        for c in df_csv.columns:
            c_low = c.lower()
            if c_low in ['lat','latitude']:
                col_map[c] = 'latitude'
            elif c_low in ['lon','longitude','long']:
                col_map[c] = 'longitude'
            elif c_low in ['depth','depth_km','kedalaman']:
                col_map[c] = 'depth'
            elif c_low in ['mag','magnitude','magnitudo']:
                col_map[c] = 'mag'

        df_csv.rename(columns=col_map, inplace=True)

        # Pastikan semua kolom penting ada
        for key, val in DEFAULTS.items():
            if key not in df_csv.columns:
                st.warning(f"Kolom '{key}' tidak ada, digunakan default: {val}")
                df_csv[key] = val

        input_data = df_csv[list(DEFAULTS.keys())].copy()
        input_data = input_data.fillna(pd.Series(DEFAULTS))

# =============================
# Prediksi & Visualisasi
# =============================
if input_data is not None and st.button("🔍 Prediksi"):
    try:
        # Prediksi kategori
        y_pred_num = model.predict(input_data)
        y_pred_str = le.inverse_transform(y_pred_num) if hasattr(le, "inverse_transform") else y_pred_num
        input_data['Kategori'] = y_pred_str

        st.subheader("📊 Hasil Prediksi")
        st.dataframe(input_data)

        # Probabilitas prediksi
        if hasattr(model, "predict_proba"):
            st.subheader("📈 Probabilitas Prediksi")
            prob = model.predict_proba(input_data)
            for i, idx in enumerate(input_data.index):
                fig, ax = plt.subplots()
                ax.bar(le.classes_, prob[i], color="skyblue")
                ax.set_ylabel("Probabilitas")
                ax.set_ylim(0,1)
                ax.set_title(f"Baris {i+1}")
                for j, v in enumerate(prob[i]):
                    ax.text(j, v+0.02, f"{v*100:.1f}%", ha="center")
                st.pyplot(fig)

        # Peta lokasi
        st.subheader("🗺️ Peta Lokasi")
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=input_data,
            get_position='[longitude, latitude]',
            get_color='[255,0,0]',
            get_radius=25000,
        )
        view_state = pdk.ViewState(
            latitude=input_data['latitude'].mean(),
            longitude=input_data['longitude'].mean(),
            zoom=5
        )
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

    except Exception as e:
        st.error(f"Kesalahan prediksi: {e}")

st.markdown("---")
st.markdown("Dikembangkan oleh **M. Hibban Ramadhan** • Model: Random Forest/XGBoost")
