import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import gdown
import pydeck as pdk
import numpy as np

# =============================
# DOWNLOAD MODEL DARI GOOGLE DRIVE
# =============================
MODEL_FILE = "bestmodel_gempa.pkl"
MODEL_ID = "1OF8OtxUcD0fFdPp6Go0fqY5nxcYw8kIi"
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

st.title("🌋 Prediksi Kategori Gempa Berdasarkan Data Input")
st.markdown("""
Aplikasi ini memprediksi **kategori gempa bumi** berdasarkan data numerik menggunakan model *Random Forest* terlatih.
Kamu bisa:
- 📤 Upload file CSV, atau
- ✍️ Masukkan data secara manual menggunakan slider dan lihat preview lokasi di peta.
""")
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
# FUNCTION FEATURE IMPORTANCE
# =============================
def plot_feature_importance(pipe, feature_names, title="Feature Importance"):
    if 'clf' in pipe.named_steps:
        clf = pipe.named_steps['clf']
    else:
        clf = pipe

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        idx = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar([feature_names[i] for i in idx], importances[idx])
        ax.set_title(title)
        ax.set_xticklabels([feature_names[i] for i in idx], rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Model tidak mendukung feature_importances_")

# =============================
# MODE INPUT
# =============================
tab1, tab2 = st.tabs(["📂 Upload CSV", "🧮 Input Manual"])

# =============================
# MODE 1: UPLOAD CSV
# =============================
with tab1:
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("📄 Data yang Diunggah")
        st.dataframe(data.head())

        if model is not None:
            try:
                if hasattr(model, "feature_names_in_"):
                    expected_features = model.feature_names_in_
                    data = data.reindex(columns=expected_features, fill_value=0)

                pred = model.predict(data)
                data["Prediksi"] = pred

                st.subheader("🔮 Hasil Prediksi")
                st.dataframe(data.head())

                st.subheader("📊 Ringkasan Prediksi")
                counts = data["Prediksi"].value_counts()
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
# MODE 2: INPUT MANUAL DENGAN SLIDER & PETA GRATIS
# =============================
with tab2:
    st.subheader("🧮 Input Manual Gempa")

    if model is not None:
        # Ambil nama fitur dari pipeline
        if hasattr(model, "feature_names_in_"):
            FEATURES = model.feature_names_in_
        else:
            FEATURES = ["lintang","bujur","magnitudo","kedalaman","tahun","bulan","hari"]

        # Slider/input manual
        magnitudo = st.slider("Magnitudo Gempa (Skala Richter)", 0.0, 10.0, 5.0, 0.1)
        kedalaman = st.slider("Kedalaman Gempa (km)", 0, 700, 10, 1)
        lintang = st.slider("Koordinat Lintang (Latitude)", -90.0, 90.0, 0.0, 0.1)
        bujur = st.slider("Koordinat Bujur (Longitude)", -180.0, 180.0, 0.0, 0.1)
        tahun = st.number_input("Tahun", min_value=1900, max_value=2100, value=2025)
        bulan = st.slider("Bulan", 1, 12, 1)
        hari = st.slider("Hari", 1, 31, 1)

        if st.button("Prediksi Sekarang 🔮"):
            try:
                # Buat input sesuai urutan FEATURES
                X_new = []
                for f in FEATURES:
                    if f == "magnitudo":
                        X_new.append(magnitudo)
                    elif f == "kedalaman":
                        X_new.append(kedalaman)
                    elif f == "lintang":
                        X_new.append(lintang)
                    elif f == "bujur":
                        X_new.append(bujur)
                    elif f == "tahun":
                        X_new.append(tahun)
                    elif f == "bulan":
                        X_new.append(bulan)
                    elif f == "hari":
                        X_new.append(hari)
                    else:
                        X_new.append(0)  # default jika fitur tidak ada

                prediction = model.predict([X_new])[0]

                st.success(f"🌍 Kategori Gempa: {prediction}")
                st.metric(label="Prediksi Akhir", value=prediction)

                # Preview peta
                st.subheader("🗺️ Lokasi Gempa")
                st.pydeck_chart(pdk.Deck(
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    initial_view_state=pdk.ViewState(
                        latitude=lintang,
                        longitude=bujur,
                        zoom=4,
                        pitch=0,
                    ),
                    layers=[
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=pd.DataFrame([{"lintang": lintang, "bujur": bujur}]),
                            get_position='[bujur, lintang]',
                            get_color='[255, 0, 0]',
                            get_radius=50000,
                            pickable=True
                        )
                    ]
                ))

                # Tampilkan feature importance jika ada
                plot_feature_importance(model, FEATURES, title="Feature Importance - Model")

            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {e}")
    else:
        st.warning("Model belum tersedia. Pastikan file pipeline ada.")
