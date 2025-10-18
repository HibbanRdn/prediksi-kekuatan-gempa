import os
import requests
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =============================
# DOWNLOAD MODEL DARI GOOGLE DRIVE (HANDLE FILE BESAR)
# =============================
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

MODEL_FILE = "bestmodel_gempa.pkl"
MODEL_ID = "1OF8OtxUcD0fFdPp6Go0fqY5nxcYw8kIi"

if not os.path.exists(MODEL_FILE):
    with st.spinner("Mengunduh model dari Google Drive..."):
        download_file_from_google_drive(MODEL_ID, MODEL_FILE)

# =============================
# CONFIG & HEADER
# =============================
st.set_page_config(
    page_title="🌋 Prediksi Kategori Gempa",
    page_icon="🌋",
    layout="wide"
)

st.title("🌋 Prediksi Kategori Gempa Berdasarkan Data Input")
st.markdown(
    """
    Aplikasi ini memprediksi **kategori gempa bumi** berdasarkan data numerik menggunakan model *Random Forest* terlatih.
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
# MODE 2: INPUT MANUAL
# =============================
with tab2:
    st.write("Masukkan data gempa secara manual di bawah ini untuk melakukan prediksi tunggal:")

    if model is not None:
        if hasattr(model, "feature_names_in_"):
            cols = model.feature_names_in_
        else:
            cols = ["magnitudo", "kedalaman", "lintang", "bujur"]

        with st.form("input_form"):
            inputs = {}
            for col in cols:
                inputs[col] = st.number_input(f"Masukkan nilai untuk {col}", value=0.0)
            submit = st.form_submit_button("Prediksi Sekarang 🔮")

        if submit:
            try:
                input_df = pd.DataFrame([inputs])
                prediction = model.predict(input_df)[0]

                st.success(f"🌍 **Kategori Gempa:** {prediction}")
                st.metric(label="Prediksi Akhir", value=prediction)

            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {e}")
    else:
        st.warning("Model belum tersedia. Pastikan file `bestmodel_gempa.pkl` ada di direktori yang sama.")
