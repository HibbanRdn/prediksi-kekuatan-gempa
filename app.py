import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import gdown

# =============================
# DOWNLOAD MODEL DARI GOOGLE DRIVE
# =============================
MODEL_FILE = "bestmodel_gempa.pkl"
MODEL_ID = "1OF8OtxUcD0fFdPp6Go0fqY5nxcYw8kIi"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

if not os.path.exists(MODEL_FILE):
    st.info("📥 Menyiapkan model..")
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
st.markdown(
    """
    Aplikasi ini memprediksi **kategori gempa bumi** berdasarkan data numerik menggunakan model *Random Forest* terlatih.
    Kamu bisa:
    - 📤 Upload file CSV, atau
    - ✍️ Masukkan data secara manual menggunakan slider.
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
# MODE 2: INPUT MANUAL DENGAN SLIDER
# =============================
with tab2:
    st.write("Masukkan data gempa secara manual dengan slider:")

    if model is not None:
        if hasattr(model, "feature_names_in_"):
            cols = model.feature_names_in_
        else:
            cols = ["magnitudo", "kedalaman", "lintang", "bujur"]

        with st.form("input_form_slider"):
            inputs = {}
            # Slider friendly untuk setiap kolom
            for col in cols:
                if col == "magnitudo":
                    inputs[col] = st.slider(
                        "Magnitudo Gempa (Skala Richter)",
                        min_value=0.0, max_value=10.0, value=5.0, step=0.1,
                        help="Skala Richter, contoh: 5.6"
                    )
                elif col == "kedalaman":
                    inputs[col] = st.slider(
                        "Kedalaman Gempa (km)",
                        min_value=0, max_value=700, value=10, step=1,
                        help="Kedalaman dari permukaan bumi dalam kilometer"
                    )
                elif col == "lintang":
                    inputs[col] = st.slider(
                        "Koordinat Lintang (Latitude)",
                        min_value=-90.0, max_value=90.0, value=0.0, step=0.1,
                        help="Positif = Utara, Negatif = Selatan"
                    )
                elif col == "bujur":
                    inputs[col] = st.slider(
                        "Koordinat Bujur (Longitude)",
                        min_value=-180.0, max_value=180.0, value=0.0, step=0.1,
                        help="Positif = Timur, Negatif = Barat"
                    )

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
