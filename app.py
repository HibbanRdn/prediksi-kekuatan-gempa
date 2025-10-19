import streamlit as st
import numpy as np
import pandas as pd
import joblib
import gdown
import os
import pydeck as pdk
import datetime
import matplotlib.pyplot as plt

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
Model dilatih menggunakan algoritma *Random Forest/XGBoost* dengan data gempa Indonesia tahun **2008–2023**. (**Data historis BMKG & USGS**)
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

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# === PILIH MODE INPUT =====================================
# ==========================================================
st.sidebar.title("🧭 Mode Input")
mode = st.sidebar.radio(
    "Pilih metode input data:",
    ("Input Langsung", "Upload File CSV")
)

# ==========================================================
# === MODE 1: INPUT LANGSUNG ===============================
# ==========================================================
if mode == "Input Langsung":
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

    if st.button("🔍 Prediksi"):
        input_data = np.array([[depth, mag]])
        try:
            pred = model.predict(input_data)
            kategori = None

            try:
                if pred.dtype.kind in ("i", "u", "f"):
                    kategori = le.inverse_transform(pred.astype(int))[0]
                else:
                    first = pred[0]
                    kategori = first if first in getattr(le, "classes_", []) else first
            except Exception:
                kategori = pred[0]

            try:
                probs = model.predict_proba(input_data)[0]
                df_probs = pd.DataFrame({
                    "Kategori": le.classes_,
                    "Probabilitas (%)": (probs * 100).round(2)
                }).sort_values(by="Probabilitas (%)", ascending=False)
            except Exception:
                df_probs = None

            st.subheader("🌏 Hasil Prediksi:")
            st.success(f"Kategori Gempa: **{kategori}**")
            if df_probs is not None:
                st.write("### 🔢 Probabilitas Tiap Kategori")
                st.dataframe(df_probs)

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

        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")

# ==========================================================
# === MODE 2: UPLOAD FILE CSV ==============================
# ==========================================================
else:
    st.header("📂 Upload File CSV")
    st.markdown("Unggah file CSV dengan kolom: **tgl, ot, lat, lon, depth, mag, remark**")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # --- Coba baca CSV ---
            try:
                df = pd.read_csv(uploaded_file, delimiter=",")
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep="\t")

            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            st.write("### 🧾 Data Awal")
            st.dataframe(df.head())

            required_cols = {"depth", "mag", "lat", "lon"}
            if not required_cols.issubset(df.columns):
                st.error("❌ Kolom wajib tidak ditemukan. Pastikan ada: depth, mag, lat, lon")
            else:
                X = df[["depth", "mag"]].values
                preds = model.predict(X)
                probs = model.predict_proba(X)

                kategori = [
                    le.inverse_transform([int(p)])[0] if str(p).isdigit() else str(p)
                    for p in preds
                ]
                df["Prediksi Kategori"] = kategori
                df["Probabilitas Maks (%)"] = (np.max(probs, axis=1) * 100).round(2)

                st.success("✅ Prediksi selesai!")

                tampil_cols = [col for col in ["tgl", "lat", "lon", "depth", "mag", "remark", "Prediksi Kategori", "Probabilitas Maks (%)"] if col in df.columns]
                st.dataframe(df[tampil_cols])

                # --- Statistik Distribusi ---
                st.markdown("### 📊 Distribusi Kategori Gempa")
                summary = df["Prediksi Kategori"].value_counts().reset_index()
                summary.columns = ["Kategori", "Jumlah"]
                summary["Persentase (%)"] = (summary["Jumlah"] / summary["Jumlah"].sum() * 100).round(2)
                st.dataframe(summary)

                color_map_hex = {
                    "Gempa Mikro": "#ADD8E6",
                    "Gempa Minor": "#64B5F6",
                    "Gempa Ringan": "#48C9B0",
                    "Gempa Sedang": "#FFCC80",
                    "Gempa Kuat": "#FFA726",
                    "Gempa Dahsyat": "#F44336",
                }
                colors = [color_map_hex.get(k, "#CCCCCC") for k in summary["Kategori"]]

                fig, ax = plt.subplots()
                ax.pie(summary["Jumlah"], labels=summary["Kategori"], colors=colors, autopct="%1.1f%%", startangle=90)
                ax.axis("equal")
                st.pyplot(fig)

                # --- Unduh hasil ---
                csv_out = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Unduh Hasil Prediksi CSV",
                    data=csv_out,
                    file_name="hasil_prediksi_gempa.csv",
                    mime="text/csv"
                )

                # --- Peta Interaktif ---
                st.markdown("### 🗺️ Visualisasi Lokasi Gempa")
                color_map = {
                    "Gempa Mikro": [173, 216, 230],
                    "Gempa Minor": [100, 181, 246],
                    "Gempa Ringan": [72, 201, 176],
                    "Gempa Sedang": [255, 204, 128],
                    "Gempa Kuat": [255, 167, 38],
                    "Gempa Dahsyat": [244, 67, 54],
                }
                df["color"] = df["Prediksi Kategori"].map(lambda x: color_map.get(x, [255, 255, 255]))

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df,
                    get_position='[lon, lat]',
                    get_fill_color='color',
                    get_radius=40000,
                    pickable=True,
                )
                view_state = pdk.ViewState(latitude=df["lat"].mean(), longitude=df["lon"].mean(), zoom=4)
                tooltip = {"text": "Kategori: {Prediksi Kategori}\nMagnitudo: {mag}"}
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

        except Exception as e:
            st.error(f"Gagal membaca atau memproses file CSV: {e}")

# ==========================================================
# === INFO TAMBAHAN ========================================
# ==========================================================
st.markdown("<br><hr><br>", unsafe_allow_html=True)
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

# ==========================================================
# === FOOTER ===============================================
# ==========================================================
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
