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
Aplikasi ini memprediksi kategori gempa berdasarkan **kedalaman (km)** dan **magnitudo (Skala Richter)**.  
Model dilatih menggunakan algoritma *Random Forest/XGBoost* dengan data gempa Indonesia tahun **2008–2023** (BMKG & USGS).
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
# === TAB NAVIGASI =========================================
# ==========================================================
tab1, tab2, tab3 = st.tabs(["🧾 Input Langsung", "📂 Upload CSV", "📘 Tentang Aplikasi"])

# ==========================================================
# === TAB 1: INPUT LANGSUNG ================================
# ==========================================================
with tab1:
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

    if st.button("🔍 Prediksi", use_container_width=True):
        input_data = np.array([[depth, mag]])
        try:
            pred = model.predict(input_data)
            try:
                kategori = le.inverse_transform(pred.astype(int))[0]
            except Exception:
                kategori = pred[0]

            st.subheader("🌏 Hasil Prediksi:")
            st.success(f"Kategori Gempa: **{kategori}**")

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
# === TAB 2: UPLOAD FILE CSV ===============================
# ==========================================================
with tab2:
    st.header("📂 Upload File CSV")
    st.markdown("Unggah file CSV dengan kolom: **tgl, ot, lat, lon, depth, mag, remark**")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
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
                kategori = [
                    le.inverse_transform([int(p)])[0] if str(p).isdigit() else str(p)
                    for p in preds
                ]
                df["Prediksi Kategori"] = kategori

                st.success("✅ Prediksi selesai!")

                tampil_cols = [col for col in ["tgl", "lat", "lon", "depth", "mag", "remark", "Prediksi Kategori"] if col in df.columns]
                st.dataframe(df[tampil_cols])

                # --- Statistik Distribusi ---
                st.markdown("### 📊 Distribusi Kategori Gempa")
                summary = df["Prediksi Kategori"].value_counts().reset_index()
                summary.columns = ["Kategori", "Jumlah"]
                summary["Persentase (%)"] = (summary["Jumlah"] / summary["Jumlah"].sum() * 100).round(2)
                st.dataframe(summary)

                # --- Bar Chart ---
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(summary["Kategori"], summary["Jumlah"], color="#4B9CD3")
                ax.set_xlabel("Kategori Gempa")
                ax.set_ylabel("Jumlah")
                ax.set_title("Distribusi Kategori Gempa", pad=10)
                plt.xticks(rotation=30)
                st.pyplot(fig, use_container_width=False)

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
# === TAB 3: TENTANG APLIKASI ==============================
# ==========================================================
with tab3:
    st.header("📘 Tentang Aplikasi")

    st.markdown("""
    Aplikasi **Prediksi Kategori Gempa Indonesia** ini dikembangkan untuk mempermudah analisis tingkat kekuatan gempa bumi
    berdasarkan parameter **magnitudo** dan **kedalaman (depth)** menggunakan pendekatan *Machine Learning*.

    ### 🔍 Dasar Perhitungan
    Model menggunakan formula empiris:
    \n> **Indeks Gempa (IG) = mag × (100 − depth) / 100**
    
    yang merepresentasikan intensitas relatif antara kekuatan dan kedalaman gempa.
    Berdasarkan nilai IG inilah model *Random Forest/XGBoost* mengkategorikan gempa menjadi:
    - Gempa Mikro
    - Gempa Minor
    - Gempa Ringan
    - Gempa Sedang
    - Gempa Kuat
    - Gempa Dahsyat

    ### 🧠 Framework Pengembangan
    Proyek ini mengikuti tahapan **CRISP–DM**:
    - **Business Understanding** → Pemahaman dampak sosial & mitigasi bencana
    - **Data Understanding** → Data gempa dari BMKG & USGS (2008–2023)
    - **Data Preparation** → Pembersihan, standarisasi, dan feature engineering (IG)
    - **Modeling** → *Random Forest Classifier* dan *XGBoost*
    - **Evaluation** → Akurasi dan interpretasi kategori
    - **Deployment** → Implementasi interaktif berbasis Streamlit

    ### 📊 Sumber Data
    - **BMKG (Badan Meteorologi, Klimatologi, dan Geofisika)**
    - **USGS Earthquake Catalog**
    - Rentang tahun **2008–2023**

    ### 👨‍💻 Pengembang
    - **Nama:** M. Hibban Ramadhan  
    - **Institusi:** Universitas Lampung  
    - **Teknologi:** Python, Streamlit, Scikit-Learn, XGBoost, PyDeck  

    ---
    """)
    
    st.subheader("🌋 Visualisasi Data Historis Gempa 2008–2023")
    url = "https://raw.githubusercontent.com/HibbanRdn/prediksi-kekuatan-gempa/refs/heads/main/data/katalog_gempa.csv"
    try:
        df_hist = pd.read_csv(url)
        df_hist.columns = [c.strip().lower().replace(" ", "_") for c in df_hist.columns]
        df_hist = df_hist.dropna(subset=["lat", "lon", "mag"])
        if "tgl" in df_hist.columns:
            df_hist["year"] = pd.to_datetime(df_hist["tgl"], errors="coerce").dt.year
        elif "tanggal" in df_hist.columns:
            df_hist["year"] = pd.to_datetime(df_hist["tanggal"], errors="coerce").dt.year
        else:
            df_hist["year"] = df_hist.index  # fallback 
    tren = df_hist.groupby("year")["mag"].mean().reset_index().dropna()
    st.markdown("#### 📈 Tren Rata-rata Magnitudo per Tahun")
    fig1, ax1 = plt.subplots(figsize=(7, 3))
    ax1.plot(tren["year"], tren["mag"], marker="o", color="#4B9CD3", linewidth=2)
    ax1.set_xlabel("Tahun")
    ax1.set_ylabel("Rata-rata Magnitudo")
    ax1.set_title("Tren Gempa Indonesia 2008-2023", pad=10)
    st.pyplot(fig1, use_container_width=False)

    # --- Heatmap berbasis magnitudo ---
    st.markdown("#### 🗺️ Heatmap Persebaran Gempa Berdasarkan Magnitudo")
    st.pydeck_chart(
        pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            initial_view_state=pdk.ViewState(
                latitude=df_hist["lat"].mean(),
                longitude=df_hist["lon"].mean(),
                zoom=4,
                pitch=30,
            ),
            layers=[
                pdk.Layer(
                    "HeatmapLayer",
                    data=df_hist,
                    get_position=["lon", "lat"],
                    get_weight="mag",
                    radiusPixels=50,
                    intensity=1,
                    threshold=0.05,
                    opacity=0.6,
                ),
            ],
        )
    )
    st.caption("Visualisasi ini menggunakan data historis 2008–2023 dari GitHub, berdasarkan magnitudo gempa di Indonesia.")
        except Exception as e:
        st.warning(f"❌ Tidak dapat memuat data historis: {e}")




# ==========================================================
# === FOOTER ===============================================
# ==========================================================
st.markdown("<br><hr><br>", unsafe_allow_html=True)
year = datetime.datetime.now().year
st.markdown(
    f"""
    <div style='text-align: center; color: gray; font-size: 0.9rem; margin-top: 10px;'>
        © {year} <b>M. Hibban Ramadhan</b> — Proyek <i>Prediksi Kategori Gempa Indonesia</i><br>
        Dibangun menggunakan <a href='https://streamlit.io' target='_blank' style='color: #4b9cd3; text-decoration: none;'>Streamlit</a>
    </div>
    """,
    unsafe_allow_html=True
)

