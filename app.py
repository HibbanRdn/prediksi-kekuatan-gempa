import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pydeck as pdk
import matplotlib.pyplot as plt

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Kategori Gempa Indonesia",
    page_icon="🌋",
    layout="wide"
)

# --- Header ---
st.markdown(
    "<h2 style='text-align:center;color:#FF5733;'>🌋 Prediksi Kategori Gempa Indonesia</h2>",
    unsafe_allow_html=True
)

# --- Muat Model ---
MODEL_PATH = "model_rf.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.warning("⚠️ Model belum tersedia, pastikan file 'model_rf.pkl' ada di folder yang sama.")
    st.stop()

# --- Fungsi Prediksi ---
def hitung_kategori_gempa(depth, mag):
    ig = mag * (100 - depth) / 100
    if ig < 1:
        return "Gempa Mikro"
    elif ig < 3:
        return "Gempa Minor"
    elif ig < 5:
        return "Gempa Ringan"
    elif ig < 7:
        return "Gempa Sedang"
    elif ig < 9:
        return "Gempa Kuat"
    else:
        return "Gempa Dahsyat"

# --- Buat Tab ---
tab1, tab2, tab3 = st.tabs(["📄 Prediksi Langsung", "📂 Upload CSV", "ℹ️ Info Apps"])

# =======================================================
# 📄 TAB 1 – INPUT LANGSUNG
# =======================================================
with tab1:
    st.subheader("Masukkan Data Gempa")

    col1, col2 = st.columns(2)
    with col1:
        depth = st.number_input("Kedalaman (km)", min_value=0.0, max_value=700.0, step=0.1)
    with col2:
        mag = st.number_input("Magnitudo (Skala Richter)", min_value=0.0, max_value=10.0, step=0.1)

    if st.button("🔍 Prediksi Kategori"):
        data = pd.DataFrame([[depth, mag]], columns=["depth", "mag"])
        pred = model.predict(data)[0]
        kategori = hitung_kategori_gempa(depth, mag)

        st.success(f"✅ Kategori Gempa: **{kategori}** (Model Prediksi: {pred})")

# =======================================================
# 📂 TAB 2 – UPLOAD CSV
# =======================================================
with tab2:
    st.subheader("Unggah File CSV Gempa")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            try:
                df = pd.read_csv(uploaded_file, delimiter=",")
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep="\t")

            required_cols = {"depth", "mag", "lat", "lon"}
            if not required_cols.issubset(df.columns):
                st.error("❌ Kolom wajib tidak ditemukan. Pastikan ada: depth, mag, lat, lon")
            else:
                df["Prediksi Model"] = model.predict(df[["depth", "mag"]])
                df["Prediksi Kategori"] = df.apply(lambda x: hitung_kategori_gempa(x["depth"], x["mag"]), axis=1)
                st.success("✅ Prediksi selesai!")

                st.dataframe(df.head())

                # Ringkasan hasil
                st.markdown("### 📊 Distribusi Kategori Gempa")
                summary = df["Prediksi Kategori"].value_counts().reset_index()
                summary.columns = ["Kategori", "Jumlah"]
                summary["Persentase (%)"] = (summary["Jumlah"] / summary["Jumlah"].sum() * 100).round(2)
                st.dataframe(summary)

                # Warna kategori
                color_map = {
                    "Gempa Mikro": "#ADD8E6",
                    "Gempa Minor": "#64B5F6",
                    "Gempa Ringan": "#48C9B0",
                    "Gempa Sedang": "#FFCC80",
                    "Gempa Kuat": "#FFA726",
                    "Gempa Dahsyat": "#F44336",
                }

                colors = [color_map.get(k, "#CCCCCC") for k in summary["Kategori"]]
                fig, ax = plt.subplots()
                ax.pie(summary["Jumlah"], labels=summary["Kategori"], colors=colors, autopct="%1.1f%%", startangle=90)
                ax.axis("equal")
                st.pyplot(fig)

                # Peta
                st.markdown("### 🗺️ Peta Sebaran Episentrum")
                df_map = df[["lat", "lon", "Prediksi Kategori"]]
                st.pydeck_chart(
                    pdk.Deck(
                        map_style="mapbox://styles/mapbox/dark-v11",
                        initial_view_state=pdk.ViewState(
                            latitude=df_map["lat"].mean(),
                            longitude=df_map["lon"].mean(),
                            zoom=4,
                            pitch=30,
                        ),
                        layers=[
                            pdk.Layer(
                                "HeatmapLayer",
                                data=df_map,
                                get_position=["lon", "lat"],
                                opacity=0.6,
                            ),
                        ],
                    )
                )
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# =======================================================
# ℹ️ TAB 3 – INFO APPS
# =======================================================
with tab3:
    st.subheader("Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini memprediksi **kategori gempa** berdasarkan **kedalaman (km)** dan **magnitudo (Skala Richter)**.  
    Kategori dihitung menggunakan **Indeks Gempa (IG = mag × (100 – depth) / 100)** dan dipelajari melalui model **Random Forest/XGBoost**.
    
    Data historis diambil dari catatan gempa Indonesia tahun **2008–2023**.
    """)

    st.divider()
    st.subheader("📊 Tren Historis Gempa 2008–2023")

    # Contoh data historis (bisa diganti dataset sebenarnya)
    np.random.seed(42)
    tahun = np.arange(2008, 2024)
    magnitudo = np.random.uniform(4.0, 6.5, len(tahun))

    df_tren = pd.DataFrame({"Tahun": tahun, "Rata-rata Magnitudo": magnitudo})

    fig, ax = plt.subplots()
    ax.plot(df_tren["Tahun"], df_tren["Rata-rata Magnitudo"], marker="o", linewidth=2)
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Rata-rata Magnitudo")
    ax.set_title("Tren Rata-rata Magnitudo Gempa di Indonesia (2008–2023)")
    st.pyplot(fig)

    # Heatmap lokasi historis (dummy)
    st.subheader("🌋 Heatmap Persebaran Gempa")
    data_heatmap = pd.DataFrame({
        "lat": np.random.uniform(-10, 6, 200),
        "lon": np.random.uniform(95, 140, 200),
    })
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v11",
            initial_view_state=pdk.ViewState(
                latitude=-2.5,
                longitude=120,
                zoom=4,
                pitch=30,
            ),
            layers=[
                pdk.Layer(
                    "HeatmapLayer",
                    data=data_heatmap,
                    get_position=["lon", "lat"],
                    opacity=0.6,
                ),
            ],
        )
    )

    st.info("📘 Dikembangkan sebagai proyek analisis dan prediksi gempa berbasis Machine Learning menggunakan Streamlit.")


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
