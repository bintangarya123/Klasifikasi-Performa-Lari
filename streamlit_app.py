import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman
st.set_page_config(
    page_title="Klasifikasi Performa Aktivitas Lari",
    layout="wide"
)

# Judul Aplikasi
st.title(" Klasifikasi Performa Aktivitas Lari")

st.write(
    "Aplikasi ini merupakan implementasi model Gaussian Naïve Bayes yang "
    "dikembangkan untuk mengklasifikasikan performa aktivitas lari "
    "berdasarkan parameter jarak, waktu, pace, dan rata-rata detak jantung."
)

st.divider()

# Memuat Model
model = joblib.load("model_gnb.pkl")

# Upload Dataset
uploaded_file = st.file_uploader(
    "Upload Dataset (.csv)",
    type=["csv"]
)

# Jika Dataset Berhasil Diunggah
if uploaded_file is not None:

    # Membaca dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview Dataset")
    st.dataframe(df, use_container_width=True)

    # Tombol Proses
    if st.button("Proses Klasifikasi"):

        # Mengambil fitur yang digunakan model
        X_pred = df[[
            "Jarak (Km)",
            "Waktu_Detik",
            "Pace_Detik",
            "Rata-rata Detak Jantung"
        ]]

        # Melakukan prediksi
        hasil_prediksi = model.predict(X_pred)

        # Menambahkan hasil prediksi
        df["Prediksi"] = hasil_prediksi

        # Menghilangkan kolom yang tidak perlu ditampilkan
        df_hasil = df.drop(columns=["Waktu_Detik", "Pace_Detik"])

        # Informasi
        st.success("Proses klasifikasi berhasil dilakukan.")

        st.info(f"Jumlah data yang diproses : {len(df_hasil)}")

        # Ringkasan Hasil
        st.subheader("Ringkasan Hasil")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Rendah",
            int((df_hasil["Prediksi"] == "Rendah").sum())
        )

        col2.metric(
            "Sedang",
            int((df_hasil["Prediksi"] == "Sedang").sum())
        )

        col3.metric(
            "Tinggi",
            int((df_hasil["Prediksi"] == "Tinggi").sum())
        )

        # Hasil Klasifikasi
        st.subheader("Hasil Klasifikasi")

        st.dataframe(df_hasil, use_container_width=True)

else:
    st.info("Silakan upload dataset terlebih dahulu.")