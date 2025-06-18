import streamlit as st
import pandas as pd
import joblib
import os

# --- Judul Aplikasi ---
st.title('Prediksi Kategori Waktu Kelulusan Mahasiswa')

st.write("Aplikasi ini memprediksi apakah seorang mahasiswa akan lulus tepat waktu (On Time) atau terlambat (Late) berdasarkan beberapa faktor.")

# --- Memuat Model ---
# Pastikan model_graduation.pkl berada di direktori yang sama dengan aplikasi Streamlit Anda
# Atau sediakan path absolut jika lokasinya berbeda
model_path = 'model_graduation.pkl'

if not os.path.exists(model_path):
    st.error(f"Error: Model '{model_path}' tidak ditemukan. Pastikan file model berada di direktori yang sama.")
else:
    try:
        model = joblib.load(model_path)
        st.success("Model berhasil dimuat!")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        st.stop() # Hentikan eksekusi jika model gagal dimuat

    # --- Input Data Baru dari Pengguna ---
    st.header("Masukkan Data Mahasiswa Baru")

    # Menggunakan st.number_input untuk input numerik
    new_ACT = st.number_input('Masukkan nilai ACT composite score:', min_value=0.0, max_value=36.0, value=25.0, step=0.1)
    new_SAT = st.number_input('Masukkan nilai SAT total score:', min_value=0.0, max_value=1600.0, value=1200.0, step=1.0)
    new_GPA = st.number_input('Masukkan nilai rata-rata SMA:', min_value=0.0, max_value=4.0, value=3.0, step=0.01)
    new_income = st.number_input('Masukkan nilai pendapatan orang tua (contoh: 50000 untuk $50,000):', min_value=0.0, value=50000.0, step=1000.0)
    new_education = st.number_input('Masukkan tingkat pendidikan orang tua (angka, contoh: 1=SMP, 2=SMA, 3=S1, dst.):', min_value=1, max_value=5, value=3, step=1)

    # --- Tombol untuk Prediksi ---
    if st.button('Prediksi Kategori Kelulusan'):
        try:
            # Buat DataFrame dari input baru
            new_data_df = pd.DataFrame(
                [[new_ACT, new_SAT, new_GPA, new_income, new_education]],
                columns=['ACT composite score', 'SAT total score', 'high school gpa', 'parental income', 'parent_edu_numerical']
            )

            # Lakukan prediksi
            predicted_code = model.predict(new_data_df)[0]  # hasilnya 0 atau 1

            # Konversi hasil prediksi ke label asli
            label_mapping = {1: 'On Time', 0: 'Late'}
            predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

            st.subheader("Hasil Prediksi:")
            if predicted_label == 'On Time':
                st.success(f"Prediksi kategori masa studi adalah: **{predicted_label}** 🎉")
            else:
                st.warning(f"Prediksi kategori masa studi adalah: **{predicted_label}** ⚠️")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

st.markdown("---")
st.write("Dibuat oleh Data Scientist")