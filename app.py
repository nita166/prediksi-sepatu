# ============================================================
# 👟 SHOE PRICE PREDICTION APP - DECISION TREE V3
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1️⃣ Load Model dan Scaler
# ============================================================
@st.cache_resource
def load_assets():
    """
    Memuat model Decision Tree yang sudah terlatih.
    Karena kita tidak menggunakan pipeline, scaler juga perlu dibuat dan
    dilatih secara terpisah.
    """
    model_path = "model_sepatu_tree.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ File model '{model_path}' tidak ditemukan. Mohon buat file tersebut terlebih dahulu.")
        st.stop()

    try:
        # Memuat model yang sudah dilatih
        trained_model = joblib.load(model_path)
        
        # Inisialisasi scaler baru
        scaler = StandardScaler()
        
        # Karena kita tidak bisa melatih scaler di sini tanpa data,
        # kita akan mengasumsikan scaler ini akan dilatih di script terpisah
        # dan juga disimpan ke file pkl.
        # Namun, untuk menyederhanakan, kita akan memuat scaler yang sudah dilatih
        # dari file lain. Kita akan membuat scaler dan model dalam satu file pkl
        # di langkah selanjutnya agar lebih mudah.
        
        # Untuk saat ini, kita akan membuat model yang lebih sederhana tanpa scaler.
        # Model Decision Tree tidak selalu memerlukan scaling.
        
        return trained_model
    except Exception as e:
        st.error(f"❌ Gagal memuat aset: {e}. Pastikan versi pustaka Anda konsisten.")
        st.stop()

model = load_assets()

# ============================================================
# 2️⃣ App Configuration
# ============================================================
st.set_page_config(
    page_title="Prediksi Harga Sepatu",
    page_icon="👟",
    layout="centered"
)

st.title("👟 Aplikasi Prediksi Harga Sepatu (Decision Tree)")
st.markdown("Masukkan detail sepatu untuk memprediksi harga.")

# ============================================================
# 3️⃣ Input Form
# ============================================================
with st.form("shoe_price_form"):
    st.subheader("📝 Masukkan Detail Sepatu")
    
    how_many_sold = st.number_input("Jumlah Terjual (misal: 2242)", min_value=0)
    rating = st.slider("Rating Produk", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
    
    submitted = st.form_submit_button("🔮 Prediksi Harga")
    
    if submitted:
        try:
            # Buat DataFrame dari input pengguna
            input_data = pd.DataFrame([{
                'How_Many_Sold': how_many_sold,
                'RATING': rating
            }])
            
            # Melakukan prediksi langsung dengan model
            prediction = model.predict(input_data)
            
            st.subheader("✅ Prediksi Berhasil!")
            st.success(f"Harga sepatu yang diprediksi adalah: ₹{prediction[0]:,.2f}")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")