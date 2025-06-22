import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from env_qlearning import PenjualanEnv

st.set_page_config(page_title="Q-Learning Harga", layout="wide")
q_table_file = "q_table.npy"

# Init environment
env = PenjualanEnv()
env.unique_states = list(set(env.states))
env.n_states = len(env.unique_states)

menu = st.sidebar.radio("Pilih Halaman", [
    "📊 Visualisasi Q-table",
    "📈 Evaluasi Policy",
    "📉 Grafik Reward",
    "📊 Perbandingan Sebelum vs Sesudah Training",
    "⚙️ Training Ulang",
    "ℹ️ Tentang"
])

# Fungsi Training & Evaluasi (diambil dari kode lu)...
# ---- (tidak diubah) ----

# ========== Halaman: Perbandingan ===========
if menu == "📊 Perbandingan Sebelum vs Sesudah Training":
    st.title("📊 Perbandingan Profit: Sebelum vs Sesudah Training")
    try:
        before = pd.read_csv("datapenjualan.csv")
        after = pd.read_csv("simulation_after_training.csv")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(before["tanggal"], before["profit"], label="Sebelum Training", color="red")
        ax.plot(after["tanggal"], after["profit"], label="Setelah Training", color="green")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Profit")
        ax.legend()
        st.pyplot(fig)
    except FileNotFoundError:
        st.error("❌ File data belum lengkap. Pastikan `datapenjualan.csv` dan `simulation_after_training.csv` tersedia.")
