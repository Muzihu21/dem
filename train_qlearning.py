import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from env_qlearning import PenjualanEnv

# ========== Setup ========== 
st.set_page_config(page_title="Q-Learning Harga", layout="wide")
q_table_file = "q_table.npy"

if os.path.exists(q_table_file):
    q_table = np.load(q_table_file)
else:
    st.warning(f"⚠️ File `{q_table_file}` belum ditemukan. Silakan jalankan training dahulu.")
    q_table = None

# ========== Load Environment ========== 
env = PenjualanEnv()
env.unique_states = list(set(env.states))
env.n_states = len(env.unique_states)

# ========== Sidebar Menu ========== 
menu = st.sidebar.radio("Pilih Halaman", [
    "📊 Visualisasi Q-table",
    "📈 Evaluasi Policy",
    "📉 Grafik Reward",
    "⚙️ Training Ulang",
    "📊 Perbandingan Profit",
    "ℹ️ Tentang"
])

# ========== Fungsi: Training ========== 
def train_q_learning(env, alpha, gamma, epsilon, episodes):
    state_to_index = {s: i for i, s in enumerate(env.unique_states)}
    q_table = np.zeros((len(env.unique_states), env.n_actions))
    rewards_per_episode = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_idx = state_to_index[state]
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(q_table[state_idx])
            result = env.step(action)

            if len(result) == 3:
                next_state, reward, done = result
            else:
                next_state, reward, done, _ = result
            next_state_idx = state_to_index[next_state]
            q_table[state_idx, action] += alpha * (
                reward + gamma * np.max(q_table[next_state_idx]) - q_table[state_idx, action]
            )
            total_reward += reward
            state = next_state
        rewards_per_episode.append(total_reward)

    return q_table, np.array(rewards_per_episode)

# ========== Fungsi: Evaluasi Policy ========== 
def evaluate_policy(env, q_table, n_trials=100):
    total_rewards = []
    for _ in range(n_trials):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_index = env.unique_states.index(state)
            action = np.argmax(q_table[state_index])
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, _ = result
            elif len(result) == 3:
                next_state, reward, done = result
            else:
                raise ValueError("env.step() return format tidak valid")
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

# ========== Fungsi: Simulasi Setelah Training ========== 
def simulate_after_training(env, q_table, state_to_index, episodes=100, output_file="simulation_after_training.csv"):
    results = []
    for _ in range(episodes):
        state = env.reset()
        total_profit = 0
        done = False
        while not done:
            state_index = state_to_index[state]
            action = np.argmax(q_table[state_index])
            result = env.step(action)
            if len(result) == 3:
                next_state, reward, done = result
            else:
                next_state, reward, done, _ = result
            total_profit += reward
            state = next_state
        results.append(total_profit)

    df = pd.DataFrame({"profit": results})
    df.to_csv(output_file, index=False)

# ==================================================
# Halaman-Halaman
# ==================================================

if menu == "📊 Visualisasi Q-table":
    st.title("📊 Strategi Harga: Q-table Heatmap")
    if q_table is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(q_table, annot=True, cmap="YlGnBu",
                    xticklabels=env.harga_list,
                    yticklabels=env.unique_states,
                    ax=ax)
        ax.set_xlabel("Harga (Action)")
        ax.set_ylabel("State")
        st.pyplot(fig)

elif menu == "📈 Evaluasi Policy":
    st.title("📈 Evaluasi Policy")
    if q_table is not None:
        trials = st.slider("Jumlah Simulasi Episode", 10, 10000, 100, step=100)
        avg_reward = evaluate_policy(env, q_table, trials)
        st.success(f"🎯 Rata-rata reward dari {trials} simulasi: **{avg_reward:.2f}**")

elif menu == "📉 Grafik Reward":
    st.title("📉 Grafik Reward per Episode")
    try:
        rewards = np.load("rewards_per_episode.npy")
        fig, ax = plt.subplots()
        ax.plot(rewards, label='Reward per Episode', color='green')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Reward per Episode (Training Progress)")
        ax.legend()
        st.pyplot(fig)
    except FileNotFoundError:
        st.error("❌ File `rewards_per_episode.npy` tidak ditemukan.")

elif menu == "⚙️ Training Ulang":
    st.title("⚙️ Training Ulang Q-Learning")
    alpha = st.number_input("Alpha (Learning rate)", 0.0, 1.0, 0.1, step=0.01)
    gamma = st.number_input("Gamma (Discount factor)", 0.0, 1.0, 0.9, step=0.01)
    epsilon = st.number_input("Epsilon (Exploration rate)", 0.0, 1.0, 0.1, step=0.01)
    episodes = st.number_input("Jumlah Episode", 100, 10000, 1000, step=100)

    if st.button("🚀 Mulai Training"):
        with st.spinner("Training sedang berjalan..."):
            q_table_new, rewards = train_q_learning(env, alpha, gamma, epsilon, episodes)
            np.save("q_table.npy", q_table_new)
            np.save("rewards_per_episode.npy", rewards)
            
            state_to_index = {s: i for i, s in enumerate(env.unique_states)}
            simulate_after_training(env, q_table_new, state_to_index)

            st.success("✅ Training selesai dan data simulasi juga disimpan.")

elif menu == "📊 Perbandingan Profit":
    st.title("📊 Perbandingan Profit Sebelum vs Setelah Training")
    before_file = "datapenjualan.csv"
    after_file = "simulation_after_training.csv"

    if not os.path.exists(after_file):
        st.warning(f"⚠️ File `{after_file}` belum ditemukan. Silakan jalankan training dahulu.")
    else:
        df_before = pd.read_csv(before_file)
        df_after = pd.read_csv(after_file)

        fig, ax = plt.subplots()
        sns.kdeplot(df_before["profit"], label="Sebelum Training", ax=ax, fill=True, color="red", alpha=0.5)
        sns.kdeplot(df_after["profit"], label="Setelah Training", ax=ax, fill=True, color="green", alpha=0.5)

        ax.set_title("Distribusi Profit Sebelum vs Setelah Training")
        ax.set_xlabel("Profit")
        ax.legend()
        st.pyplot(fig)

elif menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini mensimulasikan **Reinforcement Learning (Q-Learning)** untuk penetapan harga produk.

    **Fitur:**
    - Visualisasi Q-table (heatmap)
    - Evaluasi Policy
    - Grafik Reward per Episode
    - Training Ulang dengan hyperparameter custom
    - Perbandingan Profit Sebelum vs Setelah Training

    **Author**: Zihu — AI Engineer & Pejuang Skripsi 🧠🔥  
    **Stack**: Python, Streamlit, NumPy, Matplotlib, Seaborn
    """)

# ========== Footer ==========
st.markdown("---")
st.caption("© 2025 — Made with ❤️ by Zihu | Powered by Streamlit")
