import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Membaca dataset
df = pd.read_csv('SAHeart.csv')

# Judul aplikasi
st.title("Aplikasi Prediksi CHD Berdasarkan Obesitas dan Alkohol")

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Daftar Data", "Prediksi CHD", "Visualisasi"])

# Halaman Daftar Data
if page == "Daftar Data":
    st.subheader("Data SAHeart")
    st.dataframe(df, hide_index=True)  # Menampilkan DataFrame tanpa kolom indeks

# Halaman Prediksi CHD
elif page == "Prediksi CHD":
    st.subheader("Prediksi CHD Berdasarkan Nilai Obesitas dan Alkohol")
    
    # Pilihan input pengguna
    obesity_input = st.number_input("Masukkan Nilai Obesitas:", min_value=0.0, max_value=50.0, step=0.1)
    alcohol_input = st.number_input("Masukkan Nilai Alkohol:", min_value=0.0, max_value=100.0, step=0.1)

    # Split data dan model training
    X = df[['obesity', 'alcohol']]
    y = df['chd']

    # Membagi data ke dalam training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standarisasi data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Prediksi ketika tombol ditekan
    if st.button("Prediksi CHD"):
        # Menyiapkan input data pengguna
        input_data = scaler.transform([[obesity_input, alcohol_input]])
        
        # Melakukan prediksi
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[0][1]  # Probabilitas CHD

        # Output hasil prediksi
        if prediction[0] == 1:
            st.write(f"Hasil prediksi: Orang ini mungkin memiliki CHD. Probabilitas: {prediction_proba * 100:.2f}%")
        else:
            st.write(f"Hasil prediksi: Orang ini mungkin tidak memiliki CHD. Probabilitas: {prediction_proba * 100:.2f}%")

# Halaman Visualisasi
elif page == "Visualisasi":
    st.subheader("Visualisasi Data SAHeart")

    # Visualisasi rata-rata obesitas per usia
    st.subheader("Rata-rata Obesitas per Usia")
    avg_obesity_by_age = df.groupby('age')['obesity'].mean().reset_index()

    # Set the Seaborn style and color palette
    sns.set(style="whitegrid")

    # Create a figure for the average obesity plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='age', y='obesity', data=avg_obesity_by_age, marker='o', linewidth=2.5, ax=ax, color='teal')
    ax.set_title('Rata-rata Obesitas per Usia', fontsize=16, fontweight='bold')
    ax.set_xlabel('Usia', fontsize=12)
    ax.set_ylabel('Obesitas Rata-rata', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)

    # Show the plot in Streamlit
    st.pyplot(fig)

    # Visualisasi total konsumsi alkohol berdasarkan chd
    st.subheader("Total Konsumsi Alkohol Berdasarkan Status CHD")
    alcohol_by_chd = df.groupby('chd')['alcohol'].sum().reset_index()

    # Create a figure for the alcohol consumption plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='chd', y='alcohol', data=alcohol_by_chd, palette="muted", ax=ax)
    ax.set_title('Total Konsumsi Alkohol Berdasarkan Status CHD', fontsize=16, fontweight='bold')
    ax.set_xlabel('CHD (0: Tidak, 1: Ya)', fontsize=12)
    ax.set_ylabel('Total Konsumsi Alkohol', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)

    # Show the plot in Streamlit
    st.pyplot(fig)