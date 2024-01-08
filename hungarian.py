# Import library yang diperlukan
import itertools
import pandas as pd
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from streamlit_option_menu import option_menu

# Membaca data dari file "hungarian.data"
with open("data/hungarian.data", encoding='Latin1') as file:
    lines = [line.strip() for line in file]

# Ekstrak data yang relevan dari file dalam kelompok 10 baris
data = itertools.takewhile(
    lambda x: len(x) == 76,
    (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10)))

# Membuat DataFrame dari data yang diekstrak sebelumnya
df = pd.DataFrame.from_records(data)

# Menghilangkan kolom terakhir dari DataFrame
df = df.iloc[:, :-1]

# Menghapus kolom pertama dari DataFrame
df = df.drop(df.columns[0], axis=1)

# Mengonversi nilai-nilai dalam DataFrame menjadi tipe data float
df = df.astype(float)

# Mengganti nilai -9.0 dengan NaN (Not a Number)
df.replace(-9.0, np.NaN, inplace=True)

# Memilih subset kolom tertentu dari DataFrame
df_selected = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]

# Membuat pemetaan kolom menggunakan dictionary
column_mapping = {
    2: 'age',
    3: 'sex',
    8: 'cp',
    9: 'trestbps',
    11: 'chol',
    15: 'fbs',
    18: 'restecg',
    31: 'thalach',
    37: 'exang',
    39: 'oldpeak',
    40: 'slope',
    43: 'ca',
    50: 'thal',
    57: 'target'
}

# Mengganti nama kolom dalam DataFrame sesuai dengan pemetaan yang telah dibuat
df_selected.rename(columns=column_mapping, inplace=True)

# Kolom-kolom yang akan dihapus dari DataFrame
columns_to_drop = ['ca', 'slope', 'thal']

# Menghapus kolom-kolom yang tidak diperlukan dari DataFrame
df_selected = df_selected.drop(columns_to_drop, axis=1)

# Menghitung rata-rata dari kolom-kolom tertentu setelah menghapus nilai NaN
meanTBPS = df_selected['trestbps'].dropna().astype(float).mean()
meanChol = df_selected['chol'].dropna().astype(float).mean()
meanfbs = df_selected['fbs'].dropna().astype(float).mean()
meanRestCG = df_selected['restecg'].dropna().astype(float).mean()
meanthalach = df_selected['thalach'].dropna().astype(float).mean()
meanexang = df_selected['exang'].dropna().astype(float).mean()

# Pembulatan nilai rata-rata ke angka bulat
meanTBPS = round(meanTBPS)
meanChol = round(meanChol)
meanfbs = round(meanfbs)
meanthalach = round(meanthalach)
meanexang = round(meanexang)
meanRestCG = round(meanRestCG)

# Nilai-nilai rata-rata untuk menggantikan nilai NaN
fill_values = {
    'trestbps': meanTBPS,
    'chol': meanChol,
    'fbs': meanfbs,
    'thalach': meanthalach,
    'exang': meanexang,
    'restecg': meanRestCG
}

# Mengisi nilai-nilai NaN dalam DataFrame dengan nilai rata-rata yang telah dihitung
df_clean = df_selected.fillna(value=fill_values)

# Menghapus duplikat baris dalam DataFrame
df_clean.drop_duplicates(inplace=True)

# Memisahkan fitur (X) dan target (y)
X = df_clean.drop("target", axis=1)
y = df_clean['target']

# Melakukan oversampling menggunakan RandomOverSampler
random = RandomOverSampler(random_state=42)
X1, y1 = random.fit_resample(X, y)

# Memuat model dari file yang telah disimpan sebelumnya (pickle)
model = pickle.load(open("model/xgb_model_random.pkl", 'rb'))

# Melakukan prediksi terhadap data oversampled
y_pred = model.predict(X1)

# Menghitung akurasi dari prediksi
accuracy = accuracy_score(y1, y_pred)
accuracy = round((accuracy * 100), 2)

# Membuat DataFrame akhir dengan fitur-fitur hasil oversampling dan target
df_final = X1
df_final['target'] = y

# STREAMLIT
st.title('CAPSTONE PROJECT DATA SCIENCE')
st.write(":information_source: **Hungarian Heart Disease Dataset**")
from PIL import Image
# Mendefinisikan path gambar
gambar_path = "static/jantung.jpg"
# Membaca gambar
gambar = Image.open(gambar_path)
# Menampilkan gambar di Streamlit
st.image(gambar, caption='Maintaining a healthy heart is crucial for overall well-being. Adopting a heart-healthy lifestyle can significantly ', use_column_width=True)
st.write(":green[**The dataset is obtained from the UC Irvine Machine Learning Repository**], focusing on heart disease diagnosis. [Dataset Link](https://archive.ics.uci.edu/dataset/45/heart+disease)")
st.write("This directory contains four databases related to heart disease diagnosis. All attributes are numerical. The data is collected from the following locations:")
st.write("1. **Cleveland Clinic Foundation** (*cleveland.data*)")
st.write("2. **Hungarian Institute of Cardiology, Budapest** (*hungarian.data*)")
st.write("3. **V.A. Medical Center, Long Beach, CA** (*long-beach-va.data*)")
st.write("4. **University Hospital, Zurich, Switzerland** (*switzerland.data*)")
st.write(f":chart_with_upwards_trend: **_Model's Accuracy_** :  :green[**{accuracy}**]% (:red[_Do not copy outright_])")
# Membuat tab dengan dua opsi: "Single-predict" dan "Multi-predict"
tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])
# Pada tab1 ("Single-predict")
with tab1:
    # Menampilkan header pada sidebar untuk input pengguna
    st.sidebar.header("**User Input** Sidebar")

    # Menampilkan widget untuk menginputkan usia (age)
    age = st.sidebar.number_input(label=":violet[**Age**]", min_value=df_final['age'].min(),
                                max_value=df_final['age'].max(), key="age")
    # Menampilkan informasi range nilai yang valid untuk usia
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['age'].min()}**], :red[Max] value: :red[**{df_final['age'].max()}**]")
    st.sidebar.write("")

    # Menampilkan widget untuk memilih jenis kelamin (sex)
    sex_sb = st.sidebar.radio(label=":violet[**Sex**]", options=["Male", "Female"], key="sex")
    st.sidebar.write("")

    # Menampilkan widget untuk memilih jenis nyeri dada (chest pain type)
    cp_sb = st.sidebar.selectbox(label=":violet[**Chest pain type**]",
                                options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"], key="cp")
    st.sidebar.write("")

    # Menampilkan widget untuk menginputkan tekanan darah istirahat (trestbps)
    trestbps = st.sidebar.number_input(
        label=":violet[**Resting blood pressure** (in mm Hg on admission to the hospital)]",
        min_value=df_final['trestbps'].min(), max_value=df_final['trestbps'].max(), key="trestbps")
    # Menampilkan informasi range nilai yang valid untuk tekanan darah istirahat
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['trestbps'].min()}**], :red[Max] value: :red[**{df_final['trestbps'].max()}**]")
    st.sidebar.write("")

    # Menampilkan widget untuk menginputkan kadar kolesterol (chol)
    chol = st.sidebar.number_input(
        label=":violet[**Serum cholesterol** (in mg/dl)]",
        min_value=df_final['chol'].min(), max_value=df_final['chol'].max(), key="chol")
    # Menampilkan informasi range nilai yang valid untuk kadar kolesterol
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['chol'].min()}**], :red[Max] value: :red[**{df_final['chol'].max()}**]")
    st.sidebar.write("")

    # Menampilkan widget untuk memilih apakah gula darah puasa > 120 mg/dl (fbs)
    fbs_sb = st.sidebar.radio(label=":violet[**Fasting blood sugar > 120 mg/dl?**]", options=["False", "True"], key="fbs")
    st.sidebar.write("")

    # Menampilkan widget untuk memilih hasil elektrokardiografi saat istirahat (restecg)
    restecg_sb = st.sidebar.radio(label=":violet[**Resting electrocardiographic results**]",
                                    options=["Normal", "Having ST-T wave abnormality", "Showing left ventricular hypertrophy"], key="restecg")
    st.sidebar.write("")

    # Menampilkan widget untuk menginputkan detak jantung maksimum yang dicapai (thalach)
    thalach = st.sidebar.number_input(label=":violet[**Maximum heart rate achieved**]",
                                    min_value=df_final['thalach'].min(), max_value=df_final['thalach'].max(), key="thalach")
    # Menampilkan informasi range nilai yang valid untuk detak jantung maksimum
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['thalach'].min()}**], :red[Max] value: :red[**{df_final['thalach'].max()}**]")
    st.sidebar.write("")

    # Menampilkan widget untuk memilih apakah terjadi angina yang dipicu oleh olahraga (exang)
    exang_sb = st.sidebar.radio(label=":violet[**Exercise induced angina?**]", options=["No", "Yes"], key="exang")
    st.sidebar.write("")

    # Menampilkan widget untuk menginputkan depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat (oldpeak)
    oldpeak = st.sidebar.number_input(label=":violet[**ST depression induced by exercise relative to rest**]",
                                    min_value=df_final['oldpeak'].min(), max_value=df_final['oldpeak'].max(), key="oldpeak")
    # Menampilkan informasi range nilai yang valid untuk depresi ST
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df_final['oldpeak'].min()}**], :red[Max] value: :red[**{df_final['oldpeak'].max()}**]")
    st.sidebar.write("")

    # Membuat dictionary berisi data input pengguna
    data = {
        'Age': age,
        'Sex': sex_sb,
        'Chest pain type': cp_sb,
        'RPB': f"{trestbps} mm Hg",
        'Serum Cholestoral': f"{chol} mg/dl",
        'FBS > 120 mg/dl?': fbs_sb,
        'Resting ECG': restecg_sb,
        'Maximum heart rate': thalach,
        'Exercise induced angina?': exang_sb,
        'ST depression': oldpeak,
    }

    # Membuat DataFrame untuk preview data input pengguna
    preview_df = pd.DataFrame(data, index=['input'])

    # Menampilkan header dan dua bagian DataFrames (membagi menjadi dua bagian untuk estetika)
    st.header("User Input as DataFrame")
    st.write("")
    st.dataframe(preview_df.iloc[:, :6])
    st.write("")
    st.dataframe(preview_df.iloc[:, 6:])
    st.write("")

    # Inisialisasi variabel hasil prediksi
    result = ":violet[-]"

    # Membuat tombol "Predict"
    predict_btn = st.button("**Predict**", type="primary")

    st.write("")
    if predict_btn:
        # Mengonversi beberapa nilai string menjadi numerik
        sex = 1 if sex_sb == "Male" else 0
        cp = 1 if cp_sb == "Typical angina" else 2 if cp_sb == "Atypical angina" else 3 if cp_sb == "Non-anginal pain" else 4
        fbs = 1 if fbs_sb == "True" else 0
        restecg = 0 if restecg_sb == "Normal" else 1 if restecg_sb == "Having ST-T wave abnormality" else 2
        exang = 1 if exang_sb == "Yes" else 0
        
        # Menyiapkan input dalam bentuk list
        inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
        
        # Melakukan prediksi menggunakan model
        prediction = model.predict(inputs)[0]

        # Menampilkan progress bar saat prediksi berlangsung
        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        # Menentukan hasil prediksi berdasarkan kelas yang dihasilkan oleh model
        if prediction == 0:
            result = ":green[**Sehat**]"
        elif prediction == 1:
            result = ":orange[**Penyakit Jantung Tingkat 1**]"
        elif prediction == 2:
            result = ":orange[**Penyakit Jantung Tingkat 2**]"
        elif prediction == 3:
            result = ":red[**Penyakit Jantung Tingkat 3**]"
        elif prediction == 4:
            result = ":red[**Penyakit Jantung Tingkat 4**]"

    # Menampilkan hasil prediksi
    st.write("")
    st.write("")
    st.subheader("Prediksi:")
    st.subheader(result)

    # Grafik Countplot untuk Diagnosis Penyakit Jantung
    st.header("Heart Disease Diagnosis Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='target', data=df_clean, ax=ax)
    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Count')
    ax.set_title('Heart Disease Diagnosis')
    st.pyplot(fig)

    # Visualisasi korelasi antar fitur
    st.header("Correlation Heatmap")
    fig, ax = plt.subplots()
    correlation_matrix = df_clean.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)




# Pada tab2 ("Multi-predict")
with tab2:
    # Menampilkan header untuk prediksi data ganda
    st.header("Predict multiple data:")

    # Membuat contoh data CSV untuk diunduh
    sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode('utf-8')

    st.write("")
    # Menampilkan tombol untuk mengunduh contoh data CSV
    st.download_button("Download CSV Example", data=sample_csv, file_name='sample_heart_disease_parameters.csv',
                    mime='text/csv')

    st.write("")
    st.write("")
    
    # Menyediakan opsi untuk mengunggah file CSV
    file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

    if file_uploaded:
        # Membaca data CSV yang diunggah
        uploaded_df = pd.read_csv(file_uploaded)
        # Melakukan prediksi menggunakan model
        prediction_arr = model.predict(uploaded_df)

        # Menampilkan progress bar saat prediksi berlangsung
        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 70):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)

        # Mengonversi hasil prediksi menjadi label yang dapat dibaca
        result_arr = []

        for prediction in prediction_arr:
            if prediction == 0:
                result = "Sehat"
            elif prediction == 1:
                result = "Penyakit Jantung Tingkat 1"
            elif prediction == 2:
                result = "Penyakit Jantung Tingkat 2"
            elif prediction == 3:
                result = "Penyakit Jantung Tingkat 3"
            elif prediction == 4:
                result = "Penyakit Jantung Tingkat 4"
            result_arr.append(result)

        # Membuat DataFrame untuk hasil prediksi
        uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

        for i in range(70, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        # Menampilkan hasil prediksi dan data yang diunggah dalam dua kolom
        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(uploaded_result)
        with col2:
            st.dataframe(uploaded_df)

    # Informasi profil mahasiswa
    st.title("**Mahasiswa Profile**")
    st.write("Nama: Reyhan Vito Idham Pratama")
    st.write("NIM: A11.2020.129881")

    # Menampilkan gambar profil mahasiswa
    gambar_reyhan = "static/profile.jpg"
    gambar_reyhan = Image.open(gambar_reyhan)
    st.image(gambar_reyhan, caption='Foto Profil Reyhan Vito Idham Pratama',use_column_width=True)



    


