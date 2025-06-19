# Laporan Proyek Machine Learning - Prediksi Persetujuan Pinjaman

## Project Overview

Prediksi persetujuan pinjaman merupakan masalah krusial di industri keuangan. Sistem yang akurat dapat membantu lembaga keuangan dalam mengurangi risiko kredit macet, mempercepat proses pengajuan pinjaman, dan menggoptimalkan alokasi modal.

Menurut Valmiki Sarath Kumar dan K. Vijayalakshmi (2024), metode evaluasi tradisional terkadang masih efektif dalam kondisi tertentu, tetapi sering kali tidak cukup cepat atau akurat untuk memenuhi tuntutan pasar modern. Oleh karena itu, machine learning hadir sebagai solusi inovatif dengan pendekatan prediktif. Teknologi ini memungkinkan sistem untuk mengenali pola dalam data menggunakan model statistik dan algoritma, sehingga dapat menghasilkan prediksi atau kesimpulan dengan tingkat akurasi yang lebih tinggi. Metode ini menjadi sangat penting dalam mengelompokkan pemohon pinjaman berdasarkan kemampuan mereka dalam melunasi kredit.

Referensi:
- [https://eprajournals.com/IJMR/article/13136/download](https://eprajournals.com/IJMR/article/13136/download)

## Business Understanding

### Problem Statements
1. Bagaimana memprediksi persetujuan pinjaman berdasarkan karakteristik peminjam?

### Goals
1. Membangun model prediksi yang dapat memprediksi dengan akurasi > 80%

### Solution Approach
1. **Neural Network Classifier**:
   - Embedding layer untuk fitur kategorik
   - Dense layer untuk fitur numerik
   - Output sigmoid untuk klasifikasi biner
2. **TFX Pipeline**:
   - Otomatisasi alur kerja ML
   - Validasi data dan model terintegrasi       


## Data Understanding

Source yang digunakan dalam proyek ini adalah **Loan Prediction Problem Dataset** yang diunduh dari [https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).  

### **1. Deskripsi Dataset**        
Dataset ini berisi informasi peminjam dan kelayakan pengajuan pinjaman dengan **614 entri** dan **13 fitur**        
- **Total Kolom**: 13 (5 numerik, 8 kategorikal)    
- **Missing Value**: Terdapat pada kolom `Gender` (13), `Married` (3), `Dependents` (15), `Self_Employed` (32), `LoanAmount` (22), `Loan_Amount_Term` (14), dan `Credit_History` (50).  
- **Duplikat**: Tidak ada data duplikat (semua `Loan_ID` unik).  

---

### **2. Detail Fitur**  

#### **A. Fitur Kategorikal**  
1. **`Loan_ID`**  
   - **Deskripsi**: ID unik untuk setiap pengajuan pinjaman.  
   - **Tipe Data**: String (`object`)  
   - **Contoh Nilai**: `'LP001002'`, `'LP001003'`  
   - **Catatan**: Tidak relevan untuk analisis/modeling (hanya identifier).  

2. **`Gender`**  
   - **Deskripsi**: Jenis kelamin peminjam.  
   - **Tipe Data**: String (`object`)  
   - **Nilai Unik**: `'Male'`, `'Female'`, `NaN` (13 missing values)  

3. **`Married`**  
   - **Deskripsi**: Status pernikahan peminjam (`Yes`/`No`).  
   - **Tipe Data**: String (`object`)  
   - **Nilai Unik**: `'Yes'`, `'No'`, `NaN` (3 missing values)  

4. **`Dependents`**  
   - **Deskripsi**: Jumlah tanggungan peminjam.  
   - **Tipe Data**: String (`object`)  
   - **Nilai Unik**: `'0'`, `'1'`, `'2'`, `'3+'`, `NaN` (15 missing values)  

5. **`Education`**  
   - **Deskripsi**: Latar belakang pendidikan peminjam.  
   - **Tipe Data**: String (`object`)  
   - **Nilai Unik**: `'Graduate'`, `'Not Graduate'`  

6. **`Self_Employed`**  
   - **Deskripsi**: Status pekerjaan (wiraswasta atau tidak).  
   - **Tipe Data**: String (`object`)  
   - **Nilai Unik**: `'Yes'`, `'No'`, `NaN` (32 missing values)  

7. **`Property_Area`**  
   - **Deskripsi**: Lokasi properti yang dijaminkan.  
   - **Tipe Data**: String (`object`)  
   - **Nilai Unik**: `'Urban'`, `'Rural'`, `'Semiurban'`    

8. **`Loan_Status`** (Target)  
   - **Deskripsi**: Status persetujuan pinjaman (`Y` = Disetujui, `N` = Ditolak).  
   - **Tipe Data**: String (`object`)  
   - **Nilai Unik**: `'Y'`, `'N'`   

#### **B. Fitur Numerik**  
1. **`ApplicantIncome`**  
   - **Deskripsi**: Pendapatan peminjam (per bulan).  
   - **Tipe Data**: Integer (`int64`)    

2. **`CoapplicantIncome`**  
   - **Deskripsi**: Pendapatan pasangan/pendamping peminjam.  
   - **Tipe Data**: Float (`float64`)  

3. **`LoanAmount`**  
   - **Deskripsi**: Jumlah pinjaman (dalam ribuan).  
   - **Tipe Data**: Float (`float64`)  

4. **`Loan_Amount_Term`**  
   - **Deskripsi**: Tenor pinjaman (dalam bulan).  
   - **Tipe Data**: Float (`float64`)  
   - **Nilai Unik**: `360` (30 tahun), `180` (15 tahun), `120`, `240`, dst.  

5. **`Credit_History`**  
   - **Deskripsi**: Riwayat kredit (`1` = Baik, `0` = Buruk).  
   - **Tipe Data**: Float (`float64`)  
   - **Nilai Unik**: `1.0`, `0.0`, `NaN` (50 missing values)   

## Data Preparation

Langkah-langkah preprocessing:
1. **Handling Missing Values**:
   - Numerik: diisi dengan median
   - Kategorik: diisi dengan modus

2. **Feature Engineering**:
   - Normalisasi fitur numerik
   - Encoding fitur kategorik
   - Transformasi 'Dependents' (3+ → 4)

3. **Data Splitting**:
   - 80% training
   - 20% evaluasi

4. **TFX Transform**:
   - Otomatisasi transformasi fitur
   - Generasi metadata schema

```python
def preprocessing_fn(inputs):
    outputs = {}
    # Transformasi kategorik
    for key in CATEGORICAL_FEATURES:
        outputs[key] = tft.compute_and_apply_vocabulary(inputs[key])
    
    # Transformasi numerik
    for key in NUMERICAL_FEATURES:
        outputs[key] = tf.cast(inputs[key], tf.float32)
    
    return outputs
```

## Modeling

### Arsitektur Model

Model neural network ini menggabungkan embedding untuk fitur kategorikal dan normalisasi untuk fitur numerik. Data yang telah ditransformasikan melalui TensorFlow Transform kemudian diolah dalam beberapa lapisan Dense dengan aktivasi ReLU dan Dropout untuk regulasi. Output dari model adalah unit Dense dengan aktivasi sigmoid untuk klasifikasi biner.     
```python
# Input Layer
inputs = {
    key: layers.Input(shape=(1,), dtype=tf.int64 if key in CATEGORICAL_FEATURES else tf.float32) 
    for key in FEATURE_KEYS
}

# Embedding untuk fitur kategorik
categorical = []
for key in CATEGORICAL_FEATURES:
    emb = layers.Embedding(vocab_size, emb_dim)(inputs[key])
    categorical.append(layers.Reshape((emb_dim,))(emb))
    
# Dense untuk fitur numerik
numerical = layers.Dense(16, activation='relu')(
    layers.concatenate([inputs[key] for key in NUMERICAL_FEATURES])
)

# Gabungkan dan output
x = layers.concatenate([categorical, numerical])
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=inputs, outputs=outputs)
```

### Training Configuration
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics: Accuracy
- Callbacks: Early Stopping
- Epochs: 30
- Batch Size: 64

### Performa Model
- Akurasi Training: 82%
- Akurasi Validasi: 81%

## Evaluation

### Metrik Evaluasi
1. **Binary Accuracy**: 0.82
2. **AUC**: 0.81

```python
eval_config = tfma.EvalConfig(
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='BinaryAccuracy'),
            tfma.MetricConfig(class_name='AUC')
        ])
    ]
)
```

## Deployment

Model di-deploy dengan:
- Docker image yang berisi model dalam format TensorFlow Serving
- Dockerfilenya seperti berikut ini:        
```
FROM tensorflow/serving:latest

WORKDIR /app

COPY ./serving_model /models

ENV MODEL_NAME=loan_model
```     

Untuk menjalankan:
1. Build docker image: 
   `docker build -t loan-model-tf-serving .`
2. Run container: 
   `docker run -d -p 8081:8501 --name loan_model_container loan-model-tf-serving`

Model dapat diakses via API endpoint di `http://localhost:8081/v1/models/loan_model` 
- Contoh Request:
```json
{
    "Gender": "Male",
    "Married": "No",
    "Dependents": 0,
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5849,
    "CoapplicantIncome": 0,
    "LoanAmount": 128,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban"
}
```     
- Hasil:
```
Loan Status: Y
```

**Kesimpulan**:
Model mencapai target akurasi >80%, yakni 82% dan siap digunakan untuk otomatisasi proses persetujuan pinjaman. Integrasi sistem prediksi ini ke dalam sistem produksi dapat meningkatkan efisiensi operasional.
