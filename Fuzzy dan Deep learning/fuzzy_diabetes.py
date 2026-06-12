"""
=============================================================
  SISTEM PREDIKSI RISIKO DIABETES
  Fuzzy Mamdani & Sugeno + Deep Learning Ensemble
  Dataset : BRFSS 2015 (Kaggle)
  Fitur   : BMI, HighBP, HighChol (+ 15 fitur tambahan untuk DL)
  Output  : TIDAK / MUNGKIN / IYA
=============================================================

  ARSITEKTUR HYBRID:
  ┌─────────────────────────────────────┐
  │         INPUT PASIEN                │
  │  (BMI, HighBP, HighChol, + lainnya) │
  └──────────┬──────────────────────────┘
             │
    ┌─────────┴──────────┐
    │                    │
    ▼                    ▼
  FUZZY SISTEM      DEEP LEARNING
  (Mamdani+Sugeno)  (18 fitur, sigmoid)
    │                    │
    └─────────┬──────────┘
              ▼
        ENSEMBLE FUSION
        (weighted average)
              │
              ▼
       PREDIKSI AKHIR
       TIDAK/MUNGKIN/IYA
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib
import os

# =============================================================
#  BAGIAN 1 — FUNGSI KEANGGOTAAN DASAR  [TIDAK DIUBAH]
# =============================================================

def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    mask1 = (x >= a) & (x <= b)
    if b != a:
        result[mask1] = (x[mask1] - a) / (b - a)
    mask2 = (x > b) & (x <= c)
    if c != b:
        result[mask2] = (c - x[mask2]) / (c - b)
    result[x == b] = 1.0
    return result


def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    mask1 = (x >= a) & (x < b)
    if b != a:
        result[mask1] = (x[mask1] - a) / (b - a)
    mask2 = (x >= b) & (x <= c)
    result[mask2] = 1.0
    mask3 = (x > c) & (x <= d)
    if d != c:
        result[mask3] = (d - x[mask3]) / (d - c)
    return result


# =============================================================
#  BAGIAN 2 — VARIABEL LINGUISTIK & MEMBERSHIP FUNCTION  [TIDAK DIUBAH]
# =============================================================

def mf_bmi_kurus(x):    return trapmf(x, 10, 10, 16, 18.5)
def mf_bmi_normal(x):   return trimf(x,  16, 21.75, 24.9)
def mf_bmi_gemuk(x):    return trimf(x,  23, 27.45, 29.9)
def mf_bmi_obesitas(x): return trapmf(x, 28, 30, 70, 70)

def mf_bp_rendah(x): return trapmf(x, 0, 0, 2, 4)
def mf_bp_sedang(x): return trimf(x,  3, 5, 7)
def mf_bp_tinggi(x): return trapmf(x, 6, 8, 10, 10)

def mf_chol_rendah(x): return trapmf(x, 0, 0, 2, 4)
def mf_chol_sedang(x): return trimf(x,  3, 5, 7)
def mf_chol_tinggi(x): return trapmf(x, 6, 8, 10, 10)

def mf_tidak(x):   return trapmf(x, 0, 0, 2, 4)
def mf_mungkin(x): return trimf(x,  3, 5, 7)
def mf_iya(x):     return trapmf(x, 6, 8, 10, 10)


# =============================================================
#  BAGIAN 3 — RULE BASE (20 Rules)  [TIDAK DIUBAH]
# =============================================================

RULES = [
    (mf_bmi_normal,   mf_bp_rendah, mf_chol_rendah, mf_tidak,   2.0),
    (mf_bmi_kurus,    mf_bp_rendah, mf_chol_rendah, mf_tidak,   2.0),
    (mf_bmi_normal,   mf_bp_rendah, mf_chol_sedang, mf_tidak,   2.5),
    (mf_bmi_kurus,    mf_bp_sedang, mf_chol_rendah, mf_tidak,   2.0),
    (mf_bmi_normal,   mf_bp_sedang, mf_chol_rendah, mf_tidak,   2.5),
    (mf_bmi_gemuk,    mf_bp_rendah, mf_chol_rendah, mf_mungkin, 5.0),
    (mf_bmi_normal,   mf_bp_tinggi, mf_chol_sedang, mf_mungkin, 5.0),
    (mf_bmi_gemuk,    mf_bp_sedang, mf_chol_sedang, mf_mungkin, 5.0),
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_rendah, mf_mungkin, 5.0),
    (mf_bmi_normal,   mf_bp_sedang, mf_chol_sedang, mf_mungkin, 4.5),
    (mf_bmi_gemuk,    mf_bp_rendah, mf_chol_sedang, mf_mungkin, 5.0),
    (mf_bmi_kurus,    mf_bp_tinggi, mf_chol_sedang, mf_mungkin, 4.5),
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_tinggi, mf_iya,     8.5),
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_sedang, mf_iya,     8.5),
    (mf_bmi_gemuk,    mf_bp_tinggi, mf_chol_tinggi, mf_iya,     8.0),
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_tinggi, mf_iya,     8.0),
    (mf_bmi_gemuk,    mf_bp_tinggi, mf_chol_sedang, mf_iya,     7.5),
    (mf_bmi_normal,   mf_bp_tinggi, mf_chol_tinggi, mf_iya,     7.0),
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_tinggi, mf_iya,     7.5),
    (mf_bmi_gemuk,    mf_bp_sedang, mf_chol_tinggi, mf_iya,     7.5),
]


# =============================================================
#  BAGIAN 4 — PREPROCESSING  [TIDAK DIUBAH]
# =============================================================

def preprocess(df):
    df = df.copy()
    df['HighBP_scaled']   = df['HighBP'].map({0: 2.5, 1: 7.5})
    df['HighChol_scaled'] = df['HighChol'].map({0: 2.5, 1: 7.5})
    return df


def scale_bp(highbp_binary):
    return 7.5 if highbp_binary == 1 else 2.5


def scale_chol(highchol_binary):
    return 7.5 if highchol_binary == 1 else 2.5


# =============================================================
#  BAGIAN 5 — FUZZIFIKASI  [TIDAK DIUBAH]
# =============================================================

FN_MAP = {
    mf_bmi_kurus:    'bmi_kurus',
    mf_bmi_normal:   'bmi_normal',
    mf_bmi_gemuk:    'bmi_gemuk',
    mf_bmi_obesitas: 'bmi_obesitas',
    mf_bp_rendah:    'bp_rendah',
    mf_bp_sedang:    'bp_sedang',
    mf_bp_tinggi:    'bp_tinggi',
    mf_chol_rendah:  'chol_rendah',
    mf_chol_sedang:  'chol_sedang',
    mf_chol_tinggi:  'chol_tinggi',
}


def fuzzifikasi(bmi, bp_scaled, chol_scaled):
    def mu(fn, val):
        return float(fn(np.array([val]))[0])

    return {
        'bmi_kurus':    mu(mf_bmi_kurus,    bmi),
        'bmi_normal':   mu(mf_bmi_normal,   bmi),
        'bmi_gemuk':    mu(mf_bmi_gemuk,    bmi),
        'bmi_obesitas': mu(mf_bmi_obesitas, bmi),
        'bp_rendah':    mu(mf_bp_rendah,    bp_scaled),
        'bp_sedang':    mu(mf_bp_sedang,    bp_scaled),
        'bp_tinggi':    mu(mf_bp_tinggi,    bp_scaled),
        'chol_rendah':  mu(mf_chol_rendah,  chol_scaled),
        'chol_sedang':  mu(mf_chol_sedang,  chol_scaled),
        'chol_tinggi':  mu(mf_chol_tinggi,  chol_scaled),
    }


# =============================================================
#  BAGIAN 6 — FUZZY MAMDANI  [TIDAK DIUBAH]
# =============================================================

X_OUTPUT = np.linspace(0, 10, 1000)


def inferensi_mamdani(fuzz):
    agregat = np.zeros(len(X_OUTPUT))
    for rule in RULES:
        fn_bmi, fn_bp, fn_chol, fn_out, _ = rule
        alpha = min(
            fuzz[FN_MAP[fn_bmi]],
            fuzz[FN_MAP[fn_bp]],
            fuzz[FN_MAP[fn_chol]],
        )
        if alpha > 0:
            konsekuen = np.minimum(alpha, fn_out(X_OUTPUT))
            agregat   = np.maximum(agregat, konsekuen)
    return agregat


def defuzzifikasi_centroid(agregat):
    denom = np.sum(agregat)
    if denom == 0:
        return 5.0
    return float(np.sum(X_OUTPUT * agregat) / denom)


def prediksi_mamdani(bmi, bp_scaled, chol_scaled):
    fuzz    = fuzzifikasi(bmi, bp_scaled, chol_scaled)
    agregat = inferensi_mamdani(fuzz)
    skor    = defuzzifikasi_centroid(agregat)
    label, biner = skor_ke_label(skor)
    return skor, label, biner


# =============================================================
#  BAGIAN 7 — FUZZY SUGENO  [TIDAK DIUBAH]
# =============================================================

def inferensi_sugeno(fuzz):
    alphas, zs = [], []
    for rule in RULES:
        fn_bmi, fn_bp, fn_chol, _, z_out = rule
        alpha = min(
            fuzz[FN_MAP[fn_bmi]],
            fuzz[FN_MAP[fn_bp]],
            fuzz[FN_MAP[fn_chol]],
        )
        alphas.append(alpha)
        zs.append(z_out)
    return alphas, zs


def defuzzifikasi_weighted_average(alphas, zs):
    total = sum(alphas)
    if total == 0:
        return 5.0
    return float(sum(a * z for a, z in zip(alphas, zs)) / total)


def prediksi_sugeno(bmi, bp_scaled, chol_scaled):
    fuzz         = fuzzifikasi(bmi, bp_scaled, chol_scaled)
    alphas, zs   = inferensi_sugeno(fuzz)
    skor         = defuzzifikasi_weighted_average(alphas, zs)
    label, biner = skor_ke_label(skor)
    return skor, label, biner


# =============================================================
#  BAGIAN 8 — KONVERSI SKOR → LABEL  [TIDAK DIUBAH]
# =============================================================

def skor_ke_label(skor):
    if skor < 4.0:
        return 'TIDAK', 0
    elif skor < 6.5:
        return 'MUNGKIN', 0
    else:
        return 'IYA', 1


# =============================================================
#  BAGIAN 9 — DEEP LEARNING MODEL  [DIPERBAIKI]
# =============================================================

# Fitur yang digunakan oleh DL model
DL_FEATURES = [
    'HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits',
    'Veggies', 'HvyAlcoholConsump', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex',
    'Age', 'Education', 'Income'
]


def build_dl_model(input_dim: int) -> tf.keras.Model:
    """
    Bangun arsitektur Deep Learning.
    Dipisahkan agar bisa dipakai ulang saat load model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


def train_deep_learning(df: pd.DataFrame, epochs: int = 20, batch_size: int = 64):
    """
    Latih model Deep Learning dari DataFrame BRFSS.

    FIX dari versi sebelumnya:
    - Mengembalikan scaler dan nama fitur bersama model
      agar prediksi baru bisa di-transform dengan benar
    - Menggunakan EarlyStopping agar tidak overfit
    - Menyimpan riwayat training untuk visualisasi

    Returns:
        model   : tf.keras.Model yang sudah dilatih
        scaler  : StandardScaler yang sudah di-fit
        history : riwayat training
    """
    data = df[DL_FEATURES + ['Diabetes_012']].dropna()

    X = data[DL_FEATURES].values
    y = data['Diabetes_012'].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Hitung class weight untuk menangani imbalanced dataset BRFSS
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    class_weight = {0: 1.0, 1: n_neg / n_pos}
    print(f"  Class weight → 0: 1.0 | 1: {class_weight[1]:.2f}")

    model = build_dl_model(input_dim=X_train.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=5,
            mode='max', restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluasi
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    print(f"\n  DL Test Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred,
                                target_names=['Tidak Diabetes', 'Diabetes']))

    return model, scaler, history


def simpan_model_dl(model, scaler, path_dir='saved_model'):
    """Simpan model DL dan scaler ke disk."""
    os.makedirs(path_dir, exist_ok=True)
    model.save(os.path.join(path_dir, 'dl_model.keras'))
    joblib.dump(scaler, os.path.join(path_dir, 'scaler.pkl'))
    print(f"  Model disimpan di '{path_dir}/'")


def load_model_dl(path_dir='saved_model'):
    """Load model DL dan scaler dari disk."""
    model  = tf.keras.models.load_model(os.path.join(path_dir, 'dl_model.keras'))
    scaler = joblib.load(os.path.join(path_dir, 'scaler.pkl'))
    return model, scaler


# =============================================================
#  BAGIAN 10 — ENSEMBLE FUSION  [BARU — inti integrasi DL + Fuzzy]
# =============================================================

def ensemble_fusion(skor_fuzzy: float, dl_prob: float,
                    w_fuzzy: float = 0.4, w_dl: float = 0.6) -> float:
    """
    Gabungkan skor fuzzy (skala 0–10) dengan probabilitas DL (0–1)
    menjadi satu skor ensemble pada skala 0–10.

    Rumus:
        skor_dl_scaled = dl_prob × 10          # normalisasi ke 0–10
        skor_akhir = w_fuzzy × skor_fuzzy + w_dl × skor_dl_scaled

    Bobot default:
        w_fuzzy = 0.4  (fuzzy pakai 3 fitur, lebih interpretatif)
        w_dl    = 0.6  (DL pakai 18 fitur, lebih prediktif)

    Returns:
        skor_ensemble (float, range 0–10)
    """
    skor_dl_scaled = dl_prob * 10.0
    return w_fuzzy * skor_fuzzy + w_dl * skor_dl_scaled


# =============================================================
#  BAGIAN 11 — PREDIKSI SATU PASIEN  [DIPERBAIKI]
# =============================================================

def prediksi_pasien(bmi, highbp, highchol):
    """
    Prediksi HANYA menggunakan Fuzzy (tanpa DL).
    Tetap tersedia untuk kompatibilitas.
    """
    bp_scaled   = scale_bp(highbp)
    chol_scaled = scale_chol(highchol)
    skor_m, label_m, biner_m = prediksi_mamdani(bmi, bp_scaled, chol_scaled)
    skor_s, label_s, biner_s = prediksi_sugeno(bmi, bp_scaled, chol_scaled)

    return {
        'mamdani': {'skor': round(skor_m, 4), 'label': label_m, 'biner': biner_m},
        'sugeno':  {'skor': round(skor_s, 4), 'label': label_s, 'biner': biner_s},
    }


def prediksi_pasien_hybrid(row: dict, dl_model, scaler,
                           w_fuzzy: float = 0.4, w_dl: float = 0.6) -> dict:
    """
    Prediksi hybrid: Fuzzy (Mamdani + Sugeno) + Deep Learning.

    FIX dari versi sebelumnya:
    - dl_probability sekarang BENAR-BENAR digunakan dalam fusion
    - Menghasilkan skor_ensemble dan label_ensemble yang merupakan
      output gabungan nyata, bukan hanya menampilkan keduanya secara terpisah
    - row harus berupa dict dengan semua kolom DL_FEATURES
      (minimal: HighBP, HighChol, BMI — sisanya bisa 0 jika tidak tersedia)

    Parameter:
        row     : dict data pasien (harus mengandung DL_FEATURES)
        dl_model: tf.keras.Model yang sudah dilatih
        scaler  : StandardScaler yang sudah di-fit
        w_fuzzy : bobot untuk skor fuzzy (default 0.4)
        w_dl    : bobot untuk probabilitas DL (default 0.6)

    Returns:
        dict hasil prediksi lengkap termasuk 'ensemble'
    """
    # ── 1. DEEP LEARNING prediction ─────────────────────────
    # Pastikan semua fitur ada; isi 0 untuk kolom yang tidak ada
    row_filled = {f: row.get(f, 0) for f in DL_FEATURES}
    X_input  = pd.DataFrame([row_filled])[DL_FEATURES]
    X_scaled = scaler.transform(X_input)
    dl_prob  = float(dl_model.predict(X_scaled, verbose=0)[0][0])

    # ── 2. FUZZY prediction ──────────────────────────────────
    bmi         = row['BMI']
    bp_scaled   = scale_bp(row['HighBP'])
    chol_scaled = scale_chol(row['HighChol'])

    skor_m, label_m, biner_m = prediksi_mamdani(bmi, bp_scaled, chol_scaled)
    skor_s, label_s, biner_s = prediksi_sugeno(bmi, bp_scaled, chol_scaled)

    # Gunakan rata-rata skor Mamdani & Sugeno sebagai input fuzzy ke ensemble
    skor_fuzzy_avg = (skor_m + skor_s) / 2.0

    # ── 3. ENSEMBLE fusion ───────────────────────────────────
    skor_ensemble = ensemble_fusion(skor_fuzzy_avg, dl_prob, w_fuzzy, w_dl)
    label_ensemble, biner_ensemble = skor_ke_label(skor_ensemble)

    return {
        'dl_probability': round(dl_prob, 4),
        'mamdani': {
            'skor': round(skor_m, 4),
            'label': label_m,
            'biner': biner_m
        },
        'sugeno': {
            'skor': round(skor_s, 4),
            'label': label_s,
            'biner': biner_s
        },
        'ensemble': {
            # Output utama — gabungan fuzzy + DL
            'skor': round(skor_ensemble, 4),
            'label': label_ensemble,
            'biner': biner_ensemble,
            'bobot_fuzzy': w_fuzzy,
            'bobot_dl': w_dl,
        }
    }


# =============================================================
#  BAGIAN 12 — JALANKAN PADA DATASET  [DIPERBAIKI]
# =============================================================

def jalankan_pada_dataset(csv_path, n_sampel=5000, random_state=42,
                          dl_model=None, scaler=None):
    """
    Load dataset BRFSS, preprocessing, jalankan Mamdani, Sugeno,
    dan opsional Ensemble (jika dl_model & scaler diberikan).
    """
    df   = pd.read_csv(csv_path)
    data = df[DL_FEATURES + ['Diabetes_012']].dropna()
    data = preprocess(data)
    sample = data.sample(n=n_sampel, random_state=random_state).reset_index(drop=True)

    print(f"Dataset dimuat: {len(sample)} sampel")
    print(f"Distribusi label:\n{sample['Diabetes_012'].value_counts()}\n")

    # ── Fuzzy saja ──────────────────────────────────────────
    print("Menjalankan Fuzzy Mamdani...")
    skor_m, label_m, pred_m = [], [], []
    for _, row in sample.iterrows():
        s, l, b = prediksi_mamdani(row['BMI'], row['HighBP_scaled'], row['HighChol_scaled'])
        skor_m.append(s); label_m.append(l); pred_m.append(b)

    print("Menjalankan Fuzzy Sugeno...")
    skor_s, label_s, pred_s = [], [], []
    for _, row in sample.iterrows():
        s, l, b = prediksi_sugeno(row['BMI'], row['HighBP_scaled'], row['HighChol_scaled'])
        skor_s.append(s); label_s.append(l); pred_s.append(b)

    sample['skor_mamdani']  = skor_m
    sample['label_mamdani'] = label_m
    sample['pred_mamdani']  = pred_m
    sample['skor_sugeno']   = skor_s
    sample['label_sugeno']  = label_s
    sample['pred_sugeno']   = pred_s

    y_true = sample['Diabetes_012'].astype(int).values

    # ── Ensemble jika model tersedia ────────────────────────
    pred_ensemble = None
    if dl_model is not None and scaler is not None:
        print("Menjalankan Hybrid Ensemble (Fuzzy + DL)...")
        skor_e, label_e, pred_e = [], [], []
        for _, row in sample.iterrows():
            hasil = prediksi_pasien_hybrid(row.to_dict(), dl_model, scaler)
            skor_e.append(hasil['ensemble']['skor'])
            label_e.append(hasil['ensemble']['label'])
            pred_e.append(hasil['ensemble']['biner'])
        sample['skor_ensemble']  = skor_e
        sample['label_ensemble'] = label_e
        sample['pred_ensemble']  = pred_e
        pred_ensemble = pred_e

    # ── Evaluasi ────────────────────────────────────────────
    print("\n" + "="*55)
    print("  EVALUASI PERFORMA")
    print("="*55)
    print(f"\n🔵 FUZZY MAMDANI — Akurasi: {accuracy_score(y_true, pred_m)*100:.2f}%")
    print(classification_report(y_true, pred_m, target_names=['Tidak Diabetes', 'Diabetes']))
    print(f"\n🟠 FUZZY SUGENO  — Akurasi: {accuracy_score(y_true, pred_s)*100:.2f}%")
    print(classification_report(y_true, pred_s, target_names=['Tidak Diabetes', 'Diabetes']))
    if pred_ensemble is not None:
        print(f"\n🟢 HYBRID ENSEMBLE — Akurasi: {accuracy_score(y_true, pred_ensemble)*100:.2f}%")
        print(classification_report(y_true, pred_ensemble,
                                    target_names=['Tidak Diabetes', 'Diabetes']))

    plot_perbandingan(sample, y_true, pred_m, pred_s, pred_ensemble)
    return sample


# =============================================================
#  BAGIAN 13 — VISUALISASI  [DIPERBAIKI]
# =============================================================

def plot_membership_functions():
    """Visualisasi semua membership function."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Fungsi Keanggotaan (Membership Functions)', fontsize=15, fontweight='bold')

    ax, x = axes[0, 0], np.linspace(10, 70, 500)
    ax.plot(x, mf_bmi_kurus(x),    'b-', lw=2, label='Kurus')
    ax.plot(x, mf_bmi_normal(x),   'g-', lw=2, label='Normal')
    ax.plot(x, mf_bmi_gemuk(x),    'y-', lw=2, label='Gemuk')
    ax.plot(x, mf_bmi_obesitas(x), 'r-', lw=2, label='Obesitas')
    ax.set_title('Input 1: BMI', fontweight='bold')
    ax.set_xlabel('BMI'); ax.set_ylabel('Derajat Keanggotaan')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.1)

    ax, x = axes[0, 1], np.linspace(0, 10, 500)
    ax.plot(x, mf_bp_rendah(x), 'g-', lw=2, label='Rendah')
    ax.plot(x, mf_bp_sedang(x), 'y-', lw=2, label='Sedang')
    ax.plot(x, mf_bp_tinggi(x), 'r-', lw=2, label='Tinggi')
    ax.set_title('Input 2: HighBP (mapped 0–10)', fontweight='bold')
    ax.set_xlabel('Skor BP'); ax.set_ylabel('Derajat Keanggotaan')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.1)

    ax, x = axes[1, 0], np.linspace(0, 10, 500)
    ax.plot(x, mf_chol_rendah(x), 'g-', lw=2, label='Rendah')
    ax.plot(x, mf_chol_sedang(x), 'y-', lw=2, label='Sedang')
    ax.plot(x, mf_chol_tinggi(x), 'r-', lw=2, label='Tinggi')
    ax.set_title('Input 3: HighChol (mapped 0–10)', fontweight='bold')
    ax.set_xlabel('Skor Kolesterol'); ax.set_ylabel('Derajat Keanggotaan')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.1)

    ax, x = axes[1, 1], np.linspace(0, 10, 500)
    ax.plot(x, mf_tidak(x),   'g-', lw=2, label='TIDAK')
    ax.plot(x, mf_mungkin(x), 'y-', lw=2, label='MUNGKIN')
    ax.plot(x, mf_iya(x),     'r-', lw=2, label='IYA')
    ax.set_title('Output: Risiko Diabetes', fontweight='bold')
    ax.set_xlabel('Skor Risiko (0–10)'); ax.set_ylabel('Derajat Keanggotaan')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.savefig('membership_functions.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_history(history):
    """Plot kurva training DL model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Deep Learning Training History', fontsize=13, fontweight='bold')

    axes[0].plot(history.history['loss'],     label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['auc'],     label='Train AUC')
    axes[1].plot(history.history['val_auc'], label='Val AUC')
    axes[1].set_title('AUC'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dl_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_perbandingan(sample, y_true, pred_m, pred_s, pred_ensemble=None):
    """Visualisasi perbandingan Mamdani vs Sugeno (vs Ensemble)."""
    acc_m = accuracy_score(y_true, pred_m)
    acc_s = accuracy_score(y_true, pred_s)

    metode = ['Mamdani', 'Sugeno']
    akurasi = [acc_m * 100, acc_s * 100]
    warna   = ['#3B82F6', '#F97316']

    if pred_ensemble is not None:
        acc_e = accuracy_score(y_true, pred_ensemble)
        metode.append('Ensemble')
        akurasi.append(acc_e * 100)
        warna.append('#22C55E')

    n_col = 3 if pred_ensemble is not None else 2
    fig, axes = plt.subplots(1, n_col + 1, figsize=(5 * (n_col + 1), 5))
    fig.suptitle('Perbandingan Metode Prediksi', fontsize=14, fontweight='bold')

    # Akurasi
    ax = axes[0]
    bars = ax.bar(metode, akurasi, color=warna, width=0.4, edgecolor='white')
    for bar, val in zip(bars, akurasi):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.2f}%', ha='center', fontweight='bold')
    ax.set_title('Akurasi'); ax.set_ylabel('Akurasi (%)'); ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    # Distribusi tiap metode
    for i, (col, title) in enumerate(
        [('label_mamdani', 'Distribusi Mamdani'),
         ('label_sugeno',  'Distribusi Sugeno'),
         ('label_ensemble', 'Distribusi Ensemble')][:n_col]
    ):
        ax = axes[i + 1]
        if col in sample.columns:
            cnt = sample[col].value_counts()
            ax.bar(cnt.index, cnt.values,
                   color=['#22C55E', '#EAB308', '#EF4444'], edgecolor='white')
        ax.set_title(title); ax.set_ylabel('Jumlah Sampel')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('perbandingan_metode.png', dpi=150, bbox_inches='tight')
    plt.show()

    diff = np.abs(np.array(sample['skor_mamdani']) - np.array(sample['skor_sugeno']))
    print(f"\nRata-rata selisih skor Mamdani vs Sugeno : {diff.mean():.4f}")
    print(f"Selisih maksimum                          : {diff.max():.4f}")


# =============================================================
#  MAIN
# =============================================================

if __name__ == '__main__':
    # ── Visualisasi membership function
    plot_membership_functions()

    # ── Test prediksi fuzzy saja (tanpa DL)
    print("\n" + "="*55)
    print("  DEMO PREDIKSI FUZZY — SATU PASIEN")
    print("="*55)
    kasus = [
        ("Berisiko tinggi", 35.0, 1, 1),
        ("Normal/sehat",    22.0, 0, 0),
        ("Borderline",      27.5, 1, 0),
    ]
    for nama, bmi, bp, chol in kasus:
        hasil = prediksi_pasien(bmi, bp, chol)
        print(f"\n  [{nama}] BMI={bmi}, HighBP={bp}, HighChol={chol}")
        print(f"  🔵 Mamdani → Skor: {hasil['mamdani']['skor']} | {hasil['mamdani']['label']}")
        print(f"  🟠 Sugeno  → Skor: {hasil['sugeno']['skor']}  | {hasil['sugeno']['label']}")

    # ── Latih DL dan jalankan ensemble pada dataset ──────────
    # Aktifkan blok di bawah jika file CSV sudah tersedia

    CSV_PATH = 'Diabetes_012_health_indicators_BRFSS2015.csv'

    if os.path.exists(CSV_PATH):
        print("\n" + "="*55)
        print("  LATIH DEEP LEARNING MODEL")
        print("="*55)
        df_full = pd.read_csv(CSV_PATH)
        dl_model, scaler, history = train_deep_learning(df_full, epochs=30)
        plot_training_history(history)
        simpan_model_dl(dl_model, scaler)

        print("\n" + "="*55)
        print("  DEMO PREDIKSI HYBRID — SATU PASIEN")
        print("="*55)
        # Contoh pasien dengan fitur lengkap (18 fitur)
        pasien_contoh = {
            'BMI': 35.0, 'HighBP': 1, 'HighChol': 1,
            'Smoker': 0, 'Stroke': 0, 'HeartDiseaseorAttack': 0,
            'PhysActivity': 0, 'Fruits': 0, 'Veggies': 0,
            'HvyAlcoholConsump': 0, 'GenHlth': 4,
            'MentHlth': 0, 'PhysHlth': 5, 'DiffWalk': 1,
            'Sex': 1, 'Age': 9, 'Education': 4, 'Income': 5
        }
        hasil = prediksi_pasien_hybrid(pasien_contoh, dl_model, scaler)
        print(f"\n  Input: {pasien_contoh}")
        print(f"  🔵 Mamdani  → Skor: {hasil['mamdani']['skor']}  | {hasil['mamdani']['label']}")
        print(f"  🟠 Sugeno   → Skor: {hasil['sugeno']['skor']}   | {hasil['sugeno']['label']}")
        print(f"  🤖 DL Prob  → {hasil['dl_probability']:.4f}")
        print(f"  🟢 ENSEMBLE → Skor: {hasil['ensemble']['skor']} | {hasil['ensemble']['label']}")

        print("\n" + "="*55)
        print("  EVALUASI PADA DATASET")
        print("="*55)
        jalankan_pada_dataset(CSV_PATH, n_sampel=5000,
                              dl_model=dl_model, scaler=scaler)
    else:
        print(f"\n[INFO] File '{CSV_PATH}' tidak ditemukan.")
        print("       Letakkan file CSV BRFSS untuk menjalankan evaluasi penuh.")