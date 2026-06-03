"""
=============================================================
  SISTEM PREDIKSI RISIKO DIABETES
  Fuzzy Mamdani & Sugeno — From Scratch
  Dataset : BRFSS 2015 (Kaggle)
  Fitur   : BMI, HighBP, HighChol
  Output  : TIDAK / MUNGKIN / IYA
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# =============================================================
#  BAGIAN 1 — FUNGSI KEANGGOTAAN DASAR
# =============================================================

def trimf(x, a, b, c):
    """
    Fungsi keanggotaan SEGITIGA (Triangular Membership Function)
      a = titik kiri  (derajat 0)
      b = titik puncak (derajat 1)
      c = titik kanan (derajat 0)
    """
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
    """
    Fungsi keanggotaan TRAPESIUM (Trapezoidal Membership Function)
      a = titik kiri bawah
      b = titik kiri atas  (mulai plateau)
      c = titik kanan atas (akhir plateau)
      d = titik kanan bawah
    """
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
#  BAGIAN 2 — VARIABEL LINGUISTIK & MEMBERSHIP FUNCTION
# =============================================================

# ── INPUT 1: BMI (range 10–70) ──────────────────────────────
def mf_bmi_kurus(x):    return trapmf(x, 10, 10, 16, 18.5)
def mf_bmi_normal(x):   return trimf(x,  16, 21.75, 24.9)
def mf_bmi_gemuk(x):    return trimf(x,  23, 27.45, 29.9)
def mf_bmi_obesitas(x): return trapmf(x, 28, 30, 70, 70)

# ── INPUT 2: HighBP — diperluas ke skala 0–10 ───────────────
# Nilai asli binary: 0 → 2.5,  1 → 7.5
def mf_bp_rendah(x): return trapmf(x, 0, 0, 2, 4)
def mf_bp_sedang(x): return trimf(x,  3, 5, 7)
def mf_bp_tinggi(x): return trapmf(x, 6, 8, 10, 10)

# ── INPUT 3: HighChol — diperluas ke skala 0–10 ─────────────
# Nilai asli binary: 0 → 2.5,  1 → 7.5
def mf_chol_rendah(x): return trapmf(x, 0, 0, 2, 4)
def mf_chol_sedang(x): return trimf(x,  3, 5, 7)
def mf_chol_tinggi(x): return trapmf(x, 6, 8, 10, 10)

# ── OUTPUT: Risiko Diabetes (range 0–10) ────────────────────
def mf_tidak(x):   return trapmf(x, 0, 0, 2, 4)
def mf_mungkin(x): return trimf(x,  3, 5, 7)
def mf_iya(x):     return trapmf(x, 6, 8, 10, 10)


# =============================================================
#  BAGIAN 3 — RULE BASE (20 Rules)
# =============================================================
#
#  Format setiap rule (tuple 5 elemen):
#    (fn_bmi, fn_bp, fn_chol, fn_output_mamdani, skor_sugeno)
#
#  Skor Sugeno:
#    TIDAK   → 2.0
#    MUNGKIN → 5.0
#    IYA     → 8.5

RULES = [
    # ── RISIKO RENDAH (TIDAK) ── R01–R05
    # R01: BMI Normal  + BP Rendah + Kolesterol Rendah → TIDAK
    (mf_bmi_normal,   mf_bp_rendah, mf_chol_rendah, mf_tidak,   2.0),
    # R02: BMI Kurus   + BP Rendah + Kolesterol Rendah → TIDAK
    (mf_bmi_kurus,    mf_bp_rendah, mf_chol_rendah, mf_tidak,   2.0),
    # R03: BMI Normal  + BP Rendah + Kolesterol Sedang → TIDAK
    (mf_bmi_normal,   mf_bp_rendah, mf_chol_sedang, mf_tidak,   2.5),
    # R04: BMI Kurus   + BP Sedang + Kolesterol Rendah → TIDAK
    (mf_bmi_kurus,    mf_bp_sedang, mf_chol_rendah, mf_tidak,   2.0),
    # R05: BMI Normal  + BP Sedang + Kolesterol Rendah → TIDAK
    (mf_bmi_normal,   mf_bp_sedang, mf_chol_rendah, mf_tidak,   2.5),

    # ── RISIKO SEDANG (MUNGKIN) ── R06–R12
    # R06: BMI Gemuk   + BP Rendah + Kolesterol Rendah → MUNGKIN
    (mf_bmi_gemuk,    mf_bp_rendah, mf_chol_rendah, mf_mungkin, 5.0),
    # R07: BMI Normal  + BP Tinggi + Kolesterol Sedang → MUNGKIN
    (mf_bmi_normal,   mf_bp_tinggi, mf_chol_sedang, mf_mungkin, 5.0),
    # R08: BMI Gemuk   + BP Sedang + Kolesterol Sedang → MUNGKIN
    (mf_bmi_gemuk,    mf_bp_sedang, mf_chol_sedang, mf_mungkin, 5.0),
    # R09: BMI Obesitas+ BP Rendah + Kolesterol Rendah → MUNGKIN
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_rendah, mf_mungkin, 5.0),
    # R10: BMI Normal  + BP Sedang + Kolesterol Sedang → MUNGKIN
    (mf_bmi_normal,   mf_bp_sedang, mf_chol_sedang, mf_mungkin, 4.5),
    # R11: BMI Gemuk   + BP Rendah + Kolesterol Sedang → MUNGKIN
    (mf_bmi_gemuk,    mf_bp_rendah, mf_chol_sedang, mf_mungkin, 5.0),
    # R12: BMI Kurus   + BP Tinggi + Kolesterol Sedang → MUNGKIN
    (mf_bmi_kurus,    mf_bp_tinggi, mf_chol_sedang, mf_mungkin, 4.5),

    # ── RISIKO TINGGI (IYA) ── R13–R20
    # R13: BMI Obesitas+ BP Tinggi + Kolesterol Tinggi → IYA
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_tinggi, mf_iya,     8.5),
    # R14: BMI Obesitas+ BP Tinggi + Kolesterol Sedang → IYA
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_sedang, mf_iya,     8.5),
    # R15: BMI Gemuk   + BP Tinggi + Kolesterol Tinggi → IYA
    (mf_bmi_gemuk,    mf_bp_tinggi, mf_chol_tinggi, mf_iya,     8.0),
    # R16: BMI Obesitas+ BP Sedang + Kolesterol Tinggi → IYA
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_tinggi, mf_iya,     8.0),
    # R17: BMI Gemuk   + BP Tinggi + Kolesterol Sedang → IYA
    (mf_bmi_gemuk,    mf_bp_tinggi, mf_chol_sedang, mf_iya,     7.5),
    # R18: BMI Normal  + BP Tinggi + Kolesterol Tinggi → IYA
    (mf_bmi_normal,   mf_bp_tinggi, mf_chol_tinggi, mf_iya,     7.0),
    # R19: BMI Obesitas+ BP Rendah + Kolesterol Tinggi → IYA
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_tinggi, mf_iya,     7.5),
    # R20: BMI Gemuk   + BP Sedang + Kolesterol Tinggi → IYA
    (mf_bmi_gemuk,    mf_bp_sedang, mf_chol_tinggi, mf_iya,     7.5),
]


# =============================================================
#  BAGIAN 4 — PREPROCESSING
# =============================================================

def preprocess(df):
    """
    Mapping fitur binary ke skala 0–10:
      HighBP  : 0 → 2.5 (tidak hipertensi),  1 → 7.5 (hipertensi)
      HighChol: 0 → 2.5 (kolesterol normal),  1 → 7.5 (kolesterol tinggi)
    """
    df = df.copy()
    df['HighBP_scaled']   = df['HighBP'].map({0: 2.5, 1: 7.5})
    df['HighChol_scaled'] = df['HighChol'].map({0: 2.5, 1: 7.5})
    return df


def scale_bp(highbp_binary):
    """Konversi nilai binary HighBP → skala 0–10 untuk satu nilai."""
    return 7.5 if highbp_binary == 1 else 2.5


def scale_chol(highchol_binary):
    """Konversi nilai binary HighChol → skala 0–10 untuk satu nilai."""
    return 7.5 if highchol_binary == 1 else 2.5


# =============================================================
#  BAGIAN 5 — FUZZIFIKASI (dipakai bersama Mamdani & Sugeno)
# =============================================================

# Mapping fungsi → key dict fuzzifikasi (dipakai di inferensi)
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
    """
    FUZZIFIKASI: Nilai crisp → derajat keanggotaan semua himpunan.
    Parameter:
      bmi        : nilai BMI pasien (float, range 10–70)
      bp_scaled  : skor HighBP yang sudah di-scale (0–10)
      chol_scaled: skor HighChol yang sudah di-scale (0–10)
    Returns:
      dict berisi derajat keanggotaan setiap himpunan fuzzy
    """
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
#  BAGIAN 6 — FUZZY MAMDANI
# =============================================================

X_OUTPUT = np.linspace(0, 10, 1000)   # Universe of discourse output


def inferensi_mamdani(fuzz):
    """
    INFERENSI MAMDANI:
      - AND antecedent : min
      - Clip konsekuen : min(alpha, mf_output)
      - Agregasi       : max semua rule
    Returns:
      agregat (array 1000 elemen) — kurva output gabungan
    """
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
    """
    DEFUZZIFIKASI MAMDANI — Metode Centroid (Center of Gravity):
      z* = Σ(z · μ(z)) / Σ(μ(z))
    """
    denom = np.sum(agregat)
    if denom == 0:
        return 5.0
    return float(np.sum(X_OUTPUT * agregat) / denom)


def prediksi_mamdani(bmi, bp_scaled, chol_scaled):
    """
    Pipeline lengkap Fuzzy Mamdani untuk satu sampel.
    Returns: (skor_crisp, label_linguistik, nilai_biner)
    """
    fuzz    = fuzzifikasi(bmi, bp_scaled, chol_scaled)
    agregat = inferensi_mamdani(fuzz)
    skor    = defuzzifikasi_centroid(agregat)
    label, biner = skor_ke_label(skor)
    return skor, label, biner


# =============================================================
#  BAGIAN 7 — FUZZY SUGENO
# =============================================================

def inferensi_sugeno(fuzz):
    """
    INFERENSI SUGENO:
      - AND antecedent : min
      - Konsekuen      : nilai crisp konstan per rule
    Returns:
      alphas (list firing strength), zs (list skor output)
    """
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
    """
    DEFUZZIFIKASI SUGENO — Weighted Average:
      z* = Σ(αᵢ · zᵢ) / Σ(αᵢ)
    """
    total = sum(alphas)
    if total == 0:
        return 5.0
    return float(sum(a * z for a, z in zip(alphas, zs)) / total)


def prediksi_sugeno(bmi, bp_scaled, chol_scaled):
    """
    Pipeline lengkap Fuzzy Sugeno untuk satu sampel.
    Returns: (skor_crisp, label_linguistik, nilai_biner)
    """
    fuzz         = fuzzifikasi(bmi, bp_scaled, chol_scaled)
    alphas, zs   = inferensi_sugeno(fuzz)
    skor         = defuzzifikasi_weighted_average(alphas, zs)
    label, biner = skor_ke_label(skor)
    return skor, label, biner


# =============================================================
#  BAGIAN 8 — KONVERSI SKOR → LABEL
# =============================================================

def skor_ke_label(skor):
    """
    Konversi skor crisp (0–10) → label linguistik & nilai biner.
    Returns: (label_str, nilai_biner)
      TIDAK   → (0–4.0)  → 0
      MUNGKIN → (4.0–6.5) → 0  (dianggap negatif untuk metrik biner)
      IYA     → (6.5–10) → 1
    """
    if skor < 4.0:
        return 'TIDAK', 0
    elif skor < 6.5:
        return 'MUNGKIN', 0
    else:
        return 'IYA', 1


# =============================================================
#  BAGIAN 9 — JALANKAN PADA DATASET
# =============================================================

def jalankan_pada_dataset(csv_path, n_sampel=5000, random_state=42):
    """
    Load dataset BRFSS, preprocessing, jalankan Mamdani & Sugeno,
    lalu tampilkan evaluasi perbandingan.
    Parameter:
      csv_path    : path ke file CSV BRFSS
      n_sampel    : jumlah baris yang diproses (default 5000)
      random_state: seed untuk reproducibility
    """
    # Load & preprocessing
    df   = pd.read_csv(csv_path)
    data = df[['BMI', 'HighBP', 'HighChol', 'Diabetes_binary']].dropna()
    data = preprocess(data)
    sample = data.sample(n=n_sampel, random_state=random_state).reset_index(drop=True)

    print(f"Dataset dimuat: {len(sample)} sampel")
    print(f"Distribusi label:\n{sample['Diabetes_binary'].value_counts()}\n")

    # Inferensi Mamdani
    print("Menjalankan Fuzzy Mamdani...")
    skor_m, label_m, pred_m = [], [], []
    for _, row in sample.iterrows():
        s, l, b = prediksi_mamdani(row['BMI'], row['HighBP_scaled'], row['HighChol_scaled'])
        skor_m.append(s); label_m.append(l); pred_m.append(b)

    # Inferensi Sugeno
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

    # Evaluasi
    y_true = sample['Diabetes_binary'].astype(int).values
    print("\n" + "="*55)
    print("  EVALUASI PERFORMA")
    print("="*55)
    print(f"\n🔵 FUZZY MAMDANI — Akurasi: {accuracy_score(y_true, pred_m)*100:.2f}%")
    print(classification_report(y_true, pred_m, target_names=['Tidak Diabetes', 'Diabetes']))
    print(f"\n🟠 FUZZY SUGENO  — Akurasi: {accuracy_score(y_true, pred_s)*100:.2f}%")
    print(classification_report(y_true, pred_s, target_names=['Tidak Diabetes', 'Diabetes']))

    # Plot perbandingan
    plot_perbandingan(sample, y_true, pred_m, pred_s)

    return sample


# =============================================================
#  BAGIAN 10 — VISUALISASI
# =============================================================

def plot_membership_functions():
    """Visualisasi semua membership function."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Fungsi Keanggotaan (Membership Functions)', fontsize=15, fontweight='bold')

    # BMI
    ax, x = axes[0, 0], np.linspace(10, 70, 500)
    ax.plot(x, mf_bmi_kurus(x),    'b-',  lw=2, label='Kurus')
    ax.plot(x, mf_bmi_normal(x),   'g-',  lw=2, label='Normal')
    ax.plot(x, mf_bmi_gemuk(x),    'y-',  lw=2, label='Gemuk')
    ax.plot(x, mf_bmi_obesitas(x), 'r-',  lw=2, label='Obesitas')
    ax.set_title('Input 1: BMI', fontweight='bold')
    ax.set_xlabel('BMI'); ax.set_ylabel('Derajat Keanggotaan')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.1)

    # HighBP
    ax, x = axes[0, 1], np.linspace(0, 10, 500)
    ax.plot(x, mf_bp_rendah(x), 'g-', lw=2, label='Rendah')
    ax.plot(x, mf_bp_sedang(x), 'y-', lw=2, label='Sedang')
    ax.plot(x, mf_bp_tinggi(x), 'r-', lw=2, label='Tinggi')
    ax.set_title('Input 2: HighBP (mapped 0–10)', fontweight='bold')
    ax.set_xlabel('Skor BP'); ax.set_ylabel('Derajat Keanggotaan')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.1)

    # HighChol
    ax, x = axes[1, 0], np.linspace(0, 10, 500)
    ax.plot(x, mf_chol_rendah(x), 'g-', lw=2, label='Rendah')
    ax.plot(x, mf_chol_sedang(x), 'y-', lw=2, label='Sedang')
    ax.plot(x, mf_chol_tinggi(x), 'r-', lw=2, label='Tinggi')
    ax.set_title('Input 3: HighChol (mapped 0–10)', fontweight='bold')
    ax.set_xlabel('Skor Kolesterol'); ax.set_ylabel('Derajat Keanggotaan')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.1)

    # Output
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


def plot_perbandingan(sample, y_true, pred_m, pred_s):
    """Visualisasi perbandingan Mamdani vs Sugeno."""
    acc_m = accuracy_score(y_true, pred_m)
    acc_s = accuracy_score(y_true, pred_s)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Perbandingan Fuzzy Mamdani vs Sugeno', fontsize=14, fontweight='bold')

    # Akurasi
    ax = axes[0]
    bars = ax.bar(['Mamdani', 'Sugeno'], [acc_m*100, acc_s*100],
                  color=['#3B82F6', '#F97316'], width=0.4, edgecolor='white')
    for bar, val in zip(bars, [acc_m*100, acc_s*100]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}%', ha='center', fontweight='bold')
    ax.set_title('Akurasi'); ax.set_ylabel('Akurasi (%)'); ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    # Distribusi Mamdani
    ax = axes[1]
    cnt = sample['label_mamdani'].value_counts()
    ax.bar(cnt.index, cnt.values, color=['#22C55E', '#EAB308', '#EF4444'], edgecolor='white')
    ax.set_title('Distribusi Output Mamdani'); ax.set_ylabel('Jumlah Sampel')
    ax.grid(axis='y', alpha=0.3)

    # Distribusi Sugeno
    ax = axes[2]
    cnt = sample['label_sugeno'].value_counts()
    ax.bar(cnt.index, cnt.values, color=['#22C55E', '#EAB308', '#EF4444'], edgecolor='white')
    ax.set_title('Distribusi Output Sugeno'); ax.set_ylabel('Jumlah Sampel')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('perbandingan_mamdani_sugeno.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Selisih skor
    diff = np.abs(np.array(sample['skor_mamdani']) - np.array(sample['skor_sugeno']))
    print(f"\nRata-rata selisih skor Mamdani vs Sugeno : {diff.mean():.4f}")
    print(f"Selisih maksimum                          : {diff.max():.4f}")


# =============================================================
#  BAGIAN 11 — PREDIKSI SATU PASIEN (fungsi utama untuk Streamlit)
# =============================================================

def prediksi_pasien(bmi, highbp, highchol):
    """
    Prediksi risiko diabetes untuk satu pasien.
    Parameter:
      bmi      : nilai BMI (float, misal 28.5)
      highbp   : 0 = tidak hipertensi, 1 = hipertensi
      highchol : 0 = kolesterol normal, 1 = kolesterol tinggi
    Returns:
      dict berisi hasil Mamdani & Sugeno
    """
    bp_scaled   = scale_bp(highbp)
    chol_scaled = scale_chol(highchol)

    skor_m, label_m, biner_m = prediksi_mamdani(bmi, bp_scaled, chol_scaled)
    skor_s, label_s, biner_s = prediksi_sugeno(bmi, bp_scaled, chol_scaled)

    return {
        'mamdani': {'skor': round(skor_m, 4), 'label': label_m, 'biner': biner_m},
        'sugeno':  {'skor': round(skor_s, 4), 'label': label_s, 'biner': biner_s},
    }


# =============================================================
#  MAIN — Jalankan langsung dari terminal
# =============================================================

if __name__ == '__main__':
    # ── Visualisasi membership function
    plot_membership_functions()

    # ── Test prediksi satu pasien
    print("\n" + "="*50)
    print("  DEMO PREDIKSI SATU PASIEN")
    print("="*50)

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

    # ── Jalankan pada dataset (ganti path sesuai lokasi file CSV kamu)
    # sample = jalankan_pada_dataset('diabetes_binary_health_indicators_BRFSS2015.csv')