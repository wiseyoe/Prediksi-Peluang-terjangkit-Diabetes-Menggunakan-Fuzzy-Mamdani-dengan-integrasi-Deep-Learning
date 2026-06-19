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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
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

def mf_umur_muda(x):
    return trapmf(x,1,1,4,7)

def mf_umur_dewasa(x):
    return trimf(x,4,7,10)
def mf_umur_tua(x): return trapmf(x, 8, 11, 13, 13)

def mf_sayur_jarang(x):
    x = np.asarray(x)
    return (x == 0).astype(float)

def mf_sayur_sering(x):
    x = np.asarray(x)
    return (x == 1).astype(float)


# =============================================================
#  BAGIAN 3 — RULE BASE (20 Rules)  [TIDAK DIUBAH]
# =============================================================

RULES = [
    # ==========================================================
    # ── TIDAK — 20 Rules (Risiko Rendah, skor 1.5–3.5) ───────
    # ==========================================================

    # R01: Kurus, BP rendah, Chol rendah, Muda, Sering sayur
    (mf_bmi_kurus,   mf_bp_rendah, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_sering, mf_tidak, 1.5),

    # R02: Normal, BP rendah, Chol rendah, Muda, Sering sayur
    (mf_bmi_normal,  mf_bp_rendah, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_sering, mf_tidak, 2.0),

    # R03: Kurus, BP rendah, Chol rendah, Dewasa, Sering sayur
    (mf_bmi_kurus,   mf_bp_rendah, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_sering, mf_tidak, 2.0),

    # R04: Normal, BP rendah, Chol rendah, Dewasa, Sering sayur
    (mf_bmi_normal,  mf_bp_rendah, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_sering, mf_tidak, 2.0),

    # R05: Normal, BP rendah, Chol rendah, Tua, Sering sayur
    (mf_bmi_normal,  mf_bp_rendah, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_sering, mf_tidak, 2.5),

    # R06: Normal, BP rendah, Chol sedang, Muda, Sering sayur
    (mf_bmi_normal,  mf_bp_rendah, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_sering, mf_tidak, 2.5),

    # R07: Normal, BP sedang, Chol rendah, Muda, Sering sayur
    (mf_bmi_normal,  mf_bp_sedang, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_sering, mf_tidak, 2.5),

    # R08: Kurus, BP sedang, Chol rendah, Dewasa, Sering sayur
    (mf_bmi_kurus,   mf_bp_sedang, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_sering, mf_tidak, 2.5),

    # R09: Kurus, BP rendah, Chol sedang, Muda, Sering sayur
    (mf_bmi_kurus,   mf_bp_rendah, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_sering, mf_tidak, 2.5),

    # R10: Kurus, BP rendah, Chol rendah, Tua, Sering sayur
    (mf_bmi_kurus,   mf_bp_rendah, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_sering, mf_tidak, 2.5),

    # R11: Normal, BP rendah, Chol sedang, Dewasa, Sering sayur
    (mf_bmi_normal,  mf_bp_rendah, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_sering, mf_tidak, 3.0),

    # R12: Normal, BP sedang, Chol rendah, Dewasa, Sering sayur
    (mf_bmi_normal,  mf_bp_sedang, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_sering, mf_tidak, 3.0),

    # R13: Kurus, BP sedang, Chol sedang, Muda, Sering sayur
    (mf_bmi_kurus,   mf_bp_sedang, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_sering, mf_tidak, 3.0),

    # R14: Kurus, BP rendah, Chol sedang, Dewasa, Sering sayur
    (mf_bmi_kurus,   mf_bp_rendah, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_sering, mf_tidak, 3.0),

    # R15: Kurus, BP sedang, Chol rendah, Tua, Sering sayur
    (mf_bmi_kurus,   mf_bp_sedang, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_sering, mf_tidak, 3.0),

    # R16: Normal, BP rendah, Chol rendah, Muda, Jarang sayur
    (mf_bmi_normal,  mf_bp_rendah, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_jarang, mf_tidak, 3.0),

    # R17: Kurus, BP rendah, Chol rendah, Muda, Jarang sayur
    (mf_bmi_kurus,   mf_bp_rendah, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_jarang, mf_tidak, 3.0),

    # R18: Normal, BP sedang, Chol sedang, Muda, Sering sayur
    (mf_bmi_normal,  mf_bp_sedang, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_sering, mf_tidak, 3.0),

    # R19: Kurus, BP sedang, Chol sedang, Dewasa, Sering sayur
    (mf_bmi_kurus,   mf_bp_sedang, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_sering, mf_tidak, 3.5),

    # R20: Normal, BP rendah, Chol sedang, Tua, Sering sayur
    (mf_bmi_normal,  mf_bp_rendah, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_sering, mf_tidak, 3.5),

    # ==========================================================
    # ── MUNGKIN — 20 Rules (Risiko Sedang, skor 4.0–6.4) ─────
    # ==========================================================

    # R21: Gemuk, BP rendah, Chol rendah, Muda, Sering sayur
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_sering, mf_mungkin, 4.0),

    # R22: Normal, BP sedang, Chol sedang, Dewasa, Sering sayur
    (mf_bmi_normal,  mf_bp_sedang, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_sering, mf_mungkin, 4.0),

    # R23: Gemuk, BP rendah, Chol rendah, Dewasa, Sering sayur
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_sering, mf_mungkin, 4.0),

    # R24: Normal, BP tinggi, Chol rendah, Muda, Sering sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_sering, mf_mungkin, 4.0),

    # R25: Gemuk, BP sedang, Chol rendah, Muda, Sering sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_sering, mf_mungkin, 4.5),

    # R26: Gemuk, BP rendah, Chol sedang, Muda, Sering sayur
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_sering, mf_mungkin, 4.5),

    # R27: Gemuk, BP sedang, Chol rendah, Dewasa, Sering sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_sering, mf_mungkin, 4.5),

    # R28: Normal, BP sedang, Chol sedang, Tua, Sering sayur
    (mf_bmi_normal,  mf_bp_sedang, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_sering, mf_mungkin, 4.5),

    # R29: Normal, BP tinggi, Chol sedang, Dewasa, Sering sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_sering, mf_mungkin, 5.0),

    # R30: Gemuk, BP rendah, Chol sedang, Dewasa, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_jarang, mf_mungkin, 5.0),

    # R31: Gemuk, BP sedang, Chol sedang, Muda, Sering sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_sering, mf_mungkin, 5.0),

    # R32: Obesitas, BP rendah, Chol rendah, Dewasa, Sering sayur
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_sering, mf_mungkin, 5.0),

    # R33: Gemuk, BP sedang, Chol sedang, Dewasa, Sering sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_sering, mf_mungkin, 5.0),

    # R34: Normal, BP tinggi, Chol tinggi, Muda, Sering sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_mungkin, 5.5),

    # R35: Gemuk, BP tinggi, Chol rendah, Dewasa, Sering sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_sering, mf_mungkin, 5.5),

    # R36: Gemuk, BP sedang, Chol sedang, Tua, Sering sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_sering, mf_mungkin, 5.5),

    # R37: Obesitas, BP rendah, Chol rendah, Muda, Sering sayur
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_sering, mf_mungkin, 5.5),

    # R38: Obesitas, BP rendah, Chol sedang, Dewasa, Sering sayur
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_sering, mf_mungkin, 5.5),

    # R39: Gemuk, BP tinggi, Chol rendah, Tua, Sering sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_sering, mf_mungkin, 6.0),

    # R40: Obesitas, BP sedang, Chol rendah, Dewasa, Sering sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_sering, mf_mungkin, 6.0),

    # ==========================================================
    # ── IYA — 20 Rules (Risiko Tinggi, skor 6.5–9.5) ─────────
    # ==========================================================

    # R41: Gemuk, BP tinggi, Chol sedang, Dewasa, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_jarang, mf_iya, 7.0),

    # R42: Obesitas, BP sedang, Chol sedang, Dewasa, Jarang sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_jarang, mf_iya, 7.0),

    # R43: Gemuk, BP tinggi, Chol tinggi, Muda, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_jarang, mf_iya, 7.5),

    # R44: Obesitas, BP rendah, Chol tinggi, Dewasa, Jarang sayur
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_jarang, mf_iya, 7.5),

    # R45: Obesitas, BP tinggi, Chol rendah, Dewasa, Jarang sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_jarang, mf_iya, 7.5),

    # R46: Gemuk, BP tinggi, Chol sedang, Tua, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_jarang, mf_iya, 7.5),

    # R47: Obesitas, BP sedang, Chol tinggi, Muda, Jarang sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_jarang, mf_iya, 7.5),

    # R48: Gemuk, BP tinggi, Chol tinggi, Dewasa, Sering sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_sering, mf_iya, 8.0),

    # R49: Obesitas, BP tinggi, Chol sedang, Dewasa, Sering sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_sering, mf_iya, 8.0),

    # R50: Gemuk, BP tinggi, Chol sedang, Dewasa, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_jarang, mf_iya, 8.0),

    # R51: Obesitas, BP sedang, Chol tinggi, Dewasa, Jarang sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_jarang, mf_iya, 8.0),

    # R52: Obesitas, BP tinggi, Chol sedang, Dewasa, Jarang sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_jarang, mf_iya, 8.5),

    # R53: Gemuk, BP tinggi, Chol tinggi, Tua, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_jarang, mf_iya, 8.5),

    # R54: Obesitas, BP sedang, Chol tinggi, Tua, Jarang sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_jarang, mf_iya, 8.5),

    # R55: Obesitas, BP tinggi, Chol sedang, Tua, Jarang sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_jarang, mf_iya, 8.5),

    # R56: Obesitas, BP tinggi, Chol tinggi, Muda, Sering sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_iya, 8.5),

    # R57: Obesitas, BP tinggi, Chol tinggi, Muda, Jarang sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_jarang, mf_iya, 9.0),

    # R58: Obesitas, BP tinggi, Chol tinggi, Dewasa, Jarang sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_jarang, mf_iya, 9.0),

    # R59: Gemuk, BP tinggi, Chol tinggi, Tua, Sering sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_sering, mf_iya, 9.0),

    # R60: Obesitas, BP tinggi, Chol tinggi, Tua, Jarang sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_jarang, mf_iya, 9.5),

    # ==========================================================
    # ── TIDAK TAMBAHAN — R61–R90 (Risiko Rendah, skor 1.5–3.8)
    # ==========================================================

    # R61: Kurus, BP rendah, Chol tinggi, Muda, Sering sayur
    (mf_bmi_kurus,   mf_bp_rendah, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_tidak, 2.8),

    # R62: Kurus, BP sedang, Chol sedang, Tua, Sering sayur
    (mf_bmi_kurus,   mf_bp_sedang, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_sering, mf_tidak, 3.0),

    # R63: Normal, BP sedang, Chol sedang, Dewasa, Jarang sayur
    (mf_bmi_normal,  mf_bp_sedang, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_jarang, mf_tidak, 3.2),

    # R64: Kurus, BP rendah, Chol rendah, Dewasa, Jarang sayur
    (mf_bmi_kurus,   mf_bp_rendah, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_jarang, mf_tidak, 3.0),

    # R65: Normal, BP rendah, Chol rendah, Tua, Jarang sayur
    (mf_bmi_normal,  mf_bp_rendah, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_jarang, mf_tidak, 3.2),

    # R66: Kurus, BP tinggi, Chol rendah, Muda, Sering sayur
    (mf_bmi_kurus,   mf_bp_tinggi, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_sering, mf_tidak, 3.0),

    # R67: Kurus, BP rendah, Chol sedang, Tua, Sering sayur
    (mf_bmi_kurus,   mf_bp_rendah, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_sering, mf_tidak, 3.0),

    # R68: Normal, BP sedang, Chol rendah, Tua, Sering sayur
    (mf_bmi_normal,  mf_bp_sedang, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_sering, mf_tidak, 3.0),

    # R69: Kurus, BP sedang, Chol rendah, Muda, Jarang sayur
    (mf_bmi_kurus,   mf_bp_sedang, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_jarang, mf_tidak, 3.2),

    # R70: Normal, BP rendah, Chol sedang, Muda, Jarang sayur
    (mf_bmi_normal,  mf_bp_rendah, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_jarang, mf_tidak, 3.2),

    # R71: Kurus, BP tinggi, Chol rendah, Dewasa, Sering sayur
    (mf_bmi_kurus,   mf_bp_tinggi, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_sering, mf_tidak, 3.3),

    # R72: Kurus, BP rendah, Chol tinggi, Dewasa, Sering sayur
    (mf_bmi_kurus,   mf_bp_rendah, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_sering, mf_tidak, 3.3),

    # R73: Normal, BP tinggi, Chol rendah, Dewasa, Sering sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_sering, mf_tidak, 3.3),

    # R74: Kurus, BP sedang, Chol tinggi, Muda, Sering sayur
    (mf_bmi_kurus,   mf_bp_sedang, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_tidak, 3.3),

    # R75: Normal, BP sedang, Chol sedang, Tua, Jarang sayur
    (mf_bmi_normal,  mf_bp_sedang, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_jarang, mf_tidak, 3.5),

    # R76: Kurus, BP rendah, Chol sedang, Muda, Jarang sayur
    (mf_bmi_kurus,   mf_bp_rendah, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_jarang, mf_tidak, 3.2),

    # R77: Normal, BP sedang, Chol rendah, Dewasa, Jarang sayur
    (mf_bmi_normal,  mf_bp_sedang, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_jarang, mf_tidak, 3.3),

    # R78: Kurus, BP tinggi, Chol sedang, Muda, Sering sayur
    (mf_bmi_kurus,   mf_bp_tinggi, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_sering, mf_tidak, 3.5),

    # R79: Normal, BP tinggi, Chol rendah, Muda, Jarang sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_jarang, mf_tidak, 3.5),

    # R80: Kurus, BP sedang, Chol sedang, Muda, Jarang sayur
    (mf_bmi_kurus,   mf_bp_sedang, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_jarang, mf_tidak, 3.5),

    # R81: Normal, BP rendah, Chol sedang, Dewasa, Jarang sayur
    (mf_bmi_normal,  mf_bp_rendah, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_jarang, mf_tidak, 3.5),

    # R82: Kurus, BP tinggi, Chol rendah, Tua, Sering sayur
    (mf_bmi_kurus,   mf_bp_tinggi, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_sering, mf_tidak, 3.5),

    # R83: Normal, BP rendah, Chol tinggi, Muda, Sering sayur
    (mf_bmi_normal,  mf_bp_rendah, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_tidak, 3.5),

    # R84: Kurus, BP sedang, Chol rendah, Dewasa, Jarang sayur
    (mf_bmi_kurus,   mf_bp_sedang, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_jarang, mf_tidak, 3.3),

    # R85: Kurus, BP tinggi, Chol sedang, Dewasa, Sering sayur
    (mf_bmi_kurus,   mf_bp_tinggi, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_sering, mf_tidak, 3.8),

    # R86: Normal, BP sedang, Chol tinggi, Muda, Sering sayur
    (mf_bmi_normal,  mf_bp_sedang, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_tidak, 3.8),

    # R87: Kurus, BP rendah, Chol tinggi, Muda, Jarang sayur
    (mf_bmi_kurus,   mf_bp_rendah, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_jarang, mf_tidak, 3.5),

    # R88: Normal, BP rendah, Chol tinggi, Dewasa, Sering sayur
    (mf_bmi_normal,  mf_bp_rendah, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_sering, mf_tidak, 3.8),

    # R89: Kurus, BP sedang, Chol tinggi, Dewasa, Sering sayur
    (mf_bmi_kurus,   mf_bp_sedang, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_sering, mf_tidak, 3.8),

    # R90: Kurus, BP tinggi, Chol tinggi, Muda, Sering sayur
    (mf_bmi_kurus,   mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_tidak, 3.8),

    # ==========================================================
    # ── MUNGKIN TAMBAHAN — R91–R120 (Risiko Sedang, 4.0–6.4) ─
    # ==========================================================

    # R91: Normal, BP tinggi, Chol rendah, Tua, Sering sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_sering, mf_mungkin, 4.2),

    # R92: Gemuk, BP rendah, Chol rendah, Tua, Sering sayur
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_sering, mf_mungkin, 4.2),

    # R93: Gemuk, BP rendah, Chol sedang, Muda, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_jarang, mf_mungkin, 4.5),

    # R94: Normal, BP tinggi, Chol rendah, Dewasa, Jarang sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_jarang, mf_mungkin, 4.5),

    # R95: Gemuk, BP rendah, Chol rendah, Muda, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_jarang, mf_mungkin, 4.5),

    # R96: Normal, BP sedang, Chol sedang, Muda, Jarang sayur
    (mf_bmi_normal,  mf_bp_sedang, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_jarang, mf_mungkin, 4.5),

    # R97: Gemuk, BP sedang, Chol rendah, Tua, Sering sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_sering, mf_mungkin, 4.8),

    # R98: Obesitas, BP rendah, Chol rendah, Tua, Sering sayur
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_sering, mf_mungkin, 4.8),

    # R99: Normal, BP tinggi, Chol sedang, Muda, Jarang sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_jarang, mf_mungkin, 5.0),

    # R100: Gemuk, BP sedang, Chol rendah, Muda, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_jarang, mf_mungkin, 5.0),

    # R101: Gemuk, BP rendah, Chol sedang, Tua, Sering sayur
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_sering, mf_mungkin, 5.0),

    # R102: Obesitas, BP sedang, Chol rendah, Muda, Sering sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_sering, mf_mungkin, 5.0),

    # R103: Normal, BP tinggi, Chol tinggi, Dewasa, Sering sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_sering, mf_mungkin, 5.2),

    # R104: Gemuk, BP sedang, Chol sedang, Muda, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_jarang, mf_mungkin, 5.2),

    # R105: Obesitas, BP rendah, Chol sedang, Muda, Sering sayur
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_sering, mf_mungkin, 5.2),

    # R106: Gemuk, BP rendah, Chol tinggi, Muda, Sering sayur
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_mungkin, 5.5),

    # R107: Gemuk, BP sedang, Chol rendah, Dewasa, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_jarang, mf_mungkin, 5.5),

    # R108: Obesitas, BP rendah, Chol sedang, Tua, Sering sayur
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_sering, mf_mungkin, 5.5),

    # R109: Normal, BP tinggi, Chol sedang, Tua, Sering sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_sering, mf_mungkin, 5.5),

    # R110: Gemuk, BP tinggi, Chol rendah, Muda, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_jarang, mf_mungkin, 5.5),

    # R111: Obesitas, BP sedang, Chol rendah, Tua, Sering sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_sering, mf_mungkin, 5.8),

    # R112: Gemuk, BP rendah, Chol tinggi, Dewasa, Sering sayur
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_sering, mf_mungkin, 5.8),

    # R113: Obesitas, BP rendah, Chol tinggi, Muda, Sering sayur
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_mungkin, 5.8),

    # R114: Normal, BP tinggi, Chol tinggi, Tua, Sering sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_sering, mf_mungkin, 6.0),

    # R115: Gemuk, BP sedang, Chol sedang, Dewasa, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_jarang, mf_mungkin, 6.0),

    # R116: Obesitas, BP rendah, Chol rendah, Dewasa, Jarang sayur
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_jarang, mf_mungkin, 5.8),

    # R117: Gemuk, BP tinggi, Chol sedang, Muda, Sering sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_sering, mf_mungkin, 6.0),

    # R118: Obesitas, BP sedang, Chol sedang, Muda, Sering sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_sering, mf_mungkin, 6.0),

    # R119: Gemuk, BP rendah, Chol tinggi, Tua, Sering sayur
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_sering, mf_mungkin, 6.2),

    # R120: Obesitas, BP sedang, Chol sedang, Tua, Sering sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_sering, mf_mungkin, 6.2),

    # ==========================================================
    # ── IYA TAMBAHAN — R121–R150 (Risiko Tinggi, 6.5–9.5) ────
    # ==========================================================

    # R121: Gemuk, BP tinggi, Chol tinggi, Muda, Sering sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_iya, 7.0),

    # R122: Obesitas, BP tinggi, Chol rendah, Muda, Jarang sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_jarang, mf_iya, 7.0),

    # R123: Obesitas, BP sedang, Chol sedang, Muda, Jarang sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_jarang, mf_iya, 7.0),

    # R124: Gemuk, BP sedang, Chol tinggi, Muda, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_jarang, mf_iya, 7.0),

    # R125: Obesitas, BP tinggi, Chol rendah, Tua, Jarang sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_jarang, mf_iya, 7.5),

    # R126: Gemuk, BP sedang, Chol tinggi, Dewasa, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_jarang, mf_iya, 7.5),

    # R127: Obesitas, BP rendah, Chol tinggi, Muda, Jarang sayur
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_jarang, mf_iya, 7.5),

    # R128: Gemuk, BP tinggi, Chol rendah, Dewasa, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_jarang, mf_iya, 7.5),

    # R129: Obesitas, BP sedang, Chol tinggi, Dewasa, Sering sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_sering, mf_iya, 7.5),

    # R130: Gemuk, BP sedang, Chol tinggi, Tua, Sering sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_sering, mf_iya, 7.5),

    # R131: Obesitas, BP tinggi, Chol rendah, Dewasa, Sering sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_sering, mf_iya, 8.0),

    # R132: Gemuk, BP tinggi, Chol tinggi, Dewasa, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_jarang, mf_iya, 8.0),

    # R133: Obesitas, BP sedang, Chol sedang, Tua, Jarang sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_jarang, mf_iya, 8.0),

    # R134: Gemuk, BP sedang, Chol tinggi, Tua, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_jarang, mf_iya, 8.0),

    # R135: Obesitas, BP rendah, Chol tinggi, Tua, Jarang sayur
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_jarang, mf_iya, 8.0),

    # R136: Gemuk, BP tinggi, Chol sedang, Tua, Sering sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_sering, mf_iya, 8.0),

    # R137: Obesitas, BP tinggi, Chol rendah, Tua, Sering sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_sering, mf_iya, 8.0),

    # R138: Obesitas, BP sedang, Chol tinggi, Muda, Sering sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_iya, 8.0),

    # R139: Gemuk, BP tinggi, Chol tinggi, Muda, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_jarang, mf_iya, 8.5),

    # R140: Obesitas, BP tinggi, Chol sedang, Muda, Jarang sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_jarang, mf_iya, 8.5),

    # R141: Obesitas, BP rendah, Chol tinggi, Tua, Sering sayur
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_sering, mf_iya, 8.5),

    # R142: Gemuk, BP tinggi, Chol rendah, Tua, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_jarang, mf_iya, 8.5),

    # R143: Obesitas, BP tinggi, Chol tinggi, Dewasa, Sering sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_sering, mf_iya, 8.5),

    # R144: Gemuk, BP sedang, Chol tinggi, Muda, Sering sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_iya, 8.5),

    # R145: Obesitas, BP tinggi, Chol sedang, Muda, Sering sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_sering, mf_iya, 8.5),

    # R146: Obesitas, BP tinggi, Chol tinggi, Tua, Sering sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_sering, mf_iya, 9.0),

    # R147: Gemuk, BP tinggi, Chol tinggi, Dewasa, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_jarang, mf_iya, 9.0),

    # R148: Obesitas, BP tinggi, Chol sedang, Tua, Sering sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_sering, mf_iya, 9.0),

    # R149: Obesitas, BP sedang, Chol tinggi, Tua, Sering sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_sering, mf_iya, 9.0),

    # R150: Obesitas, BP tinggi, Chol tinggi, Tua, Sering sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_sering, mf_iya, 9.5),

    # ==========================================================
    # ── TIDAK TAMBAHAN — R151–R172 (Risiko Rendah, skor 1.5–3.9)
    # ==========================================================

    # R151: Kurus, BP tinggi, Chol rendah, Muda, Jarang sayur
    (mf_bmi_kurus,   mf_bp_tinggi, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_jarang, mf_tidak, 3.5),

    # R152: Normal, BP tinggi, Chol rendah, Tua, Jarang sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_jarang, mf_tidak, 3.6),

    # R153: Kurus, BP sedang, Chol tinggi, Tua, Sering sayur
    (mf_bmi_kurus,   mf_bp_sedang, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_sering, mf_tidak, 3.7),

    # R154: Normal, BP sedang, Chol tinggi, Dewasa, Sering sayur
    (mf_bmi_normal,  mf_bp_sedang, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_sering, mf_tidak, 3.8),

    # R155: Kurus, BP tinggi, Chol sedang, Tua, Sering sayur
    (mf_bmi_kurus,   mf_bp_tinggi, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_sering, mf_tidak, 3.8),

    # R156: Normal, BP tinggi, Chol sedang, Muda, Sering sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_sering, mf_tidak, 3.8),

    # R157: Kurus, BP rendah, Chol tinggi, Tua, Sering sayur
    (mf_bmi_kurus,   mf_bp_rendah, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_sering, mf_tidak, 3.5),

    # R158: Normal, BP rendah, Chol tinggi, Tua, Sering sayur
    (mf_bmi_normal,  mf_bp_rendah, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_sering, mf_tidak, 3.7),

    # R159: Kurus, BP tinggi, Chol tinggi, Dewasa, Sering sayur
    (mf_bmi_kurus,   mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_sering, mf_tidak, 3.9),

    # R160: Normal, BP sedang, Chol tinggi, Tua, Sering sayur
    (mf_bmi_normal,  mf_bp_sedang, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_sering, mf_tidak, 3.9),

    # R161: Kurus, BP rendah, Chol sedang, Dewasa, Jarang sayur
    (mf_bmi_kurus,   mf_bp_rendah, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_jarang, mf_tidak, 3.3),

    # R162: Normal, BP rendah, Chol sedang, Tua, Jarang sayur
    (mf_bmi_normal,  mf_bp_rendah, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_jarang, mf_tidak, 3.5),

    # R163: Kurus, BP sedang, Chol rendah, Tua, Jarang sayur
    (mf_bmi_kurus,   mf_bp_sedang, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_jarang, mf_tidak, 3.3),

    # R164: Normal, BP sedang, Chol rendah, Tua, Jarang sayur
    (mf_bmi_normal,  mf_bp_sedang, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_jarang, mf_tidak, 3.5),

    # R165: Kurus, BP tinggi, Chol rendah, Dewasa, Jarang sayur
    (mf_bmi_kurus,   mf_bp_tinggi, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_jarang, mf_tidak, 3.6),

    # R166: Kurus, BP rendah, Chol rendah, Tua, Jarang sayur
    (mf_bmi_kurus,   mf_bp_rendah, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_jarang, mf_tidak, 3.2),

    # R167: Normal, BP tinggi, Chol sedang, Muda, Jarang sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_jarang, mf_tidak, 3.8),

    # R168: Kurus, BP sedang, Chol sedang, Tua, Jarang sayur
    (mf_bmi_kurus,   mf_bp_sedang, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_jarang, mf_tidak, 3.6),

    # R169: Normal, BP rendah, Chol tinggi, Muda, Jarang sayur
    (mf_bmi_normal,  mf_bp_rendah, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_jarang, mf_tidak, 3.7),

    # R170: Kurus, BP tinggi, Chol sedang, Muda, Jarang sayur
    (mf_bmi_kurus,   mf_bp_tinggi, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_jarang, mf_tidak, 3.7),

    # R171: Normal, BP tinggi, Chol tinggi, Muda, Jarang sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_jarang, mf_tidak, 3.9),

    # R172: Kurus, BP tinggi, Chol tinggi, Tua, Sering sayur
    (mf_bmi_kurus,   mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_sering, mf_tidak, 3.9),

    # ==========================================================
    # ── MUNGKIN TAMBAHAN — R173–R194 (Risiko Sedang, 4.0–6.4) ─
    # ==========================================================

    # R173: Gemuk, BP rendah, Chol rendah, Dewasa, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_jarang, mf_mungkin, 4.5),

    # R174: Normal, BP tinggi, Chol rendah, Tua, Jarang sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_jarang, mf_mungkin, 4.8),

    # R175: Gemuk, BP rendah, Chol rendah, Tua, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_jarang, mf_mungkin, 4.8),

    # R176: Obesitas, BP rendah, Chol rendah, Muda, Jarang sayur
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_jarang, mf_mungkin, 5.0),

    # R177: Normal, BP sedang, Chol sedang, Dewasa, Jarang sayur
    (mf_bmi_normal,  mf_bp_sedang, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_jarang, mf_mungkin, 5.0),

    # R178: Gemuk, BP sedang, Chol sedang, Tua, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_jarang, mf_mungkin, 5.2),

    # R179: Normal, BP tinggi, Chol sedang, Dewasa, Jarang sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_jarang, mf_mungkin, 5.5),

    # R180: Gemuk, BP rendah, Chol sedang, Muda, Jarang sayur (sudah ada yg mirip, beda umur)
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_jarang, mf_mungkin, 5.5),

    # R181: Obesitas, BP rendah, Chol rendah, Tua, Jarang sayur
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_jarang, mf_mungkin, 5.5),

    # R182: Normal, BP tinggi, Chol tinggi, Dewasa, Jarang sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_jarang, mf_mungkin, 5.8),

    # R183: Gemuk, BP tinggi, Chol rendah, Muda, Sering sayur (tercover sebelumnya tapi beda kombinasi)
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_mungkin, 5.5),

    # R184: Obesitas, BP sedang, Chol rendah, Muda, Jarang sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_jarang, mf_mungkin, 5.5),

    # R185: Normal, BP tinggi, Chol tinggi, Tua, Jarang sayur
    (mf_bmi_normal,  mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_jarang, mf_mungkin, 6.0),

    # R186: Gemuk, BP tinggi, Chol sedang, Muda, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_jarang, mf_mungkin, 6.0),

    # R187: Obesitas, BP sedang, Chol sedang, Dewasa, Sering sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_sering, mf_mungkin, 6.0),

    # R188: Gemuk, BP rendah, Chol tinggi, Muda, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_jarang, mf_mungkin, 6.0),

    # R189: Obesitas, BP rendah, Chol sedang, Dewasa, Jarang sayur
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_jarang, mf_mungkin, 6.0),

    # R190: Gemuk, BP tinggi, Chol tinggi, Muda, Sering sayur (kondisi sedang-tinggi batas)
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_sering, mf_mungkin, 6.2),

    # R191: Obesitas, BP tinggi, Chol rendah, Muda, Sering sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_rendah,
     mf_umur_muda,   mf_sayur_sering, mf_mungkin, 6.2),

    # R192: Gemuk, BP rendah, Chol tinggi, Dewasa, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_jarang, mf_mungkin, 6.2),

    # R193: Obesitas, BP sedang, Chol rendah, Tua, Jarang sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_jarang, mf_mungkin, 6.2),

    # R194: Gemuk, BP tinggi, Chol rendah, Tua, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_jarang, mf_mungkin, 6.4),

    # ==========================================================
    # ── IYA TAMBAHAN — R195–R216 (Risiko Tinggi, 6.5–9.5) ────
    # ==========================================================

    # R195: Gemuk, BP sedang, Chol tinggi, Dewasa, Sering sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_sering, mf_iya, 7.0),

    # R196: Obesitas, BP sedang, Chol sedang, Muda, Sering sayur (batas atas mungkin→iya)
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_rendah,
     mf_umur_dewasa, mf_sayur_sering, mf_iya, 7.0),

    # R197: Gemuk, BP tinggi, Chol rendah, Muda, Sering sayur (batas atas mungkin)
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_iya, 7.5),

    # R198: Obesitas, BP rendah, Chol tinggi, Dewasa, Sering sayur
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_sering, mf_iya, 7.5),

    # R199: Gemuk, BP sedang, Chol sedang, Muda, Jarang sayur (konfirmasi risiko)
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_sering, mf_iya, 7.5),

    # R200: Obesitas, BP sedang, Chol tinggi, Tua, Sering sayur (duplikat R149 disingkirkan)
    (mf_bmi_obesitas, mf_bp_rendah, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_iya, 7.5),

    # R201: Gemuk, BP tinggi, Chol sedang, Dewasa, Sering sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_sedang,
     mf_umur_dewasa, mf_sayur_sering, mf_iya, 8.0),

    # R202: Obesitas, BP tinggi, Chol sedang, Dewasa, Jarang sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_sering, mf_iya, 7.8),

    # R203: Gemuk, BP rendah, Chol tinggi, Tua, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_rendah, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_jarang, mf_iya, 8.0),

    # R204: Obesitas, BP tinggi, Chol rendah, Tua, Sering sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_rendah,
     mf_umur_tua,    mf_sayur_sering, mf_iya, 8.0),

    # R205: Gemuk, BP sedang, Chol tinggi, Muda, Sering sayur
    (mf_bmi_gemuk,   mf_bp_sedang, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_iya, 8.0),

    # R206: Obesitas, BP sedang, Chol tinggi, Muda, Jarang sayur (berat)
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_jarang, mf_iya, 9.0),

    # R207: Gemuk, BP tinggi, Chol tinggi, Tua, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_jarang, mf_iya, 9.0),

    # R208: Obesitas, BP tinggi, Chol sedang, Tua, Jarang sayur
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_sedang,
     mf_umur_tua,    mf_sayur_jarang, mf_iya, 9.0),

    # R209: Gemuk, BP tinggi, Chol sedang, Muda, Jarang sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_sedang,
     mf_umur_muda,   mf_sayur_jarang, mf_iya, 8.5),

    # R210: Obesitas, BP sedang, Chol tinggi, Tua, Jarang sayur
    (mf_bmi_obesitas, mf_bp_sedang, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_iya, 8.0),

    # R211: Gemuk, BP sedang, Chol tinggi, Dewasa, Sering sayur
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_sering, mf_iya, 9.0),

    # R212: Obesitas, BP tinggi, Chol tinggi, Dewasa, Jarang sayur (very high)
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_jarang, mf_iya, 9.5),

    # R213: Gemuk, BP tinggi, Chol tinggi, Dewasa, Sering sayur (sering tapi risiko tetap tinggi)
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_jarang, mf_iya, 9.5),

    # R214: Obesitas, BP tinggi, Chol tinggi, Muda, Sering sayur (extreme)
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_muda,   mf_sayur_sering, mf_iya, 8.8),

    # R215: Obesitas, BP tinggi, Chol tinggi, Tua, Jarang sayur (worst case)
    (mf_bmi_obesitas, mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_tua,    mf_sayur_jarang, mf_iya, 9.5),

    # R216: Gemuk, BP tinggi, Chol tinggi, Tua, Jarang sayur (very high risk)
    (mf_bmi_gemuk,   mf_bp_tinggi, mf_chol_tinggi,
     mf_umur_dewasa, mf_sayur_jarang, mf_iya, 9.0),
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
    # 0 → 2 (zona rendah), 1 → 9 (zona tinggi, µ tinggi=1.0)
    return 2 if highbp_binary == 0 else 9

def scale_chol(highchol_binary):
    # 0 → 2 (zona rendah), 1 → 9 (zona tinggi, µ tinggi=1.0)
    return 2 if highchol_binary == 0 else 9


# =============================================================
#  BAGIAN 5 — FUZZIFIKASI  [TIDAK DIUBAH]
# =============================================================

FN_MAP = {
    # BMI
    mf_bmi_kurus:    'bmi_kurus',
    mf_bmi_normal:   'bmi_normal',
    mf_bmi_gemuk:    'bmi_gemuk',
    mf_bmi_obesitas: 'bmi_obesitas',

    # Blood Pressure
    mf_bp_rendah:    'bp_rendah',
    mf_bp_sedang:    'bp_sedang',
    mf_bp_tinggi:    'bp_tinggi',

    # Cholesterol
    mf_chol_rendah:  'chol_rendah',
    mf_chol_sedang:  'chol_sedang',
    mf_chol_tinggi:  'chol_tinggi',

    # Age
    mf_umur_muda:    'umur_muda',
    mf_umur_dewasa:  'umur_dewasa',
    mf_umur_tua:     'umur_tua',

    # Veggies
    mf_sayur_jarang: 'sayur_jarang',
    mf_sayur_sering: 'sayur_sering',
}


def fuzzifikasi(
    bmi,
    bp_scaled,
    chol_scaled,
    age,
    veggies
):
    def mu(fn, val):
        result = fn(np.array([val]))[0]

        if isinstance(result, tuple):
            result = result[0]

        return float(result)

    return {

        # BMI
        'bmi_kurus':    mu(mf_bmi_kurus, bmi),
        'bmi_normal':   mu(mf_bmi_normal, bmi),
        'bmi_gemuk':    mu(mf_bmi_gemuk, bmi),
        'bmi_obesitas': mu(mf_bmi_obesitas, bmi),

        # Blood Pressure
        'bp_rendah':    mu(mf_bp_rendah, bp_scaled),
        'bp_sedang':    mu(mf_bp_sedang, bp_scaled),
        'bp_tinggi':    mu(mf_bp_tinggi, bp_scaled),

        # Cholesterol
        'chol_rendah':  mu(mf_chol_rendah, chol_scaled),
        'chol_sedang':  mu(mf_chol_sedang, chol_scaled),
        'chol_tinggi':  mu(mf_chol_tinggi, chol_scaled),

        # Age
        'umur_muda':    mu(mf_umur_muda, age),
        'umur_dewasa':  mu(mf_umur_dewasa, age),
        'umur_tua':     mu(mf_umur_tua, age),

        # Veggies
        'sayur_jarang': mu(mf_sayur_jarang, veggies),
        'sayur_sering': mu(mf_sayur_sering, veggies),
    }


# =============================================================
#  BAGIAN 6 — FUZZY MAMDANI  [TIDAK DIUBAH]
# =============================================================

X_OUTPUT = np.linspace(0, 10, 1000)


def inferensi_mamdani(fuzz):
    agregat = np.zeros(len(X_OUTPUT))

    for rule in RULES:

        (
            fn_bmi,
            fn_bp,
            fn_chol,
            fn_umur,
            fn_sayur,
            fn_out,
            _
        ) = rule

        alpha = min(
            fuzz[FN_MAP[fn_bmi]],
            fuzz[FN_MAP[fn_bp]],
            fuzz[FN_MAP[fn_chol]],
            fuzz[FN_MAP[fn_umur]],
            fuzz[FN_MAP[fn_sayur]]
        )

        if alpha > 0:
            konsekuen = np.minimum(alpha, fn_out(X_OUTPUT))
            agregat = np.maximum(agregat, konsekuen)

    return agregat


def defuzzifikasi_centroid(agregat):
    denom = np.sum(agregat)
    if denom == 0:
        return 5.0
    return float(np.sum(X_OUTPUT * agregat) / denom)


def prediksi_mamdani(bmi, bp_scaled, chol_scaled,age,veggies):
    fuzz    = fuzzifikasi(
    bmi,
    bp_scaled,
    chol_scaled,
    age,
    veggies
)
    agregat = inferensi_mamdani(fuzz)
    skor    = defuzzifikasi_centroid(agregat)
    label, biner = skor_ke_label(skor)
    return skor, label, biner


# =============================================================
#  BAGIAN 7 — FUZZY SUGENO  [TIDAK DIUBAH]
# =============================================================

def inferensi_sugeno(fuzz):
    alphas = []
    zs = []

    for rule in RULES:

        (
            fn_bmi,
            fn_bp,
            fn_chol,
            fn_umur,
            fn_sayur,
            _,
            z_out
        ) = rule
        alpha = float(min([
            fuzz[FN_MAP[fn_bmi]],
            fuzz[FN_MAP[fn_bp]],
            fuzz[FN_MAP[fn_chol]],
            fuzz[FN_MAP[fn_umur]],
            fuzz[FN_MAP[fn_sayur]]
        ]))
        
        alphas.append(alpha)
        zs.append(z_out)

    return alphas, zs


def defuzzifikasi_weighted_average(alphas, zs):
    total = sum(alphas)
    if total == 0:
        return 5.0
    return float(sum(a * z for a, z in zip(alphas, zs)) / total)


def prediksi_sugeno(bmi, bp_scaled, chol_scaled, age, veggies):
    fuzz         =fuzzifikasi(
    bmi,
    bp_scaled,
    chol_scaled,
    age,
    veggies
)
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

    data['Diabetes_012'] = (
        data['Diabetes_012'] > 0
    ).astype(int)

    X = data[DL_FEATURES].values
    y = data['Diabetes_012'].values
    

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

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    cm = confusion_matrix(y_test, y_pred)

    print("\n===== HASIL EVALUASI DEEP LEARNING =====")
    print(f"Accuracy  : {accuracy*100:.2f}%")
    print(f"Precision : {precision*100:.2f}%")
    print(f"Recall    : {recall*100:.2f}%")
    print(f"F1 Score  : {f1*100:.2f}%")
    print(f"ROC-AUC   : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                'Tidak Diabetes',
                'Diabetes'
            ]
        )
    )
    return (
        model,
        scaler,
        history,
        accuracy,
        precision,
        recall,
        f1,
        auc,
        cm
    )

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

def prediksi_pasien(bmi, highbp, highchol, age, veggies):
    """
    Prediksi HANYA menggunakan Fuzzy (tanpa DL).
    Tetap tersedia untuk kompatibilitas.
    """
    bp_scaled   = scale_bp(highbp)
    chol_scaled = scale_chol(highchol)
    skor_m, label_m, biner_m = prediksi_mamdani(bmi, bp_scaled, chol_scaled, age, veggies)
    skor_s, label_s, biner_s = prediksi_sugeno(bmi, bp_scaled, chol_scaled, age, veggies)

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
    veggies     = row['Veggies']
    age         = row['Age']
    bp_scaled   = scale_bp(row['HighBP'])
    chol_scaled = scale_chol(row['HighChol'])

    skor_m, label_m, biner_m = prediksi_mamdani(bmi, bp_scaled, chol_scaled, age, veggies)
    skor_s, label_s, biner_s = prediksi_sugeno(bmi, bp_scaled, chol_scaled, age, veggies)

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
        s, l, b = prediksi_mamdani(row['BMI'], row['HighBP_scaled'], row['HighChol_scaled'], row['Age'], row['Veggies'])
        skor_m.append(s); label_m.append(l); pred_m.append(b)

    print("Menjalankan Fuzzy Sugeno...")
    skor_s, label_s, pred_s = [], [], []
    for _, row in sample.iterrows():
        s, l, b = prediksi_sugeno(row['BMI'], row['HighBP_scaled'], row['HighChol_scaled'], row['Age'], row['Veggies'])
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

    # ── Visualisasi membership function ──────────────────────
    plot_membership_functions()

    # ── Demo prediksi fuzzy ──────────────────────────────────
    print("\n" + "=" * 55)
    print("  DEMO PREDIKSI FUZZY — SATU PASIEN")
    print("=" * 55)

    kasus = [
        ("Berisiko tinggi", 35.0, 1, 1, 12, 0),
        ("Normal/sehat",    22.0, 0, 0, 3, 1),
        ("Borderline",      27.5, 1, 0, 7, 1),
    ]

    for nama, bmi, bp, chol, age, veggies in kasus:

        hasil = prediksi_pasien(
            bmi,
            bp,
            chol,
            age,
            veggies
        )

        print(
            f"\n[{nama}] "
            f"BMI={bmi}, "
            f"HighBP={bp}, "
            f"HighChol={chol}, "
            f"Age={age}, "
            f"Veggies={veggies}"
        )

        print(
            f"🔵 Mamdani → "
            f"Skor: {hasil['mamdani']['skor']} | "
            f"{hasil['mamdani']['label']}"
        )

        print(
            f"🟠 Sugeno → "
            f"Skor: {hasil['sugeno']['skor']} | "
            f"{hasil['sugeno']['label']}"
        )

    # ── Deep Learning & Evaluasi Dataset ─────────────────────
    CSV_PATH = 'Diabetes_012_health_indicators_BRFSS2015.csv'

    if os.path.exists(CSV_PATH):

        print("\n" + "=" * 55)
        print("  LATIH DEEP LEARNING MODEL")
        print("=" * 55)

        df_full = pd.read_csv(CSV_PATH)

        dl_model, scaler, history = train_deep_learning(
            df_full,
            epochs=30
        )

        plot_training_history(history)
        simpan_model_dl(dl_model, scaler)

        print("\n" + "=" * 55)
        print("  DEMO PREDIKSI HYBRID — SATU PASIEN")
        print("=" * 55)

        pasien_contoh = {
            'BMI': 35.0,
            'HighBP': 1,
            'HighChol': 1,
            'Smoker': 0,
            'Stroke': 0,
            'HeartDiseaseorAttack': 0,
            'PhysActivity': 0,
            'Fruits': 0,
            'Veggies': 0,
            'HvyAlcoholConsump': 0,
            'GenHlth': 4,
            'MentHlth': 0,
            'PhysHlth': 5,
            'DiffWalk': 1,
            'Sex': 1,
            'Age': 9,
            'Education': 4,
            'Income': 5
        }

        hasil = prediksi_pasien_hybrid(
            pasien_contoh,
            dl_model,
            scaler
        )

        print(f"\nInput: {pasien_contoh}")
        print(
            f"🔵 Mamdani  → "
            f"Skor: {hasil['mamdani']['skor']} | "
            f"{hasil['mamdani']['label']}"
        )

        print(
            f"🟠 Sugeno   → "
            f"Skor: {hasil['sugeno']['skor']} | "
            f"{hasil['sugeno']['label']}"
        )

        print(
            f"🤖 DL Prob → "
            f"{hasil['dl_probability']:.4f}"
        )

        print(
            f"🟢 Ensemble → "
            f"Skor: {hasil['ensemble']['skor']} | "
            f"{hasil['ensemble']['label']}"
        )

        print("\n" + "=" * 55)
        print("  EVALUASI PADA DATASET")
        print("=" * 55)

        jalankan_pada_dataset(
            CSV_PATH,
            n_sampel=5000,
            dl_model=dl_model,
            scaler=scaler
        )

    else:

        print(f"\n[INFO] File '{CSV_PATH}' tidak ditemukan.")
        print("       Training Deep Learning dilewati.")
        print("       Evaluasi dataset dilewati.")
        print("       Fuzzy Mamdani & Sugeno tetap dapat digunakan.")