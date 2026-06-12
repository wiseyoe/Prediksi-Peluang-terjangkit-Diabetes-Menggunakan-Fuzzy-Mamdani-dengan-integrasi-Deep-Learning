"""
DiabetesRisk AI — Streamlit App
Fuzzy Mamdani + Sugeno + Deep Learning Ensemble
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, io

from fuzzy_diabetes import (
    mf_bmi_kurus, mf_bmi_normal, mf_bmi_gemuk, mf_bmi_obesitas,
    mf_bp_rendah, mf_bp_sedang, mf_bp_tinggi,
    mf_chol_rendah, mf_chol_sedang, mf_chol_tinggi,
    mf_tidak, mf_mungkin, mf_iya,
    prediksi_pasien, prediksi_pasien_hybrid,
    fuzzifikasi, inferensi_mamdani, defuzzifikasi_centroid,
    inferensi_sugeno, defuzzifikasi_weighted_average,
    skor_ke_label, scale_bp, scale_chol,
    preprocess, simpan_model_dl, load_model_dl,
    build_dl_model, DL_FEATURES, RULES, FN_MAP,
    X_OUTPUT,
)

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Sistem Prediksi Peluang Terjangkit Diabetes",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS — hanya untuk badge warna ─────────────────────
st.markdown("""
<style>
  #MainMenu, footer { visibility: hidden; }
  .badge {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1rem;
  }
  .badge-tidak   { background: #DCFCE7; color: #166534; }
  .badge-mungkin { background: #FEF3C7; color: #92400E; }
  .badge-iya     { background: #FEE2E2; color: #991B1B; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────
def badge_html(label):
    cls = {"TIDAK": "badge-tidak", "MUNGKIN": "badge-mungkin", "IYA": "badge-iya"}
    return f'<span class="badge {cls.get(label, "")}">{label}</span>'

def label_emoji(label):
    return {"TIDAK": "✅", "MUNGKIN": "⚠️", "IYA": "🔴"}.get(label, "")

def plot_mf(title, x_range, curves, vline=None):
    """Generic membership function plot."""
    fig, ax = plt.subplots(figsize=(5, 2.8))
    colors = ["#2563EB", "#16A34A", "#D97706", "#DC2626"]
    for i, (fn, lbl) in enumerate(curves):
        x = np.linspace(*x_range, 500)
        ax.plot(x, fn(x), color=colors[i % len(colors)], lw=2, label=lbl)
    if vline is not None:
        ax.axvline(vline, color="#111827", lw=1.5, ls="--", label=f"nilai={vline}")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.set_facecolor("#F9FAFB")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig

def plot_mamdani(agregat, skor):
    fig, ax = plt.subplots(figsize=(6, 3))
    x = X_OUTPUT
    ax.fill_between(x, agregat, alpha=0.25, color="#2563EB")
    ax.plot(x, agregat, "#2563EB", lw=1.8, label="Agregat")
    ax.axvline(skor, color="#DC2626", lw=2, ls="--", label=f"Defuzz = {skor:.3f}")
    for fn, lbl, c in [(mf_tidak,"TIDAK","#16A34A"),(mf_mungkin,"MUNGKIN","#D97706"),(mf_iya,"IYA","#DC2626")]:
        ax.plot(x, fn(x), color=c, lw=1, ls=":", alpha=0.6, label=lbl)
    ax.set_title("Mamdani — Agregat & Defuzzifikasi Centroid", fontsize=9, fontweight="bold")
    ax.set_xlabel("Skor Risiko (0–10)"); ax.set_ylabel("μ(x)")
    ax.set_ylim(-0.05, 1.15); ax.legend(fontsize=7); ax.grid(True, alpha=0.2, linestyle=":")
    ax.set_facecolor("#F9FAFB"); fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig

def plot_sugeno(alphas, zs):
    nonzero = [(a, z) for a, z in zip(alphas, zs) if a > 0.001]
    if not nonzero:
        return None
    alphas_nz, zs_nz = zip(*nonzero)
    labels = [f"z={z:.1f}" for z in zs_nz]
    colors = ["#16A34A" if z < 4 else "#D97706" if z < 6.5 else "#DC2626" for z in zs_nz]
    fig, ax = plt.subplots(figsize=(5.5, max(2.5, len(nonzero) * 0.42)))
    bars = ax.barh(labels, alphas_nz, color=colors, edgecolor="white", height=0.6)
    for bar, a in zip(bars, alphas_nz):
        ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height()/2,
                f"{a:.3f}", va="center", fontsize=7.5)
    ax.set_xlim(0, 1.2); ax.set_xlabel("Firing Strength (α)")
    ax.set_title("Sugeno — Firing Strength per Rule", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.2, linestyle=":")
    ax.set_facecolor("#F9FAFB"); fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


# ── Session state ──────────────────────────────────────────────
for k, v in [("dl_model", None), ("dl_scaler", None), ("hasil", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════
#  SIDEBAR — INPUT
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("Pengaturan Data")
    st.divider()

    st.subheader("Filter Utama")
    bmi_val  = st.slider("BMI", 10.0, 70.0, 27.5, 0.5,
                          help="Body Mass Index pasien")
    highbp   = st.selectbox("Tekanan Darah Tinggi", [0, 1],
                             format_func=lambda x: "Tidak" if x == 0 else "Ya")
    highchol = st.selectbox("Kolesterol Tinggi", [0, 1],
                             format_func=lambda x: "Tidak" if x == 0 else "Ya")

    st.divider()
    st.subheader("Filter Tambahan")

    col1, col2 = st.columns(2)
    with col1:
        smoker    = st.selectbox("Perokok",    [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        stroke    = st.selectbox("Stroke",     [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        heart_dis = st.selectbox("Jantung",    [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        phys_act  = st.selectbox("Aktif Fisik",[1,0], format_func=lambda x: "Ya" if x else "Tidak")
        fruits    = st.selectbox("Buah",       [1,0], format_func=lambda x: "Ya" if x else "Tidak")
    with col2:
        veggies   = st.selectbox("Sayur",      [1,0], format_func=lambda x: "Ya" if x else "Tidak")
        heavy_alc = st.selectbox("Alkohol",    [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        diff_walk = st.selectbox("Sulit Jalan",[0,1], format_func=lambda x: "Ya" if x else "Tidak")
        sex       = st.selectbox("Kelamin",    [0,1], format_func=lambda x: "P" if x==0 else "L")

    gen_hlth  = st.select_slider("Kesehatan Umum (1=Baik–5=Buruk)", [1,2,3,4,5], 3)
    ment_hlth = st.slider("Hari Kes. Mental Buruk /30hr", 0, 30, 0)
    phys_hlth = st.slider("Hari Kes. Fisik Buruk /30hr",  0, 30, 0)
    age       = st.select_slider("Kel. Usia (1=18–24 … 13=80+)", list(range(1,14)), 7)
    education = st.select_slider("Pendidikan (1–6)", [1,2,3,4,5,6], 4)
    income    = st.select_slider("Pendapatan (1–8)", list(range(1,9)), 5)

    st.divider()
    predict_btn = st.button("Kirim", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.title("🩺 Sistem Prediksi Peluang Terjangkit Diabetes")
st.divider()


# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab_pred, tab_mf, tab_dl, tab_eval, tab_info = st.tabs([
    "📊 Prediksi",
    "📐 Keanggotaan & Rules",
    "🤖 Deep Learning",
    "📁 Evaluasi Dataset",
    "ℹ️ Tentang",
])


# ──────────────────────────────────────────────────────────────
#  TAB 1 — PREDIKSI
# ──────────────────────────────────────────────────────────────
with tab_pred:
    if predict_btn:
        bp_s   = scale_bp(highbp)
        chol_s = scale_chol(highchol)

        # Fuzzy
        hasil_f = prediksi_pasien(bmi_val, highbp, highchol)

        # Detail untuk plot
        fuzz    = fuzzifikasi(bmi_val, bp_s, chol_s)
        agregat = inferensi_mamdani(fuzz)
        skor_m  = defuzzifikasi_centroid(agregat)
        al_s, zs_s = inferensi_sugeno(fuzz)
        skor_s  = defuzzifikasi_weighted_average(al_s, zs_s)

        # Hybrid (jika model ada)
        row_dict = {
            "BMI": bmi_val, "HighBP": highbp, "HighChol": highchol,
            "Smoker": smoker, "Stroke": stroke, "HeartDiseaseorAttack": heart_dis,
            "PhysActivity": phys_act, "Fruits": fruits, "Veggies": veggies,
            "HvyAlcoholConsump": heavy_alc, "GenHlth": gen_hlth,
            "MentHlth": ment_hlth, "PhysHlth": phys_hlth, "DiffWalk": diff_walk,
            "Sex": sex, "Age": age, "Education": education, "Income": income,
        }
        hasil_h = None
        if st.session_state.dl_model is not None:
            hasil_h = prediksi_pasien_hybrid(row_dict, st.session_state.dl_model, st.session_state.dl_scaler)

        st.session_state.hasil = {
            "hasil_f": hasil_f, "hasil_h": hasil_h,
            "fuzz": fuzz, "agregat": agregat, "skor_m": skor_m,
            "al_s": al_s, "zs_s": zs_s, "skor_s": skor_s,
            "bmi_val": bmi_val, "bp_s": bp_s, "chol_s": chol_s,
        }

    if st.session_state.hasil is None:
        st.info("👈 Atur data pasien di panel kiri, lalu klik **Prediksi**.")
        st.stop()

    d       = st.session_state.hasil
    hasil_f = d["hasil_f"]
    hasil_h = d["hasil_h"]

    # ── Kartu Hasil ──────────────────────────────────────────
    st.subheader("Hasil Prediksi")
    n_col = 3 if hasil_h else 2
    cols  = st.columns(n_col)

    lbl_m = hasil_f["mamdani"]["label"]
    with cols[0]:
        st.metric("🔵 Fuzzy Mamdani", f"{hasil_f['mamdani']['skor']:.3f} / 10")
        st.markdown(badge_html(lbl_m), unsafe_allow_html=True)
        st.caption("Centroid of Gravity")

    lbl_s = hasil_f["sugeno"]["label"]
    with cols[1]:
        st.metric("🟠 Fuzzy Sugeno", f"{hasil_f['sugeno']['skor']:.3f} / 10")
        st.markdown(badge_html(lbl_s), unsafe_allow_html=True)
        st.caption("Weighted Average")

    if hasil_h:
        lbl_e = hasil_h["ensemble"]["label"]
        with cols[2]:
            st.metric("🟢 Hybrid Ensemble", f"{hasil_h['ensemble']['skor']:.3f} / 10",
                      delta=f"DL prob: {hasil_h['dl_probability']:.1%}")
            st.markdown(badge_html(lbl_e), unsafe_allow_html=True)
            st.caption(f"Fuzzy×0.4 + DL×0.6")
    else:
        with cols[1] if n_col == 2 else cols[2]:
            pass
        

    st.divider()

    # ── Derajat Keanggotaan ───────────────────────────────────
    st.subheader("Derajat Keanggotaan Input")
    fuzz = d["fuzz"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("**BMI**")
        for k, lbl in [("bmi_kurus","Kurus"),("bmi_normal","Normal"),
                        ("bmi_gemuk","Gemuk"),("bmi_obesitas","Obesitas")]:
            v = fuzz[k]
            st.markdown(f"`{lbl}` &nbsp; **{v:.4f}**", unsafe_allow_html=True)
            st.progress(float(v))
    with c2:
        st.caption("**Tekanan Darah**")
        for k, lbl in [("bp_rendah","Rendah"),("bp_sedang","Sedang"),("bp_tinggi","Tinggi")]:
            v = fuzz[k]
            st.markdown(f"`{lbl}` &nbsp; **{v:.4f}**", unsafe_allow_html=True)
            st.progress(float(v))
    with c3:
        st.caption("**Kolesterol**")
        for k, lbl in [("chol_rendah","Rendah"),("chol_sedang","Sedang"),("chol_tinggi","Tinggi")]:
            v = fuzz[k]
            st.markdown(f"`{lbl}` &nbsp; **{v:.4f}**", unsafe_allow_html=True)
            st.progress(float(v))

    st.divider()

    # ── Plot Inferensi ────────────────────────────────────────
    st.subheader("Visualisasi Inferensi")
    p1, p2 = st.columns(2)
    with p1:
        fig = plot_mamdani(d["agregat"], d["skor_m"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with p2:
        fig2 = plot_sugeno(d["al_s"], d["zs_s"])
        if fig2:
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
        else:
            st.warning("Semua rule memiliki firing strength = 0.")


# ──────────────────────────────────────────────────────────────
#  TAB 2 — KEANGGOTAAN & RULES
# ──────────────────────────────────────────────────────────────
with tab_mf:
    st.subheader("Fungsi Keanggotaan")
    st.caption("Garis putus-putus = posisi nilai input pasien saat ini.")

    bp_s_cur   = scale_bp(highbp)
    chol_s_cur = scale_chol(highchol)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig = plot_mf("BMI", (10, 70),
                      [(mf_bmi_kurus,"Kurus"),(mf_bmi_normal,"Normal"),
                       (mf_bmi_gemuk,"Gemuk"),(mf_bmi_obesitas,"Obesitas")],
                      vline=bmi_val)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with r1c2:
        fig = plot_mf("Tekanan Darah (HighBP)", (0, 10),
                      [(mf_bp_rendah,"Rendah"),(mf_bp_sedang,"Sedang"),(mf_bp_tinggi,"Tinggi")],
                      vline=bp_s_cur)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig = plot_mf("Kolesterol (HighChol)", (0, 10),
                      [(mf_chol_rendah,"Rendah"),(mf_chol_sedang,"Sedang"),(mf_chol_tinggi,"Tinggi")],
                      vline=chol_s_cur)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with r2c2:
        fig = plot_mf("Output — Risiko Diabetes", (0, 10),
              [(mf_tidak,"TIDAK"),(mf_mungkin,"MUNGKIN"),(mf_iya,"IYA")],
              vline=d["skor_m"])
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.divider()
    st.subheader("Rule Base (20 Rules)")

    fn_label = {
        "bmi_kurus":"Kurus","bmi_normal":"Normal","bmi_gemuk":"Gemuk","bmi_obesitas":"Obesitas",
        "bp_rendah":"Rendah","bp_sedang":"Sedang","bp_tinggi":"Tinggi",
        "chol_rendah":"Rendah","chol_sedang":"Sedang","chol_tinggi":"Tinggi",
    }
    out_name = {id(mf_tidak):"TIDAK", id(mf_mungkin):"MUNGKIN", id(mf_iya):"IYA"}

    rows = []
    for i, (fb, fp, fc, fo, z) in enumerate(RULES, 1):
        rows.append({
            "No": f"R{i:02d}",
            "BMI": fn_label.get(FN_MAP[fb], "?"),
            "Tekanan Darah": fn_label.get(FN_MAP[fp], "?"),
            "Kolesterol": fn_label.get(FN_MAP[fc], "?"),
            "Output Mamdani": out_name.get(id(fo), "?"),
            "z Sugeno": z,
        })

    df_rules = pd.DataFrame(rows)
    st.dataframe(df_rules, hide_index=True, use_container_width=True,
                 column_config={
                     "No": st.column_config.TextColumn(width="small"),
                     "z Sugeno": st.column_config.NumberColumn(format="%.1f"),
                 })


# ──────────────────────────────────────────────────────────────
#  TAB 3 — DEEP LEARNING
# ──────────────────────────────────────────────────────────────
with tab_dl:
    st.subheader("Deep Learning Model")

    # Status
    if st.session_state.dl_model is not None:
        st.success("✅ Model Deep Learning sudah aktif. Prediksi Ensemble tersedia.")
    else:
        st.warning("⚠️ Model belum dimuat. Latih atau muat model dari file.")

    st.divider()

    # Muat model tersimpan
    if os.path.exists("saved_model/dl_model.keras"):
        if st.button("📂 Muat Model Tersimpan", use_container_width=True):
            with st.spinner("Memuat model..."):
                m, s = load_model_dl()
                st.session_state.dl_model  = m
                st.session_state.dl_scaler = s
            st.success("Model berhasil dimuat!")
            st.rerun()

    st.divider()
    st.subheader("Latih Model Baru")

    st.info("Upload file CSV yang akan digunakan untuk melatih model.")

    uploaded = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded:
        col_ep, col_bs = st.columns(2)
        epochs = col_ep.slider("Epochs", 5, 50, 20)
        batch  = col_bs.select_slider("Batch Size", [32, 64, 128, 256], 64)

        if st.button("🚀 Mulai Training", use_container_width=True, type="primary"):
            import tensorflow as tf
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            with st.spinner("Membaca dataset..."):
                df_up = pd.read_csv(uploaded)
            st.caption(f"Dataset: {len(df_up):,} baris")

            prog  = st.progress(0, text="Memulai training...")
            chart = st.empty()
            logs_ = []

            class CB(tf.keras.callbacks.Callback):
                def on_epoch_end(self, ep, logs=None):
                    pct = int((ep+1) / epochs * 100)
                    prog.progress(pct, text=f"Epoch {ep+1}/{epochs}  —  loss: {logs.get('loss',0):.4f}  |  val_auc: {logs.get('val_auc',0):.4f}")
                    logs_.append({"loss": logs.get("loss"), "val_loss": logs.get("val_loss"),
                                  "val_auc": logs.get("val_auc")})
                    if len(logs_) > 1:
                        chart.line_chart(pd.DataFrame(logs_), y=["loss","val_loss"])

            data  = df_up[DL_FEATURES + ["Diabetes_012"]].dropna()
            X     = data[DL_FEATURES].values
            y     = data["Diabetes_012"].astype(int).values
            sc    = StandardScaler()
            X_sc  = sc.fit_transform(X)
            X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2,
                                                        random_state=42, stratify=y)
            cw = {0: 1.0, 1: float(np.sum(y_tr==0)/np.sum(y_tr==1))}

            model = build_dl_model(X_tr.shape[1])
            model.fit(X_tr, y_tr, validation_data=(X_te, y_te),
                      epochs=epochs, batch_size=batch,
                      class_weight=cw, callbacks=[CB()], verbose=0)

            st.session_state.dl_model  = model
            st.session_state.dl_scaler = sc
            simpan_model_dl(model, sc)
            st.success("✅ Training selesai! Model disimpan dan Ensemble aktif.")

    st.divider()
    st.subheader("Arsitektur & Formula Ensemble")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Arsitektur MLP (18 fitur input)**")
        arch = pd.DataFrame([
            {"Layer": "Input → Dense", "Unit": 64, "Aktivasi": "ReLU", "Reg": "BatchNorm + Dropout(0.3)"},
            {"Layer": "Dense",         "Unit": 32, "Aktivasi": "ReLU", "Reg": "BatchNorm + Dropout(0.2)"},
            {"Layer": "Dense",         "Unit": 16, "Aktivasi": "ReLU", "Reg": "—"},
            {"Layer": "Output",        "Unit":  1, "Aktivasi": "Sigmoid", "Reg": "—"},
        ])
        st.dataframe(arch, hide_index=True, use_container_width=True)
        st.caption("Loss: Binary Crossentropy · Optimizer: Adam · EarlyStopping: val_auc")

    with c2:
        st.markdown("**Formula Ensemble Fusion**")
        st.code(
            "skor_dl       = dl_probability × 10\n"
            "skor_fuzzy    = (skor_mamdani + skor_sugeno) / 2\n"
            "skor_ensemble = 0.4 × skor_fuzzy\n"
            "              + 0.6 × skor_dl\n\n"
            "# Threshold label:\n"
            "# skor < 4.0  → TIDAK\n"
            "# 4.0 – 6.5   → MUNGKIN\n"
            "# ≥ 6.5       → IYA",
            language="python"
        )


# ──────────────────────────────────────────────────────────────
#  TAB 4 — EVALUASI DATASET
# ──────────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("Evaluasi Batch pada Dataset BRFSS 2015")
    st.info("Upload dataset CSV untuk membandingkan akurasi Mamdani vs Sugeno secara batch.")

    up_eval  = st.file_uploader("Pilih file CSV", type=["csv"], key="eval_csv")
    n_sampel = st.slider("Jumlah Sampel Evaluasi", 500, 10_000, 2_000, 500)

    if up_eval and st.button("▶️ Jalankan Evaluasi", use_container_width=True, type="primary"):
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from fuzzy_diabetes import prediksi_mamdani, prediksi_sugeno

        with st.spinner("Memproses..."):
            df_ev  = pd.read_csv(up_eval)
            cols_need = list(set(["BMI","HighBP","HighChol","Diabetes_012"] + DL_FEATURES))
            cols_ok   = [c for c in cols_need if c in df_ev.columns]
            data_ev   = df_ev[cols_ok].dropna()
            data_ev   = preprocess(data_ev)
            sample    = data_ev.sample(n=min(n_sampel, len(data_ev)), random_state=42).reset_index(drop=True)

            pred_m, pred_s = [], []
            bar = st.progress(0)
            total = len(sample)
            for idx, (_, row) in enumerate(sample.iterrows()):
                _, _, bm = prediksi_mamdani(row["BMI"], row["HighBP_scaled"], row["HighChol_scaled"])
                _, _, bs = prediksi_sugeno(row["BMI"], row["HighBP_scaled"], row["HighChol_scaled"])
                pred_m.append(bm); pred_s.append(bs)
                if idx % 200 == 0:
                    bar.progress(min(idx/total, 1.0))
            bar.progress(1.0)

        y_true = sample["Diabetes_012"].astype(int).values
        acc_m  = accuracy_score(y_true, pred_m)
        acc_s  = accuracy_score(y_true, pred_s)

        st.divider()
        st.subheader("Hasil Evaluasi")

        mc1, mc2 = st.columns(2)
        mc1.metric("🔵 Fuzzy Mamdani", f"{acc_m*100:.2f}%", f"{len(sample):,} sampel")
        mc2.metric("🟠 Fuzzy Sugeno",  f"{acc_s*100:.2f}%", f"{len(sample):,} sampel")

        # Bar chart
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["Mamdani","Sugeno"], [acc_m*100, acc_s*100],
               color=["#2563EB","#EA580C"], width=0.4, edgecolor="white")
        for i, v in enumerate([acc_m*100, acc_s*100]):
            ax.text(i, v+0.5, f"{v:.2f}%", ha="center", fontweight="bold", fontsize=9)
        ax.set_ylim(0, 110); ax.set_ylabel("Akurasi (%)")
        ax.set_title("Perbandingan Akurasi", fontweight="bold", fontsize=10)
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.set_facecolor("#F9FAFB"); fig.patch.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("📋 Classification Report Lengkap"):
            st.markdown("**Mamdani**")
            st.code(classification_report(y_true, pred_m,
                    target_names=["Tidak Diabetes","Mungkin Diabetes","Diabetes"]))
            st.markdown("**Sugeno**")
            st.code(classification_report(y_true, pred_s,
                    target_names=["Tidak Diabetes","Mungkin Diabetes","Diabetes"]))

        # Distribusi label
        with st.expander("📊 Distribusi Output"):
            dc1, dc2 = st.columns(2)
            for col, preds, title in [(dc1, pred_m,"Mamdani"),(dc2, pred_s,"Sugeno")]:
                cnt = pd.Series(preds).value_counts().rename({0:"Tidak Diabetes",1:"Diabetes"})
                col.caption(f"**{title}**")
                col.bar_chart(cnt)


# ──────────────────────────────────────────────────────────────
#  TAB 5 — TENTANG
# ──────────────────────────────────────────────────────────────
with tab_info:
    st.subheader("Tentang Sistem")

    st.markdown("""
**Sistem Prediksi Peluang Terjangkit Diabetes** adalah sistem prediksi risiko diabetes berbasis **Fuzzy Logic**
(Mamdani & Sugeno) yang diperkuat dengan **Deep Learning** dalam skema Hybrid Ensemble.
    """)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Dataset")
        st.markdown("""
- **BRFSS 2015** (Behavioral Risk Factor Surveillance System)
- 253.680 responden dewasa Amerika Serikat
- Label: `Diabetes_012` (0 = tidak, 1 = diabetes/prediabetes)
        """)

        st.markdown("#### Fitur Fuzzy Input (3 fitur)")
        st.dataframe(pd.DataFrame([
            {"Fitur":"BMI",      "Tipe":"Kontinu (10–70)", "Himpunan":"Kurus · Normal · Gemuk · Obesitas"},
            {"Fitur":"HighBP",   "Tipe":"Binary → 0–10",   "Himpunan":"Rendah · Sedang · Tinggi"},
            {"Fitur":"HighChol", "Tipe":"Binary → 0–10",   "Himpunan":"Rendah · Sedang · Tinggi"},
        ]), hide_index=True, use_container_width=True)

    with c2:
        st.markdown("#### Metode")
        st.markdown("""
- **Mamdani** — AND: min · Agregasi: max · Defuzz: Centroid CoG
- **Sugeno** — AND: min · Konsekuen: konstan · Defuzz: Weighted Average
- **Deep Learning** — 18 fitur · MLP 4 layer · BatchNorm · EarlyStopping
- **Ensemble** — `0.4 × Fuzzy + 0.6 × DL`
        """)

        st.markdown("#### Threshold Klasifikasi")
        st.dataframe(pd.DataFrame([
            {"Skor":"< 4.0",      "Label":"TIDAK",   "Interpretasi":"Risiko rendah"},
            {"Skor":"4.0 – 6.5",  "Label":"MUNGKIN", "Interpretasi":"Risiko sedang"},
            {"Skor":"≥ 6.5",      "Label":"IYA",     "Interpretasi":"Risiko tinggi"},
        ]), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("#### 20 Rule Base")
    fn_lbl = {
        "bmi_kurus":"Kurus","bmi_normal":"Normal","bmi_gemuk":"Gemuk","bmi_obesitas":"Obesitas",
        "bp_rendah":"Rendah","bp_sedang":"Sedang","bp_tinggi":"Tinggi",
        "chol_rendah":"Rendah","chol_sedang":"Sedang","chol_tinggi":"Tinggi",
    }
    out_nm = {id(mf_tidak):"TIDAK", id(mf_mungkin):"MUNGKIN", id(mf_iya):"IYA"}
    rows_info = []
    for i, (fb, fp, fc, fo, z) in enumerate(RULES, 1):
        rows_info.append({
            "No": f"R{i:02d}",
            "JIKA BMI": fn_lbl.get(FN_MAP[fb],"?"),
            "DAN BP": fn_lbl.get(FN_MAP[fp],"?"),
            "DAN Kolesterol": fn_lbl.get(FN_MAP[fc],"?"),
            "MAKA": out_nm.get(id(fo),"?"),
            "z": z,
        })
    st.dataframe(pd.DataFrame(rows_info), hide_index=True, use_container_width=True)
