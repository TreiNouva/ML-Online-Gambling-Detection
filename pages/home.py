import streamlit as st

def show():
    st.markdown("""
    <div style="text-align:center;padding:30px 0 16px">
        <div style="font-size:4rem;">🛡️</div>
        <h1 style="font-size:2.6rem;margin:8px 0;">JudolGuard</h1>
        <p style="font-size:1.15rem;color:#94a3b8;max-width:580px;margin:auto;">
            Sistem Deteksi Komentar Promosi Judi Online berbasis
            <b style="color:#60a5fa;">Natural Language Processing</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Fitur ────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc, hint in [
        (c1,"🔍","Deteksi Komentar",
         "Ketik atau tempel satu komentar dan langsung lihat hasilnya dalam hitungan detik.",
         "→ Menu kiri: 🔍 Deteksi Komentar"),
        (c2,"📂","Moderasi File",
         "Upload file CSV atau Excel berisi ratusan komentar, proses sekaligus, lalu download hasilnya.",
         "→ Menu kiri: 📂 Moderasi File CSV/Excel"),
        (c3,"📊","Dashboard",
         "Visualisasi performa model, confusion matrix, distribusi data, dan analisis kata kunci.",
         "→ Menu kiri: 📊 Dashboard Performa Model"),
    ]:
        col.markdown(f"""
        <div style="background:#0f172a;border:1px solid #1e3a5f;
                    border-radius:14px;padding:22px;height:100%;">
            <div style="font-size:2.2rem;">{icon}</div>
            <h3 style="margin:10px 0 6px;">{title}</h3>
            <p style="color:#94a3b8;font-size:0.9rem;">{desc}</p>
            <p style="color:#60a5fa;font-size:0.85rem;margin-top:12px;">{hint}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Kenapa SVM ───────────────────────────────────────────
    st.markdown("### 🤖 Model yang Digunakan: SVM")

    col_info, col_why = st.columns([1, 1])

    with col_info:
        st.markdown("""
        <div style="background:#0f172a;border:2px solid #38bdf8;
                    border-radius:14px;padding:22px;text-align:center;">
            <div style="font-size:2.5rem;">⚙️</div>
            <h3 style="color:#38bdf8;margin:8px 0;">Support Vector Machine</h3>
            <div style="font-size:2.8rem;font-weight:700;color:white;margin:10px 0;">96.55%</div>
            <div style="color:#94a3b8;font-size:0.9rem;">Accuracy pada data test</div>
            <div style="color:#60a5fa;font-size:1rem;margin-top:6px;font-weight:600;">
                F1-Score: 96.55%
            </div>
        </div>""", unsafe_allow_html=True)

    with col_why:
        st.markdown("""
        <div style="background:#0f172a;border:1px solid #1e3a5f;
                    border-radius:14px;padding:22px;">
            <h4 style="color:#f1f5f9;margin:0 0 14px;">Alasan Pemilihan SVM</h4>
            <div style="display:flex;flex-direction:column;gap:10px;">
                <div style="display:flex;gap:10px;align-items:flex-start;">
                    <span style="color:#38bdf8;font-size:1.1rem;margin-top:1px;">①</span>
                    <span style="color:#cbd5e1;font-size:0.92rem;">
                        <b style="color:#f1f5f9;">F1-Score & Accuracy tertinggi</b> dibanding 3 model lain
                        yang diuji (LR, Naive Bayes, Random Forest)
                    </span>
                </div>
                <div style="display:flex;gap:10px;align-items:flex-start;">
                    <span style="color:#38bdf8;font-size:1.1rem;margin-top:1px;">②</span>
                    <span style="color:#cbd5e1;font-size:0.92rem;">
                        <b style="color:#f1f5f9;">Unggul pada teks berdimensi tinggi</b> — SVM dirancang
                        untuk data sparse seperti hasil TF-IDF (12.000 fitur)
                    </span>
                </div>
                <div style="display:flex;gap:10px;align-items:flex-start;">
                    <span style="color:#38bdf8;font-size:1.1rem;margin-top:1px;">③</span>
                    <span style="color:#cbd5e1;font-size:0.92rem;">
                        <b style="color:#f1f5f9;">Robust terhadap noise</b> — memaksimalkan margin
                        sehingga tahan terhadap variasi penulisan komentar judol
                    </span>
                </div>
                <div style="display:flex;gap:10px;align-items:flex-start;">
                    <span style="color:#38bdf8;font-size:1.1rem;margin-top:1px;">④</span>
                    <span style="color:#cbd5e1;font-size:0.92rem;">
                        <b style="color:#f1f5f9;">Seimbang di kedua kelas</b> — performa baik
                        mendeteksi Judol maupun Aman tanpa bias berlebih
                    </span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline ─────────────────────────────────────────────
    st.markdown("### ⚙️ Pipeline Deteksi")
    steps = [
        ("📥","Input","Teks komentar"),
        ("🧹","Preprocessing","Cleaning + normalisasi\nobfuscation handling"),
        ("📐","TF-IDF","Ekstraksi fitur\nunigram + bigram"),
        ("🤖","SVM","Klasifikasi"),
        ("📋","Output","Label + Confidence\n+ Risk Level"),
    ]
    scols = st.columns(len(steps))
    for col, (icon, title, desc) in zip(scols, steps):
        col.markdown(f"""
        <div style="text-align:center;background:#0f172a;border:1px solid #1e3a5f;
                    border-radius:10px;padding:14px;">
            <div style="font-size:1.8rem;">{icon}</div>
            <b style="font-size:0.95rem;">{title}</b>
            <p style="color:#94a3b8;font-size:0.78rem;margin:4px 0 0;white-space:pre-line;">{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Kelompok 13 · NLP Project · Dataset: TikTok & YouTube comments · Model: SVM (F1 96.55%)")
