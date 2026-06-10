import streamlit as st
from utils.model_utils import load_model, predict_one

EXAMPLES = [
    ("🚫 Judol — Langsung",    "Daftar sekarang! Slot gacor maxwin terpercaya, link di bio!"),
    ("🚫 Judol — Obfuscation", "s l o t gacor hari ini deposit murah w1n terbesar buktikan"),
    ("🚫 Judol — Terselubung", "Mau penghasilan tambahan? DM admin ya ada info menarik banget"),
    ("✅ Komentar Biasa",       "Mantap videonya kak, sangat informatif dan bermanfaat sekali!"),
    ("✅ Sebut Judi tapi Aman", "Pemerintah harus tegas memberantas judi online di Indonesia!"),
]

def show():
    st.markdown("## 🔍 Deteksi Komentar")
    st.markdown("Ketik atau tempel komentar di kolom bawah, lalu klik **Deteksi Sekarang**.")

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f3460,#1a1a2e);
                border:1px solid #38bdf8;border-radius:10px;
                padding:10px 18px;margin-bottom:18px;">
        <span style="color:#e2e8f0;">⭐ Model aktif:
        <b style="color:#38bdf8;">SVM</b>
        &nbsp;—&nbsp; Accuracy <b style="color:#38bdf8;">96.55%</b>
        &nbsp;|&nbsp; F1-Score <b style="color:#38bdf8;">96.55%</b></span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    bundle = load_model()

    # Contoh komentar
    with st.expander("💡 Pilih contoh komentar (klik untuk buka)"):
        cols = st.columns(len(EXAMPLES))
        for col, (label, txt) in zip(cols, EXAMPLES):
            if col.button(label, use_container_width=True, key=f"ex_{label}"):
                st.session_state["komentar_input"] = txt
                st.rerun()

    # Input
    st.markdown("### ✍️ Input Komentar")
    teks = st.text_area(
        label="Teks komentar:",
        value=st.session_state.get("komentar_input", ""),
        height=130,
        placeholder="Ketik atau tempel komentar di sini...\n\nContoh: Daftar slot gacor link di bio...",
        key="komentar_input",
        label_visibility="collapsed",
    )

    deteksi = st.button("🔎  Deteksi Sekarang", type="primary", use_container_width=True)

    # Hasil
    if deteksi:
        if not teks.strip():
            st.warning("⚠️ Kolom komentar kosong. Ketik komentar atau pilih contoh di atas.")
            return

        with st.spinner("Menganalisis..."):
            res = predict_one(teks, bundle)

        st.markdown("---")
        st.markdown("### 📋 Hasil Deteksi")

        label    = res["label"]
        conf_pct = res["conf"] * 100
        is_judol = label == 1

        if is_judol:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#450a0a,#7f1d1d);
                        border:2px solid #ef4444;border-radius:14px;
                        padding:28px;text-align:center;margin-bottom:16px;">
                <div style="font-size:3.5rem;">🚫</div>
                <h2 style="color:#fca5a5;margin:8px 0;">JUDOL TERDETEKSI</h2>
                <p style="color:#fecaca;">Komentar ini terindikasi promosi judi online</p>
                <p style="color:#ddd;margin-top:10px;">
                    Confidence: <b style="color:white;font-size:1.2rem;">{conf_pct:.1f}%</b>
                    &nbsp;·&nbsp; Model: <b style="color:#38bdf8;">SVM</b>
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#052e16,#14532d);
                        border:2px solid #22c55e;border-radius:14px;
                        padding:28px;text-align:center;margin-bottom:16px;">
                <div style="font-size:3.5rem;">✅</div>
                <h2 style="color:#86efac;margin:8px 0;">KOMENTAR AMAN</h2>
                <p style="color:#bbf7d0;">Tidak terdeteksi sebagai promosi judi online</p>
                <p style="color:#ddd;margin-top:10px;">
                    Confidence: <b style="color:white;font-size:1.2rem;">{conf_pct:.1f}%</b>
                    &nbsp;·&nbsp; Model: <b style="color:#38bdf8;">SVM</b>
                </p>
            </div>""", unsafe_allow_html=True)

        # Confidence bar
        bar_color = "#ef4444" if is_judol else "#22c55e"
        st.markdown(f"""
        <div style="margin:12px 0 20px;">
            <div style="display:flex;justify-content:space-between;
                        color:#94a3b8;font-size:.85rem;margin-bottom:5px;">
                <span>Confidence Score</span><span>{conf_pct:.1f}%</span>
            </div>
            <div style="background:#1e293b;border-radius:8px;height:16px;">
                <div style="background:{bar_color};width:{conf_pct:.1f}%;
                            height:16px;border-radius:8px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Risk level
        score = conf_pct if is_judol else 0
        r1, r2, r3 = st.columns(3)
        for col, lbl, rng, active in [
            (r1, "✅ Aman",         "Confidence 0–40%",   score < 40),
            (r2, "⚠️ Perlu Review", "Confidence 40–70%",  40 <= score < 70),
            (r3, "🚫 Berbahaya",    "Confidence 70–100%", score >= 70),
        ]:
            bdr = "2px solid #f1f5f9" if active else "1px solid #1e3a5f"
            op  = "1" if active else "0.35"
            col.markdown(f"""
            <div style="background:#0f172a;border:{bdr};opacity:{op};
                        border-radius:10px;padding:12px;text-align:center;">
                <b>{lbl}</b><br><small style="color:#94a3b8;">{rng}</small>
            </div>""", unsafe_allow_html=True)

        with st.expander("🔧 Lihat teks setelah preprocessing"):
            st.code(res["clean"] or "(kosong setelah cleaning)", language=None)
