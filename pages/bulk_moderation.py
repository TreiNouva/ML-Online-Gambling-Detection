import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.model_utils import load_model, predict_one

def show():
    st.markdown("## 📂 Moderasi File CSV / Excel")
    st.markdown("Upload file komentar → sistem memberi label otomatis → download hasilnya.")

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f3460,#1a1a2e);
                border:1px solid #38bdf8;border-radius:10px;
                padding:10px 18px;margin-bottom:18px;">
        <span style="color:#e2e8f0;">⭐ Model aktif:
        <b style="color:#38bdf8;">SVM</b>
        — Accuracy <b style="color:#38bdf8;">96.55%</b>
        | F1-Score <b style="color:#38bdf8;">96.55%</b></span>
    </div>""", unsafe_allow_html=True)

    bundle = load_model()

    # Step 1: Format
    st.markdown("### 📄 Langkah 1 — Format File")
    st.info("File harus memiliki kolom bernama **`text`** (atau `comment` / `komentar`). Satu komentar per baris.")
    sample = pd.DataFrame({"text": [
        "Daftar slot gacor maxwin link di bio!",
        "Video sangat bermanfaat terima kasih",
        "s l o t deposit murah win besar",
        "Semangat terus kontennya bagus",
    ]})
    st.download_button("⬇️ Download Template CSV",
        sample.to_csv(index=False).encode(), "template_komentar.csv","text/csv")

    # Step 2: Upload
    st.markdown("### 📤 Langkah 2 — Upload File")
    uploaded = st.file_uploader("Pilih file CSV atau Excel:", type=["csv","xlsx","xls"])

    if uploaded is None:
        st.markdown("""
        <div style="border:2px dashed #2d5a8e;border-radius:12px;
                    padding:32px;text-align:center;color:#64748b;margin-top:8px;">
            <div style="font-size:2.5rem;">📁</div>
            <p>Belum ada file. Klik <b>Browse files</b> di atas atau drag & drop ke sini.<br>
            <small>Format: CSV, XLSX · Maks 200MB</small></p>
        </div>""", unsafe_allow_html=True)
        return

    try:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"❌ Gagal membaca file: {e}"); return

    st.success(f"✅ File dibaca — **{len(df):,} baris**, **{len(df.columns)} kolom**: "
               f"{', '.join(f'`{c}`' for c in df.columns)}")

    # Step 3: Konfigurasi
    st.markdown("### ⚙️ Langkah 3 — Konfigurasi")
    detected = None
    for cand in ["text","comment","komentar","teks","isi","content"]:
        for col in df.columns:
            if col.strip().lower() == cand: detected = col; break
        if detected: break

    c1, c2 = st.columns(2)
    with c1:
        idx = df.columns.tolist().index(detected) if detected else 0
        text_col = st.selectbox("Kolom teks komentar:", df.columns.tolist(), index=idx)
    with c2:
        max_rows = st.number_input("Maksimum baris diproses:",
                                   min_value=10, max_value=5000,
                                   value=min(500,len(df)), step=50)

    filter_opt = st.radio("Filter tampilan hasil:",
                          ["Semua","Judol saja","Aman saja"], horizontal=True)

    st.markdown("#### 👁️ Preview 5 Baris Pertama")
    st.dataframe(df[[text_col]].head(5), use_container_width=True)

    # Step 4: Proses
    st.markdown("### ▶️ Langkah 4 — Jalankan Moderasi")
    run = st.button("🚀  Proses Sekarang", type="primary", use_container_width=True)
    if not run: return

    df_w  = df.head(int(max_rows)).copy()
    total = len(df_w)
    prog  = st.progress(0, text="Memulai...")
    labels, confs, risks = [], [], []

    for i, txt in enumerate(df_w[text_col].astype(str)):
        r    = predict_one(txt, bundle)
        lbl  = "Judol" if r["label"]==1 else "Aman"
        conf = round(r["conf"]*100, 1)
        sc   = conf if r["label"]==1 else 0
        risk = "Berbahaya" if sc>=70 else ("Perlu Review" if sc>=40 else "Aman")
        labels.append(lbl); confs.append(conf); risks.append(risk)
        prog.progress(int((i+1)/total*100), text=f"Memproses {i+1}/{total}...")

    prog.empty()
    st.success(f"✅ Selesai! **{total:,}** komentar dianalisis menggunakan **SVM**.")

    df_w["prediksi"]   = labels
    df_w["confidence"] = confs
    df_w["risk_level"] = risks

    # Step 5: Ringkasan
    st.markdown("---")
    st.markdown("### 📊 Langkah 5 — Ringkasan Hasil")

    judol_n = labels.count("Judol")
    aman_n  = labels.count("Aman")
    bhs_n   = risks.count("Berbahaya")
    rev_n   = risks.count("Perlu Review")

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,val,lbl,color in [
        (c1,total,   "Total",           "#60a5fa"),
        (c2,judol_n, "🚫 Judol",        "#f87171"),
        (c3,aman_n,  "✅ Aman",         "#4ade80"),
        (c4,bhs_n,   "⛔ Berbahaya",    "#ef4444"),
        (c5,rev_n,   "⚠️ Perlu Review", "#fbbf24"),
    ]:
        col.markdown(f"""
        <div style="background:#0f172a;border:1px solid {color}55;
                    border-radius:10px;padding:12px;text-align:center;">
            <div style="font-size:1.9rem;font-weight:700;color:{color};">{val}</div>
            <div style="color:#94a3b8;font-size:.82rem;">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig, axes = plt.subplots(1,2,figsize=(9,3.2))
    fig.patch.set_facecolor('#0f172a')
    for ax in axes:
        ax.set_facecolor('#0f172a')
        for sp in ax.spines.values(): sp.set_edgecolor('#1e3a5f')

    if judol_n>0 or aman_n>0:
        axes[0].pie([judol_n,aman_n], labels=["Judol","Aman"],
                    colors=["#ef4444","#22c55e"], autopct="%1.1f%%", startangle=90,
                    textprops={"color":"white","size":11},
                    wedgeprops={"edgecolor":"#0f172a","linewidth":1.5})
    axes[0].set_title("Judol vs Aman", color="white")

    rk_lbl = ["Aman","Perlu Review","Berbahaya"]
    rk_val = [risks.count(r) for r in rk_lbl]
    bars = axes[1].bar(rk_lbl, rk_val, color=["#22c55e","#f59e0b","#ef4444"], edgecolor="#0f172a")
    for b,v in zip(bars,rk_val):
        axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+.3,
                     str(v), ha="center", color="white", fontsize=10)
    axes[1].set_title("Distribusi Risk Level", color="white")
    axes[1].tick_params(colors="white")
    axes[1].set_ylabel("Jumlah", color="#94a3b8")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Tabel
    st.markdown("### 📋 Tabel Hasil Moderasi")
    df_show = df_w.copy()
    if filter_opt=="Judol saja":  df_show = df_show[df_show["prediksi"]=="Judol"]
    elif filter_opt=="Aman saja": df_show = df_show[df_show["prediksi"]=="Aman"]

    def hl(row):
        if row.get("prediksi")=="Judol": return ["background:rgba(239,68,68,.15)"]*len(row)
        return [""]*len(row)
    st.dataframe(df_show.style.apply(hl,axis=1), use_container_width=True)
    st.caption(f"Menampilkan {len(df_show):,} dari {total:,} baris.")

    # Download
    st.markdown("### ⬇️ Langkah 6 — Download Hasil")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("📥 Download Semua Hasil (CSV)",
            df_w.to_csv(index=False).encode(),
            "hasil_moderasi_semua.csv","text/csv", use_container_width=True)
    with d2:
        judol_df = df_w[df_w["prediksi"]=="Judol"]
        st.download_button(f"🚫 Download Judol Saja ({judol_n} baris)",
            judol_df.to_csv(index=False).encode(),
            "hasil_judol_only.csv","text/csv", use_container_width=True)
