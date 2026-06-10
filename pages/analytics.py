import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from utils.model_utils import load_model

MODELS  = ["SVM","Logistic Regression","Naive Bayes","Random Forest"]
COLORS  = ["#38bdf8","#f472b6","#34d399","#fb923c"]

def dark_ax(ax):
    ax.set_facecolor('#0f172a')
    ax.tick_params(colors='#cbd5e1')
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('white')
    for sp in ax.spines.values(): sp.set_edgecolor('#1e3a5f')

def show():
    st.markdown("## 📊 Dashboard Performa Model")
    bundle  = load_model()
    results = bundle['eval_results']

    tab1, tab2, tab3 = st.tabs(["📈 Performa","🔢 Confusion Matrix","🔑 Top Keywords"])

    # ══ TAB 1: PERFORMA ═════════════════════════════════════
    with tab1:
        st.markdown("### Ringkasan Metrik — 4 Model")
        cols = st.columns(4)
        for col, name, color in zip(cols, MODELS, COLORS):
            r = results[name]
            star = " ⭐" if name=="SVM" else ""
            col.markdown(f"""
            <div style="background:#0f172a;border:2px solid {color}44;
                        border-radius:12px;padding:16px;text-align:center;">
                <b style="color:{color};">{name}{star}</b><br>
                <span style="font-size:2rem;font-weight:700;color:white;">
                    {r['accuracy']*100:.2f}%</span><br>
                <small style="color:#94a3b8;">Accuracy</small><br>
                <span style="color:#60a5fa;font-size:.9rem;">F1: {r['f1']*100:.2f}%</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Grouped bar
        st.markdown("### Grouped Bar Chart — Semua Metrik")
        metrics = ['accuracy','precision','recall','f1']
        mlbl    = ['Accuracy','Precision','Recall','F1-Score']
        x, w    = np.arange(4), 0.18
        fig, ax = plt.subplots(figsize=(11,4.5))
        fig.patch.set_facecolor('#0f172a'); dark_ax(ax)
        for i,(name,color) in enumerate(zip(MODELS,COLORS)):
            vals = [results[name][m] for m in metrics]
            bars = ax.bar(x+i*w, vals, w, label=name, color=color, alpha=.88)
            for b,v in zip(bars,vals):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+.004,
                        f'{v*100:.1f}', ha='center', fontsize=7.5, color='white', fontweight='bold')
        ax.set_xticks(x+w*1.5); ax.set_xticklabels(mlbl, color='white')
        ax.set_ylim(0.75,1.12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f'{v:.0%}'))
        ax.legend(facecolor='#0f172a',edgecolor='#1e3a5f',labelcolor='white',fontsize=9)
        ax.set_title('Perbandingan Performa 4 Model', color='white', pad=12)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Tabel ringkasan
        st.markdown("### Tabel Perbandingan")
        rows = []
        for name in MODELS:
            r = results[name]
            rows.append({"Model": name + (" ⭐" if name=="SVM" else ""),
                "Accuracy":f"{r['accuracy']*100:.2f}%",
                "Precision":f"{r['precision']*100:.2f}%",
                "Recall":f"{r['recall']*100:.2f}%",
                "F1-Score":f"{r['f1']*100:.2f}%"})
        def hl_svm(row):
            if '⭐' in str(row.get('Model','')):
                return ['background:rgba(56,189,248,.15);font-weight:bold']*len(row)
            return ['']*len(row)
        st.dataframe(pd.DataFrame(rows).style.apply(hl_svm,axis=1),
                     use_container_width=True, hide_index=True)
        st.caption("⭐ SVM dipilih sebagai model utama berdasarkan F1-Score tertinggi.")

        # Radar
        st.markdown("### Radar Chart")
        fig2, ax2 = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
        fig2.patch.set_facecolor('#0f172a'); ax2.set_facecolor('#0f172a')
        cats = ['Accuracy','Precision','Recall','F1']; N=4
        angles = [n/N*2*np.pi for n in range(N)]+[0]
        ax2.set_xticks(angles[:-1]); ax2.set_xticklabels(cats, color='white', size=9)
        ax2.set_ylim(0.75,1); ax2.set_yticks([.80,.85,.90,.95,1])
        ax2.set_yticklabels(['80%','85%','90%','95%','100%'], color='#94a3b8', size=7)
        ax2.grid(color='#1e3a5f',lw=.6); ax2.spines['polar'].set_color('#1e3a5f')
        for name, color in zip(MODELS, COLORS):
            r = results[name]
            vals = [r['accuracy'],r['precision'],r['recall'],r['f1'],r['accuracy']]
            lw = 2.5 if name=="SVM" else 1.2
            ax2.plot(angles, vals,'o-', lw=lw, color=color,
                     label=name+(' ⭐' if name=='SVM' else ''), markersize=4 if name=='SVM' else 2)
            ax2.fill(angles, vals, alpha=.12 if name=='SVM' else .04, color=color)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.5,1.2),
                   facecolor='#0f172a',edgecolor='#1e3a5f',labelcolor='white',fontsize=8)
        ax2.set_title('Radar Chart Perbandingan', color='white', pad=18)
        rc1,_ = st.columns([1,1])
        with rc1: st.pyplot(fig2); plt.close()

    # ══ TAB 2: CONFUSION MATRIX ════════════════════════════
    with tab2:
        st.markdown("### Confusion Matrix — Semua Model")
        st.info("**TN** = Benar Aman · **FP** = Salah alarm · **FN** = Judol lolos · **TP** = Benar Judol terdeteksi")

        fig4, axes4 = plt.subplots(2,2,figsize=(11,8))
        fig4.patch.set_facecolor('#0f172a')
        for ax, name, color in zip(axes4.flatten(), MODELS, COLORS):
            cm = np.array(results[name]['cm'])
            sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                        cmap=sns.light_palette(color, as_cmap=True),
                        xticklabels=['Non-Judol','Judol'],
                        yticklabels=['Non-Judol','Judol'],
                        linewidths=.5, linecolor='#0f172a',
                        annot_kws={"size":14,"weight":"bold"})
            r = results[name]
            star = ' ⭐' if name=='SVM' else ''
            ax.set_title(f'{name}{star}\nAcc {r["accuracy"]*100:.2f}%  F1 {r["f1"]*100:.2f}%',
                         color='white', pad=8)
            ax.set_xlabel('Prediksi',color='#94a3b8'); ax.set_ylabel('Aktual',color='#94a3b8')
            ax.tick_params(colors='white'); ax.set_facecolor('#0f172a')
        plt.tight_layout(pad=2.5); st.pyplot(fig4); plt.close()

        st.markdown("### Detail per Model")
        sel = st.selectbox("Pilih model:", MODELS)
        cm  = np.array(results[sel]['cm'])
        TN,FP,FN,TP = cm[0,0],cm[0,1],cm[1,0],cm[1,1]
        s1,s2,s3,s4 = st.columns(4)
        for col,val,lbl,color in [
            (s1,TP,"True Positive\n(Judol ✓)","#22c55e"),
            (s2,TN,"True Negative\n(Aman ✓)","#60a5fa"),
            (s3,FP,"False Positive\n(Salah alarm)","#fbbf24"),
            (s4,FN,"False Negative\n(Judol lolos 🚨)","#ef4444"),
        ]:
            col.markdown(f"""
            <div style="background:#0f172a;border:1px solid {color};
                        border-radius:10px;padding:14px;text-align:center;">
                <div style="font-size:2rem;font-weight:700;color:{color};">{val}</div>
                <small style="color:#94a3b8;white-space:pre-line;">{lbl}</small>
            </div>""", unsafe_allow_html=True)

    # ══ TAB 3: TOP KEYWORDS ════════════════════════════════
    with tab3:
        st.markdown("### Top 25 Kata Kunci — berdasarkan Koefisien TF-IDF (Logistic Regression)")
        tfidf = bundle['tfidf']
        lr    = bundle['models']['Logistic Regression']
        feats = tfidf.get_feature_names_out()
        coef  = lr.coef_[0]
        top_j = [(feats[i],coef[i])     for i in np.argsort(coef)[-25:][::-1]]
        top_n = [(feats[i],abs(coef[i])) for i in np.argsort(coef)[:25]]

        cj, cn = st.columns(2)
        for col, title, data, bar_color in [
            (cj,"🚫 Kata Paling Judol",    top_j, "#ef4444"),
            (cn,"✅ Kata Paling Non-Judol", top_n, "#22c55e"),
        ]:
            with col:
                st.markdown(f"#### {title}")
                fig5,ax5 = plt.subplots(figsize=(5,7))
                fig5.patch.set_facecolor('#0f172a'); dark_ax(ax5)
                ws=[w for w,_ in data]; vs=[v for _,v in data]
                ax5.barh(range(len(ws)), vs, color=bar_color, edgecolor='#0f172a', height=.7)
                ax5.set_yticks(range(len(ws))); ax5.set_yticklabels(ws, color='white', size=9)
                ax5.set_xlabel('Koefisien', color='#94a3b8')
                plt.tight_layout(); st.pyplot(fig5); plt.close()

        st.markdown("### Tabel Kata Kunci Lengkap")
        st.dataframe(pd.DataFrame({
            "Rank":range(1,26),
            "Kata (Judol)":[w for w,_ in top_j],
            "Skor Judol":[round(v,4) for _,v in top_j],
            "Kata (Non-Judol)":[w for w,_ in top_n],
            "Skor Non-Judol":[round(v,4) for _,v in top_n],
        }), use_container_width=True, hide_index=True)
