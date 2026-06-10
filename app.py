import streamlit as st

st.set_page_config(
    page_title="JudolGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sembunyikan semua page links bawaan Streamlit di sidebar
st.markdown("""
<style>
/* Sembunyikan multipage nav otomatis Streamlit */
[data-testid="stSidebarNav"] { display: none !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f172a 0%,#1e293b 100%);
}
[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
div[data-testid="stSidebar"] .stRadio label {
    background: #1e3a5f33;
    border: 1px solid #2d5a8e55;
    border-radius: 8px;
    padding: 10px 14px !important;
    font-size: 15px !important;
    display: block;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#2563eb,#1d4ed8);
    color: white; border: none; border-radius: 10px;
    padding: 12px 28px; font-size: 16px; font-weight: 600;
    width: 100%;
}
[data-testid="stFileUploader"] {
    border: 2px dashed #2d5a8e !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🛡️ JudolGuard")
    st.markdown("*Deteksi Komentar Judi Online*")
    st.markdown("---")
    page = st.radio(
        "Menu",
        ["🏠  Beranda",
         "🔍  Deteksi Komentar",
         "📂  Moderasi File CSV/Excel",
         "📊  Dashboard Performa Model"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Kelompok 13 · NLP Project")

if   "Beranda"   in page:
    from pages.home             import show; show()
elif "Deteksi"   in page:
    from pages.single_detection import show; show()
elif "Moderasi"  in page:
    from pages.bulk_moderation  import show; show()
elif "Dashboard" in page:
    from pages.analytics        import show; show()
