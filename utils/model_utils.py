import pickle, re, numpy as np
import streamlit as st

# ── Regex: PROMO pattern (commercial / recruitment intent) ────
# v2: tambah pola "main di situs aku/ku" — promosi terselubung yang
# menyamar sebagai keluhan ("nyesel main di tempat lain...")
PROMO_RE = re.compile(
    r'\balex\w*17\b|4lexis\w*17|\bmandalika\s*77\b|\bweton\s*88\b|\bpulau\s*77\w\b|\bpulauwin\b|'
    r'\bsl[o0]t\b.{0,50}\b(gacor|maxwin|terpercaya|daftar|deposit|bonus|scatter|x\d+)\b|'
    r'\b(gacor|maxwin|terpercaya|daftar|deposit|bonus|scatter)\b.{0,50}\bsl[o0]t\b|'
    r'\b(daftar|gabung|join|dm|hubungi|chat|klik|link)\b.{0,40}\b(slot|judi|judol|gacor|maxwin|admin|bio|wa)\b|'
    r'\b(slot|judi|judol|gacor)\b.{0,40}\b(daftar|gabung|join|dm|hubungi|klik|link|bio|wa|admin)\b|'
    r'\bm[4a]xw[1i]n\b|\bsl[o0]t\s+g[a4]c[o0]r\b|\bg[a4]c[o0]r\s+sl[o0]t\b|'
    r'\b(deposit|depo|wd|withdraw)\b.{0,40}\b(slot|judi|gacor|maxwin|terpercaya)\b|'
    r'\b(modal\s*kecil|rezeki\s*mengalir|anti\s*rungkad|gampang\s*jepe|gampang\s*menang)\b|'
    r'\b[sS]\s+[lL]\s+[oO]\s+[tT]\b|\b[jJ]\s+[uU]\s+[dD]\s+[iI]\b|\b[gG]\s+[aA]\s+[cC]\s+[oO]\s+[rR]\b|'
    r'\b(zeus|olympus|starlight\s*princess|scatter\s*hitam|pgsoft|pragmatic)\b.{0,30}\b(x\d+|gacor|maxwin|bocoran)\b|'
    r'\bbocoran\s+(slot|gacor|hari\s*ini|malam\s*ini)\b|'
    r'\blink\s+(gacor|slot|judi|judol|terpercaya)\b|'
    r'\b(buruan|langsung|sekarang|hari\s*ini)\b.{0,30}\b(daftar|gabung|slot|gacor|maxwin)\b|'
    r'\bmain\s+di\s+(situs|tempat)\s+(aku|ku|saya|sini|gue|punyaku)\b|'
    r'\b(situs|tempat)\s+(aku|ku|saya|punyaku)\b.{0,30}\b(main|gacor|menang|terpercaya)\b',
    re.IGNORECASE
)

# Regex: ANTI-judol pattern (negative/critical/prohibition sentiment)
# v2: tambah kata "dilarang/terlarang/larang", dan perketat "nyesel/kapok"
# agar hanya cocok jika eksplisit terkait konteks judi (bukan kata berdiri sendiri)
ANTI_RE = re.compile(
    r'\b(haram|bahaya|dosa|rugi|bangkrut|hancur|rusak|tobat|insaf|kapok)\b|'
    r'\b(dilarang|terlarang|jangan|stop|hindari|tinggalkan|berhenti|keluar|lepas|larang)\b.{0,50}\b(judi|judol|slot|gambling|main)\b|'
    r'\b(judi|judol|slot|gambling)\b.{0,50}\b(dilarang|terlarang|jangan|stop|hindari|haram|dosa|bahaya|rugi|bangkrut|larang)\b|'
    r'\b(berantas|memberantas|hapus|tutup|blokir|perangi|lawan|basmi)\b.{0,50}\b(judi|judol|slot)\b|'
    r'\b(judi|judol|slot)\b.{0,50}\b(berantas|hapus|tutup|blokir|perangi|lawan|basmi)\b|'
    r'\bkorban\s+(judi|judol|slot)\b|\b(judi|judol|slot)\s+korban\b|'
    r'\b(merusak|menghancurkan|merugikan).{0,40}\b(keluarga|rumah\s*tangga|hidup|masa\s*depan)\b|'
    r'\binget\s+mati\b|\bingat\s+akhirat\b|\bsemoga\s+(sadar|tobat|insaf|berhenti)\b|'
    r'\b(awas|waspada)\b.{0,30}\b(judi|judol|slot)\b|'
    r'\b(nyesel|menyesal|kapok)\b.{0,40}\b(judi|judol|slot|gacor|maxwin)\b|'
    r'\b(judi|judol|slot|gacor|maxwin)\b.{0,40}\b(nyesel|menyesal|kapok)\b',
    re.IGNORECASE
)

@st.cache_resource
def load_model():
    with open("model/judol_model.pkl", "rb") as f:
        return pickle.load(f)

NORM = {
    'yg':'yang','dgn':'dengan','gk':'tidak','ga':'tidak','gak':'tidak',
    'nggak':'tidak','ngga':'tidak','krn':'karena','karna':'karena',
    'utk':'untuk','tdk':'tidak','tp':'tapi','lg':'lagi','udh':'sudah',
    'sdh':'sudah','dg':'dengan','sm':'sama','jd':'jadi','aja':'saja',
    'bgt':'banget','emg':'memang','emang':'memang','bs':'bisa','org':'orang',
    'dr':'dari','klo':'kalau','kl':'kalau','sy':'saya','gw':'saya',
    'gue':'saya','lo':'kamu','lu':'kamu','wkwkwk':'tertawa','wkwk':'tertawa',
    'sl0t':'slot','s1ot':'slot','jud1':'judi','b0coran':'bocoran','maxw1n':'maxwin',
    'g4cor':'gacor','g4c0r':'gacor','gac0r':'gacor','4lexis':'alexis',
    'alex1s':'alexis','m4xwin':'maxwin',
}

def _normalize_spaced(text):
    """Tangani obfuscation spasi: 's l o t' → 'slot'"""
    def collapse(m): return re.sub(r'[\s._-]','',m.group(0))
    return re.sub(r'(?<!\w)([a-zA-Z](?:[\s._-][a-zA-Z]){2,})(?!\w)', collapse, text)

def preprocess(text):
    if not text or str(text).strip()=="": return ""
    text = str(text).encode('ascii','ignore').decode('ascii').lower()
    text = _normalize_spaced(text)
    for s,d in [('0','o'),('1','i'),('3','e'),('4','a'),('5','s')]: text=text.replace(s,d)
    text = re.sub(r'http\S+|www\S+','',text)
    text = re.sub(r'@\w+|#\w+','',text)
    text = re.sub(r'\d+','',text)
    text = re.sub(r'[^a-z\s]',' ',text)
    words = [NORM.get(w,w) for w in text.split()]
    return re.sub(r'\s+',' ',' '.join(words)).strip()

BEST_MODEL = "SVM"

def predict_one(text, bundle):
    """
    Hybrid prediction:
    1. Cek ANTI_RE dulu  → jika cocok (dan tidak ada promo), label 0 (Aman)
    2. Cek PROMO_RE       → jika cocok (dan tidak ada anti), label 1 (Judol)
    3. Keduanya cocok (konflik) → SVM diberi prioritas, tapi bias ke 0 jika SVM ragu
    4. Tidak ada regex cocok → pakai prediksi SVM murni
    """
    original_text = str(text)
    clean = preprocess(original_text)

    is_anti  = bool(ANTI_RE.search(original_text))
    is_promo = bool(PROMO_RE.search(original_text))

    if is_anti and not is_promo:
        return {"label": 0, "conf": 0.97, "clean": clean,
                "override": "anti-judol sentiment"}
    if is_promo and not is_anti:
        return {"label": 1, "conf": 0.97, "clean": clean,
                "override": "promo pattern"}

    model = bundle['models'][BEST_MODEL]
    vec   = bundle['tfidf'].transform([clean])
    label = int(model.predict(vec)[0])
    if hasattr(model, 'predict_proba'):
        conf = float(model.predict_proba(vec)[0][label])
    elif hasattr(model, 'decision_function'):
        raw  = model.decision_function(vec)[0]
        conf = float(1 / (1 + np.exp(-abs(raw))))
    else:
        conf = 1.0

    if is_anti and is_promo:
        label = 0
        conf  = max(conf, 0.75)

    return {"label": label, "conf": conf, "clean": clean, "override": None}
