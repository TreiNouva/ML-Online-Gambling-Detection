import pickle, re, numpy as np
import streamlit as st

# ── Regex: PROMO pattern (commercial / recruitment intent) ────
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

# Stopword Bahasa Indonesia netral — dihapus supaya tidak dianggap fitur
# pembeda topik oleh TF-IDF (mis. "lagi", "yang", "kalo" yang ternyata
# punya koefisien tinggi di SVM hanya karena gaya tulisan, bukan makna)
STOPWORDS_ID = set("""
yang dan di ke dari pada untuk dengan atau ini itu juga lagi saja akan
sudah belum masih bisa ada tidak tak ga gak engga nggak ngga jadi karena
karna kalo kalau klo kl jika kl bila maka tapi tp namun walau walaupun
biar agar supaya hanya cuma cuman pun lah kah deh dong sih nih situ sini
sana mereka kita kami saya aku gw gue lo lu kamu anda dia ia nya mu ku
pak bu bang kak adek mas mbak om tante banget bgt sekali amat terlalu
paling lebih kurang antara sambil sampai hingga sebelum sesudah setelah
ketika saat waktu tahun bulan hari jam menit detik nya an kan in
""".split())

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
    words = [w for w in words if w not in STOPWORDS_ID]
    return re.sub(r'\s+',' ',' '.join(words)).strip()

BEST_MODEL = "SVM"

# ── Lapisan Intent (v8) ──────────────────────────────────────
# Menutupi keterbatasan semantik: komentar yang menyebut kata terkait
# judi tapi sebenarnya berisi kritik/penegakan hukum, curhat pribadi,
# atau diskusi berita/isu sosial — bukan promosi maupun topik biasa.
KRITIK_HUKUM_RE = re.compile(
    r'\btangkap\b.{0,40}\b(bandar|artis|penjudi|streamer|pelaku|admin)\b|'
    r'\b(bandar|artis|penjudi|streamer|pelaku)\b.{0,40}\btangkap\b|'
    r'\bdiberantas\b.{0,40}\b(situs|bandar|judi)\b|'
    r'\b(situs|bandar|judi)\b.{0,40}\bdiberantas\b|'
    r'\bdi\s*brantas\b.{0,40}\b(judi|situs|bandar)\b|'
    r'\b(judi|situs|bandar)\b.{0,40}\bdi\s*brantas\b|'
    r'\busut\w*\b.{0,80}\b(judi|kasus|bandar)\b|'
    r'\b(judi|kasus|bandar)\b.{0,80}\busut\w*\b|'
    r'\bditangkap\b.{0,60}\bpromosi\b.{0,40}\b(judi|judol|slot)\b|'
    r'\bpromosi\b.{0,40}\b(judi|judol|slot)\b.{0,60}\bditangkap\b',
    re.IGNORECASE | re.DOTALL
)
CURHAT_RE = re.compile(
    r'\bketemu\s+penjudi\b|\bkenal\s+penjudi\b|\bsuami\s+(tukang\s+)?judi\b|'
    r'\bistri\s+(tukang\s+)?judi\b|\bnasehat\w*\b.{0,80}\bpenjudi\b|'
    r'\bpenjudi\b.{0,80}\bnasehat\w*\b|\bsusah\s+dinasehat\w*\b|'
    r'\bbreak\s+dulu\b.{0,30}\bpenjudi\b|\bmusuhi\b.{0,40}\bpenjudi\b|'
    r'\bbebal\b.{0,40}\bpenjudi\b|\bpenjudi\b.{0,40}\bbebal\b',
    re.IGNORECASE | re.DOTALL
)
DISKUSI_BERITA_RE = re.compile(
    r'\bbansos\b.{0,100}\b(judi|judol|penjudi)\b|\b(judi|judol|penjudi)\b.{0,100}\bbansos\b|'
    r'\bkasus\b.{0,80}\b(judi|judol)\b|\b(judi|judol)\b.{0,80}\bkasus\b|'
    r'\b\d+\s*[tT]\b.{0,40}\bjudi\b|\bjudi\b.{0,40}\b\d+\s*[tT]\b|'
    r'\bpajak\b.{0,30}\bjudi\b|\bjudi\b.{0,30}\bpajak\b|'
    r'\bkorupsi\b.{0,40}\bjudi\b|\bjudi\b.{0,40}\bkorupsi\b',
    re.IGNORECASE | re.DOTALL
)

INTENT_LABELS = {
    "kritik_hukum": "Kritik/Penegakan Hukum",
    "curhat": "Curhat/Pengalaman Pribadi",
    "diskusi_berita": "Diskusi Berita/Isu Sosial",
}
INTENT_CONF = 0.90          # confidence tetap untuk hasil match pola intent
SVM_DOUBT_THRESHOLD = 0.70  # di bawah ini, SVM dianggap "ragu"

def predict_one(text, bundle):
    """
    Hybrid prediction (v8 — dengan lapisan Intent):
    1. Cek ANTI_RE dulu  → jika cocok (dan tidak ada promo), label 0 (Aman). Ini
       sinyal paling dipercaya, tidak akan ditimpa oleh lapisan Intent.
    2. Cek KRITIK_HUKUM_RE / CURHAT_RE / DISKUSI_BERITA_RE → jika salah satu cocok,
       maka komentar dianggap BUKAN promosi judol, walau ada kata terkait judi.
       Lapisan ini bisa menimpa hasil PROMO_RE (karena PROMO_RE cukup general dan
       rawan salah tangkap konteks panjang) maupun hasil SVM yang ragu.
    3. Cek PROMO_RE → jika cocok (dan tidak ada anti/intent lain), label 1 (Judol).
    4. Tidak ada regex cocok sama sekali, dan SVM juga ragu (confidence rendah)
       → beri label tambahan "Uncategorized" sebagai sinyal bahwa keputusan
       model kurang yakin dan tidak match pola apapun yang dikenal sistem.
    5. Sisanya → pakai prediksi SVM murni.
    """
    original_text = str(text)
    clean = preprocess(original_text)

    is_anti  = bool(ANTI_RE.search(original_text))
    is_promo = bool(PROMO_RE.search(original_text))

    is_kh = bool(KRITIK_HUKUM_RE.search(original_text))
    is_ch = bool(CURHAT_RE.search(original_text))
    is_db = bool(DISKUSI_BERITA_RE.search(original_text))
    intent_key = "kritik_hukum" if is_kh else ("curhat" if is_ch else ("diskusi_berita" if is_db else None))
    intent = INTENT_LABELS.get(intent_key) if intent_key else None

    # ANTI_RE paling dipercaya — selesai di sini, tidak perlu cek intent lagi
    if is_anti and not is_promo:
        return {"label": 0, "conf": 0.97, "clean": clean,
                "override": "anti-judol sentiment", "intent": None}

    # Hitung dulu hasil dasar (PROMO_RE atau SVM) sebagai pembanding
    if is_promo and not is_anti:
        base_label, base_conf, base_override = 1, 0.97, "promo pattern"
    else:
        model = bundle['models'][BEST_MODEL]
        vec   = bundle['tfidf'].transform([clean])
        base_label = int(model.predict(vec)[0])
        if hasattr(model, 'predict_proba'):
            base_conf = float(model.predict_proba(vec)[0][base_label])
        elif hasattr(model, 'decision_function'):
            raw = model.decision_function(vec)[0]
            base_conf = float(1 / (1 + np.exp(-abs(raw))))
        else:
            base_conf = 1.0
        base_override = None
        if is_anti and is_promo:
            base_label, base_conf = 0, max(base_conf, 0.75)
            base_override = None

    # Lapisan Intent: override jika cukup kuat dibanding hasil dasar
    if intent is not None:
        came_from_promo_regex = (base_override == "promo pattern")
        svm_is_doubtful = base_conf < SVM_DOUBT_THRESHOLD
        intent_stronger_than_svm = (base_override is None) and (INTENT_CONF >= base_conf)

        if came_from_promo_regex or svm_is_doubtful or intent_stronger_than_svm:
            return {"label": 0, "conf": INTENT_CONF, "clean": clean,
                    "override": f"intent: {intent}", "intent": intent}
        else:
            return {"label": base_label, "conf": base_conf, "clean": clean,
                    "override": base_override, "intent": intent}

    if base_override is not None:
        return {"label": base_label, "conf": base_conf, "clean": clean,
                "override": base_override, "intent": None}

    # Tidak ada regex apapun yang cocok, dan SVM juga ragu -> Uncategorized
    if base_conf < SVM_DOUBT_THRESHOLD:
        return {"label": base_label, "conf": base_conf, "clean": clean,
                "override": None, "intent": "Uncategorized"}

    return {"label": base_label, "conf": base_conf, "clean": clean,
            "override": None, "intent": None}
