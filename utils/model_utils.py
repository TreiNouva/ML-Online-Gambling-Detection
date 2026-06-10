import pickle, re, numpy as np
import streamlit as st

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
    clean = preprocess(text)
    vec   = bundle['tfidf'].transform([clean])
    model = bundle['models'][BEST_MODEL]
    label = int(model.predict(vec)[0])
    if hasattr(model, 'predict_proba'):
        conf = float(model.predict_proba(vec)[0][label])
    elif hasattr(model, 'decision_function'):
        raw  = model.decision_function(vec)[0]
        conf = float(1 / (1 + np.exp(-abs(raw))))
    else:
        conf = 1.0
    return {"label": label, "conf": conf, "clean": clean}
