"""
train_model.py  — Jalankan untuk melatih ulang model dari awal.
Usage: python train_model.py

File yang dibutuhkan di folder yang sama:
  - judol_train_dataset.csv
  - judol_test_dataset.csv
  - judi_online_comments.xlsx   (opsional, akan disertakan jika ada)
"""
import pandas as pd, numpy as np, re, pickle, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

JUDOL_RE = re.compile(
    r"alex\s*1?s?\s*17|4lexis17|alexis\s*17|a\s*l\s*e\s*x\s*i\s*s\s*1\s*7|"
    r"mandalika\s*77|weton\s*88|pulau\s*777|pulau\s*win|pulauwin|"
    r"sl[o0]t|s\s*l\s*o\s*t|g[a4]c[o0]r|g\s*a\s*c\s*o\s*r|"
    r"max\s*w?in|m4xw1n|maxw1n|maks?\s*win|"
    r"jepe?y?|\bjp\b|ngecuan|cuan\b|depo(sit)?|\bwd\b|withdraw|"
    r"scatter|zeus|olympus|pgsoft|pragmatic|starlight|rungk(ad)?|"
    r"\bbonus\b|\bmenang\b|ketagihan|\bsitus\b|sketer|gampang|"
    r"\blanding\b|\bcair\b|\bgabung\b|\bjoin\b|\bdaftar\b|"
    r"terpercaya|bocoran|bocor\b|modal\s*(kecil|receh)|rezeki\s*mengalir|"
    r"petir\s*x|zeus\s*x|scatter\s*hitam|\b(77|88|777|17)\b",
    re.IGNORECASE
)

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

def normalize_spaced(text):
    def collapse(m): return re.sub(r'[\s._-]','',m.group(0))
    return re.sub(r'(?<!\w)([a-zA-Z](?:[\s._-][a-zA-Z]){2,})(?!\w)', collapse, text)

def preprocess(text):
    if pd.isna(text) or str(text).strip()=="": return ""
    text = str(text).encode('ascii','ignore').decode('ascii').lower()
    text = normalize_spaced(text)
    for s,d in [('0','o'),('1','i'),('3','e'),('4','a'),('5','s')]: text=text.replace(s,d)
    text = re.sub(r'http\S+|www\S+','',text); text = re.sub(r'@\w+|#\w+','',text)
    text = re.sub(r'\d+','',text); text = re.sub(r'[^a-z\s]',' ',text)
    return re.sub(r'\s+',' ',' '.join([NORM.get(w,w) for w in text.split()])).strip()

print("="*55); print("  JudolGuard — Training Pipeline"); print("="*55)

train = pd.read_csv('judol_train_dataset.csv')
test  = pd.read_csv('judol_test_dataset.csv')
print(f"Train: {len(train):,} | Test: {len(test):,}")

# Tambahkan xlsx jika ada
xlsx_file = 'judi_online_comments.xlsx'
if os.path.exists(xlsx_file):
    df_xlsx = pd.read_excel(xlsx_file)
    col = 'komentar' if 'komentar' in df_xlsx.columns else df_xlsx.columns[0]
    rows = []
    for txt in df_xlsx[col].astype(str):
        lbl = 1 if JUDOL_RE.search(txt) else 0
        if lbl==0 and re.search(r'ð[^\s]{4,}.*\d{2}|\d{2}.*ð[^\s]{4,}', txt): lbl=1
        if lbl==0 and re.search(r'[Aa]\s+[Ll]\s+[Ee]\s+[Xx]\s+[Ii]\s+[Ss]\s+\d', txt): lbl=1
        rows.append({'text':txt,'label':lbl})
    df_xlsx_labeled = pd.DataFrame(rows)
    train = pd.concat([train, df_xlsx_labeled], ignore_index=True)
    print(f"+ XLSX: {len(df_xlsx_labeled)} baris ditambahkan")

train['clean'] = train['text'].apply(preprocess)
test['clean']  = test['text'].apply(preprocess)
train = train[train['clean'].str.len()>0].reset_index(drop=True)
test  = test[test['clean'].str.len()>0].reset_index(drop=True)

tfidf = TfidfVectorizer(max_features=12000, ngram_range=(1,2), min_df=2, sublinear_tf=True)
X_train = tfidf.fit_transform(train['clean']); X_test = tfidf.transform(test['clean'])
y_train, y_test = train['label'], test['label']

models = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "SVM":                  LinearSVC(C=1.0, max_iter=2000, random_state=42),
    "Naive Bayes":          MultinomialNB(alpha=0.1),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}
eval_results, trained = {}, {}
for name, model in models.items():
    model.fit(X_train, y_train); y_pred = model.predict(X_test)
    eval_results[name] = {
        'accuracy':round(accuracy_score(y_test,y_pred),4),
        'precision':round(precision_score(y_test,y_pred,average='weighted'),4),
        'recall':round(recall_score(y_test,y_pred,average='weighted'),4),
        'f1':round(f1_score(y_test,y_pred,average='weighted'),4),
        'cm':confusion_matrix(y_test,y_pred).tolist(),
    }
    trained[name] = model
    r = eval_results[name]
    print(f"  {name:25} Acc={r['accuracy']*100:.2f}%  F1={r['f1']*100:.2f}%")

best = max(eval_results, key=lambda n: eval_results[n]['f1'])
print(f"\n⭐ Model terbaik: {best} (F1={eval_results[best]['f1']*100:.2f}%)")
os.makedirs('model',exist_ok=True)
pickle.dump({'tfidf':tfidf,'models':trained,'eval_results':eval_results,'best_model':best},
            open('model/judol_model.pkl','wb'))
print("✅ Model disimpan ke model/judol_model.pkl")
print("   Jalankan: streamlit run app.py")
print("="*55)
