import streamlit as st
import os
import joblib
import pandas as pd
import re
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")
st.title("Spam Detector")

MODEL_BNB = 'model_bnb_spam.pkl'
MODEL_SGD = 'model_sgd_spam.pkl'
MODEL_ENSEMBLE = 'model_ensemble_spam.pkl'
LE_PATH = 'label_encoder_spam.pkl'
DATA_PATH = '/mnt/data/spam.csv'

@st.cache_resource
def load_artifacts():
    artifacts = {}
    if os.path.exists(MODEL_BNB):
        artifacts['bnb'] = joblib.load(MODEL_BNB)
    if os.path.exists(MODEL_SGD):
        artifacts['sgd'] = joblib.load(MODEL_SGD)
    if os.path.exists(MODEL_ENSEMBLE):
        artifacts['ensemble'] = joblib.load(MODEL_ENSEMBLE)
    if os.path.exists(LE_PATH):
        artifacts['le'] = joblib.load(LE_PATH)
        return artifacts

art = load_artifacts()

def clean_text(x):
    if not isinstance(x, str):
        return ""
    x = re.sub(r'http\S+|www\S+|https\S+', '', x)
    x = re.sub(r'[^A-Za-z0-9\s]', ' ', x)
    x = x.lower().strip()
    x = re.sub(r'\s+', ' ', x)
    return x

st.sidebar.header('Mode')
mode = st.sidebar.radio('Pilih mode', ['Ensemble (Voting)', 'Head-to-Head'])
if mode == 'Head-to-Head':
    sel = st.sidebar.selectbox('Pilih model', ['BernoulliNB', 'SGDClassifier'])

le = art.get('le')

st.subheader('Masukkan teks untuk prediksi')
text = st.text_area('', height=140)
if st.button('Prediksi'):
    if text.strip() == '':
        st.warning('Masukkan teks terlebih dahulu')
    else:
        txt = clean_text(text)
        pred = None
        probs = None
        if mode == 'Ensemble (Voting)':
            m = art.get('ensemble')
            if m is None:
                st.error('Model ensemble tidak ditemukan. Pastikan file model_ensemble_spam.pkl tersedia.')
            else:
                pred = m.predict([txt])[0]
                try:
                    probs = m.predict_proba([txt])[0]
                except Exception:
                    probs = None
        else:
            if sel == 'BernoulliNB':
                m = art.get('bnb')
            else:
                m = art.get('sgd')
            if m is None:
                st.error('Model yang dipilih tidak ditemukan. Jalankan training dan simpan model terlebih dahulu.')
            else:
                pred = m.predict([txt])[0]
                try:
                    probs = m.predict_proba([txt])[0]
                except Exception:
                    probs = None

        if pred is not None:
            if le is not None:
                label = le.inverse_transform([pred])[0]
            else:
                label = str(pred)
            st.markdown('### Hasil')
            st.write('Label:', label)
            if probs is not None and le is not None:
                classes = le.classes_
                pairs = sorted(list(zip(classes, probs)), key=lambda x: x[1], reverse=True)
                st.write('Probabilitas:')
                for c, p in pairs:
                    st.write(f'- {c}: {p:.3f}')

st.write('---')
st.subheader('Evaluasi singkat (jika dataset asli ada)')
if os.path.exists(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH, encoding='latin-1')
        df = df.dropna(axis=1)
        df = df.rename(columns={df.columns[0]: 'label', df.columns[1]: 'text'})
        df = df.drop_duplicates().reset_index(drop=True)
        df['clean'] = df['text'].apply(clean_text)
        if le is not None:
            df['label_enc'] = le.transform(df['label'])
        else:
            df['label_enc'] = pd.factorize(df['label'])[0]
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(df['clean'].values, df['label_enc'].values, test_size=0.2, random_state=42, stratify=df['label_enc'].values)
        st.write('Dataset ditemukan di', DATA_PATH)
        if 'bnb' in art:
            preds = art['bnb'].predict(X_test)
            st.write('Akurasi BernoulliNB:', accuracy_score(y_test, preds))
        if 'sgd' in art:
            preds = art['sgd'].predict(X_test)
            st.write('Akurasi SGD:', accuracy_score(y_test, preds))
        if 'ensemble' in art:
            preds = art['ensemble'].predict(X_test)
            st.write('Akurasi Ensemble:', accuracy_score(y_test, preds))
    except Exception as e:
        st.write('Gagal load dataset untuk evaluasi:', e)
else:
    st.write(f'Dataset tidak ditemukan di {DATA_PATH}. Untuk menampilkan akurasi otomatis, unggah dataset ke path tersebut atau edit DATA_PATH.')

st.write('---')
st.write('Jalankan: streamlit run streamlit_main.py')
