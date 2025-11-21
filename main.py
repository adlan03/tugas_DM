import streamlit as st
import joblib
import re
import numpy as np

st.set_page_config(page_title="SMS Spam Detector (Ensemble) — Improved", page_icon="📩", layout="centered")
st.title("SMS Spam Detector — Dynamic Duo (NB + SVM)")
st.markdown("**Deskripsi:** Aplikasi memprediksi apakah pesan SMS adalah *spam* atau *ham*. Model: MultinomialNB + Calibrated SVM ensembled (soft voting).")

# ----- Load artifacts (pastikan file-file ini berada pada folder yang sama dengan app.py) -----
tfidf = joblib.load("vectorizer_tfidf.pkl")
nb = joblib.load("model_multinomial_nb.pkl")
svm = joblib.load("model_calibrated_svm.pkl")
ensemble = joblib.load("model_ensemble_voting.pkl")

# ----- Sidebar: performance (opsional: edit angka jika mau) -----
with st.sidebar:
    st.header("Model Performance (test split)")
    # Jika Anda punya metadata.json, Anda bisa memuatnya dan menampilkan nilai nyata
    try:
        meta = joblib.load("metadata.json") if False else None
    except Exception:
        meta = None
    st.write("Akurasi MultinomialNB: (lihat training logs)")
    st.write("Akurasi Calibrated SVM: (lihat training logs)")
    st.write("Akurasi Ensemble (soft): (lihat training logs)")
    st.caption("Catatan: angka akurasi asli disimpan di metadata.json saat training.")

st.markdown('---')
st.subheader("Masukkan SMS / Pesan untuk dianalisis")
preset = st.selectbox("Pilih contoh cepat:", ["-- Ketik manual --", "Free entry: win money", "Reminder: meeting at 10", "Congrats you won", "Call me later"])
text = st.text_area("Masukkan teks:", value="" if preset=="-- Ketik manual --" else preset, height=120)
compare_models = st.checkbox("Bandingkan model (tampilkan prediksi tiap model)", value=True)
detail_preproc = st.checkbox("Tampilkan detail preprocessing (clean text & token check)", value=False)
show_top_features = st.checkbox("Tampilkan top fitur SVM (attempt)", value=False)

MIN_LEN = 3

# ----- Helpers -----
def preprocess(text):
    s = str(text).lower()
    s = re.sub(r'http\S+', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def predict_with_model(vec, model):
    # returns (pred_label, confidence (0..1 or None), prob_dict)
    try:
        probs = model.predict_proba(vec)[0]
        classes = model.classes_
        best_idx = int(np.argmax(probs))
        return classes[best_idx], float(probs[best_idx]), dict(zip(classes, probs))
    except Exception:
        # fallback: only predict
        pred = model.predict(vec)[0]
        # try to approximate confidence using decision_function (sigmoid)
        try:
            df = model.decision_function(vec)
            # scalar or array
            if hasattr(df, 'shape'):
                val = float(df[0]) if df.ndim == 1 else float(df[0][0])
                prob = 1.0/(1.0 + np.exp(-val))
                return pred, prob, {pred: prob}
        except Exception:
            return pred, None, {pred: None}

def top_features_for_svm(vectorizer, calibrated_svm, class_label, top_n=10):
    try:
        # calibrated_svm wraps a base estimator (LinearSVC)
        base = calibrated_svm.base_estimator_
        coef = base.coef_  # shape: (n_classes, n_features) for multiclass
        classes = calibrated_svm.classes_
        idx = list(classes).index(class_label)
        feat_names = vectorizer.get_feature_names_out()
        topn = np.argsort(coef[idx])[-top_n:][::-1]
        return [(feat_names[i], float(coef[idx][i])) for i in topn]
    except Exception:
        return None

# ----- Main interaction -----
if st.button("Analisis"):
    user_text = text.strip()
    if len(user_text) < MIN_LEN:
        st.warning(f"Teks terlalu pendek — harap masukkan sedikitnya {MIN_LEN} karakter atau tambahkan konteks.")
    else:
        cleaned = preprocess(user_text)
        vec = tfidf.transform([cleaned])

        tokens = cleaned.split()
        known = sum(1 for t in tokens if t in tfidf.vocabulary_)

        if detail_preproc:
            st.write("Hasil preprocessing:", cleaned)
            st.write("Token asli:", tokens)
            st.write("Token yang dikenali oleh model:", known, "dari", len(tokens))

        if known == 0:
            st.warning("Tidak ada token yang dikenali oleh model — hasil mungkin tidak akurat. Coba kata yang lebih umum atau konteks lebih panjang.")

        # perform predictions
        if compare_models:
            p_nb, c_nb, probs_nb = predict_with_model(vec, nb)
            p_svm, c_svm, probs_svm = predict_with_model(vec, svm)
            p_ens, c_ens, probs_ens = predict_with_model(vec, ensemble)

            st.markdown("### Hasil Perbandingan Model")
            if c_nb is not None:
                st.write(f"• MultinomialNB → {p_nb.upper()} (confidence: {c_nb*100:.2f}%)")
            else:
                st.write(f"• MultinomialNB → {p_nb.upper()}")

            if c_svm is not None:
                st.write(f"• CalibratedSVM → {p_svm.upper()} (confidence: {c_svm*100:.2f}%)")
            else:
                st.write(f"• CalibratedSVM → {p_svm.upper()}")

            if c_ens is not None:
                st.write(f"• Ensemble → {p_ens.upper()} (confidence: {c_ens*100:.2f}%)")
            else:
                st.write(f"• Ensemble → {p_ens.upper()}")

        else:
            p_ens, c_ens, probs_ens = predict_with_model(vec, ensemble)
            st.markdown("### Prediksi (Ensemble)")
            if c_ens is not None:
                st.success(f"Label: {p_ens.upper()} — confidence: {c_ens*100:.2f}%")
            else:
                st.success(f"Label: {p_ens.upper()}")

        # show probs if available
        if 'probs_ens' in locals() and probs_ens:
            st.write("Probabilitas (Ensemble):", probs_ens)

        # show top features for spam if requested
        if show_top_features:
            st.markdown('---')
            st.write("Top fitur yang menyumbang untuk kelas 'spam' (SVM) — jika ada:")
            top_spam = top_features_for_svm(tfidf, svm, 'spam', top_n=12)
            if top_spam:
                st.table(top_spam)
            else:
                st.write("Tidak dapat menampilkan fitur — struktur model mungkin berbeda atau model bukan linear L2.")

st.markdown('---')
st.caption("Catatan: Model dilatih pada dataset SMS Spam. Untuk teks pendek atau kata yang jarang muncul, hasil mungkin kurang akurat.")
