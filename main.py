import streamlit as st
import joblib
import re
import numpy as np

st.set_page_config(page_title="Spam Detector (BNB + SGD + Ensemble)", page_icon="📩", layout="centered")
st.title("📩 SMS Spam Detector — Dynamic Duo Model")
st.write("Aplikasi ini menggunakan 2 model + ensemble untuk mendeteksi pesan **spam** atau **ham**.")

# ===========================
#  LOAD MODEL YANG ADA
# ===========================

MODEL_BNB_PATH = "model_bnb_spam.pkl"
MODEL_SGD_PATH = "model_sgd_spam.pkl"
MODEL_ENSEMBLE_PATH = "model_ensemble_spam.pkl"
LABEL_ENCODER_PATH = "label_encoder_spam.pkl"

try:
    model_bnb = joblib.load(MODEL_BNB_PATH)
    model_sgd = joblib.load(MODEL_SGD_PATH)
    model_ensemble = joblib.load(MODEL_ENSEMBLE_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
except Exception as e:
    st.error("❌ Error loading model: " + str(e))
    st.stop()

# ===========================
#  PREPROCESSING
# ===========================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+", " ", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


# ===========================
#  PREDICT FUNCTION
# ===========================

def predict_model(model, text):
    pred_encoded = model.predict([text])[0]
    try:
        proba = model.predict_proba([text])[0]
        confidence = np.max(proba)
    except:
        confidence = None

    label = label_encoder.inverse_transform([pred_encoded])[0]
    return label, confidence


# ===========================
#  UI FORM
# ===========================

st.subheader("Masukkan SMS untuk dianalisis")
text = st.text_area("Teks pesan:", height=120)

model_choice = st.selectbox(
    "Pilih model:",
    ["Ensemble", "BernoulliNB", "SGDClassifier"]
)

show_preproc = st.checkbox("Tampilkan detail preprocessing")


# ===========================
#  ANALISIS
# ===========================

if st.button("🔍 Analisis"):
    if text.strip() == "":
        st.warning("Silakan isi teks terlebih dahulu.")
        st.stop()

    cleaned = clean_text(text)

    if show_preproc:
        st.write("### 🔧 Hasil Preprocessing:")
        st.code(cleaned)

    # pilih model
    if model_choice == "BernoulliNB":
        label, conf = predict_model(model_bnb, cleaned)
    elif model_choice == "SGDClassifier":
        label, conf = predict_model(model_sgd, cleaned)
    else:
        label, conf = predict_model(model_ensemble, cleaned)

    st.subheader("🎯 Hasil Prediksi:")

    if label.lower() == "spam":
        st.error(f"🚨 SPAM! (Confidence: {conf:.2f})" if conf else "🚨 SPAM!")
    else:
        st.success(f"✔ HAM (bukan spam) — (Confidence: {conf:.2f})" if conf else "✔ HAM (bukan spam)")

