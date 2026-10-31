import joblib
import streamlit as st

st.set_page_config(page_title="Toxic Comment Detector", page_icon="🛡️")
st.title("🛡️ Toxic Comment Detector")
st.write("Type a comment and the model will estimate how likely it is to be toxic.")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


try:
    model = load_model()
except FileNotFoundError:
    st.error("model.joblib not found. Run `python train.py` first.")
    st.stop()

comment = st.text_area("Your comment", height=120, placeholder="Type something here...")

if st.button("Check comment"):
    if not comment.strip():
        st.warning("Please type a comment first.")
    else:
        prob = float(model.predict_proba([comment])[0][1])
        st.progress(prob)
        st.metric("Toxicity probability", f"{prob:.0%}")
        if prob >= 0.5:
            st.error("⚠️ This comment looks toxic.")
        else:
            st.success("✅ This comment looks fine.")

st.caption("Built with scikit-learn (TF-IDF + Logistic Regression). A learning project, not a real moderation tool.")
