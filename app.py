import streamlit as st
import pandas as pd
import joblib
import os

# Page config
st.set_page_config(page_title="Academic Warning Predictor", page_icon="🎓", layout="wide")

st.title("🎓 Academic Warning Prediction System")

# Check if model file exists
if "academic_warning_model.pkl" not in os.listdir("."):
    st.error("❌ **Missing model file!**")
    st.info("Upload `academic_warning_model.pkl` to fix this.")
    st.stop()

@st.cache_resource
def load_model():
    """Load model safely"""
    try:
        model = joblib.load("academic_warning_model.pkl")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)[:100]}")
        st.info("Try retraining with: `joblib.dump(model, 'academic_warning_model.pkl', protocol=4)`")
        st.stop()

# Load model
model = load_model()

# Inputs - Match your original exactly
st.header("📝 Enter Student Data")

col1, col2 = st.columns(2)

with col1:
    gpa = st.number_input("**GPA**", 0.0, 4.0, 2.5, 0.1)
    credits = st.number_input("**Credits Registered**", 0, 30, 15)

with col2:
    absences = st.number_input("**Absences**", 0, 50, 5)
    major = st.selectbox("**Major**", ["IT", "Business", "Economics", "Engineering"])

# Create exact same DataFrame as original
data = pd.DataFrame({
    "gpa": [gpa],
    "credits": [credits],
    "absences": [absences],
    "major": [major]
})

# Show inputs
st.markdown("---")
st.subheader("📊 Input Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("GPA", f"{gpa:.2f}")
col2.metric("Credits", credits)
col3.metric("Absences", absences)
col4.metric("Major", major)

# Predict button
if st.button("🔮 **Make Prediction**", type="primary", use_container_width=True):
    try:
        prediction = model.predict(data)[0]
        
        st.markdown("### 📈 **Prediction Result**")
        col_result, col_prob = st.columns([2, 1])
        
        with col_result:
            if prediction == 1:
                st.error("🚨 **ACADEMIC WARNING**")
                st.markdown("**Student is at risk of academic warning**")
            else:
                st.success("✅ **GOOD STANDING**")
                st.markdown("**Student is safe academically**")
        
        # Probability if available
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(data)[0]
            with col_prob:
                st.metric("Risk Probability", f"{probs[1]*100:.1f}%")
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

st.markdown("---")
st.caption("🎓 Academic Success Prediction System")
