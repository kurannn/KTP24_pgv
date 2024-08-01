#"""
#@author:  Fahrettin Kuran
#Title:    Turkiye-Specific Ground Motion Model for Peak Ground Velocity (PGV)
#Paper:    Kuran F, Tanircan G, Pashaei E (2024) Developing machine learning-based ground motion models to predict peak ground velocity in Turkiye.
#Streamlit app
#"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd


scaler = joblib.load('scaler.pkl')
model = joblib.load('ktp24_model.sav')

def normalize_input(user_input, scaler):
    input_array = np.array(user_input).reshape(1, -1)
    normalized_input = scaler.transform(input_array)
    return normalized_input

primaryColor="#426EDA"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#31333F"
font="Sans serif"

col1, col2 = st.columns([8, 2])
col1.subheader("Turkiye-Specific Ground Motion Model for Peak Ground Velocity (PGV) (KTP24)")
col2.image("Kandilli_logo.png", use_column_width=True)

text = """
<strong>Gradient Boosting</strong> algorithm is implemented to predict the geomean <strong>Peak Ground Velocity (PGV)</strong> of two horizontal components.
A Turkiye-based regional ground motion model is obtained using <strong>The expanded New Turkish Strong Motion Database (The expanded N-TSMD)</strong>.
"""
line_height = "1.2"
font_size = "14px"
st.markdown(f"<p style='line-height: {line_height}; font-size: {font_size};'>{text}</p>", unsafe_allow_html=True)

st.subheader("Estimator parameters")
input_info = [
    "<strong>M<sub>w</sub>:</strong> Moment Magnitude",
    "<strong>Vs<sub>30</sub>:</strong> Shear wave velocity of the top 30 m of the soil (m/s)",
    "<strong>δ:</strong> Dip angle (°)",
    "<strong>R<sub>JB</sub>:</strong> Joyner-Boore distance (km)",
    "<strong>R<sub>rup</sub>:</strong> Rupture distance (km)",
    "<strong>R<sub>epi</sub>:</strong> Epicentral distance (km)",
    "<strong>SoF:</strong> Style-of-faulting"
]

line_height = "0.4"
font_size = "14px"
for line in input_info:
    st.markdown(f"<p style='line-height: {line_height}; font-size: {font_size};'>{line}</p>", unsafe_allow_html=True)

# Streamlit app
def main():

    # User inputs
    Mw = st.sidebar.slider(r"$M_{w}$", 3.5, 7.8, step=0.1)
    Vs30 = st.sidebar.slider(r"$Vs_{30}$", 131, 1862)
    Dip = st.sidebar.slider("δ", 30, 90)
    Rjb = st.sidebar.slider(r"$R_{JB}$", 0.0, 200.0, step=0.01)
    Rrup = st.sidebar.slider(r"$R_{rup}$", 0.0, 200.0, step=0.01)
    Repi = st.sidebar.slider(r"$R_{epi}$", 0.0, 200.0, step=0.01)
    SoF = st.sidebar.selectbox(r"$SoF$", ["Strike-slip", "Normal", "Reverse"])

    normal = 1 if SoF == "Normal" else 0
    strike_slip = 1 if SoF == "Strike-slip" else 0
    thrust = 1 if SoF == "Reverse" else 0

    user_input = [Mw, Vs30, Dip, Rjb, Rrup, Repi, normal, strike_slip, thrust]

    input_summary = pd.DataFrame({
        'Mw': [Mw],
        'Vs30': [Vs30],
        'Dip': [Dip],
        'Rjb': [Rjb],
        'Rrup': [Rrup],
        'Repi': [Repi],
        'SoF': [SoF]
    })

    if st.button('Submit'):
       
        normalized_input = normalize_input(user_input, scaler)
        
        
        prediction = model.predict(normalized_input)
        st.subheader('Summary of your inputs')
        st.table(input_summary)
        st.write(f'PGV = {prediction[0]:.5f} cm/s')

if __name__ == '__main__':
    main()


paper = """
<strong>Publication:</strong> Kuran F, Tanircan G, Pashaei E (2024) Developing machine learning-based ground motion models to predict peak ground velocity in Turkiye.
"""
line_height = "1.2"
font_size = "14px"
st.markdown(f"<p style='line-height: {line_height}; font-size: {font_size};'>{paper}</p>", unsafe_allow_html=True)
