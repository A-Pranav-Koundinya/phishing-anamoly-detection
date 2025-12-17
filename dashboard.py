# src/dashboard.py
import streamlit as st
import requests

st.title("🛡️ Network Anomaly Detection System")

with st.form("prediction_form"):
    st.header("Manual Packet Inspection")
    col1, col2 = st.columns(2)
    
    with col1:
        port = st.number_input("Destination Port", value=80)
        duration = st.number_input("Flow Duration", value=1000.0)
        
    with col2:
        fwd_pkts = st.number_input("Total Fwd Packets", value=5)
        bwd_pkts = st.number_input("Total Bwd Packets", value=0)
        
    # Add inputs for other features here...
    
    submitted = st.form_submit_button("Analyze Packet")

if submitted:
    payload = {
        "Destination_Port": port,
        "Flow_Duration": duration,
        "Total_Fwd_Packets": fwd_pkts,
        "Total_Backward_Packets": bwd_pkts
    }
    
    try:
        # Call the API container
        res = requests.post("http://localhost:8000/predict", json=payload)
        result = res.json()
        
        if result['prediction'] == 'Benign':
            st.success(f"✅ Traffic is BENIGN (Confidence: {result['confidence']:.2f})")
        else:
            st.error(f"🚨 MALICIOUS Traffic Detected! (Error: {result['reconstruction_error']:.2f})")
            
    except Exception as e:
        st.error(f"Connection Error: {e}")