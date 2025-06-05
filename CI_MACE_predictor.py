import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('cph.pkl')  # 加载训练好的模型


# Streamlit UI
st.title("Chronotropic Incompetence Risk Stratifier for CAD​​")  # 预测器
st.write('请输入患者的临床指标来评估心血管不良事件风险')
# Sidebar for input options
st.sidebar.header("Input Sample Data")  # 侧边栏输入样本数据

# Risk factors input section
st.sidebar.header("Patient Information")

# Binary risk factors
diabetes = st.sidebar.selectbox("Diabetes:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
dyslipidaemia = st.sidebar.selectbox("Dyslipidaemia:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
apas = st.sidebar.selectbox("Antiplatelet Agents (APAs):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
bbs = st.sidebar.selectbox("Beta-Blockers (BBs):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Convert numerical factors to binary inputs
tpd = st.sidebar.selectbox("Total Perfusion Deficit (TPD ≥ 5%):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
sms = st.sidebar.selectbox("Summed Motion Score (SMS ≥ 2):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
hrr3 = st.sidebar.selectbox("Heart Rate Recovery at 3 mins (HRR3 ≥ 34):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Risk score and risk group definition
def calculate_risk_score(features):
    """Calculate risk score based on features"""
    score = 0
    if features['Diabetes'] == 1:
        score += 2
    if features['Dyslipidaemia'] == 1:
        score += 1
    if features['APAs'] == 1:
        score += 1
    if features['BBs'] == 1:
        score += 2
    if features['TPD'] == 1:
        score += 4
    if features['SMS'] == 1:
        score += 1
    if features['HRR3'] == 1:
        score -= 2  # Protective factor
    return score

def get_risk_group(score):
    """Determine risk group based on score"""
    if score <= -1:
        return "Low-risk group"
    elif score <= 2:
        return "Moderate-risk group"
    else:
        return "High-risk group"

# Predefined risk probabilities for each risk group
RISK_PROBABILITIES = {
    "Low-risk group": {"3-year": 0.002, "5-year": 0.012},
    "Moderate-risk group": {"3-year": 0.09, "5-year": 0.156},
    "High-risk group": {"3-year": 0.336, "5-year": 0.611}
}

# Collect feature data
feature_values = {
    'Diabetes': diabetes,
    'Dyslipidaemia': dyslipidaemia,
    'APAs': apas,
    'BBs': bbs,
    'TPD': tpd,
    'SMS': sms,
    'HRR3': hrr3
}


features = np.array([feature_values])  # 转换为NumPy数组

if st.button("Make Prediction"):  # 如果点击了预测按钮
    # Calculate risk score
    risk_score = calculate_risk_score(feature_values)
    
    # Determine risk group
    risk_group = get_risk_group(risk_score)
    
    # Get risk probabilities for the risk group
    risk_probs = RISK_PROBABILITIES[risk_group]
    
    # Display results
    st.subheader("预测结果")
    st.write(f"**评分:** {risk_score}")
    st.write(f"**风险组:** {risk_group}")
    
    # Display risk probabilities
    st.write("**时间风险概率:**")
    st.write(f"3年风险概率: {risk_probs['3-year']:.3f} (即 {risk_probs['3-year']*100:.1f}%)")
    st.write(f"5年风险概率: {risk_probs['5-year']:.3f} (即 {risk_probs['5-year']*100:.1f}%)")
