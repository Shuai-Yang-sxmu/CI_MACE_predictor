import streamlit as st
#import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 添加错误处理避免加载失败
#try:
    #model = joblib.load('cph.pkl')
    #model_loaded = True
#except:
    #model_loaded = False
    #st.warning("Model file not found. Risk scores will be calculated using the scoring system only.")


# Streamlit UI
st.title("Chronotropic Incompetence Risk Stratifier for CAD")  # 预测器
st.write('Please input patient clinical parameters to assess the risk of major adverse cardiac events (MACE).')

# Risk factors input section
st.sidebar.header("Patient Information")
st.sidebar.markdown("""
### Risk Factors
Please select all that apply to the patient:
""")

# 更有组织的风险因素分组
st.sidebar.markdown("**Patient Characteristics**")
diabetes = st.sidebar.selectbox("Diabetes:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
dyslipidaemia = st.sidebar.selectbox("Dyslipidaemia:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

st.sidebar.markdown("**Medications**")
apas = st.sidebar.selectbox("Antiplatelet Agents (APAs):", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
bbs = st.sidebar.selectbox("Beta-Blockers (BBs):", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

st.sidebar.markdown("**Imaging & Functional Parameters**")
tpd = st.sidebar.selectbox("Total Perfusion Deficit (TPD ≥ 5%):", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
sms = st.sidebar.selectbox("Summed Motion Score (SMS ≥ 2):", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
hrr3 = st.sidebar.selectbox("Heart Rate Recovery at 3 mins (HRR3 ≥ 34):", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')


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


def display_risk_results(risk_score, risk_group, risk_probs):
    # 基于风险组的颜色映射（完全匹配图片中的配色）
    risk_colors = {
        "High-risk group": "#FFD1DC",  # 浅粉色 - 匹配第1张图
        "Moderate-risk group": "#FFFF99",  # 浅黄色 - 匹配第2张图
        "Low-risk group": "#90EE90"  # 浅绿色 - 匹配第3张图
    }
    
    # 创建顶部横幅（使用图片中完全一致的布局）
    st.markdown(
        f"""
        <div style="
            background-color: {risk_colors[risk_group]};
            padding: 10px 15px;
            border-radius: 5px;
            font-family: Arial, sans-serif;
            font-size: 16px;
            color: #333333;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <div><strong>Total score / Risk Group:</strong></div>
            <div><strong>{risk_score} ({risk_group})</strong></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 创建数据表格（与图片完全一致的布局）
    st.markdown(
        """
        <div style="
            margin-bottom: 5px;
            font-family: Arial, sans-serif;
            font-size: 18px;
            text-align: center;
            font-weight: bold;
        ">
            Cumulative Incidence of MACE
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 创建表格（使用与图片相同的三列布局）
    prob_table = pd.DataFrame({
        "3-year": [f"{risk_probs['3-year']*100:.1f}%"],
        "5-year": [f"{risk_probs['5-year']*100:.1f}%"]
    })
    
    # 自定义表格样式以匹配图片
    st.dataframe(
        prob_table,
        hide_index=True,
        column_config={
            "3-year": st.column_config.TextColumn("3-year", width="medium"),
            "5-year": st.column_config.TextColumn("5-year", width="medium")
        },
        use_container_width=True
    )

def create_clinical_waterfall(score_details):
    """创建医疗专用的瀑布图，清晰展示风险构建"""
    # 计算累计分数
    cumulative = [0]
    for i, (_, value) in enumerate(score_details):
        cumulative.append(cumulative[-1] + value)
    
    # 设置医疗专业颜色
    risk_colors = ['#E63946' if v > 0 else '#457B9D' for _, v in score_details]
    protective_colors = ['#457B9D' if v < 0 else '#E63946' for _, v in score_details]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制瀑布主体
    for i, ((name, value), base) in enumerate(zip(score_details, cumulative[:-1])):
        color = protective_colors[i] if value < 0 else risk_colors[i]
        ax.bar(name, value, bottom=base, color=color, alpha=0.8, edgecolor='black')
    
    # 添加总得分
    ax.bar("Total Score", cumulative[-1], color="#FDAE61", alpha=0.9, edgecolor='black', hatch='//')
    
    # 医疗专业样式
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylabel("Risk Score Contribution", fontsize=12, fontweight='bold')
    ax.set_title("Constructing Cardiac Risk Score", fontsize=14, pad=15)
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig


if st.button("Calculate Risk"):
    # 计算风险评分和组别
    risk_score = calculate_risk_score(feature_values)
    risk_group = get_risk_group(risk_score)
    risk_probs = RISK_PROBABILITIES[risk_group]
    
    # 显示结果
    display_risk_results(risk_score, risk_group, risk_probs)

    # 添加临床建议
    if risk_group == "Low-risk group":
        st.markdown("""
        **Clinical Recommendation:**
        Low risk of MACE. Consider standard follow-up according to guidelines.
        """)
    elif risk_group == "Moderate-risk group":
        st.markdown("""
        **Clinical Recommendation:**
        Moderate risk of MACE. Consider more frequent follow-up and risk factor management.
        """)
    else:  # High-risk
        st.markdown("""
        **Clinical Recommendation:**
        High risk of MACE. Consider aggressive risk factor modification and potential revascularization evaluation.
        """)


    # 创建风险分解瀑布图数据
    score_details = [
        ("Diabetes", 2 if feature_values['Diabetes'] else 0),
        ("Dyslipidaemia", 1 if feature_values['Dyslipidaemia'] else 0),
        ("APAs", 1 if feature_values['APAs'] else 0),
        ("BBs", 2 if feature_values['BBs'] else 0),
        ("TPD", 4 if feature_values['TPD'] else 0),
        ("SMS", 1 if feature_values['SMS'] else 0),
        ("HRR3", -2 if feature_values['HRR3'] else 0)
    ]
    
    # 过滤掉零值项(可选)
    score_details = [(name, value) for name, value in score_details if value != 0]
    
    if score_details:  # 只有当有非零项时才显示图表
        st.subheader("Risk Score Breakdown")
        waterfall_fig = create_clinical_waterfall(score_details)
        st.pyplot(waterfall_fig)
        
        # 添加临床解释
        st.markdown("**Clinical Interpretation:**")
        st.write("- 🔴 *Red bars* show risk-increasing factors")
        st.write("- 🔵 *Blue bars* show protective factors")
        st.write("- 🟠 *Orange bar* represents final risk score")
