import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="CI-MACE Risk Predictor",
    page_icon="🫀",
    layout="wide"
)

# ==================== 常量配置 ====================
FEATURE_NAMES = ['Hypertension', 'Dyslipidaemia', 'Diabetes', 'SMS', 'TPD', 'HRR3']
CUTOFF_1 = -0.17  # Low vs Medium
CUTOFF_2 = 2.3    # Medium vs High
TIME_3YR = 36     # 3年 = 36个月
TIME_5YR = 60     # 5年 = 60个月

# ==================== 加载模型和背景数据 ====================
@st.cache_resource
def load_model_and_background():
    """加载模型和SHAP背景数据"""
    model = joblib.load('gbsa_model.pkl')
    background = joblib.load('shap_background.pkl')
    return model, background

try:
    model, background = load_model_and_background()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model loading failed: {e}")

# ==================== 辅助函数 ====================
def get_risk_group(score):
    """根据风险评分确定风险分层"""
    if score <= CUTOFF_1:
        return "Low-risk", "🟢", "#2f8e2f", "#D1FAE5"
    elif score <= CUTOFF_2:
        return "Medium-risk", "🟡", "#ef7307", "#FEF3C7"
    else:
        return "High-risk", "🔴", "#bf1b1b", "#FEE2E2"

def calculate_survival_probability(model, input_data, time_point):
    """计算特定时间点的生存概率"""
    surv_funcs = model.predict_survival_function(input_data)
    surv_func = surv_funcs[0]
    # 使用插值获取指定时间点的概率
    prob = surv_func(time_point)
    return float(prob)

def create_shap_explainer(model, background):
    """动态创建SHAP解释器"""
    def model_predict(data):
        if isinstance(data, pd.DataFrame):
            return model.predict(data)
        else:
            return model.predict(pd.DataFrame(data, columns=FEATURE_NAMES))
    
    explainer = shap.KernelExplainer(model_predict, background)
    return explainer

# ==================== 自定义CSS ====================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        border: 2px solid;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #4B5563;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 标题 ====================
st.markdown('<p class="main-title">🫀 CI-MACE Risk Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Interpretable Machine Learning for Risk Stratification in Patients with Chronotropic Incompetence Undergoing Exercise Stress SPECT-MPI</p>', unsafe_allow_html=True)

st.markdown("---")

# ==================== 侧边栏输入 ====================
st.sidebar.header("📝 Patient Parameters")
st.sidebar.markdown("Please input the six model predictors:")

st.sidebar.markdown("#### 🩺 Clinical History")

hypertension = st.sidebar.selectbox(
    "Hypertension",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
    help="History of hypertension"
)

dyslipidaemia = st.sidebar.selectbox(
    "Dyslipidaemia",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
    help="History of dyslipidaemia"
)

diabetes = st.sidebar.selectbox(
    "Diabetes Mellitus",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
    help="History of diabetes mellitus"
)

st.sidebar.markdown("#### 🔬 SPECT-MPI Parameters")

sms = st.sidebar.slider(
    "SMS (Summed Motion Score)",
    min_value=0,
    max_value=68,
    value=0,
    step=1,
    help="Summed Motion Score from gated SPECT-MPI"
)

tpd = st.sidebar.slider(
    "TPD (%)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.1,
    help="Total Perfusion Deficit from SPECT-MPI"
)

st.sidebar.markdown("#### ❤️ Exercise Parameter")

hrr3 = st.sidebar.slider(
    "HRR3 (bpm)",
    min_value=0,
    max_value=80,
    value=30,
    step=1,
    help="Heart Rate Recovery at 3 minutes post-exercise"
)

st.sidebar.markdown("---")

# 预测按钮
predict_button = st.sidebar.button("🔮 Calculate Risk", type="primary", use_container_width=True)

# ==================== 主区域 ====================
if predict_button and model_loaded:
    # 准备输入数据
    input_data = pd.DataFrame({
        'Hypertension': [hypertension],
        'Dyslipidaemia': [dyslipidaemia],
        'Diabetes': [diabetes],
        'SMS': [sms],
        'TPD': [tpd],
        'HRR3': [hrr3]
    })[FEATURE_NAMES]
    
    # 计算风险评分
    risk_score = model.predict(input_data)[0]
    
    # 确定风险分层
    risk_group, risk_emoji, risk_color, bg_color = get_risk_group(risk_score)
    
    # 计算生存概率
    prob_3yr_survival = calculate_survival_probability(model, input_data, TIME_3YR)
    prob_5yr_survival = calculate_survival_probability(model, input_data, TIME_5YR)
    
    # MACE概率 = 1 - 生存概率
    prob_3yr_mace = (1 - prob_3yr_survival) * 100
    prob_5yr_mace = (1 - prob_5yr_survival) * 100
    
    # ==================== 显示结果 ====================
    st.markdown("## 📊 Risk Assessment Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="result-card" style="background-color: #F3F4F6; border-color: #D1D5DB;">
            <p class="metric-label">Risk Score</p>
            <p class="metric-value" style="color: {risk_color};">{risk_score:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="result-card" style="background-color: {bg_color}; border-color: {risk_color};">
            <p class="metric-label">Risk Stratum</p>
            <p class="metric-value">{risk_emoji} {risk_group}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="result-card" style="background-color: #F3F4F6; border-color: #D1D5DB;">
            <p class="metric-label">MACE Probability</p>
            <p style="font-size: 1.2rem; margin: 0.3rem 0;"><b>3-Year:</b> {prob_3yr_mace:.1f}%</p>
            <p style="font-size: 1.2rem; margin: 0.3rem 0;"><b>5-Year:</b> {prob_5yr_mace:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==================== SHAP解释 ====================
    st.markdown("## 🔍 Model Explanation (SHAP Analysis)")
    st.markdown("""
    The SHAP (SHapley Additive exPlanations) visualization below decomposes the prediction for this individual patient, 
    quantifying how each predictor contributes to the final risk score.
    """)
    
    # 动态创建explainer并计算SHAP值
    with st.spinner("Calculating SHAP values (this may take a few seconds)..."):
        explainer = create_shap_explainer(model, background)
        shap_values = explainer.shap_values(input_data, nsamples=100)
    
    # SHAP图布局
    col_shap1, col_shap2 = st.columns(2)
    
    with col_shap1:
        st.markdown("### Waterfall Plot")
        st.markdown("*Shows how each feature pushes the prediction from baseline.*")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_data.values[0],
                feature_names=FEATURE_NAMES
            ),
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
    
    with col_shap2:
        st.markdown("### Force Plot")
        st.markdown("*Red = increases risk; Blue = decreases risk.*")
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_data.values[0],
            feature_names=FEATURE_NAMES,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    
    # ==================== 特征贡献表 ====================
    st.markdown("### Feature Contributions")
    
    contribution_df = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'Input Value': input_data.values[0],
        'SHAP Value': shap_values[0],
        'Direction': ['↑ Increases Risk' if sv > 0 else '↓ Decreases Risk' for sv in shap_values[0]]
    })
    contribution_df['|SHAP|'] = np.abs(contribution_df['SHAP Value'])
    contribution_df = contribution_df.sort_values('|SHAP|', ascending=False).drop(columns=['|SHAP|'])
    contribution_df['SHAP Value'] = contribution_df['SHAP Value'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(contribution_df, use_container_width=True, hide_index=True)
    
    # ==================== 生存曲线 ====================
    st.markdown("---")
    st.markdown("## 📈 Individual Survival Curve")
    
    surv_funcs = model.predict_survival_function(input_data)
    surv_func = surv_funcs[0]
    times = surv_func.x
    probs = surv_func.y
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    
    # 绘制生存曲线
    ax3.step(times, probs, where='post', color='#1E3A8A', linewidth=2.5, label='Survival Probability')
    ax3.fill_between(times, probs, step='post', alpha=0.15, color='#1E3A8A')
    
    # 标记3年和5年
    ax3.axvline(x=TIME_3YR, color='#10B981', linestyle='--', linewidth=1.5, alpha=0.8)
    ax3.axvline(x=TIME_5YR, color='#EF4444', linestyle='--', linewidth=1.5, alpha=0.8)
    ax3.scatter([TIME_3YR], [prob_3yr_survival], color='#10B981', s=100, zorder=5)
    ax3.scatter([TIME_5YR], [prob_5yr_survival], color='#EF4444', s=100, zorder=5)
    
    ax3.annotate(f'3-Year: {prob_3yr_survival:.1%}', 
                 xy=(TIME_3YR, prob_3yr_survival), 
                 xytext=(TIME_3YR + 3, prob_3yr_survival + 0.05),
                 fontsize=11, color='#10B981', fontweight='bold')
    ax3.annotate(f'5-Year: {prob_5yr_survival:.1%}', 
                 xy=(TIME_5YR, prob_5yr_survival), 
                 xytext=(TIME_5YR + 3, prob_5yr_survival - 0.08),
                 fontsize=11, color='#EF4444', fontweight='bold')
    
    ax3.set_xlabel('Time (Months)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('MACE-free Survival Probability', fontsize=12, fontweight='bold')
    ax3.set_title('Predicted Survival Curve for This Patient', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.set_xlim(0, max(times) + 5)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='lower left', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

elif not predict_button:
    # ==================== 默认页面 ====================
    st.info("👈 **Please enter patient parameters in the sidebar and click 'Calculate Risk' to generate predictions.**")
    
    st.markdown("---")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        ### 📋 About This Tool
        
        This web-based calculator implements a **Gradient Boosting Survival Analysis (GBSA)** 
        model for predicting Major Adverse Cardiovascular Events (MACE) in patients with 
        **chronotropic incompetence** undergoing exercise stress SPECT-MPI.
        
        The model was developed and externally validated using data from two medical centers.
        """)
        
        st.markdown("""
        ### 🎯 Model Predictors
        
        | Predictor | Description |
        |-----------|-------------|
        | **Hypertension** | History of hypertension |
        | **Dyslipidaemia** | History of dyslipidaemia |
        | **Diabetes** | History of diabetes mellitus |
        | **SMS** | Summed Motion Score |
        | **TPD** | Total Perfusion Deficit (%) |
        | **HRR3** | Heart Rate Recovery at 3 min (bpm) |
        """)
    
    with col_info2:
        st.markdown("""
        ### 📊 Risk Stratification
        
        Patients are stratified into three groups based on GBSA risk score:
        
        - 🟢 **Low-risk**: Score ≤ -0.17
        - 🟡 **Medium-risk**: -0.17 < Score ≤ 2.3
        - 🔴 **High-risk**: Score > 2.3
        """)
        
        st.markdown("""
        ### 📈 Output
        
        The calculator provides:
        
        1. **Risk Score** - Continuous GBSA prognostic index
        2. **Risk Stratum** - Low, Medium, or High
        3. **MACE Probability** - Estimated at 3 and 5 years
        4. **SHAP Analysis** - Individual feature contributions
        5. **Survival Curve** - Personalized MACE-free survival
        """)

# ==================== 页脚 ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.85rem; padding: 1rem 0;">
    <p><b>CI-MACE Risk Predictor</b> | Gradient Boosting Survival Analysis Model</p>
    <p>⚠️ <i>This tool is for research purposes only. Clinical decisions should be made in consultation with qualified healthcare professionals.</i></p>
    <p>© 2024 | Developed for: <i>Development and external validation of an interpretable machine learning for risk stratification of patients with chronotropic incompetence undergoing exercise stress SPECT-MPI</i></p>
</div>
""", unsafe_allow_html=True)