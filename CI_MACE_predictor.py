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
    page_title="MACE Risk Predictor for Chronotropic Incompetence",
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
        return "Low-risk", "🟢", "#059669", "#D1FAE5", "#10B981"
    elif score <= CUTOFF_2:
        return "Medium-risk", "🟡", "#D97706", "#FEF3C7", "#F59E0B"
    else:
        return "High-risk", "🔴", "#DC2626", "#FEE2E2", "#EF4444"

def calculate_survival_probability(model, input_data, time_point):
    """计算特定时间点的生存概率"""
    surv_funcs = model.predict_survival_function(input_data)
    surv_func = surv_funcs[0]
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
        font-size: 42px !important;
        font-weight: 800 !important;
        color: #1E3A8A !important;
        text-align: center !important;
        margin-bottom: 8px !important;
        line-height: 1.2 !important;
    }
    .sub-title {
        font-size: 16px !important;
        color: #4B5563 !important;
        text-align: center !important;
        margin-bottom: 8px !important;
    }
    .research-note {
        font-size: 14px !important;
        color: #6B7280 !important;
        text-align: center !important;
        font-style: italic !important;
        margin-bottom: 20px !important;
    }
    .section-header {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #1F2937 !important;
        margin: 16px 0 12px 0 !important;
    }
    .result-card {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 14px !important;
        color: #4B5563 !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    .metric-value-large {
        font-size: 32px !important;
        font-weight: 700 !important;
        margin: 6px 0 !important;
    }
    .prob-item {
        font-size: 18px !important;
        margin: 6px 0 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 侧边栏输入 ====================
st.sidebar.header("Patient Parameters")
st.sidebar.markdown("Enter the six clinical and imaging predictors below:")

st.sidebar.markdown("#### Clinical History")

hypertension = st.sidebar.selectbox(
    "Hypertension",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
    help="Diagnosis or history of hypertension"
)

dyslipidaemia = st.sidebar.selectbox(
    "Dyslipidaemia",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
    help="Diagnosis or history of dyslipidaemia"
)

diabetes = st.sidebar.selectbox(
    "Diabetes Mellitus",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
    help="Diagnosis or history of diabetes mellitus"
)

st.sidebar.markdown("#### SPECT-MPI Parameters")

sms = st.sidebar.slider(
    "SMS (Summed Motion Score)",
    min_value=0,
    max_value=20,
    value=0,
    step=1,
    help="Extent and severity of wall motion abnormalities from gated SPECT-MPI"
)

tpd = st.sidebar.slider(
    "TPD (Total Perfusion Deficit, %)",
    min_value=0,
    max_value=20,
    value=0,
    step=1,
    help="Global measure of myocardial perfusion severity from SPECT-MPI"
)

st.sidebar.markdown("#### Exercise Parameter")

hrr3 = st.sidebar.slider(
    "HRR3 (Heart Rate Recovery at 3 min, bpm)",
    min_value=10,
    max_value=70,
    value=30,
    step=1,
    help="Heart rate reduction at 3 minutes into recovery phase"
)

st.sidebar.markdown("---")

# 预测按钮
predict_button = st.sidebar.button("Calculate Risk", type="primary", use_container_width=True)

# ==================== 主区域：标题 ====================
st.markdown('<h1 class="main-title">MACE Risk Predictor for Chronotropic Incompetence</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">An interpretable machine learning tool for stratifying cardiovascular risk in patients with chronotropic incompetence undergoing exercise stress SPECT-MPI.</p>', unsafe_allow_html=True)
st.markdown('<p class="research-note">Designed for Research Verification & Educational Purposes</p>', unsafe_allow_html=True)

st.markdown("---")

# ==================== 主区域：结果显示 ====================
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
    risk_group, risk_emoji, risk_color, bg_color, border_color = get_risk_group(risk_score)
    
    # 计算生存概率
    prob_3yr_survival = calculate_survival_probability(model, input_data, TIME_3YR)
    prob_5yr_survival = calculate_survival_probability(model, input_data, TIME_5YR)
    
    # MACE概率 = 1 - 生存概率
    prob_3yr_mace = (1 - prob_3yr_survival) * 100
    prob_5yr_mace = (1 - prob_5yr_survival) * 100
    
    # ==================== Prognostic Assessment Report ====================
    st.markdown('<p class="section-header">📊 Prognostic Assessment Report</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="result-card" style="background: linear-gradient(135deg, {bg_color} 0%, white 100%); border: 3px solid {border_color};">
            <p class="metric-label">Risk Stratum</p>
            <p class="metric-value-large" style="color: {risk_color};">{risk_emoji} {risk_group}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="result-card" style="background: linear-gradient(135deg, #F0F9FF 0%, white 100%); border: 3px solid #3B82F6;">
            <p class="metric-label">MACE Probability</p>
            <p class="prob-item"><span style="color: #059669;">●</span> <b>3-Year:</b> <span style="color: #059669; font-weight: 700;">{prob_3yr_mace:.1f}%</span></p>
            <p class="prob-item"><span style="color: #DC2626;">●</span> <b>5-Year:</b> <span style="color: #DC2626; font-weight: 700;">{prob_5yr_mace:.1f}%</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== Risk Interpretation ====================
    st.markdown("##### 📝 Risk Interpretation")
    
    if risk_group == "Low-risk":
        interpretation = f"**Favorable Prognosis.** Based on the input parameters, this patient is classified as **low-risk** with a 3-year MACE probability of {prob_3yr_mace:.1f}% and a 5-year probability of {prob_5yr_mace:.1f}%. Standard clinical follow-up and routine cardiovascular risk factor management are typically indicated."
        st.success(interpretation, icon="✅")
    elif risk_group == "Medium-risk":
        interpretation = f"**Intermediate Risk Profile.** Based on the input parameters, this patient is classified as **medium-risk** with a 3-year MACE probability of {prob_3yr_mace:.1f}% and a 5-year probability of {prob_5yr_mace:.1f}%. Closer monitoring of cardiovascular risk factors and optimization of preventive medical therapy may be warranted."
        st.warning(interpretation, icon="⚠️")
    else:  # High-risk
        interpretation = f"**Elevated Risk Profile.** Based on the input parameters, this patient is classified as **high-risk** with a 3-year MACE probability of {prob_3yr_mace:.1f}% and a 5-year probability of {prob_5yr_mace:.1f}%. Intensive risk factor modification, close surveillance, and evaluation for potential ischemia-driven interventions should be considered."
        st.error(interpretation, icon="🚨")
    
    st.markdown("---")
    
    # ==================== 下方两栏：SHAP + Survival Curve ====================
    col_left, col_right = st.columns(2)
    
    # ===== 左侧：SHAP Waterfall Plot =====
    with col_left:
        st.markdown('<p class="section-header">🔍 Feature Contribution Analysis</p>', unsafe_allow_html=True)
        st.caption("Breakdown of how individual predictors push the risk score towards higher (red) or lower (blue) values relative to the population baseline.")
        
        # 动态创建explainer并计算SHAP值
        with st.spinner("Calculating SHAP values..."):
            explainer = create_shap_explainer(model, background)
            shap_values = explainer.shap_values(input_data, nsamples=100)
        
        fig1 = plt.figure(figsize=(8, 5))
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
        st.pyplot(plt.gcf())
        plt.close()
    
    # ===== 右侧：Survival Curve =====
    with col_right:
        st.markdown('<p class="section-header">📈 Personalized Survival Projection</p>', unsafe_allow_html=True)
        st.caption("Estimated probability of remaining free from MACE over a 5-year follow-up horizon based on the input parameters.")
        
        surv_funcs = model.predict_survival_function(input_data)
        surv_func = surv_funcs[0]
        times = surv_func.x
        probs = surv_func.y
        
        # 筛选60个月以内的数据
        mask = times <= 60
        times_plot = times[mask]
        probs_plot = probs[mask]
        
        # 创建生存曲线
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        
        # 绘制主生存曲线
        ax3.step(times_plot, probs_plot, where='post', color='#1E40AF', linewidth=2.5, label='MACE-free Survival')
        
        # 添加3年和5年标记点
        ax3.scatter([TIME_3YR], [prob_3yr_survival], color='#2f8e2f', s=100, zorder=5, marker='o')
        ax3.scatter([TIME_5YR], [prob_5yr_survival], color='#bf1b1b', s=100, zorder=5, marker='o')
        
        # 添加虚线
        ax3.axhline(y=prob_3yr_survival, xmin=0, xmax=TIME_3YR/62, color='#2f8e2f', linestyle='--', linewidth=1.2, alpha=0.6)
        ax3.axhline(y=prob_5yr_survival, xmin=0, xmax=TIME_5YR/62, color='#bf1b1b', linestyle='--', linewidth=1.2, alpha=0.6)
        ax3.axvline(x=TIME_3YR, ymin=0, ymax=prob_3yr_survival, color='#2f8e2f', linestyle='--', linewidth=1.2, alpha=0.6)
        ax3.axvline(x=TIME_5YR, ymin=0, ymax=prob_5yr_survival, color='#bf1b1b', linestyle='--', linewidth=1.2, alpha=0.6)
        
        # 文字标注
        ax3.text(TIME_3YR + 1.5, prob_3yr_survival + 0.02, f'3-Year: {prob_3yr_survival:.1%}', 
                 fontsize=11, color='#2f8e2f', fontweight='bold', va='bottom')
        ax3.text(TIME_5YR + 1.5, prob_5yr_survival - 0.02, f'5-Year: {prob_5yr_survival:.1%}', 
                 fontsize=11, color='#bf1b1b', fontweight='bold', va='top')
        
        # 设置坐标轴
        ax3.set_xlabel('Time (Months)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
        ax3.set_ylim([0, 1.01])
        ax3.set_xlim([0, 62])
        ax3.set_xticks([0, 12, 24, 36, 48, 60])
        ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax3.grid(True, linestyle='--', alpha=0.4)
        ax3.legend(loc='lower left', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

elif not predict_button:
    # ==================== 默认页面 ====================
    st.info("👈 **Action Required:** Please enter patient parameters in the sidebar and click **'Calculate Risk'** to generate the prognostic report.", icon="ℹ️")
    
    st.markdown("---")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        ### 📖 About This Tool
        
        This web-based prototype implements a **Gradient Boosting Survival Analysis (GBSA)** model 
        to predict the probability of Major Adverse Cardiovascular Events (MACE) in patients with 
        **chronotropic incompetence** undergoing exercise stress SPECT myocardial perfusion imaging.
        
        **Key Features:**
        - **Derivation Cohort:** Developed on 765 patients from a primary medical center
        - **Validation Cohort:** Externally validated on 295 patients from an independent center
        - **Objective:** To assist in risk stratification and inform clinical decision-making
        """)
        
        st.markdown("""
        ### 📋 Model Predictors
        
        | Predictor | Description |
        |-----------|-------------|
        | **TPD** | Total Perfusion Deficit (%) from SPECT-MPI |
        | **HRR3** | Heart Rate Recovery at 3 minutes (bpm) |
        | **SMS** | Summed Motion Score from gated SPECT |
        | **Clinical Factors** | History of Diabetes, Hypertension, Dyslipidaemia |
        """)
    
    with col_info2:
        st.markdown("""
        ### 📊 Risk Stratification
        
        Patients are stratified into three prognostic groups based on the GBSA risk score:
        
        - 🟢 **Low-risk** (Score ≤ -0.17) — Favorable prognosis
        - 🟡 **Medium-risk** (-0.17 < Score ≤ 2.3) — Intermediate prognosis
        - 🔴 **High-risk** (Score > 2.3) — Elevated risk requiring attention
        """)
        
        st.markdown("""
        ### 📈 Output
        
        The calculator provides the following results:
        
        1. **Risk Stratum** — Classification into Low, Medium, or High risk
        2. **MACE Probability** — Estimated cumulative incidence at 3 and 5 years
        3. **Risk Interpretation** — Clinical context for the predicted risk level
        4. **Feature Contribution** — SHAP analysis showing predictor contributions
        5. **Survival Projection** — Personalized MACE-free survival curve
        """)

# ==================== 页脚 ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem; padding: 1rem 0; line-height: 1.7;">
    <p style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.8rem; color: #374151;">
        Research Prototype | Not for Clinical Decision Making
    </p>
    <p style="margin: 1rem auto; padding: 0 1rem; max-width: 900px; text-align: justify;">
        ⚠️ <b>DISCLAIMER:</b> This tool is a <b>research prototype</b> developed to demonstrate the application of 
        interpretable machine learning in cardiovascular risk stratification. It has <b>not</b> been cleared or approved 
        by any regulatory authority for clinical use. The predictions generated are probability estimates based on 
        retrospective data and should <b>not</b> be used as the sole basis for clinical decisions. This tool is intended 
        to supplement, not replace, professional clinical judgment. All patient management decisions must be made by 
        qualified healthcare professionals based on comprehensive clinical evaluation.
    </p>
    <p style="font-size: 0.85rem; color: #9CA3AF; margin-top: 1rem;">
        © 2025 | Developed for: <i>Development and external validation of an interpretable machine learning model 
        for risk stratification of patients with chronotropic incompetence undergoing exercise stress SPECT-MPI: 
        A multicenter retrospective study</i>
    </p>
</div>
""", unsafe_allow_html=True)