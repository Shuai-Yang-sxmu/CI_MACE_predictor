from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb


warnings.filterwarnings("ignore", category=FutureWarning)


# ==================== Page configuration ====================
st.set_page_config(
    page_title="MACE Risk Predictor for Chronotropic Incompetence",
    page_icon="🫀",
    layout="wide",
)


# ==================== Frozen model configuration ====================
APP_DIR = Path(__file__).resolve().parent
SPEC_PATH = APP_DIR / "web_model_specification.json"

FEATURE_NAMES = ["Dyslipidaemia", "Diabetes", "SMS", "TPD", "HRR3"]
TIME_3YR = 36.0
TIME_5YR = 60.0


# ==================== Custom CSS ====================
# The visual hierarchy intentionally follows the original research prototype.
st.markdown(
    """
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
        margin-bottom: 16px !important;
    }
    .eligibility-note {
        font-size: 14px !important;
        color: #5B4A2F !important;
        background: #FFF8E1 !important;
        border-left: 4px solid #F59E0B !important;
        border-radius: 6px !important;
        padding: 10px 14px !important;
        margin: 6px auto 18px auto !important;
        max-width: 1180px !important;
        line-height: 1.55 !important;
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
        min-height: 162px;
        display: flex;
        flex-direction: column;
        justify-content: center;
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
    .technical-note {
        font-size: 12.5px !important;
        color: #7C8593 !important;
        line-height: 1.45 !important;
        margin-top: 4px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==================== Model loading and QA ====================
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@st.cache_resource
def load_frozen_model():
    """Load the frozen final XGBoost model and Breslow baseline hazard."""
    with SPEC_PATH.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)

    if spec["features"] != FEATURE_NAMES:
        raise RuntimeError(
            f"Unexpected feature order in deployment specification: {spec['features']}"
        )

    model_path = APP_DIR / spec["model_artifact"]
    baseline_path = APP_DIR / spec["baseline_artifact"]

    for path in (model_path, baseline_path):
        if not path.exists():
            raise FileNotFoundError(f"Required deployment artifact not found: {path.name}")

    if sha256_file(model_path) != spec["model_sha256"]:
        raise RuntimeError("XGBoost model artifact failed the SHA-256 integrity check.")
    if sha256_file(baseline_path) != spec["baseline_sha256"]:
        raise RuntimeError("Breslow baseline artifact failed the SHA-256 integrity check.")

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    if model.get_booster().num_features() != len(FEATURE_NAMES):
        raise RuntimeError("Model feature count does not match the deployment specification.")

    baseline = pd.read_csv(baseline_path).sort_values("EventTime").reset_index(drop=True)
    required_columns = {
        "EventTime",
        "BaselineHazardIncrement",
        "CumulativeBaselineHazard",
    }
    if not required_columns.issubset(baseline.columns):
        raise RuntimeError("Breslow baseline file has unexpected columns.")
    if baseline["EventTime"].duplicated().any():
        raise RuntimeError("Breslow baseline contains duplicate event times.")
    if (baseline["BaselineHazardIncrement"] < 0).any():
        raise RuntimeError("Breslow baseline contains a negative hazard increment.")
    if (np.diff(baseline["CumulativeBaselineHazard"].to_numpy()) < -1e-12).any():
        raise RuntimeError("Breslow cumulative baseline hazard is not monotone.")

    return model, baseline, spec


# ==================== Prediction helpers ====================
def prepare_model_input(
    dyslipidaemia: int,
    diabetes: int,
    sms: int,
    tpd: int,
    hrr3: int,
    spec: dict,
) -> pd.DataFrame:
    """Apply the frozen final-model preprocessing required by the five predictors."""
    hrr3_cfg = spec["preprocessing"]["HRR3"]
    hrr3_used = float(
        np.clip(
            float(hrr3),
            float(hrr3_cfg["winsorize_lower"]),
            float(hrr3_cfg["winsorize_upper"]),
        )
    )

    input_data = pd.DataFrame(
        {
            "Dyslipidaemia": [int(dyslipidaemia)],
            "Diabetes": [int(diabetes)],
            "SMS": [float(sms)],
            "TPD": [float(tpd)],
            "HRR3": [hrr3_used],
        }
    )[FEATURE_NAMES]

    if not np.isfinite(input_data.to_numpy(dtype=float)).all():
        raise ValueError("All predictor values must be finite.")
    return input_data


def cumulative_baseline_hazard(baseline: pd.DataFrame, time_point: float) -> float:
    eligible = baseline.loc[
        baseline["EventTime"] <= float(time_point), "CumulativeBaselineHazard"
    ]
    return 0.0 if eligible.empty else float(eligible.iloc[-1])


def calculate_survival_probability(
    raw_score: float, baseline: pd.DataFrame, time_point: float
) -> float:
    """S(t|x) = exp[-H0(t) * exp(eta)] using the frozen Breslow baseline."""
    h0 = cumulative_baseline_hazard(baseline, time_point)
    relative_hazard = float(np.exp(np.clip(raw_score, -30.0, 30.0)))
    return float(np.exp(-h0 * relative_hazard))


def get_risk_group(score: float, spec: dict):
    cutoff_1 = float(spec["risk_group_cutpoints"]["low_to_medium"])
    cutoff_2 = float(spec["risk_group_cutpoints"]["medium_to_high"])

    if score <= cutoff_1:
        return "Low-risk", "🟢", "#059669", "#D1FAE5", "#10B981"
    if score <= cutoff_2:
        return "Medium-risk", "🟡", "#D97706", "#FEF3C7", "#F59E0B"
    return "High-risk", "🔴", "#DC2626", "#FEE2E2", "#EF4444"


def create_tree_shap_explanation(model: xgb.XGBRegressor, input_data: pd.DataFrame):
    """Use native XGBoost TreeSHAP values; SHAP is used only for waterfall rendering."""
    booster = model.get_booster()
    contributions = booster.predict(xgb.DMatrix(input_data), pred_contribs=True)[0]

    feature_values = np.asarray(contributions[:-1], dtype=float)
    base_value = float(contributions[-1])
    shap_score = float(np.sum(contributions, dtype=float))

    explanation = shap.Explanation(
        values=feature_values,
        base_values=base_value,
        data=input_data.iloc[0].to_numpy(dtype=float),
        feature_names=FEATURE_NAMES,
    )
    return explanation, shap_score


def personalized_survival_curve(raw_score: float, baseline: pd.DataFrame):
    curve = baseline.loc[baseline["EventTime"] <= TIME_5YR].copy()
    times = np.r_[0.0, curve["EventTime"].to_numpy(dtype=float)]
    h0 = np.r_[0.0, curve["CumulativeBaselineHazard"].to_numpy(dtype=float)]
    relative_hazard = float(np.exp(np.clip(raw_score, -30.0, 30.0)))
    probabilities = np.exp(-h0 * relative_hazard)
    return times, probabilities


try:
    model, baseline, spec = load_frozen_model()
    model_loaded = True
except Exception as exc:
    model_loaded = False
    st.error(f"Model loading failed: {exc}")


# ==================== Sidebar inputs ====================
st.sidebar.header("Patient Parameters")
st.sidebar.markdown("Enter the five clinical, exercise, and imaging predictors below:")

st.sidebar.markdown("#### Clinical History")

dyslipidaemia = st.sidebar.selectbox(
    "Dyslipidaemia",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
    help="Diagnosis or history of dyslipidaemia",
)

diabetes = st.sidebar.selectbox(
    "Diabetes Mellitus",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
    help="Diagnosis or history of diabetes mellitus",
)

st.sidebar.markdown("#### SPECT-MPI Parameters")

sms = st.sidebar.slider(
    "SMS (Summed Motion Score)",
    min_value=0,
    max_value=20,
    value=0,
    step=1,
    help="Extent and severity of wall motion abnormalities from gated SPECT-MPI",
)

tpd = st.sidebar.slider(
    "TPD (Total Perfusion Deficit, %)",
    min_value=0,
    max_value=20,
    value=0,
    step=1,
    help="Global measure of myocardial perfusion severity from SPECT-MPI",
)

st.sidebar.markdown("#### Exercise Parameter")

hrr3 = st.sidebar.slider(
    "HRR3 (Heart Rate Recovery at 3 min, bpm)",
    min_value=10,
    max_value=70,
    value=30,
    step=1,
    help="Heart rate reduction at 3 minutes into recovery phase",
)

st.sidebar.markdown("---")

predict_button = st.sidebar.button(
    "Calculate Risk",
    type="primary",
    use_container_width=True,
    disabled=not model_loaded,
)


# ==================== Main title ====================
st.markdown(
    '<h1 class="main-title">MACE Risk Stratification for Chronotropic Incompetence</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">An explainable XGBoost survival model developed and externally '
    'validated for predicting MACE in patients with chronotropic incompetence undergoing '
    'exercise stress SPECT-MPI.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="research-note">Designed for Research Verification &amp; Educational Purposes</p>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="eligibility-note">
        <b>Intended-use population:</b> Patients with chronotropic incompetence undergoing
        exercise stress SPECT-MPI who meet the study eligibility criteria. This model should
        <b>not</b> be applied to patients with prior myocardial infarction or prior coronary
        revascularization.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# ==================== Prediction results ====================
if predict_button and model_loaded:
    input_data = prepare_model_input(
        dyslipidaemia=dyslipidaemia,
        diabetes=diabetes,
        sms=sms,
        tpd=tpd,
        hrr3=hrr3,
        spec=spec,
    )

    # The locked X-tile cut-points are defined on the XGBoost output margin.
    risk_score = float(model.predict(input_data, output_margin=True)[0])

    risk_group, risk_emoji, risk_color, bg_color, border_color = get_risk_group(
        risk_score, spec
    )

    prob_3yr_survival = calculate_survival_probability(
        risk_score, baseline, TIME_3YR
    )
    prob_5yr_survival = calculate_survival_probability(
        risk_score, baseline, TIME_5YR
    )

    prob_3yr_mace = (1.0 - prob_3yr_survival) * 100.0
    prob_5yr_mace = (1.0 - prob_5yr_survival) * 100.0

    # ==================== Prognostic Assessment Report ====================
    st.markdown(
        '<p class="section-header">📊 Prognostic Assessment Report</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="result-card" style="background: linear-gradient(135deg, {bg_color} 0%, white 100%); border: 3px solid {border_color};">
                <p class="metric-label">Risk Stratum</p>
                <p class="metric-value-large" style="color: {risk_color};">{risk_emoji} {risk_group}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="result-card" style="background: linear-gradient(135deg, #F0F9FF 0%, white 100%); border: 3px solid #3B82F6;">
                <p class="metric-label">MACE Probability</p>
                <p class="prob-item"><span style="color: #059669;">●</span> <b>3-Year:</b> <span style="color: #059669; font-weight: 700;">{prob_3yr_mace:.1f}%</span></p>
                <p class="prob-item"><span style="color: #DC2626;">●</span> <b>5-Year:</b> <span style="color: #DC2626; font-weight: 700;">{prob_5yr_mace:.1f}%</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==================== Risk Interpretation ====================
    st.markdown("##### 📝 Risk Interpretation")

    if risk_group == "Low-risk":
        interpretation = (
            f"**Favorable Prognosis.** Based on the input parameters, this patient is "
            f"classified as **low-risk** with a 3-year MACE probability of "
            f"{prob_3yr_mace:.1f}% and a 5-year probability of {prob_5yr_mace:.1f}%. "
            "This model-based estimate should be interpreted together with the patient's "
            "overall clinical assessment."
        )
        st.success(interpretation, icon="✅")
    elif risk_group == "Medium-risk":
        interpretation = (
            f"**Intermediate Risk Profile.** Based on the input parameters, this patient "
            f"is classified as **medium-risk** with a 3-year MACE probability of "
            f"{prob_3yr_mace:.1f}% and a 5-year probability of {prob_5yr_mace:.1f}%. "
            "This model-based estimate should be interpreted together with the patient's "
            "overall clinical assessment."
        )
        st.warning(interpretation, icon="⚠️")
    else:
        interpretation = (
            f"**Elevated Risk Profile.** Based on the input parameters, this patient is "
            f"classified as **high-risk** with a 3-year MACE probability of "
            f"{prob_3yr_mace:.1f}% and a 5-year probability of {prob_5yr_mace:.1f}%. "
            "This model-based estimate should be interpreted together with the patient's "
            "overall clinical assessment."
        )
        st.error(interpretation, icon="🚨")

    st.markdown("---")

    # ==================== Feature contribution + survival curve ====================
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            '<p class="section-header">🔍 Feature Contribution Analysis</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Breakdown of how individual predictors push the XGBoost risk score towards "
            "higher (red) or lower (blue) values relative to the model baseline."
        )

        with st.spinner("Calculating TreeSHAP contributions..."):
            explanation, shap_score = create_tree_shap_explanation(model, input_data)

        plt.figure(figsize=(8, 5))
        shap.plots.waterfall(
            explanation,
            max_display=len(FEATURE_NAMES),
            show=False,
        )
        fig1 = plt.gcf()
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        st.markdown(
            '<p class="technical-note">TreeSHAP values represent contributions to the '
            'raw XGBoost log-risk margin and should not be interpreted as changes in '
            'absolute MACE probability.</p>',
            unsafe_allow_html=True,
        )

        if abs(shap_score - risk_score) > 1e-4:
            st.warning(
                "Internal TreeSHAP additivity check exceeded the expected numerical tolerance."
            )

    with col_right:
        st.markdown(
            '<p class="section-header">📈 Personalized Survival Projection</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Estimated probability of remaining free from MACE over a 5-year follow-up "
            "horizon based on the frozen final XGBoost model."
        )

        times_plot, probs_plot = personalized_survival_curve(risk_score, baseline)

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.step(
            times_plot,
            probs_plot,
            where="post",
            color="#1E40AF",
            linewidth=2.5,
            label="MACE-free Survival",
        )

        ax3.scatter(
            [TIME_3YR],
            [prob_3yr_survival],
            color="#2f8e2f",
            s=100,
            zorder=5,
            marker="o",
        )
        ax3.scatter(
            [TIME_5YR],
            [prob_5yr_survival],
            color="#bf1b1b",
            s=100,
            zorder=5,
            marker="o",
        )

        ax3.hlines(
            prob_3yr_survival,
            xmin=0,
            xmax=TIME_3YR,
            color="#2f8e2f",
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
        )
        ax3.hlines(
            prob_5yr_survival,
            xmin=0,
            xmax=TIME_5YR,
            color="#bf1b1b",
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
        )
        ax3.vlines(
            TIME_3YR,
            ymin=0,
            ymax=prob_3yr_survival,
            color="#2f8e2f",
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
        )
        ax3.vlines(
            TIME_5YR,
            ymin=0,
            ymax=prob_5yr_survival,
            color="#bf1b1b",
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
        )

        ax3.text(
            TIME_3YR + 1.5,
            min(prob_3yr_survival + 0.02, 0.995),
            f"3-Year: {prob_3yr_survival:.1%}",
            fontsize=11,
            color="#2f8e2f",
            fontweight="bold",
            va="bottom",
        )
        ax3.text(
            TIME_5YR - 1.0,
            max(prob_5yr_survival - 0.025, 0.02),
            f"5-Year: {prob_5yr_survival:.1%}",
            fontsize=11,
            color="#bf1b1b",
            fontweight="bold",
            va="top",
            ha="right",
        )

        ax3.set_xlabel("Time (Months)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Survival Probability", fontsize=12, fontweight="bold")
        ax3.set_ylim([0, 1.01])
        ax3.set_xlim([0, 62])
        ax3.set_xticks([0, 12, 24, 36, 48, 60])
        ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax3.grid(True, linestyle="--", alpha=0.4)
        ax3.legend(loc="lower left", fontsize=10)

        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)


elif not predict_button:
    # ==================== Default page ====================
    st.info(
        "👈 **Action Required:** Please enter patient parameters in the sidebar and click "
        "**'Calculate Risk'** to generate the prognostic report.",
        icon="ℹ️",
    )

    st.markdown("---")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown(
            """
            ### 📖 About This Tool

            This web-based research prototype implements the frozen final **XGBoost survival model**
            to estimate the probability of Major Adverse Cardiovascular Events (MACE) in patients
            with **chronotropic incompetence** undergoing exercise stress SPECT myocardial
            perfusion imaging.

            **Key Features:**
            - **Derivation Cohort:** Developed on 765 patients from the derivation center
            - **Validation Cohort:** Externally validated on 295 patients from an independent center
            - **Model Status:** Frozen final model; no runtime refitting or recalibration
            """
        )

        st.markdown(
            """
            ### 📋 Model Predictors

            | Predictor | Description |
            |-----------|-------------|
            | **TPD** | Total Perfusion Deficit (%) from SPECT-MPI |
            | **HRR3** | Heart Rate Recovery at 3 minutes (bpm) |
            | **SMS** | Summed Motion Score from gated SPECT |
            | **Dyslipidaemia** | History of dyslipidaemia |
            | **Diabetes** | History of diabetes mellitus |
            """
        )

    with col_info2:
        st.markdown(
            """
            ### 📊 Risk Stratification

            Patients are stratified into three prognostic groups using the **XGBoost raw risk score**
            and the cut-points prespecified from the derivation cohort:

            - 🟢 **Low-risk** (Score ≤ −0.480) — Favorable prognostic stratum
            - 🟡 **Medium-risk** (−0.480 < Score ≤ 1.043) — Intermediate prognostic stratum
            - 🔴 **High-risk** (Score > 1.043) — High prognostic stratum
            """
        )

        st.markdown(
            """
            ### 📈 Output

            The calculator provides the following model outputs:

            1. **Risk Stratum** — Classification into Low, Medium, or High risk
            2. **MACE Probability** — Estimated cumulative MACE risk at 3 and 5 years
            3. **Risk Interpretation** — Model-based summary of the predicted risk level
            4. **Feature Contribution** — Patient-level TreeSHAP explanation
            5. **Survival Projection** — Predicted MACE-free survival through 5 years
            """
        )


# ==================== Footer ====================
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #6B7280; font-size: 0.9rem; padding: 1rem 0; line-height: 1.7;">
    <p style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.8rem; color: #374151;">
        Research Risk Stratification Tool | For Research Use Only
    </p>
    <p style="margin: 1rem auto; padding: 0 1rem; max-width: 1000px; text-align: justify;">
        ⚠️ <b>DISCLAIMER:</b> This tool is a <b>research prototype</b> developed to demonstrate
        the application of explainable machine learning in cardiovascular risk stratification.
        It has <b>not</b> been cleared or approved by any regulatory authority for clinical use.
        The predictions generated are probability estimates based on retrospective data and
        should <b>not</b> be used as the sole basis for clinical decisions. The model should not
        be applied outside the study eligibility criteria, including to patients with prior
        myocardial infarction or prior coronary revascularization. All patient-management
        decisions must be based on comprehensive clinical evaluation by qualified healthcare
        professionals.
    </p>
    <p style="font-size: 0.85rem; color: #9CA3AF; margin-top: 1rem;">
        © 2026 | Developed for: <i>Development and external validation of an explainable machine
        learning model for risk stratification of patients with chronotropic incompetence
        undergoing exercise stress SPECT-MPI: a dual-center study</i>
    </p>
</div>
""",
    unsafe_allow_html=True,
)

