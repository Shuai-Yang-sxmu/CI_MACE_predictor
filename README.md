# CI_MACE_predictor

An interactive web-based research tool for individualized MACE risk stratification in patients with chronotropic incompetence undergoing exercise stress SPECT myocardial perfusion imaging (SPECT-MPI).

The calculator implements the frozen final XGBoost survival model developed in a derivation cohort of 765 patients and externally validated in an independent cohort of 295 patients.

## Model Predictors

The final model uses five predictors:

- Dyslipidaemia
- Diabetes
- Summed Motion Score (SMS)
- Total Perfusion Deficit (TPD)
- Heart Rate Recovery at 3 minutes (HRR3)

## Calculator Outputs

The calculator provides:

- Low-, medium-, or high-risk stratification
- Estimated 3-year and 5-year MACE probabilities
- Individualized TreeSHAP-based feature contributions
- Personalized MACE-free survival projection

## Online Calculator

https://cimacepredictor.streamlit.app/

## Intended Use

This calculator is intended for patients with chronotropic incompetence undergoing exercise stress SPECT-MPI who meet the study eligibility criteria. It should not be applied to patients with prior myocardial infarction or prior coronary revascularization.

## Disclaimer

This tool is provided for research and educational purposes only. It has not been cleared or approved by any regulatory authority for clinical use. Model predictions should not be used as the sole basis for clinical decision-making and should be interpreted together with comprehensive clinical assessment.
