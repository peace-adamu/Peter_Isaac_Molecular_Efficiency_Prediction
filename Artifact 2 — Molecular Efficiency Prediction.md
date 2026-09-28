# Artifact 2 — Molecular Efficiency Prediction

This Streamlit application deploys the final Artifact 2 workflow from `Artifact2_Final_corr (1).ipynb`.

## Model and inputs

The uploaded `artifact2_best_model.pkl` is a pipeline containing a **Random Forest Regressor**. It predicts Inhibition Efficiency (`IE%`) from these nine Artifact 2 features:

- `Ionization_Potential`
- `Chemical_Hardness`
- `Molecular_Weight`
- `Num_Aromatic_Ring`
- `Num_S`
- `E_ads_0K`
- `E_ads_298K`
- `E_ads_333K`
- `MSD_CoM_393K`

CDU suitability is evaluated independently using the notebook rule:

```text
MSD_CoM_393K <= 5 Å²
```

## Deployment files

Upload these files to the repository root:

- `app.py`
- `requirements.txt`
- `runtime.txt`
- `artifact2_best_model.pkl`
- `artifact2_step2_full_features.csv`
- `artifact2_regression_metrics.csv`
- `artifact2_feature_relevance.csv`
- `artifact2_shap_importance.csv`
- `artifact2_CDU_candidates.csv`

The two surrogate artifacts are retained in the project for provenance and future expansion:

- `surrogates.pkl`
- `surrogate_scaler.pkl`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## App pages

- **Prediction:** reference-molecule selection, manual nine-feature entry, and batch CSV prediction/download.
- **Master Dataset:** 80-row Artifact 2 table with measured/surrogate MD source labels.
- **Model Performance:** notebook-reported CV and held-out metrics.
- **Feature Relevance:** correlation-filter table and global SHAP importance table.
- **CDU Candidates:** candidates satisfying the 5 Å² MSD CoM threshold.

## Interpretation note

The master table contains eight measured MD rows and 72 surrogate-filled rows. Surrogate values are estimates based on a very small measured set and should be treated as directional screening information, not as confirmed simulation results. The model metrics are exploratory and should not be presented as proof of broad generalization.
