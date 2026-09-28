from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Molecular Efficiency, MD Integration and CDU
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifact2_best_model.pkl"
MASTER_PATH = BASE_DIR / "artifact2_step2_full_features.csv"
METRICS_PATH = BASE_DIR / "artifact2_regression_metrics.csv"
RELEVANCE_PATH = BASE_DIR / "artifact2_feature_relevance.csv"
SHAP_PATH = BASE_DIR / "artifact2_shap_importance.csv"
CDU_PATH = BASE_DIR / "artifact2_CDU_candidates.csv"

MODEL_FEATURES = [
    "Ionization_Potential",
    "Chemical_Hardness",
    "Molecular_Weight",
    "Num_Aromatic_Ring",
    "Num_S",
    "E_ads_0K",
    "E_ads_298K",
    "E_ads_333K",
    "MSD_CoM_393K",
]

FEATURE_LABELS = {
    "Ionization_Potential": "Ionization Potential (eV)",
    "Chemical_Hardness": "Chemical Hardness (eV)",
    "Molecular_Weight": "Molecular Weight (g/mol)",
    "Num_Aromatic_Ring": "Number of Aromatic Rings",
    "Num_S": "Number of Sulfur Atoms",
    "E_ads_0K": "Adsorption Energy at 0 K (kcal/mol)",
    "E_ads_298K": "Adsorption Energy at 298 K (kcal/mol)",
    "E_ads_333K": "Adsorption Energy at 333 K (kcal/mol)",
    "MSD_CoM_393K": "MSD CoM at 393 K (Å²)",
}

st.set_page_config(
    page_title="Molecular Efficiency Prediction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading prediction model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("artifact2_best_model.pkl is missing.")
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_master():
    if not MASTER_PATH.exists():
        raise FileNotFoundError("artifact2_step2_full_features.csv is missing.")
    return pd.read_csv(MASTER_PATH)


@st.cache_data(show_spinner=False)
def load_optional_csv(path: Path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


try:
    model = load_model()
    master_df = load_master()
except Exception as exc:
    st.error("The prediction model and reference data could not be loaded.")
    st.exception(exc)
    st.info("Confirm that app.py, artifact2_best_model.pkl, and artifact2_step2_full_features.csv are in the same repository folder.")
    st.stop()

metrics_df = load_optional_csv(METRICS_PATH)
relevance_df = load_optional_csv(RELEVANCE_PATH)
shap_df = load_optional_csv(SHAP_PATH)
cdu_df = load_optional_csv(CDU_PATH)

missing_model_features = [feature for feature in MODEL_FEATURES if feature not in master_df.columns]
if missing_model_features:
    st.error("The master table is missing model features: " + ", ".join(missing_model_features))
    st.stop()


# ============================================================
# Prediction helpers
# ============================================================
def make_feature_frame(values: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        missing = [feature for feature in MODEL_FEATURES if feature not in values.columns]
        if missing:
                raise ValueError("Missing required model features: " + ", ".join(missing))
        frame = values[MODEL_FEATURES].copy()
    else:
        frame = pd.DataFrame([{feature: values[feature] for feature in MODEL_FEATURES}])

    for feature in MODEL_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    if frame.isna().any().any():
        invalid = frame.columns[frame.isna().any()].tolist()
        raise ValueError("All nine model inputs must be numeric and non-empty: " + ", ".join(invalid))
    return frame


def predict_frame(frame: pd.DataFrame) -> pd.DataFrame:
    predicted_ie = np.asarray(model.predict(frame), dtype=float)
    # IE% is physically bounded. Clipping here prevents an impossible display
    # value if a user supplies an out-of-distribution descriptor profile.
    predicted_ie = np.clip(predicted_ie, 0.0, 100.0)
    msd = frame["MSD_CoM_393K"].to_numpy(dtype=float)
    cdu_pass = msd <= 5.0
    result = pd.DataFrame(
        {
            "Predicted_IE_pct": predicted_ie,
            "CDU_Suitability": np.where(cdu_pass, "Pass — CDU suitable", "Fail — exceeds 5 Å² threshold"),
            "CDU_pass": cdu_pass,
            "MSD_CoM_393K": msd,
        },
        index=frame.index,
    )
    return result


def render_result_card(row: pd.Series):
    ie = float(row["Predicted_IE_pct"])
    cdu_pass = bool(row["CDU_pass"])
    status = "CDU suitable" if cdu_pass else "Not CDU suitable"
    color = "#198754" if cdu_pass else "#b42318"
    st.markdown(
        f"""
        <div style='border:1px solid #d7dde5;border-left:8px solid {color};border-radius:10px;padding:1.15rem 1.3rem;background:#ffffff;margin:.5rem 0 1rem 0;'>
          <div style='font-size:.78rem;color:#667085;text-transform:uppercase;letter-spacing:.08em;'>Prediction result</div>
          <div style='font-size:2rem;font-weight:750;color:#16324f;margin:.15rem 0 .55rem 0;'>Predicted IE%: {ie:.2f}%</div>
          <div style='display:flex;gap:2.8rem;flex-wrap:wrap;'>
            <div><span style='color:#667085;'>CDU suitability</span><br><b style='color:{color};'>{status}</b></div>
            <div><span style='color:#667085;'>MSD CoM at 393 K</span><br><b>{float(row["MSD_CoM_393K"]):,.2f} Å²</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_context(frame: pd.DataFrame):
    st.subheader("Model input summary")
    summary = frame.T.reset_index()
    summary.columns = ["Model feature", "Value"]
    summary["Descriptor"] = summary["Model feature"].map(FEATURE_LABELS)
    summary = summary[["Descriptor", "Model feature", "Value"]]
    st.dataframe(summary.style.format({"Value": "{:,.4f}"}), use_container_width=True, hide_index=True)


# ============================================================
# Sidebar navigation
# ============================================================
st.sidebar.title("Molecular Efficiency Prediction")
st.sidebar.caption("Molecular dynamics integration and CDU suitability screening")
page = st.sidebar.radio(
    "Navigate",
    ["Prediction", "Master Dataset", "Model Performance", "Feature Relevance", "CDU Candidates"],
)
st.sidebar.divider()
st.sidebar.markdown("**Verified model**")
st.sidebar.write("Random Forest Regressor")
st.sidebar.write("Nine model features")
st.sidebar.caption("CDU rule: MSD CoM at 393 K ≤ 5 Å²")


# ============================================================
# Prediction page
# ============================================================
if page == "Prediction":
    st.markdown(
        """
        <div style='text-align:center;padding:.6rem 0 1.1rem 0;'>
          <div style='font-size:2rem;font-weight:800;letter-spacing:.03em;color:#16324f;margin-top:.35rem;'>MOLECULAR EFFICIENCY PREDICTION</div>
          <div style='font-size:1.03rem;color:#475467;margin-top:.3rem;'>An intelligent computational application for predicting molecular inhibitor efficiency and screening candidate molecules for desorption suitability.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## INTRODUCTION")
    st.markdown(
        """
        **This project is owned by:**  
        **Okenye Isaac Ebipade**  
        Final Year Student, Department of Chemical Engineering  
        Federal University of Petroleum Resources, Effurun (FUPRE)  
        *In partial fulfillment of the requirements for the award of Bachelor of Engineering (B.Eng) Degree*

        ### MOLECULAR EFFICIENCY PREDICTION

        *An intelligent application for the machine-learning prediction of molecular inhibitor efficiency, integration of molecular-dynamics descriptors, and computational screening of candidate molecules for desorption suitability.*
        """
    )

    with st.expander("About this project and how to use this app", expanded=True):
        st.markdown(
            """
            ### Project overview

            This chemical-engineering project applies computational chemistry and supervised machine learning to the preliminary assessment of molecular inhibitor performance. Molecular and electronic descriptors are combined with adsorption-energy and molecular-dynamics information to estimate **inhibition efficiency (IE%)** and support the screening of candidate molecules before experimental or extended simulation work.

            The reference study contains 80 molecules distributed across four chemical groups. Molecular-dynamics results are available for eight molecules, while surrogate models provide estimates for the remaining entries. These estimated values are identified in the dataset and should be interpreted as directional screening information rather than confirmed simulation measurements.

            The prediction model is a Random Forest regression model. Computational desorption suitability is assessed independently using the stated criterion **MSD CoM at 393 K ≤ 5 Å²**. A passing result indicates suitability under this screening rule only; it does not replace laboratory validation, adsorption testing, or a complete molecular-dynamics investigation.

            ### How to use this application

            1. Select **Reference Molecule** to load the descriptors of one of the 80 study molecules.
            2. Select **Manual 9-Feature Entry** when all model inputs are available for a new case.
            3. Select **Batch CSV Upload** to process several molecules at once and download the prediction results.
            4. Review the predicted IE%, the MSD-based CDU decision, and the input-feature summary.
            5. Use the supporting pages to examine the reference dataset, model performance, feature relevance, and CDU candidate list.
            """
        )

    input_mode = st.radio("Input method", ["Reference Molecule", "Manual 9-Feature Entry", "Batch CSV Upload"], horizontal=True)

    if input_mode == "Batch CSV Upload":
        st.subheader("Batch prediction")
        st.write("Upload a CSV containing the nine model features below. Any extra columns are preserved in the downloadable output.")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded is not None:
            batch = pd.read_csv(uploaded)
            st.write(f"Loaded **{len(batch):,}** rows.")
            missing = [feature for feature in MODEL_FEATURES if feature not in batch.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                st.dataframe(batch.head(10), use_container_width=True, hide_index=True)
                if st.button("Run batch prediction", type="primary"):
                    try:
                        frame = make_feature_frame(batch)
                        predictions = predict_frame(frame)
                        export = pd.concat([batch.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
                        st.success(f"Predicted {len(export):,} rows.")
                        st.dataframe(export, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download prediction results",
                            export.to_csv(index=False).encode("utf-8"),
                            "artifact2_predictions.csv",
                            "text/csv",
                        )
                    except Exception as exc:
                        st.error(str(exc))

        template = master_df[MODEL_FEATURES].head(1).to_csv(index=False).encode("utf-8")
        st.download_button("Download 9-feature CSV template", template, "artifact2_9_feature_template.csv", "text/csv")

    else:
        selected_row = None
        if input_mode == "Reference Molecule":
            molecule_names = master_df["Molecule"].astype(str).tolist()
            selected_name = st.selectbox("Select a molecule from the 80-row reference dataset", molecule_names)
            selected_row = master_df.loc[master_df["Molecule"].astype(str) == selected_name].iloc[0]
            st.caption(
                f"Group: {selected_row.get('Group', '—')} · "
                f"MD source: {selected_row.get('MD_source', '—')} · "
                f"Observed IE%: {float(selected_row.get('IE_pct', np.nan)):.2f}%"
            )

        with st.form("artifact2_prediction_form"):
            st.subheader("Model input features")
            columns = st.columns(3)
            values = {}
            for i, feature in enumerate(MODEL_FEATURES):
                default = None if selected_row is None else float(selected_row[feature])
                with columns[i % 3]:
                    values[feature] = st.number_input(
                        FEATURE_LABELS[feature],
                        value=default,
                        format="%.6f",
                        help=f"Prediction-model feature: {feature}",
                    )
            submitted = st.form_submit_button("Run prediction", type="primary", use_container_width=True)

        if submitted:
            if any(value is None for value in values.values()):
                st.error("All nine model features are required.")
            else:
                try:
                    frame = make_feature_frame(values)
                    result = predict_frame(frame).iloc[0]
                    st.divider()
                    render_result_card(result)
                    render_feature_context(frame)
                    st.info("The main model predicts IE% from the nine supplied features. CDU suitability is determined independently by the 5 Å² MSD CoM threshold.")
                except Exception as exc:
                    st.error(str(exc))


# ============================================================
# Master dataset page
# ============================================================
elif page == "Master Dataset":
    st.title("Reference Dataset")
    st.write("The 80-row full-feature table after measured molecular-dynamics integration and surrogate filling.")
    st.warning("Rows marked `surrogate` contain estimated MD features. They should be treated as directional screening values, not as measured simulation results.")
    source_options = sorted(master_df.get("MD_source", pd.Series(dtype=str)).dropna().unique().tolist())
    selected_sources = st.multiselect("Filter by MD source", source_options, default=source_options)
    view = master_df[master_df["MD_source"].isin(selected_sources)].copy() if "MD_source" in master_df else master_df.copy()
    preferred = [
        "ID", "Molecule", "Group", "IE_pct", "IE_pct_predicted",
        "MD_source", "E_ads_0K", "E_ads_298K", "E_ads_333K", "MSD_CoM_393K", "DECISION"
    ]
    display_cols = [column for column in preferred if column in view.columns]
    st.dataframe(view[display_cols], use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(view)} of {len(master_df)} molecules.")
    st.download_button("Download master dataset", master_df.to_csv(index=False).encode("utf-8"), "artifact2_master_dataset.csv", "text/csv")


# ============================================================
# Performance page
# ============================================================
elif page == "Model Performance":
    st.title("Model Performance")
    st.warning("These are the reported cross-validation and held-out test metrics. The model was selected using Test Spearman ρ, the primary ranking metric for this study.")
    if metrics_df.empty:
        st.info("artifact2_regression_metrics.csv was not found.")
    else:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            **Interpretation note.** Test Spearman ρ is used as the primary selection metric because the objective is ranking molecular efficiency. The dataset is small, and 72 of the 80 rows contain surrogate-filled MD features; therefore, these results should be interpreted as exploratory rather than as proof of broad generalization.
            """
        )


# ============================================================
# Feature relevance page
# ============================================================
elif page == "Feature Relevance":
    st.title("Feature Relevance and Explainability")
    st.write("Correlation-based feature selection and global SHAP importance for the prediction model.")
    if not relevance_df.empty:
        st.subheader("Correlation-based feature relevance")
        st.caption("The notebook applied p < 0.10 to DFT/RDKit features. All four MD features were included unconditionally by design.")
        st.dataframe(relevance_df, use_container_width=True, hide_index=True)
    if not shap_df.empty:
        st.subheader("Global SHAP importance")
        st.caption("Mean absolute SHAP values describe the model's global feature importance. They explain model behavior; they do not establish causal chemistry or guarantee generalization.")
        st.dataframe(shap_df, use_container_width=True, hide_index=True)


# ============================================================
# CDU page
# ============================================================
else:
    st.title("CDU Suitability Candidates")
    st.write("Candidates satisfying the computational screening rule: **MSD CoM at 393 K ≤ 5 Å²**.")
    st.warning("Only one measured molecule is reported by the notebook as passing this threshold. Any surrogate-predicted pass is unverified and should be treated as directional.")
    if cdu_df.empty:
        st.info("artifact2_CDU_candidates.csv was not found.")
    else:
        st.dataframe(cdu_df, use_container_width=True, hide_index=True)
        st.download_button("Download CDU candidate table", cdu_df.to_csv(index=False).encode("utf-8"), "artifact2_CDU_candidates.csv", "text/csv")
