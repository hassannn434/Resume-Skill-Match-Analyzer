"""
Smart Student Accommodation & Rental Price Estimator
Streamlit Web Application
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path

MODEL_PATH = Path("model_artifacts/rental_model.joblib")
METRICS_PATH = Path("model_artifacts/metrics.json")

st.set_page_config(
    page_title="Student Rental Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { text-align: center; padding: 1rem 0; }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 12px; text-align: center;
        color: white; margin: 1rem 0;
    }
    .prediction-price { font-size: 3rem; font-weight: 700; margin: 0.5rem 0; }
    .metric-card {
        background: #f8f9fa; padding: 1rem; border-radius: 8px;
        text-align: center; border-left: 4px solid #667eea;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def main():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("Smart Student Accommodation & Rental Price Estimator")
    st.markdown("Predict monthly rental prices for student accommodations based on property features.")
    st.markdown("</div>", unsafe_allow_html=True)

    pipeline = load_model()
    metrics = load_metrics()

    if pipeline is None:
        st.error(
            "Model not found. Please run `python model.py` first to train and save the model."
        )
        st.stop()

    st.sidebar.header("Property Features")
    st.sidebar.markdown("Adjust the parameters below to get a price estimate.")

    st.sidebar.subheader("Basic Info")
    square_footage = st.sidebar.slider(
        "Square Footage", min_value=300, max_value=2000, value=750, step=50,
        help="Total area of the rental unit in square feet."
    )
    bedrooms = st.sidebar.selectbox(
        "Bedrooms", options=[1, 2, 3, 4], index=1,
        help="Number of bedrooms in the unit."
    )
    distance = st.sidebar.slider(
        "Distance to Campus (km)", min_value=0.2, max_value=10.0, value=2.0, step=0.1,
        help="Distance from the nearest campus entrance."
    )

    st.sidebar.subheader("Amenities")
    wifi_included = st.sidebar.selectbox(
        "Wi-Fi Included", options=["Yes", "No"], index=0,
    )
    furnished = st.sidebar.selectbox(
        "Furnished", options=["Yes", "No"], index=0,
    )
    parking_available = st.sidebar.selectbox(
        "Parking Available", options=["Yes", "No"], index=0,
    )
    utilities_included = st.sidebar.selectbox(
        "Utilities Included", options=["Yes", "No"], index=0,
    )
    pet_friendly = st.sidebar.selectbox(
        "Pet Friendly", options=["Yes", "No"], index=1,
    )

    st.sidebar.subheader("Property Details")
    laundry = st.sidebar.selectbox(
        "Laundry", options=["in_unit", "in_building", "none"], index=1,
        help="Laundry facility availability."
    )
    property_type = st.sidebar.selectbox(
        "Property Type", options=["apartment", "house", "studio", "shared_house"], index=0,
    )

    input_data = pd.DataFrame(
        {
            "square_footage": [square_footage],
            "bedrooms": [bedrooms],
            "distance_to_campus_km": [distance],
            "wifi_included": [1 if wifi_included == "Yes" else 0],
            "furnished": [1 if furnished == "Yes" else 0],
            "parking_available": [1 if parking_available == "Yes" else 0],
            "utilities_included": [1 if utilities_included == "Yes" else 0],
            "pet_friendly": [1 if pet_friendly == "Yes" else 0],
            "laundry": [laundry],
            "property_type": [property_type],
        }
    )

    col_predict, col_info = st.columns([1, 1])

    with col_predict:
        if st.button("Predict Monthly Rent", type="primary", use_container_width=True):
            prediction = pipeline.predict(input_data)[0]
            st.markdown(
                f"""
                <div class="prediction-box">
                    <p style="font-size:1.1rem; margin-bottom:0;">Estimated Monthly Rent</p>
                    <p class="prediction-price">${prediction:,.0f}</p>
                    <p style="font-size:0.9rem; opacity:0.8;">per month</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            price_sqft = prediction / square_footage
            st.success(
                f"**Price per sq ft:** ${price_sqft:.2f}/sq ft | "
                f"**Annual cost:** ${prediction * 12:,.0f}"
            )

    with col_info:
        st.subheader("Property Summary")
        summary_data = {
            "Feature": [
                "Square Footage",
                "Bedrooms",
                "Distance to Campus",
                "Wi-Fi",
                "Furnished",
                "Parking",
                "Utilities Included",
                "Pet Friendly",
                "Laundry",
                "Property Type",
            ],
            "Value": [
                f"{square_footage} sq ft",
                bedrooms,
                f"{distance} km",
                wifi_included,
                furnished,
                parking_available,
                utilities_included,
                pet_friendly,
                laundry.replace("_", " ").title(),
                property_type.replace("_", " ").title(),
            ],
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    if metrics:
        st.markdown("---")
        st.subheader("Model Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test R²", f"{metrics['test_R2']:.4f}")
        m2.metric("Test RMSE", f"${metrics['test_RMSE']:,.0f}")
        m3.metric("Test MAE", f"${metrics['test_MAE']:,.0f}")
        m4.metric("CV R² (mean)", f"{metrics['cv_r2_mean']:.4f}")

    st.markdown("---")
    st.subheader("Feature Importance")
    try:
        voting_reg = pipeline.named_steps["regressor"]
        feature_names_num = ["square_footage", "bedrooms", "distance_to_campus_km"]
        ohe = pipeline.named_steps["preprocessor"].transformers_[1][1]
        feature_names_cat = list(ohe.get_feature_names_out(
            ["wifi_included", "furnished", "parking_available", "utilities_included",
             "pet_friendly", "laundry", "property_type"]
        ))
        all_features = feature_names_num + feature_names_cat

        gb_model = voting_reg.named_estimators_["gb"]
        importances = gb_model.feature_importances_
        imp_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=False)
        st.bar_chart(imp_df.set_index("Feature")["Importance"], horizontal=True)
    except Exception as e:
        st.info(f"Feature importance display unavailable: {e}")

    st.markdown("---")
    st.caption("Built with Scikit-Learn and Streamlit | Smart Student Rental Price Estimator")


if __name__ == "__main__":
    main()
