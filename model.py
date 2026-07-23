"""
Smart Student Accommodation & Rental Price Estimator
Model Training Pipeline
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os
from pathlib import Path

RANDOM_STATE = 42
OUTPUT_DIR = Path("model_artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_dataset(n_samples: int = 2000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    np.random.seed(seed)

    square_footage = np.random.normal(850, 250, n_samples).clip(300, 2000).astype(int)
    bedrooms = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.30, 0.40, 0.20, 0.10])
    distance_to_campus = np.random.exponential(2.5, n_samples).clip(0.2, 10.0).round(1)
    wifi_included = np.random.choice([0, 1], size=n_samples, p=[0.20, 0.80])
    furnished = np.random.choice([0, 1], size=n_samples, p=[0.35, 0.65])
    parking_available = np.random.choice([0, 1], size=n_samples, p=[0.40, 0.60])
    utilities_included = np.random.choice([0, 1], size=n_samples, p=[0.30, 0.70])
    pet_friendly = np.random.choice([0, 1], size=n_samples, p=[0.70, 0.30])
    laundry = np.random.choice(
        ["in_unit", "in_building", "none"], size=n_samples, p=[0.20, 0.50, 0.30]
    )
    property_type = np.random.choice(
        ["apartment", "house", "studio", "shared_house"],
        size=n_samples,
        p=[0.40, 0.15, 0.25, 0.20],
    )

    base_price = (
        400
        + 0.35 * square_footage
        + 120 * bedrooms
        - 30 * distance_to_campus
        + 80 * wifi_included
        + 100 * furnished
        + 50 * parking_available
        + 60 * utilities_included
        + 40 * pet_friendly
    )

    laundry_effect = np.where(laundry == "in_unit", 50, np.where(laundry == "in_building", 20, 0))
    type_effect = np.where(
        property_type == "apartment",
        30,
        np.where(
            property_type == "house",
            60,
            np.where(property_type == "studio", -40, 10),
        ),
    )

    noise = np.random.normal(0, 45, n_samples)
    monthly_rent = (base_price + laundry_effect + type_effect + noise).clip(350, 2500).round(0)

    df = pd.DataFrame(
        {
            "square_footage": square_footage,
            "bedrooms": bedrooms,
            "distance_to_campus_km": distance_to_campus,
            "wifi_included": wifi_included,
            "furnished": furnished,
            "parking_available": parking_available,
            "utilities_included": utilities_included,
            "pet_friendly": pet_friendly,
            "laundry": laundry,
            "property_type": property_type,
            "monthly_rent": monthly_rent,
        }
    )
    return df


def build_pipeline(numeric_features, categorical_features):
    from sklearn.preprocessing import PolynomialFeatures

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    gb = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", VotingRegressor(
                estimators=[("rf", rf), ("gb", gb)],
                weights=[0.4, 0.6],
                n_jobs=-1,
            )),
        ]
    )
    return pipeline


def evaluate(y_true, y_pred, prefix=""):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results = {
        f"{prefix}MSE": round(mse, 2),
        f"{prefix}RMSE": round(rmse, 2),
        f"{prefix}MAE": round(mae, 2),
        f"{prefix}R2": round(r2, 4),
    }
    return results


def main():
    print("=" * 60)
    print(" Smart Student Accommodation & Rental Price Estimator")
    print(" Model Training Pipeline")
    print("=" * 60)

    print("\n[1/5] Generating synthetic dataset...")
    df = generate_dataset(n_samples=2000)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Monthly rent range: ${df['monthly_rent'].min():.0f} - ${df['monthly_rent'].max():.0f}")
    df.to_csv(OUTPUT_DIR / "dataset.csv", index=False)
    print(f"  Saved dataset to {OUTPUT_DIR / 'dataset.csv'}")

    print("\n[2/5] Preparing features and target...")
    target = "monthly_rent"
    numeric_features = [
        "square_footage",
        "bedrooms",
        "distance_to_campus_km",
    ]
    categorical_features = [
        "wifi_included",
        "furnished",
        "parking_available",
        "utilities_included",
        "pet_friendly",
        "laundry",
        "property_type",
    ]
    X = df[numeric_features + categorical_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"  Train size: {len(X_train)} | Test size: {len(X_test)}")

    print("\n[3/5] Training Random Forest Regressor...")
    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)
    print("  Training complete.")

    print("\n[4/5] Evaluating model...")
    y_pred_test = pipeline.predict(X_test)
    test_metrics = evaluate(y_test, y_pred_test, prefix="test_")
    print(f"  Test MSE:  {test_metrics['test_MSE']}")
    print(f"  Test RMSE: {test_metrics['test_RMSE']}")
    print(f"  Test MAE:  {test_metrics['test_MAE']}")
    print(f"  Test R2:   {test_metrics['test_R2']}")

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    print(f"  5-Fold CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    train_metrics = evaluate(y_train, pipeline.predict(X_train), prefix="train_")
    metrics = {**train_metrics, **test_metrics, "cv_r2_mean": round(cv_scores.mean(), 4), "cv_r2_std": round(cv_scores.std(), 4)}
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {OUTPUT_DIR / 'metrics.json'}")

    print("\n[5/5] Saving model artifacts...")
    joblib.dump(pipeline, OUTPUT_DIR / "rental_model.joblib")
    print(f"  Model saved to {OUTPUT_DIR / 'rental_model.joblib'}")

    sample_input = X_test.iloc[:5].to_dict(orient="records")
    with open(OUTPUT_DIR / "sample_input.json", "w") as f:
        json.dump(sample_input, f, indent=2)
    print(f"  Sample input saved to {OUTPUT_DIR / 'sample_input.json'}")

    print("\n" + "=" * 60)
    print(" Pipeline complete. All artifacts saved in model_artifacts/")
    print("=" * 60)


if __name__ == "__main__":
    main()
