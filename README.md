# Smart Student Accommodation & Rental Price Estimator

A machine learning-powered web application that predicts monthly rental prices for student accommodations based on property features, amenities, and location data.

## Project Overview

Finding affordable student housing is a common challenge. This project uses a **Random Forest Regressor** trained on a realistic synthetic dataset to estimate monthly rental prices across various property configurations. Users interact with a clean Streamlit dashboard, adjusting property features to receive instant price predictions.

### Key Features

- **Synthetic Dataset Generation** -- Realistic rental data with 10 features and 2,000 samples
- **ML Pipeline** -- Scikit-Learn `Pipeline` with preprocessing and model for clean inference
- **Streamlit Dashboard** -- Interactive UI with sliders, selectboxes, and real-time predictions
- **Model Artifacts** -- Serialized model, metrics, and dataset saved for reproducibility
- **Feature Importance** -- Visual breakdown of which factors drive rental prices

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | Scikit-Learn |
| Web Framework | Streamlit |
| Data Handling | Pandas, NumPy |
| Serialization | Joblib |

## Project Structure

```
smart-student-rental-estimator/
├── model.py                  # Dataset generation, training, evaluation
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── model_artifacts/          # Generated after running model.py
    ├── rental_model.joblib   # Serialized trained model
    ├── metrics.json          # Evaluation metrics
    ├── dataset.csv           # Synthetic dataset
    └── sample_input.json     # Sample input for reference
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smart-student-rental-estimator.git
   cd smart-student-rental-estimator
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run Locally

### Step 1 -- Train the Model

```bash
python model.py
```

This generates the synthetic dataset, trains a Random Forest Regressor, evaluates performance, and saves all artifacts to `model_artifacts/`.

### Step 2 -- Launch the Web App

```bash
streamlit run app.py
```

Open the local URL (default: `http://localhost:8501`) in your browser.

### Step 3 -- Use the Dashboard

1. Adjust property features in the sidebar
2. Click **Predict Monthly Rent**
3. View the estimated price, per-sq-ft cost, and annual breakdown
4. Review model performance metrics and feature importance chart

## Model Performance

| Metric | Value |
|---|---|
| Test R² | 0.8867 |
| Test RMSE | $58.39 |
| Test MAE | $46.97 |
| 5-Fold CV R² | 0.8888 |

## Features Used

| Feature | Type | Description |
|---|---|---|
| `square_footage` | Numeric | Total area in sq ft |
| `bedrooms` | Numeric | Number of bedrooms (1-4) |
| `distance_to_campus_km` | Numeric | Distance to nearest campus (km) |
| `wifi_included` | Binary | Wi-Fi availability |
| `furnished` | Binary | Furnished status |
| `parking_available` | Binary | Parking availability |
| `utilities_included` | Binary | Utilities in rent |
| `pet_friendly` | Binary | Pet policy |
| `laundry` | Categorical | in_unit / in_building / none |
| `property_type` | Categorical | apartment / house / studio / shared_house |

## Future Enhancements

- **Real-world data integration** -- Scrape or connect to actual rental listing APIs (Zillow, Craigslist)
- **Geospatial features** -- Add latitude/longitude for neighborhood-level pricing
- **XGBoost / LightGBM comparison** -- Benchmark against gradient boosting models
- **Neural network model** -- Experiment with PyTorch/TensorFlow regressors
- **Docker deployment** -- Containerize the app for cloud deployment
- **CI/CD pipeline** -- GitHub Actions for automated testing and deployment
- **Multi-city support** -- Extend dataset to cover multiple metropolitan areas
- **Time-series pricing** -- Track rental price trends over semesters
- **Unit tests** -- Add pytest suite for model and API validation

## License

MIT License. See `LICENSE` for details.
