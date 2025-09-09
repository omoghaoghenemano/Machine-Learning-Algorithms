cd ml-templates; $env:PYTHONPATH="."; python homework1/task2.py
pip install spacy
python -m spacy download en_core_web_sm

## Documentation Report

### Feature Engineering
- Handles missing data robustly for both categorical and numeric columns.
- Extracts advanced features from job titles (TF-IDF n-grams), company size, remote ratio, and region matching.
- Encodes categorical variables with rare category grouping and label encoding.
- Adds interaction features (e.g., job_title_x_company_size, experience_level_x_employment_type).
- Standardizes numeric features and creates polynomial features.
- No clustering (KMeans) or sklearn clustering is used, in compliance with requirements.

### Model Pipeline
- Uses a custom `FeatureExtractor` for all feature engineering and encoding.
- Trains a linear regression model (log-target regression) using only allowed libraries.
- Performs grid search over n-gram and feature parameters, then ensembles the top 3 models for robust predictions.
- Model is saved with compression if large, and is ready for evaluation.

### Compliance
- No use of sklearn KMeans or any clustering in the final pipeline.
- All features are sanitized to avoid NaN/inf values.
- The code is robust, modular, and ready for further extension or deployment.