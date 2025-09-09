
## Documentation Report

### Feature Engineering
- Text is preprocessed by lowercasing, removing non-alphabetic characters, and filtering stopwords.
- Features extracted include TF/TF-IDF n-gram counts (from markdown), and engineered features such as text length, word count, average word length, sentence/exclamation/question counts, uppercase word count, digit count, unique word count, and unique word ratio.
- All features are concatenated for model input.

### Model Pipeline
- Uses a custom `TextPreprocessor` for text vectorization and a `FeatureEngineer` for additional features.
- Handles class imbalance by oversampling minority classes to match the majority class size.
- Trains a Multinomial Naive Bayes classifier with hyperparameter tuning (alpha grid search).
- Model is evaluated on a validation set and the best model is selected based on accuracy.
- Model and feature pipeline are saved with compression if large.

### Compliance & Robustness
- No use of forbidden libraries; all code is custom or uses allowed sklearn components.
- Handles missing data and class imbalance robustly.
- Modular and extensible for further improvements.