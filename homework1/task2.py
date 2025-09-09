# c4d8a9b2-7f3e-4c1b-9a2d-6e8f1c5b9d3a
#!/usr/bin/env python3

import os
import sys
import pickle
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
# === MLAB PATH SETUP ===
current_dir = os.path.dirname(__file__)
mlab_path = os.path.join(current_dir, 'mlab')
if mlab_path not in sys.path:
    sys.path.insert(0, mlab_path)
from mlab.regression._linear import LinearRegression

# === FEATURE EXTRACTOR ===
class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Advanced feature extractor for salary prediction: handles missing data, advanced feature engineering, target encoding, clustering, and robust encoding."""
    def __init__(self, max_features=500, ngram_range=(1,2), n_clusters=8, rare_thresh=10):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.n_clusters = n_clusters
        self.rare_thresh = rare_thresh
        self.job_title_vocab = None
        self.job_title_idf = None
        self.exp_encoder = None
        self.emp_type_encoder = None
        self.emp_res_encoder = None
        self.comp_loc_encoder = None
        self.comp_size_encoder = None
        self.seniority_map = {
            'intern': 0.5, 'internship': 0.5, 'junior': 1, 'senior': 3,
            'lead': 4, 'manager': 5, 'director': 6, 'principal': 7
        }
        self.job_title_x_comp_size_encoder = None
        self.exp_lvl_x_emp_type_encoder = None
        self.job_title_clusters = None
        self.job_title_cluster_model = None
        self.target_encodings = {}
        self.numeric_means = {}
        self.numeric_stds = {}
        self.rare_categories = {}

    def _handle_missing_data(self, data):
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].fillna('unknown')
            else:
                data[col] = data[col].fillna(data[col].median() if data[col].dtype != 'O' else 0)
        return data

    def _group_rare_categories(self, series):
        counts = series.value_counts()
        rare = counts[counts < self.rare_thresh].index
        return series.apply(lambda x: 'Other' if x in rare else x)

    def _categorical_encoding(self, series, encoder=None, fit=False):
        # Always convert to string/object to avoid Categorical assignment errors
        series = series.astype(str)
        series = self._group_rare_categories(series)
        if fit:
            values = series.fillna('unknown').astype(str).tolist()
            values.extend(['unknown', 'OTHER', 'UNSEEN', 'Other'])
            encoder = LabelEncoder()
            encoder.fit(values)
            return encoder
        else:
            encoded = []
            for value in series.fillna('unknown').astype(str):
                if value in encoder.classes_:
                    encoded.append(encoder.transform([value])[0])
                elif 'unknown' in encoder.classes_:
                    encoded.append(encoder.transform(['unknown'])[0])
                else:
                    encoded.append(0)
            return np.array(encoded)

    def _fit_job_title(self, x):
        # Fit TF-IDF for job_title
        from collections import Counter
        vocab = Counter()
        doc_freq = Counter()
        n_docs = 0
        for doc in x['job_title'].astype(str):
            tokens = doc.lower().split()
            ngrams = []
            for n in range(self.ngram_range[0], self.ngram_range[1]+1):
                ngrams += [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            vocab.update(ngrams)
            n_docs += 1
            for t in set(ngrams):
                doc_freq[t] += 1
        self.job_title_vocab = {word: idx for idx, (word, _) in enumerate(vocab.most_common(self.max_features))}
        self.job_title_idf = np.zeros(len(self.job_title_vocab))
        for word, idx in self.job_title_vocab.items():
            df = doc_freq.get(word, 1)
            self.job_title_idf[idx] = np.log((1 + n_docs) / (1 + df)) + 1

    def _transform_job_title(self, x):
        features = np.zeros((len(x), len(self.job_title_vocab)))
        for i, doc in enumerate(x['job_title'].astype(str)):
            tokens = doc.lower().split()
            ngrams = []
            for n in range(self.ngram_range[0], self.ngram_range[1]+1):
                ngrams += [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            tf = Counter(ngrams)
            for t in ngrams:
                idx = self.job_title_vocab.get(t)
                if idx is not None:
                    features[i, idx] = tf[t] * self.job_title_idf[idx]
        return features

    def _target_encode(self, series, target, name):
        # Ensure target is a pandas Series aligned with series
        if not isinstance(target, pd.Series):
            target = pd.Series(target, index=series.index)
        means = series.groupby(series).apply(lambda s: target.loc[s.index].mean())
        self.target_encodings[name] = means
        return series.map(means).fillna(target.mean()).values.reshape(-1,1)

    def _target_encode_transform(self, series, name, global_mean):
        means = self.target_encodings.get(name, None)
        if means is not None:
            return series.map(means).fillna(global_mean).values.reshape(-1,1)
        else:
            return np.full((len(series),1), global_mean)

    def _feature_engineering(self, x, y=None, fit=False):
        x = x.copy()
        # Add seniority feature
        if 'job_title' in x.columns:
            x['seniority'] = 0
            for key, val in self.seniority_map.items():
                mask = x['job_title'].str.lower().str.contains(key, na=False)
                x.loc[mask, 'seniority'] = val
        else:
            x['seniority'] = 0
        # Add company_size_num
        size_map = {
            '1-10': 5, '11-50': 30, '51-200': 125, '201-500': 350,
            '501-1000': 750, '1001-5000': 3000, '5001-10,000': 7500, '10,001+': 15000
        }
        if 'company_size' in x.columns:
            x['company_size_num'] = x['company_size'].map(size_map).fillna(0)
        else:
            x['company_size_num'] = 0
        # Add remote
        if 'remote_ratio' in x.columns:
            x['remote'] = (x['remote_ratio'] > 0).astype(int)
        else:
            x['remote'] = 0
        # Add region features
        x['company_region'] = x['company_location'].map(self._region_map)
        x['employee_region'] = x['employee_residence'].map(self._region_map)
        x['region_match'] = (x['company_region'] == x['employee_region']).astype(int)
        # Add interaction features
        x['job_title_x_company_size'] = x['job_title'].astype(str) + '_' + x['company_size'].astype(str)
        x['experience_level_x_employment_type'] = x['experience_level'].astype(str) + '_' + x['employment_type'].astype(str)
        # More interaction features
        x['seniority_x_company_size'] = x['seniority'] * x['company_size_num']
        x['remote_x_company_size'] = x['remote'] * x['company_size_num']
        x['seniority_x_remote'] = x['seniority'] * x['remote']
        x['work_year_x_company_size'] = x['work_year'].astype(float) * x['company_size_num'] if 'work_year' in x.columns else 0
        # Experience level numeric
        exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
        x['experience_level_num'] = x['experience_level'].map(exp_map).fillna(-1)
        # Remote ratio binning
        if 'remote_ratio' in x.columns:
            x['remote_ratio_bin'] = pd.cut(x['remote_ratio'], bins=[-1,0,50,100], labels=['none','partial','full'])
        else:
            x['remote_ratio_bin'] = 'none'
        return x

    def fit(self, x, y=None):
        x = self._handle_missing_data(x.copy())
        x = self._feature_engineering(x, y, fit=True)
        self._fit_job_title(x)
        # REMOVED: _fit_job_title_clusters and _job_title_cluster_feature (no longer needed)
        self.exp_encoder = self._categorical_encoding(x['experience_level'], fit=True)
        self.emp_type_encoder = self._categorical_encoding(x['employment_type'], fit=True)
        self.emp_res_encoder = self._categorical_encoding(x['employee_residence'], fit=True)
        self.comp_loc_encoder = self._categorical_encoding(x['company_location'], fit=True)
        self.comp_size_encoder = self._categorical_encoding(x['company_size'], fit=True)
        self.job_title_x_comp_size_encoder = self._categorical_encoding(x['job_title_x_company_size'], fit=True)
        self.exp_lvl_x_emp_type_encoder = self._categorical_encoding(x['experience_level_x_employment_type'], fit=True)
        self.remote_ratio_bin_encoder = self._categorical_encoding(x['remote_ratio_bin'], fit=True)
        # Target encoding for job_title and company_location
        if y is not None:
            self._target_encode(x['job_title'], y, 'job_title')
            self._target_encode(x['company_location'], y, 'company_location')
        # Standardize numeric features
        num_cols = ['seniority','company_size_num','seniority_x_company_size','remote_x_company_size','seniority_x_remote','work_year_x_company_size','experience_level_num','remote_ratio']
        for col in num_cols:
            if col in x.columns:
                self.numeric_means[col] = x[col].mean()
                self.numeric_stds[col] = x[col].std() if x[col].std() > 0 else 1
        # Compute and store non-constant columns mask
        job_title_feats = self._transform_job_title(x)
        features = np.hstack([
            job_title_feats,
            self._target_encode_transform(x['job_title'], 'job_title', y.mean() if y is not None else 0),
            self._target_encode_transform(x['company_location'], 'company_location', y.mean() if y is not None else 0),
            self._categorical_encoding(x['experience_level'], self.exp_encoder).reshape(-1,1),
            self._categorical_encoding(x['employment_type'], self.emp_type_encoder).reshape(-1,1),
            self._categorical_encoding(x['employee_residence'], self.emp_res_encoder).reshape(-1,1),
            self._categorical_encoding(x['company_location'], self.comp_loc_encoder).reshape(-1,1),
            self._categorical_encoding(x['company_size'], self.comp_size_encoder).reshape(-1,1),
            self._categorical_encoding(x['job_title_x_company_size'], self.job_title_x_comp_size_encoder).reshape(-1,1),
            self._categorical_encoding(x['experience_level_x_employment_type'], self.exp_lvl_x_emp_type_encoder).reshape(-1,1),
            self._categorical_encoding(x['remote_ratio_bin'], self.remote_ratio_bin_encoder).reshape(-1,1),
            x['remote_ratio'].astype(float).values.reshape(-1,1) if 'remote_ratio' in x.columns else np.zeros((len(x),1)),
            x['work_year'].astype(float).values.reshape(-1,1) if 'work_year' in x.columns else np.zeros((len(x),1)),
            (x['employee_residence'] == x['company_location']).astype(int).values.reshape(-1,1),
            self._standardize(x, 'seniority'),
            self._standardize(x, 'company_size_num'),
            self._standardize(x, 'seniority_x_company_size'),
            self._standardize(x, 'remote_x_company_size'),
            self._standardize(x, 'seniority_x_remote'),
            self._standardize(x, 'work_year_x_company_size'),
            self._standardize(x, 'experience_level_num'),
            self._standardize(x, 'remote_ratio')
        ])
        self.non_constant = np.any(features != 0, axis=0)
        return self

    def _standardize(self, x, col):
        if col in x.columns:
            mean = self.numeric_means.get(col, 0)
            std = self.numeric_stds.get(col, 1)
            return ((x[col] - mean) / std).values.reshape(-1,1)
        else:
            return np.zeros((len(x),1))

    def transform(self, x):
        x = self._handle_missing_data(x.copy())
        x = self._feature_engineering(x, fit=False)
        job_title_feats = self._transform_job_title(x)
        features = np.hstack([
            job_title_feats,
            self._target_encode_transform(x['job_title'], 'job_title', 0),
            self._target_encode_transform(x['company_location'], 'company_location', 0),
            self._categorical_encoding(x['experience_level'], self.exp_encoder).reshape(-1,1),
            self._categorical_encoding(x['employment_type'], self.emp_type_encoder).reshape(-1,1),
            self._categorical_encoding(x['employee_residence'], self.emp_res_encoder).reshape(-1,1),
            self._categorical_encoding(x['company_location'], self.comp_loc_encoder).reshape(-1,1),
            self._categorical_encoding(x['company_size'], self.comp_size_encoder).reshape(-1,1),
            self._categorical_encoding(x['job_title_x_company_size'], self.job_title_x_comp_size_encoder).reshape(-1,1),
            self._categorical_encoding(x['experience_level_x_employment_type'], self.exp_lvl_x_emp_type_encoder).reshape(-1,1),
            self._categorical_encoding(x['remote_ratio_bin'], self.remote_ratio_bin_encoder).reshape(-1,1),
            x['remote_ratio'].astype(float).values.reshape(-1,1) if 'remote_ratio' in x.columns else np.zeros((len(x),1)),
            x['work_year'].astype(float).values.reshape(-1,1) if 'work_year' in x.columns else np.zeros((len(x),1)),
            (x['employee_residence'] == x['company_location']).astype(int).values.reshape(-1,1),
            self._standardize(x, 'seniority'),
            self._standardize(x, 'company_size_num'),
            self._standardize(x, 'seniority_x_company_size'),
            self._standardize(x, 'remote_x_company_size'),
            self._standardize(x, 'seniority_x_remote'),
            self._standardize(x, 'work_year_x_company_size'),
            self._standardize(x, 'experience_level_num'),
            self._standardize(x, 'remote_ratio')
        ])
        features = features[:, self.non_constant]
        return features

    def _region_map(self, loc):
        region_dict = {
            'US': 'Americas', 'CA': 'Americas', 'MX': 'Americas', 'BR': 'Americas', 'AR': 'Americas',
            'GB': 'Europe', 'DE': 'Europe', 'FR': 'Europe', 'ES': 'Europe', 'IT': 'Europe', 'NL': 'Europe', 'PL': 'Europe', 'PT': 'Europe', 'IE': 'Europe', 'CH': 'Europe', 'BE': 'Europe', 'AT': 'Europe', 'SE': 'Europe', 'NO': 'Europe', 'DK': 'Europe', 'FI': 'Europe', 'UA': 'Europe', 'RO': 'Europe', 'HU': 'Europe', 'CZ': 'Europe', 'GR': 'Europe', 'BG': 'Europe', 'RS': 'Europe', 'HR': 'Europe', 'SK': 'Europe', 'SI': 'Europe', 'LT': 'Europe', 'LV': 'Europe', 'EE': 'Europe',
            'IN': 'Asia', 'CN': 'Asia', 'JP': 'Asia', 'SG': 'Asia', 'KR': 'Asia', 'VN': 'Asia', 'PH': 'Asia', 'ID': 'Asia', 'TH': 'Asia', 'MY': 'Asia', 'HK': 'Asia', 'TW': 'Asia', 'PK': 'Asia', 'BD': 'Asia', 'IL': 'Asia', 'TR': 'Asia', 'AE': 'Asia', 'SA': 'Asia', 'IR': 'Asia',
            'ZA': 'Africa', 'NG': 'Africa', 'EG': 'Africa', 'KE': 'Africa', 'MA': 'Africa', 'GH': 'Africa',
            'AU': 'Oceania', 'NZ': 'Oceania',
        }
        return region_dict.get(loc, 'Other')

# === MLAB ALGORITHM WRAPPER ===
class MLabAlgorithmWrapper(BaseEstimator):
    """Sklearn-compatible wrapper for mlab LinearRegression with log-target regression support."""
    def __init__(self, model_type='linear', log_target=True):
        self.model_type = model_type
        self.model = None
        self.log_target = log_target
        self.target_mean = None

    def fit(self, X, y=None):
        if self.log_target:
            y = np.log1p(y)
        self.model = LinearRegression()
        self.model.fit(X, y)
        if self.log_target:
            self.target_mean = np.mean(y)
        return self

    def predict(self, X_input):
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        preds = self.model.predict(X_input)
        if self.log_target:
            preds = np.expm1(preds)
            preds = np.clip(preds, 0, None)
        return np.array(preds, dtype=float)

    def predict_proba(self, X_input):
        # Not applicable for regression, but included for evaluator compatibility
        raise NotImplementedError("predict_proba is not implemented for regression.")

    def fit_predict(self, X_input, y=None):
        self.fit(X_input, y)
        return self.predict(X_input)

# === ENSEMBLE PIPELINE ===
class EnsemblePipeline:
    def __init__(self, pipelines):
        self.pipelines = pipelines
    def fit(self, X, y=None):
        for p in self.pipelines:
            p.fit(X, y)
        return self
    def predict(self, X):
        preds = np.column_stack([p.predict(X) for p in self.pipelines])
        return np.mean(preds, axis=1)

# === COMPLETE PIPELINE ===
class CompletePipeline(BaseEstimator):
    """Complete pipeline: feature extraction + model training/prediction."""
    def __init__(self, model_type='linear', max_features=500, ngram_range=(1,2), log_target=True):
        self.feature_pipeline = FeatureExtractor(max_features=max_features, ngram_range=ngram_range)
        self.algorithm_model = MLabAlgorithmWrapper(model_type=model_type, log_target=log_target)

    def fit(self, X_input, y=None):
        x_features = self.feature_pipeline.fit_transform(X_input, y)
        self.algorithm_model.fit(x_features, y)
        return self

    def predict(self, X_input):
        x_features = self.feature_pipeline.transform(X_input)
        preds = self.algorithm_model.predict(x_features)
        return np.maximum(preds, 0)

    def predict_proba(self, X_input):
        return self.algorithm_model.predict_proba(self.feature_pipeline.transform(X_input))

    def fit_predict(self, X_input, y=None):
        self.fit(X_input, y)
        return self.predict(X_input)

def create_pipeline(ngram_range=(1,2), max_features=500, log_target=True):
    """Create your complete ML pipeline"""
    return CompletePipeline(model_type='linear', max_features=max_features, ngram_range=ngram_range, log_target=log_target)

# === MODEL SAVE ===
def save_model_with_compression(model_package, filepath, size_threshold_mb=25):
    import gzip
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        pickle.dump(model_package, tmp_file)
        tmp_file_path = tmp_file.name
    file_size_mb = os.path.getsize(tmp_file_path) / (1024 * 1024)
    if file_size_mb > size_threshold_mb:
        compressed_path = filepath if filepath.endswith('.gz') else filepath + '.gz'
        with open(tmp_file_path, 'rb') as f_in, gzip.open(compressed_path, 'wb') as f_out:
            f_out.write(f_in.read())
        os.remove(tmp_file_path)
        print(f"Model saved and compressed to {compressed_path} ({file_size_mb:.2f} MB > {size_threshold_mb} MB)")
    else:
        os.replace(tmp_file_path, filepath)
        print(f"Model saved to {filepath} ({file_size_mb:.2f} MB)")

# === MAIN ===
def main():
    try:
        data_dir = os.path.dirname(__file__)
        train_path = os.path.join(data_dir, 'train.parquet')
        val_path = os.path.join(data_dir, 'validation.parquet')
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        train_df = train_df.dropna(subset=['salary_in_usd'])
        # Outlier removal: cap salaries at 1st/99th percentiles
        lower, upper = np.percentile(train_df['salary_in_usd'], [1, 99])
        train_df = train_df[(train_df['salary_in_usd'] >= lower) & (train_df['salary_in_usd'] <= upper)]
        X_train = train_df.copy()
        y_train = train_df['salary_in_usd'].values
        X_val = val_df.copy()
        y_val = val_df['salary_in_usd'].values
        # Hyperparameter grid for grid search/ensemble
        param_grid = [
            {'ngram_range': (1,1), 'max_features': 300, 'n_clusters': 6, 'rare_thresh': 5, 'n_svd_components': 4},
            {'ngram_range': (1,2), 'max_features': 300, 'n_clusters': 8, 'rare_thresh': 10, 'n_svd_components': 6},
            {'ngram_range': (1,2), 'max_features': 500, 'n_clusters': 10, 'rare_thresh': 15, 'n_svd_components': 8},
            {'ngram_range': (1,1), 'max_features': 500, 'n_clusters': 8, 'rare_thresh': 10, 'n_svd_components': 8},
            {'ngram_range': (1,2), 'max_features': 400, 'n_clusters': 12, 'rare_thresh': 8, 'n_svd_components': 5},
        ]
        # Train all models in param_grid, rank by MAE, and ensemble top 3
        results = []
        for params in param_grid:
            pipeline = create_pipeline(ngram_range=params['ngram_range'], max_features=params['max_features'], log_target=True)
            # Set additional FeatureExtractor params
            pipeline.feature_pipeline.n_clusters = params['n_clusters']
            pipeline.feature_pipeline.rare_thresh = params['rare_thresh']
            if hasattr(pipeline.feature_pipeline, 'n_svd_components'):
                pipeline.feature_pipeline.n_svd_components = params.get('n_svd_components', 8)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            r2 = r2_score(y_val, y_pred)
            print(f"Model=linear, params={params} | MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.3f}")
            results.append({'pipeline': pipeline, 'mae': mae, 'rmse': rmse, 'r2': r2, 'params': params})
        # Rank by MAE (lower is better)
        import operator
        results = sorted(results, key=operator.itemgetter('mae'))
        # Ensemble top 3
        top_pipelines = [res['pipeline'] for res in results[:3]]
        ensemble = EnsemblePipeline(top_pipelines)
        y_pred = ensemble.predict(X_val)
        print('Final Ensemble Model Performance:')
        print(f"MAE: {mean_absolute_error(y_val, y_pred):.2f}")
        print(f"RMSE: {mean_squared_error(y_val, y_pred, squared=False):.2f}")
        print(f"R2: {r2_score(y_val, y_pred):.3f}")
        # Save ensemble model
        model_package = {
            'pipeline': ensemble,
            'model_type': 'EnsembleLinearRegression',
            'feature_extractors': [p.feature_pipeline for p in top_pipelines],
            'feature_names': [list(p.feature_pipeline.job_title_vocab.keys()) for p in top_pipelines]
        }
        model_path = os.path.join(data_dir, 'task2.pkl')
        print(f"Saving model to: {model_path}")
        save_model_with_compression(model_package, model_path)
        print("Pipeline ready for evaluation!")
        return
        # ...existing code for single best_pipeline can be removed or commented...
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()
