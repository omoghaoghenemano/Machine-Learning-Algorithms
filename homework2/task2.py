#260c86b6-e733-4104-bf23-8f2c52906eb1
#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pickle
import gzip
import tempfile
import sys
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
mlab_path = os.path.join(parent_dir, 'mlab')
if mlab_path not in sys.path:
    sys.path.insert(0, mlab_path)
# Also add the parent directory itself    
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from mlab.neural_networks._mlp import MLPClassifier


# ===================================================================================================
# COMPLETE PIPELINE CLASS FOR INCOME PREDICTION
# ===================================================================================================

class CompletePipeline(BaseEstimator, TransformerMixin):
    """
    Complete ML pipeline for income prediction that handles raw demographic data
    and returns final predictions as strings ('>50K' or '<=50K')
    """
    
    def __init__(self, hidden_layer_sizes=(128, 64, 32), alpha=0.001, random_state=42):
        self.preprocessor = IncomePreprocessor()
        self.model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,  # Deeper architecture
        activation='relu',
        solver='adam',
        alpha=alpha,  # L2 regularization
        batch_size='auto',
        random_state=random_state,
        class_weight='balanced',
        epochs=200,
        early_stopping=False,  # Disable to avoid issues
    )
        self.fitted = False
        self.classes_ = ['<=50K', '>50K']
        
    def fit(self, X, y=None):
        """
        Fit the complete pipeline on raw demographic data
        
        Args:
            X (DataFrame): Raw demographic data with original column names including target
            y (Series/array): Target income values as strings ('>50K' or '<=50K') - optional if included in X
        """
        # If y is provided separately, add it to X for preprocessing
        if y is not None and 'income' not in X.columns:
            X_with_target = X.copy()
            X_with_target['income'] = y
        else:
            X_with_target = X.copy()
        
        # Fit preprocessor and transform data
        X_processed, y_processed = self.preprocessor.fit_transform(X_with_target)
        
        # Fit the model
        self.model.fit(X_processed, y_processed)
        
        self.fitted = True
        return self
    
    def predict(self, X):
        """
        Predict income categories for raw demographic data
        
        Args:
            X (DataFrame): Raw demographic data with original column names
            
        Returns:
            array: Predictions as strings ('>50K' or '<=50K')
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        
        # Transform data using fitted preprocessor
        X_processed, _ = self.preprocessor.transform(X)
        
        # Get predictions from model (0 or 1)
        y_pred_binary = self.model.predict(X_processed)
        
        # Convert to string format
        y_pred_strings = np.where(y_pred_binary == 1, '>50K', '<=50K')
        
        return y_pred_strings
    
    def predict_proba(self, X):
        """
        Predict class probabilities for raw demographic data
        
        Args:
            X (DataFrame): Raw demographic data with original column names
            
        Returns:
            array: Class probabilities (shape: n_samples x 2)
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        
        # Transform data using fitted preprocessor
        X_processed, _ = self.preprocessor.transform(X)
        
        # Get probabilities from model
        return self.model.predict_proba(X_processed)


# ===================================================================================================
# INCOME PREPROCESSOR CLASS
# ===================================================================================================

class IncomePreprocessor:
    """Complete preprocessing pipeline for income prediction"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.selected_features = None
        self.capital_gain_bins = None
        self.capital_loss_bins = None
        
    def create_advanced_features(self, df):
        """Create advanced feature engineering"""
        df_enhanced = df.copy()
        
        # 1. Age-based features
        df_enhanced['age_squared'] = df_enhanced['age'] ** 2
        df_enhanced['age_log'] = np.log1p(df_enhanced['age'])
        
        # Age groups
        df_enhanced['is_prime_earning'] = ((df_enhanced['age'] >= 35) & (df_enhanced['age'] <= 55)).astype(int)
        df_enhanced['is_early_career'] = ((df_enhanced['age'] >= 22) & (df_enhanced['age'] <= 35)).astype(int)
        df_enhanced['is_senior_worker'] = (df_enhanced['age'] > 55).astype(int)
        
        # 2. Education features
        df_enhanced['education_squared'] = df_enhanced['educational-num'] ** 2
        df_enhanced['is_graduate'] = (df_enhanced['educational-num'] >= 13).astype(int)
        df_enhanced['is_postgrad'] = (df_enhanced['educational-num'] >= 15).astype(int)
        
        # 3. Work hours features
        df_enhanced['hours_squared'] = df_enhanced['hours-per-week'] ** 2
        df_enhanced['is_part_time'] = (df_enhanced['hours-per-week'] < 35).astype(int)
        df_enhanced['is_overtime'] = (df_enhanced['hours-per-week'] > 45).astype(int)
        
        # 4. Work experience proxy
        df_enhanced['work_experience'] = np.maximum(df_enhanced['age'] - df_enhanced['educational-num'] - 6, 0)
        df_enhanced['experience_ratio'] = df_enhanced['work_experience'] / (df_enhanced['age'] + 1)
        
        # 5. Key interactions
        df_enhanced['age_education'] = df_enhanced['age'] * df_enhanced['educational-num']
        df_enhanced['age_hours'] = df_enhanced['age'] * df_enhanced['hours-per-week']
        df_enhanced['education_hours'] = df_enhanced['educational-num'] * df_enhanced['hours-per-week']
        
        # 6. Capital features
        df_enhanced['total_capital'] = df_enhanced['capital_gain_bin'] + df_enhanced['capital_loss_bin']
        df_enhanced['has_capital'] = (df_enhanced['total_capital'] > 0).astype(int)
        
        # 7. Professional indicators
        df_enhanced['likely_professional'] = ((df_enhanced['educational-num'] >= 15) & 
                                             (df_enhanced['hours-per-week'] >= 40)).astype(int)
        
        return df_enhanced
    
    def preprocess_data(self, df, is_training=True):
        """Complete preprocessing pipeline"""
        df_processed = df.copy()
        
        # Handle missing values denoted by '?' 
        df_processed = df_processed.replace('?', np.nan)
        
        # Fill missing values for categorical columns with mode
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'income':  # Don't fill target variable
                mode_val = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown'
                df_processed[col] = df_processed[col].fillna(mode_val)
        
        # Fill missing values for numerical columns with median
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != 'income':  # Don't fill target variable
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
        
        # Handle capital gains/losses
        if 'capital-gain' in df_processed.columns:
            if is_training:
                # Create bins for capital gain
                self.capital_gain_bins = [-1, 0.0, 2000, 10000, df_processed['capital-gain'].max() + 1]
                self.capital_loss_bins = [-1, 0.0, 1000, 3000, df_processed['capital-loss'].max() + 1]
            
            gain_labels = ['None', 'Low', 'Medium', 'High']
            loss_labels = ['None', 'Low', 'Medium', 'High']
            
            df_processed['capital_gain_bin'] = pd.cut(
                df_processed['capital-gain'], bins=self.capital_gain_bins, 
                labels=gain_labels, include_lowest=True
            )
            df_processed['capital_loss_bin'] = pd.cut(
                df_processed['capital-loss'], bins=self.capital_loss_bins,
                labels=loss_labels, include_lowest=True
            )
            
            # Convert to numeric
            df_processed['capital_gain_bin'] = pd.Categorical(df_processed['capital_gain_bin']).codes
            df_processed['capital_loss_bin'] = pd.Categorical(df_processed['capital_loss_bin']).codes
            
            # Drop original columns
            df_processed.drop(columns=['capital-gain', 'capital-loss'], inplace=True)
        
        # Convert income to binary (only if present - for training data)
        if 'income' in df_processed.columns:
            df_processed['income'] = df_processed['income'].apply(
                lambda x: 1 if str(x).strip() == '>50K' else 0
            )
        
        # Transform fnlwgt (sampling weights) with log transformation
        if 'fnlwgt' in df_processed.columns:
            df_processed['fnlwgt'] = np.log1p(df_processed['fnlwgt'])
        
        # Identify categorical columns for encoding
        numerical_cols = ['age', 'fnlwgt', 'educational-num', 'hours-per-week']
        if 'income' in df_processed.columns:
            numerical_cols.append('income')
        if 'capital_gain_bin' in df_processed.columns:
            numerical_cols.extend(['capital_gain_bin', 'capital_loss_bin'])
        
        categorical_cols = [col for col in df_processed.columns if col not in numerical_cols]
        
        # Encode categorical variables
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories by assigning a default value
                    le = self.label_encoders[col]
                    unique_vals = df_processed[col].astype(str).unique()
                    unseen = set(unique_vals) - set(le.classes_)
                    if unseen:
                        # Map unseen values to most frequent class
                        most_frequent = le.classes_[0]  # Use first class as default
                        df_processed[col] = df_processed[col].astype(str).apply(
                            lambda x: most_frequent if x in unseen else x
                        )
                    df_processed[col] = le.transform(df_processed[col].astype(str))
                else:
                    df_processed[col] = 0  # Default value for missing encoder
        
        return df_processed
    
    def fit_transform(self, df):
        """Fit preprocessing pipeline and transform training data"""
        # Basic preprocessing
        df_processed = self.preprocess_data(df, is_training=True)
        
        # Create advanced features
        df_enhanced = self.create_advanced_features(df_processed)
        
        # Separate features and target
        X = df_enhanced.drop(columns=['income'])
        y = df_enhanced['income']
        
        # Remove constant features
        var_threshold = VarianceThreshold(threshold=0.01)
        X_var = var_threshold.fit_transform(X)
        valid_features = X.columns[var_threshold.get_support()]
        X = pd.DataFrame(X_var, columns=valid_features, index=X.index)
        
        # Select important features based on domain knowledge
        important_features = [
            'age', 'fnlwgt', 'educational-num', 'hours-per-week',
            'workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'gender', 'native-country',
            'capital_gain_bin', 'capital_loss_bin',
            'age_squared', 'age_log', 'education_squared',
            'is_prime_earning', 'is_early_career', 'is_senior_worker',
            'is_graduate', 'is_postgrad', 'is_part_time', 'is_overtime',
            'work_experience', 'experience_ratio', 'age_education',
            'age_hours', 'education_hours', 'total_capital', 'has_capital',
            'likely_professional', 'hours_squared'
        ]
        
        # Keep only available features from important list
        self.selected_features = [f for f in important_features if f in X.columns]
        X_selected = X[self.selected_features]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        X_final = pd.DataFrame(X_scaled, columns=self.selected_features, index=X.index)
        
        print(f"✅ Training preprocessing complete:")
        print(f"   - Original features: {df.shape[1]}")
        print(f"   - Enhanced features: {len(X.columns)}")
        print(f"   - Selected features: {len(self.selected_features)}")
        print(f"   - Final dataset shape: {X_final.shape}")
        
        return X_final, y
    
    def transform(self, df):
        """Transform validation/test data using fitted pipeline"""
        # Basic preprocessing
        df_processed = self.preprocess_data(df, is_training=False)
        
        # Create advanced features
        df_enhanced = self.create_advanced_features(df_processed)
        
        # Separate features and target
        if 'income' in df_enhanced.columns:
            X = df_enhanced.drop(columns=['income'])
            y = df_enhanced['income']
        else:
            X = df_enhanced
            y = None
        
        # Select only the features used in training
        missing_features = set(self.selected_features) - set(X.columns)
        if missing_features:
            print(f"⚠️  Missing features in validation data: {missing_features}")
            # Add missing features with zeros
            for feature in missing_features:
                X[feature] = 0
        
        # Keep only selected features
        X_selected = X[self.selected_features]
        
        # Scale features
        X_scaled = self.scaler.transform(X_selected)
        X_final = pd.DataFrame(X_scaled, columns=self.selected_features, index=X.index)
        
        print(f"✅ Validation preprocessing complete:")
        print(f"   - Final dataset shape: {X_final.shape}")
        
        return X_final, y


# ===================================================================================================
# PIPELINE CREATION FUNCTION
# ===================================================================================================

def create_pipeline():
    """Create your complete ML pipeline"""
    pipeline = CompletePipeline(
        hidden_layer_sizes=(256, 128, 64, 32),
        alpha=0.001,
        random_state=42
    )
    return pipeline


# ===================================================================================================
# MODEL SAVING UTILITIES
# ===================================================================================================

def save_model_with_compression(model_package, filepath, size_threshold_mb=25):
    """
    Save model with automatic compression for large files.
    If the file size exceeds size_threshold_mb, compress with gzip and add .gz extension.
    """
    # Save to a temporary file first
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        pickle.dump(model_package, tmp_file)
        tmp_file_path = tmp_file.name
    
    file_size_mb = os.path.getsize(tmp_file_path) / (1024 * 1024)
    
    if file_size_mb > size_threshold_mb:
        # Compress and save as .gz
        compressed_path = filepath if filepath.endswith('.gz') else filepath + '.gz'
        with open(tmp_file_path, 'rb') as f_in, gzip.open(compressed_path, 'wb') as f_out:
            f_out.write(f_in.read())
        os.remove(tmp_file_path)
        print(f"Model saved and compressed to {compressed_path} ({file_size_mb:.2f} MB > {size_threshold_mb} MB)")
    else:
        # Save as regular pickle
        os.replace(tmp_file_path, filepath)
        print(f"Model saved to {filepath} ({file_size_mb:.2f} MB)")


# ===================================================================================================
# MAIN TRAINING AND EVALUATION PIPELINE
# ===================================================================================================

def main():
    """Main training and evaluation pipeline"""
    try:
        print("🚀 Starting Income Prediction Pipeline...")
        print("=" * 80)
        
        # === LOAD DATA ===
        print("📊 Loading training and validation data...")
        
        data_dir = os.path.dirname(__file__)
        train_path = os.path.join(data_dir, 'train.parquet')
        val_path = os.path.join(data_dir, 'validation.parquet')
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found at {val_path}")
        
        df_train = pd.read_parquet(train_path)
        df_val = pd.read_parquet(val_path)
        
        print(f"Training data shape: {df_train.shape}")
        print(f"Validation data shape: {df_val.shape}")
        
        # === CREATE AND TRAIN PIPELINE ===
        print("\n🤖 Creating and training ML pipeline...")
        
        pipeline = create_pipeline()
        
        # Train the pipeline
        pipeline.fit(df_train, df_train['income'])
        
        print("✅ Training completed!")
        print(f"   Total iterations: {pipeline.model.n_iter_}")
        print(f"   Final loss: {pipeline.model.loss_curve_[-1]:.6f}")
        
        # === EVALUATE ON VALIDATION SET ===
        print("\n📊 Evaluating on validation set...")
        
        # Make predictions
        y_pred = pipeline.predict(df_val)
        y_pred_proba = pipeline.predict_proba(df_val)
        
        # Calculate metrics
        if 'income' in df_val.columns:
            y_val = df_val['income']
            accuracy = accuracy_score(y_val, y_pred)
            print(f"Validation Accuracy: {accuracy:.4f}")
            
            print("\nClassification Report:")
            print(classification_report(y_val, y_pred))
            
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_val, y_pred))
        
        # === SAVE MODEL ===
        print("\n💾 Saving trained pipeline...")
        
        model_package = {
            'pipeline': pipeline,
            'model_type': 'MLPClassifier',
            'classes': ['<=50K', '>50K'],
            'feature_names': pipeline.preprocessor.selected_features
        }
        
        model_path = os.path.join(data_dir, 'task2.pkl')
        save_model_with_compression(model_package, model_path)
        
        print("\n🎉 Pipeline training and evaluation completed successfully!")
        
        # === PLOT TRAINING HISTORY ===
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(pipeline.model.loss_curve_)
        plt.title('Training Loss Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        if len(pipeline.model.validation_scores_) > 0:
            plt.subplot(1, 2, 2)
            plt.plot(pipeline.model.validation_scores_)
            plt.title('Validation Score Curve')
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return pipeline
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    pipeline = main()
