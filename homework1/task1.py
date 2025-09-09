# a7f3c8e9-4d2a-4b6f-9e1a-8c3f5d9e2b7a
#!/usr/bin/env python3

import os
import sys
import pickle
import pandas as pd
import numpy as np
import gzip
import re
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix


# === MLAB PATH SETUP ===
current_dir = os.path.dirname(__file__)
mlab_path = os.path.join(current_dir, 'mlab')
if mlab_path not in sys.path:
    sys.path.insert(0, mlab_path)

from mlab.naive_bayes._naive_bayes import MultinomialNB


def oversample_minority(X, y, random_state=42):
    np.random.seed(random_state)
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    Xy_balanced = []
    for label in unique:
        X_class = X[y == label]
        n_samples = X_class.shape[0]
        if n_samples < max_count:
            idxs = np.random.choice(n_samples, size=max_count, replace=True)
            X_upsampled = X_class.iloc[idxs]
        else:
            X_upsampled = X_class
        y_upsampled = np.full((max_count,), label)
        Xy_balanced.append((X_upsampled, y_upsampled))
    X_bal = pd.concat([x for x, _ in Xy_balanced], axis=0).reset_index(drop=True)
    y_bal = np.concatenate([y for _, y in Xy_balanced])
    y_bal = pd.DataFrame(y_bal, columns=['domain'])
    return X_bal, y_bal


class TextPreprocessor:
    def __init__(self, text_columns=None, use_tfidf=False):
        self.text_columns = text_columns if text_columns is not None else ['markdown']
        self.vocab = {}
        self.stopwords = set(['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with', 'as', 'by', 'at', 'from', 'thing'])
        self.use_tfidf = use_tfidf
        self.idf = None

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    def fit(self, x):
        vocab = Counter()
        doc_freq = Counter()
        n_docs = 0
        for col in self.text_columns:
            for doc in x[col].astype(str):
                tokens = self.clean_text(doc)
                vocab.update(tokens)
                n_docs += 1
                for t in set(tokens):
                    doc_freq[t] += 1
        self.vocab = {word: idx for idx, (word, _) in enumerate(vocab.most_common(1000))}
        if self.use_tfidf:
            self.idf = np.zeros(len(self.vocab))
            for word, idx in self.vocab.items():
                df = doc_freq.get(word, 1)
                self.idf[idx] = np.log((1 + n_docs) / (1 + df)) + 1

    def transform(self, x):
        features = np.zeros((len(x), len(self.vocab)), dtype=float if self.use_tfidf else int)
        for i, row in x.iterrows():
            for col in self.text_columns:
                tokens = self.clean_text(str(row[col]))
                tf = Counter(tokens)
                for t in tokens:
                    idx = self.vocab.get(t)
                    if idx is not None:
                        if self.use_tfidf:
                            features[i, idx] = tf[t] * self.idf[idx]
                        else:
                            features[i, idx] += 1
        return features

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class FeatureEngineer:
    def __init__(self):
        self.engineered_cols = [
            'text_length', 'word_count', 'avg_word_length', 'num_sentences',
            'num_exclamations', 'num_questions', 'num_uppercase_words',
            'num_digits', 'num_unique_words', 'unique_word_ratio'
        ]

    def transform(self, x):
        x = x.copy()
        x['text_length'] = x['markdown'].str.len()
        x['word_count'] = x['markdown'].str.split().apply(len)
        x['avg_word_length'] = x['markdown'].apply(lambda v: np.mean([len(w) for w in str(v).split()]) if len(str(v).split()) > 0 else 0)
        x['num_sentences'] = x['markdown'].apply(lambda v: str(v).count('.') + str(v).count('!') + str(v).count('?'))
        x['num_exclamations'] = x['markdown'].apply(lambda v: str(v).count('!'))
        x['num_questions'] = x['markdown'].apply(lambda v: str(v).count('?'))
        x['num_uppercase_words'] = x['markdown'].apply(lambda v: sum(1 for w in str(v).split() if w.isupper()))
        x['num_digits'] = x['markdown'].apply(lambda v: sum(c.isdigit() for c in str(v)))
        x['num_unique_words'] = x['markdown'].apply(lambda v: len(set(str(v).split())))
        x['unique_word_ratio'] = x['num_unique_words'] / (x['word_count'] + 1e-6)
        return x[self.engineered_cols].to_numpy()


class CompletePipeline(BaseEstimator):
    def __init__(self, text_columns=None, alpha=0.7, use_tfidf=False):
        self.text_columns = text_columns if text_columns is not None else ['markdown']
        self.text_preprocessor = TextPreprocessor(text_columns=self.text_columns, use_tfidf=use_tfidf)
        self.feature_engineer = FeatureEngineer()
        self.alpha = alpha
        self.use_tfidf = use_tfidf
        self.clf = MultinomialNB(alpha=self.alpha)
        self.fitted = False
        self.label_encoder = None

    def fit(self, x, y):
        x_text = self.text_preprocessor.fit_transform(x)
        x_eng = self.feature_engineer.transform(x)
        x_all = np.concatenate([x_text, x_eng], axis=1)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.clf = MultinomialNB(alpha=self.alpha)
        self.clf.fit(x_all, y_enc)
        self.fitted = True
        return self

    def transform(self, x):
        x_text = self.text_preprocessor.transform(x)
        x_eng = self.feature_engineer.transform(x)
        x_all = np.concatenate([x_text, x_eng], axis=1)
        return x_all

    def predict(self, x):
        x_all = self.transform(x)
        y_pred = self.clf.predict(x_all)
        return self.classes_[y_pred]

    def predict_proba(self, x):
        x_all = self.transform(x)
        return self.clf.predict_proba(x_all)


def load_dataset():
    data_dir = os.path.dirname(__file__)
    train_path = os.path.join(data_dir, "train1.parquet")
    if os.path.exists(train_path):
        df = pd.read_parquet(train_path)
        return df
    else:
        raise FileNotFoundError(f"Training data not found at {train_path}")

def save_model_with_compression(model_package, filepath, size_threshold_mb=25):
    """
    Save model with automatic compression for large files.
    If the file size exceeds size_threshold_mb, compress with gzip and add .gz extension.
    """
    import tempfile
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

def main():
    try:
        # === LOAD AND PREPARE DATA ===
        train_df = load_dataset()
        
        # Remove missing targets
        train_df = train_df.dropna(subset=['domain'])
        x_train = train_df
        y_train = train_df['domain'].values
        
        # === OVERSAMPLE MINORITY CLASSES ===
        x_train_os, y_train_os = oversample_minority(x_train, y_train)
        y_train_os = y_train_os.values.flatten() if hasattr(y_train_os, 'values') else y_train_os

        # === CREATE AND TRAIN PIPELINE ===
        text_cols = ['markdown']
        best_score = 0
        best_alpha = 1.0
        best_pipeline = None
        for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
            pipeline = CompletePipeline(text_columns=text_cols, alpha=alpha, use_tfidf=True)
            pipeline.fit(x_train_os, y_train_os)
            val_path = os.path.join(os.path.dirname(__file__), 'validation1.parquet')
            if os.path.exists(val_path):
                val_df = pd.read_parquet(val_path)
                x_val = val_df
                y_val = val_df['domain'].values
                y_pred = pipeline.predict(x_val)
                score = np.mean(y_pred == y_val)
                print(f"alpha={alpha:.2f} | accuracy={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
                    best_pipeline = pipeline
        print(f"Best alpha: {best_alpha}, Best accuracy: {best_score}")
        pipeline = best_pipeline

        # === EVALUATE ON VALIDATION SET ===
        val_path = os.path.join(os.path.dirname(__file__), 'validation1.parquet')
        if os.path.exists(val_path):
            val_df = pd.read_parquet(val_path)
            x_val = val_df
            y_val = val_df['domain'].values
            y_pred = pipeline.predict(x_val)
            print(classification_report(y_val, y_pred))
            print('Confusion Matrix:')
            print(confusion_matrix(y_val, y_pred))
        

        # === SAVE MODEL ===
        model_package = {
            'pipeline': pipeline,
            'model_type': 'MultinomialNB',
            'feature_engineer': pipeline.feature_engineer,
            'feature_extractor': pipeline.text_preprocessor,
            'classes': list(pipeline.classes_)
        }
        model_path = os.path.join(os.path.dirname(__file__), "task1.pkl")
        print(f"Saving model to: {model_path}")
        save_model_with_compression(model_package, model_path)
        print(f"Model save attempted at: {model_path}")

        print("Pipeline ready for evaluation!")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
