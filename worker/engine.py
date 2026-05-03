import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod

# --- STRATEGY INTERFACE ---
class ImputationStrategy(ABC):
    @abstractmethod
    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# --- IMPLEMENTATIONS ---
class MeanModeStrategy(ImputationStrategy):
    def impute(self, df):
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        df_out = df.copy()
        if not num_cols.empty:
            df_out[num_cols] = SimpleImputer(strategy='mean').fit_transform(df[num_cols])
        if not cat_cols.empty:
            df_out[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])
        return df_out

class StandardKNNStrategy(ImputationStrategy):
    def impute(self, df):
        imputer = KNNImputer(n_neighbors=5)
        return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

class AKNNTIQRStrategy(ImputationStrategy):
    def impute(self, df):
        adaptive_k = max(3, int(np.sqrt(len(df)) / 2))
        imputer = KNNImputer(n_neighbors=adaptive_k)
        return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# --- THE MAIN ENGINE ---
class OptiCleanEngine:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        self.encoders = {}

    def encode_and_remove_noise(self):
        df_clean = self.df.copy()
        
        # 1. Bulletproof Categorical Encoding
        for col in self.categorical_cols:
            # Find exactly where the valid data is (ignoring NaNs, Nones, etc.)
            mask = df_clean[col].notna()
            
            # If the column is completely empty, skip it to prevent crashes
            if not mask.any():
                continue
                
            # Force all valid data to be standard Python strings
            valid_data = df_clean.loc[mask, col].astype(str)
            
            # Encode the strings
            le = LabelEncoder()
            encoded_values = le.fit_transform(valid_data)
            
            # Create a brand new column of empty NaNs
            new_numeric_col = pd.Series(np.nan, index=df_clean.index)
            
            # Safely inject the numeric codes back into the correct rows
            new_numeric_col.loc[mask] = encoded_values
            
            # Overwrite the original column
            df_clean[col] = new_numeric_col
            self.encoders[col] = le

        # 2. TIQR Noise Removal
        for col in self.numerical_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            df_clean.loc[(df_clean[col] < Q1 - 1.5*IQR) | (df_clean[col] > Q3 + 1.5*IQR), col] = np.nan
            
        return df_clean

    def create_masked_test(self, df_clean):
        complete_rows = df_clean.dropna()
        if len(complete_rows) < 10:
            raise ValueError("Not enough complete rows to benchmark.")
            
        test_df = complete_rows.copy()
        for col in test_df.columns:
            mask = np.random.rand(len(test_df)) < 0.1
            test_df.loc[mask, col] = np.nan
            
        return test_df, complete_rows