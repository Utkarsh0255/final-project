import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from abc import ABC, abstractmethod
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

class ImputationStrategy(ABC):
    @abstractmethod
    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class MeanModeStrategy(ImputationStrategy):
    def impute(self, df):
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        df_out = df.copy()
        if not num_cols.empty:
            df_out[num_cols] = SimpleImputer(strategy='mean', keep_empty_features=True).fit_transform(df[num_cols])
        if not cat_cols.empty:
            df_out[cat_cols] = SimpleImputer(strategy='most_frequent', keep_empty_features=True).fit_transform(df[cat_cols])
        return df_out

class StandardKNNStrategy(ImputationStrategy):
    def impute(self, df):
        # As per your script: n_neighbors=5 is standard
        imputer = KNNImputer(n_neighbors=5, keep_empty_features=True)
        return pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)

class AKNNTIQRStrategy(ImputationStrategy):
    def impute(self, df):
        df_out = df.copy()
        num_cols = df_out.select_dtypes(include=[np.number]).columns
        
        for col in num_cols:
            col_min = df_out[col].min()
            shift_val = 0 if col_min > 0 else abs(col_min) + 0.0001
            data_shifted = df_out[col] + shift_val
            
            valid_mask = data_shifted.notna()
            if valid_mask.sum() < 10:
                continue
                
            try:
                valid_data = data_shifted[valid_mask]
                transformed_data, lmbda = stats.boxcox(valid_data)
                
                Q1, Q3 = np.percentile(transformed_data, [25, 75])
                IQR = Q3 - Q1
                
                clipped_tx = np.clip(transformed_data, Q1 - 1.5*IQR, Q3 + 1.5*IQR)
                
                clipped_original = inv_boxcox(clipped_tx, lmbda) - shift_val
                df_out.loc[valid_mask, col] = clipped_original
            except Exception as e:
                continue

        # 2. Apply Adaptive KNN (Using unsupervised heuristic instead of CV)
        adaptive_k = max(3, int(np.sqrt(len(df_out)) / 2))
        imputer = KNNImputer(n_neighbors=adaptive_k, keep_empty_features=True)
        return pd.DataFrame(imputer.fit_transform(df_out), columns=df_out.columns, index=df_out.index)

class MICEStrategy(ImputationStrategy):
    def impute(self, df):
        # As per your script: modeling each feature as a function of others
        imputer = IterativeImputer(max_iter=10, random_state=42, keep_empty_features=True)
        return pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)

class IsolationForestStrategy(ImputationStrategy):
    def impute(self, df):
        df_out = df.copy()
        num_cols = df_out.select_dtypes(include=[np.number]).columns
        
        if not num_cols.empty:
            # Temp fill NaNs because Isolation Forest crashes on them
            temp_imputer = SimpleImputer(strategy='median')
            temp_df = pd.DataFrame(temp_imputer.fit_transform(df_out[num_cols]), columns=num_cols)
            
            # As per your script: contamination=0.05
            iso = IsolationForest(contamination=0.05, random_state=42)
            outliers = iso.fit_predict(temp_df)
            
            # Instead of dropping rows (which breaks RMSE benchmarking), mask them as NaN
            for col in num_cols:
                df_out.loc[outliers == -1, col] = np.nan
                
        # Fill the newly created structural holes using KNN
        imputer = KNNImputer(n_neighbors=5, keep_empty_features=True)
        return pd.DataFrame(imputer.fit_transform(df_out), columns=df_out.columns, index=df_out.index)

class OptiCleanEngine:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.df.columns = self.df.columns.astype(str).str.strip().str.strip("'\"")
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

    def encode_and_remove_noise(self):
        df_clean = self.df.copy()
        
        # 1. Bulletproof Categorical Encoding
        for col in self.categorical_cols:
            mask = df_clean[col].notna()
            if not mask.any(): continue
                
            valid_data = df_clean.loc[mask, col].astype(str)
            le = LabelEncoder()
            encoded_values = le.fit_transform(valid_data)
            
            new_numeric_col = pd.Series(np.nan, index=df_clean.index)
            new_numeric_col.loc[mask] = encoded_values
            df_clean[col] = new_numeric_col
            
        return df_clean

    # Replace your old create_masked_test with this one:
    def create_masked_test(self, df_clean, target_col):
        complete_rows = df_clean.dropna()
        if len(complete_rows) < 10:
            raise ValueError("Not enough complete rows to benchmark.")
            
        test_df = complete_rows.copy()
        for col in test_df.columns:
            if col == target_col: 
                continue # Never mask the target column!
            
            # Randomly hide 10% of the data to test imputation accuracy
            mask = np.random.rand(len(test_df)) < 0.1
            test_df.loc[mask, col] = np.nan
            
        return test_df, complete_rows
    # Paste this directly under your create_masked_test function!
    def evaluate_downstream_model(self, imputed_df, target_col):
        # Separate features and target
        X = imputed_df.drop(columns=[target_col])
        y = imputed_df[target_col].astype(int)

        # Scale data for standard KNN Classifier
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Use Standard K=5 Classifier for fair evaluation
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Calculate metrics (zero_division=0 prevents crashes if a class is completely missed)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist() # Convert numpy array to JSON-safe list
        }
