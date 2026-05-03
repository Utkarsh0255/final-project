import os
import redis
import json
import time
import numpy as np
import re
from urllib.parse import quote
from engine import (
    OptiCleanEngine, 
    MeanModeStrategy, 
    StandardKNNStrategy, 
    AKNNTIQRStrategy,
    MICEStrategy,
    IsolationForestStrategy
)

STRATEGIES = {
    "Mean": MeanModeStrategy(),
    "KNN": StandardKNNStrategy(),
    "AKNN_TIQR": AKNNTIQRStrategy(),
    "MICE": MICEStrategy(),
    "Isolation Forest": IsolationForestStrategy()
}

def safe_filename(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('_') or 'imputed'

def build_output_path(filepath, output_dir, tournament_id, method_name):
    if filepath and os.path.exists(filepath):
        base_dir = os.path.dirname(filepath)
    else:
        base_dir = output_dir or os.getenv('IMPUTED_OUTPUT_DIR', './data')

    output_dir = os.path.join(base_dir, 'imputed')
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{safe_filename(tournament_id)}_{safe_filename(method_name)}_imputed.csv"
    return os.path.join(output_dir, filename)

if __name__ == '__main__':
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    r = redis.from_url(redis_url)
    
    print(f"Worker is listening for raw Redis tasks on {redis_url}...")
    
    while True:
        _, message = r.brpop('impute_queue')
        task = json.loads(message.decode('utf-8'))
        
        filepath = task.get('filepath')
        method_name = task.get('method')
        tournament_id = task.get('tournamentId')
        output_dir = task.get('outputDir')
        
        # EXPECT THE TARGET COLUMN FROM NODE (Fallback to 'Outcome' if missing)
        target_col = task.get('targetCol', 'Outcome')
        if isinstance(target_col, str):
            target_col = target_col.strip().strip("'\"")
        
        if method_name not in STRATEGIES:
            print(f"Warning: Unknown method '{method_name}' requested.")
            error_result = {
                "method": method_name, 
                "error": "Method not implemented in worker", 
                "rmse": 999999999 
            }
            r.hset(f"results:{tournament_id}", method_name, json.dumps(error_result))
            continue

        print(f"Started processing {method_name} from {filepath}...")
        start_time = time.time()
        
        try:
            engine = OptiCleanEngine(filepath)
            df_clean = engine.encode_and_remove_noise()
            preview_cols = ", ".join(df_clean.columns.astype(str)[:10])
            print(f"Columns found ({len(df_clean.columns)} total): {preview_cols}")

            if target_col not in df_clean.columns:
                available_cols = ", ".join(df_clean.columns.astype(str))
                raise ValueError(f"Target column '{target_col}' was not found. Available columns: {available_cols}")
            
            # Pass target_col so it isn't hidden
            test_df, ground_truth = engine.create_masked_test(df_clean, target_col)
            
            strategy = STRATEGIES[method_name]
            feature_cols = [col for col in test_df.columns if col != target_col]
            imputed_features = strategy.impute(test_df[feature_cols])
            imputed_df = imputed_features.copy()
            imputed_df[target_col] = test_df[target_col].values
            imputed_df = imputed_df[test_df.columns]

            full_feature_cols = [col for col in df_clean.columns if col != target_col]
            full_imputed_features = strategy.impute(df_clean[full_feature_cols])
            full_imputed_df = full_imputed_features.copy()
            full_imputed_df[target_col] = df_clean[target_col].values
            full_imputed_df = full_imputed_df[df_clean.columns]

            output_path = build_output_path(filepath, output_dir, tournament_id, method_name)
            full_imputed_df.to_csv(output_path, index=False)
            
            # 1. Calculate Imputation RMSE (Only for non-target columns)
            rmse = np.sqrt(((ground_truth[feature_cols] - imputed_df[feature_cols]) ** 2).mean().mean())
            
            # 2. Calculate Downstream Classification Metrics
            clf_metrics = engine.evaluate_downstream_model(imputed_df, target_col)
            
            result = {
                "method": method_name,
                "rmse": float(rmse),
                "accuracy": float(clf_metrics["accuracy"]),
                "precision": float(clf_metrics["precision"]),
                "recall": float(clf_metrics["recall"]),
                "f1_score": float(clf_metrics["f1_score"]),
                "confusion_matrix": clf_metrics["confusion_matrix"],
                "csvPath": output_path,
                "downloadUrl": f"/download/{quote(str(tournament_id), safe='')}/{quote(method_name, safe='')}",
                "execution_time": round(time.time() - start_time, 2)
            }
            
            r.hset(f"results:{tournament_id}", method_name, json.dumps(result))
            print(f"Finished {method_name} | RMSE: {rmse:.4f} | Acc: {clf_metrics['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Failed {method_name} Error: {str(e)}")
            failed_result = {
                "method": method_name,
                "error": str(e),
                "rmse": 999999999, 
                "execution_time": round(time.time() - start_time, 2)
            }
            r.hset(f"results:{tournament_id}", method_name, json.dumps(failed_result))
