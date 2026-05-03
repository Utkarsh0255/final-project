import os
import redis
import json
import time
import numpy as np
from engine import OptiCleanEngine, MeanModeStrategy, StandardKNNStrategy, AKNNTIQRStrategy

STRATEGIES = {
    "Mean": MeanModeStrategy(),
    "KNN": StandardKNNStrategy(),
    "AKNN_TIQR": AKNNTIQRStrategy()
}

if __name__ == '__main__':
    # URL is set to localhost for bare-metal testing
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    r = redis.from_url(redis_url)
    
    print("Worker is listening for raw Redis tasks on localhost...")
    
    while True:
        # 1. Block and wait for a task to appear in 'impute_queue'
        _, message = r.brpop('impute_queue')
        task = json.loads(message.decode('utf-8'))
        
        filepath = task['filepath']
        method_name = task['method']
        tournament_id = task['tournamentId']
        
        print(f"Started processing {method_name} from {filepath}...")
        start_time = time.time()
        
        try:
            # 2. Run your algorithms
            engine = OptiCleanEngine(filepath)
            df_clean = engine.encode_and_remove_noise()
            test_df, ground_truth = engine.create_masked_test(df_clean)
            
            strategy = STRATEGIES[method_name]
            imputed_df = strategy.impute(test_df)
            
            # 3. Calculate Error
            rmse = np.sqrt(((ground_truth - imputed_df) ** 2).mean().mean())
            
            result = {
                "method": method_name,
                "rmse": float(rmse),
                "execution_time": round(time.time() - start_time, 2)
            }
            
            # 4. Save result back to Redis so Node.js can find it
            r.hset(f"results:{tournament_id}", method_name, json.dumps(result))
            print(f"Finished {method_name} with RMSE: {rmse}")
            
        except Exception as e:
            print(f"Failed {method_name} Error: {str(e)}")