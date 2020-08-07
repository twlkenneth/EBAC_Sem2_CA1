import pandas as pd

from q2_fraud_detection.model import *

def comparison() -> pd.DataFrame:
    model_names = ['LRegression', 'DecisionTree', 'NaiveBayesClassifier', 'RandomForest', 'XGBoost', 'TensorflowANN', 'LightGBM']
    df_model_names = pd.DataFrame(model_names, columns=['model'])

    lr_results = pd.DataFrame([LRegression().run()])
    dt_results = pd.DataFrame([DecisionTree().run()])
    nbc_results = pd.DataFrame([NaiveBayesClassifier().run()])
    rf_results = pd.DataFrame([RandomForest().run()])
    xgb_results = pd.DataFrame([XGBoost().run()])
    tf_results = pd.DataFrame([TensorflowANN().run()])
    lgbm_results = pd.DataFrame([LightGBM().run()])
    df = pd.concat([lr_results, dt_results, nbc_results, rf_results, xgb_results, tf_results, lgbm_results])
    df = df.reset_index(drop=True)

    leaderboard = pd.concat([df_model_names, df], axis=1)
    leaderboard = leaderboard.sort_values(by=['matthew_corr_valid', 'f1_score_valid'], ascending=False)

    return leaderboard

if __name__ == "__main__":
    comparison()