import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize_features(df, continuous_features):
    scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])
    return df, scaler

def one_hot_encode_features(df, categorical_features):
    return pd.get_dummies(df, columns=categorical_features, drop_first=True)

def compute_metrics(preds, targets):
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets)**2))
    ss_res = np.sum((targets - preds)**2)
    ss_tot = np.sum((targets - np.mean(targets))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mae, rmse, r2
