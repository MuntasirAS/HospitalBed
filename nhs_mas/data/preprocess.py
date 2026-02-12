
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

def prepare_rate_series(df_occ: pd.DataFrame, df_avail: pd.DataFrame):
    """Aggregate to a single time series of occupancy rates (0..1)."""
    f = 'Effective_Snapshot_Date'
    df_occ = df_occ.copy()
    df_avail = df_avail.copy()
    df_occ[f] = pd.to_datetime(df_occ[f], errors='coerce')
    df_avail[f] = pd.to_datetime(df_avail[f], errors='coerce')
    occ_ts = df_occ.groupby(f)['Number_Of_Beds'].sum()
    avail_ts = df_avail.groupby(f)['Number_Of_Beds'].sum()
    aligned = pd.concat([occ_ts, avail_ts], axis=1, keys=['occ', 'avail']).dropna()
    aligned = aligned[aligned['avail']>0]
    rate = (aligned['occ']/aligned['avail']).values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    rate_scaled = scaler.fit_transform(rate)
    dates = aligned.index
    return dates, rate, rate_scaled, scaler

def create_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(len(data)-lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

def train_validation_split(ts_scaled: np.ndarray, lookback: int, val_frac: float = 0.2):
    n = len(ts_scaled)
    val_size = max(lookback+1, int(n*val_frac))
    train = ts_scaled[:-val_size]
    val = ts_scaled[-val_size:]
    X_train, y_train = create_sequences(train, lookback)
    X_val, y_val = create_sequences(val, lookback)
    return (
        torch.FloatTensor(X_train), torch.FloatTensor(y_train),
        torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    )
