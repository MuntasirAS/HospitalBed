
import pandas as pd
import numpy as np

def load_data(cfg):
    """Return (df_occ, df_avail, used_synthetic: bool)."""
    try:
        df_occ = pd.read_csv(cfg.occupied_csv)
        df_avail = pd.read_csv(cfg.available_csv)
        return df_occ, df_avail, False
    except FileNotFoundError:
        if not cfg.use_synthetic_if_missing:
            raise
        return generate_synthetic_pair(cfg), True

def generate_synthetic_pair(cfg):
    dates = pd.date_range(start='2020-01-01', periods=100, freq=cfg.freq)
    occ = pd.DataFrame({
        'Effective_Snapshot_Date': np.tile(dates, 2),
        'Number_Of_Beds': np.random.randint(50, 100, 200) + np.sin(np.linspace(0, 10, 200))*10,
        'Organisation_Code': ['ORG1']*100 + ['ORG2']*100,
        'Sector': ['General & Acute']*200
    })
    avail = occ.copy()
    avail['Number_Of_Beds'] = 110  # fixed availability per snapshot
    return occ, avail
