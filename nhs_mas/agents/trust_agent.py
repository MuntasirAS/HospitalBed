
import numpy as np
import torch

class TrustAgent:
    def __init__(self, org_code: str, region: str, sectors: dict):
        self.org_code = org_code
        self.region = region
        # sectors: {'Sector': {'avail': float, 'occ': float}}
        self.sectors = sectors

    def get_occ_rate(self, sector: str) -> float:
        s = self.sectors.get(sector, {})
        avail = s.get('avail', 0)
        occ = s.get('occ', 0)
        if avail <= 0:
            return 1.0
        return float(occ)/float(avail)

    def forecast_rate_risk(self, sector: str, model, scaler, lookback: int, threshold: float):
        if sector not in self.sectors:
            return False, 0.0
        occ = self.sectors[sector]['occ']
        avail = self.sectors[sector]['avail']
        if avail <= 0:
            return False, 0.0
        current_rate = occ/avail
        hist = np.linspace(current_rate*0.8, current_rate, lookback).reshape(-1,1)
        x_scaled = scaler.transform(hist)
        x = torch.FloatTensor(x_scaled).reshape(1, lookback, 1)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(x).numpy().reshape(-1)[0]
        pred_rate = scaler.inverse_transform([[pred_scaled]])[0][0]
        return (pred_rate > threshold), float(pred_rate)
