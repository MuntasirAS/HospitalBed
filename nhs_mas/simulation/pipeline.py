
import pandas as pd
from nhs_mas.agents.manager import NHSMASManager

def run_simulation(agents, model, scaler, mas_cfg, sectors=('General & Acute','Maternity')):
    mas = NHSMASManager(agents, model, scaler, mas_cfg)
    for s in sectors:
        mas.balance(s)
    rows = []
    for a in agents:
        for s, vals in a.sectors.items():
            avail = vals['avail'] if vals['avail']>0 else 1
            rows.append({
                'Trust': a.org_code,
                'Region': a.region,
                'Sector': s,
                'Final_Occupancy': vals['occ']/avail
            })
    return pd.DataFrame(rows)
