
# nhs_mas — LSTM‑driven Multi‑Agent Load Balancing - Muntasir Al-Asfoor

A modular Python package that trains an LSTM forecaster on **occupancy rates** and runs a 
**multi‑agent system** (MAS) to balance NHS bed occupancy across trusts and specialties. 

## Key improvements over the monolithic notebook
- **Unit consistency**: model trains *and* predicts **rates**, fixing the earlier scaler mismatch 
  between global bed counts and sector rates.  
- **Configurable MAS**: dynamic transfer amounts, risk and capacity thresholds.  
- **Reproducible**: seeding; single entry point; versionable outputs.  

## Install (editable)
```bash
pip install -e .
```

## Run
```bash
python -m nhs_mas.main --epochs 100 --lookback 4 --risk_threshold 0.9 --provider_threshold 0.75 --transfer_pct 0.07
```

The script writes `simulation_output.csv` with post‑balancing occupancy by trust/sector.

## Package layout
See `nhs_mas/` for modules: `config/`, `data/`, `models/`, `agents/`, `simulation/`, `utils/`.

 
