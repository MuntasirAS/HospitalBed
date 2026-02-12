
        import argparse
        import torch
        import pandas as pd
        from nhs_mas.config.config import LSTMConfig, MASConfig, DataConfig
        from nhs_mas.data.loader import load_data
        from nhs_mas.data.preprocess import prepare_rate_series, train_validation_split
        from nhs_mas.models.lstm_forecaster import LSTMForecaster
        from nhs_mas.models.metrics import evaluate
        from nhs_mas.utils.seed import set_seed
        from nhs_mas.agents.trust_agent import TrustAgent
        from nhs_mas.simulation.pipeline import run_simulation

        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument('--epochs', type=int, default=100)
            p.add_argument('--hidden_dim', type=int, default=64)
            p.add_argument('--num_layers', type=int, default=2)
            p.add_argument('--dropout', type=float, default=0.2)
            p.add_argument('--learning_rate', type=float, default=0.005)
            p.add_argument('--lookback', type=int, default=4)
            p.add_argument('--risk_threshold', type=float, default=0.90)
            p.add_argument('--provider_threshold', type=float, default=0.75)
            p.add_argument('--transfer_pct', type=float, default=0.07)
            p.add_argument('--seed', type=int, default=42)
            return p.parse_args()

        def main():
            args = parse_args()
            set_seed(args.seed)

            lstm_cfg = LSTMConfig(hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                                  dropout=args.dropout, learning_rate=args.learning_rate,
                                  epochs=args.epochs, lookback=args.lookback)
            mas_cfg = MASConfig(risk_threshold=args.risk_threshold,
                                provider_capacity_threshold=args.provider_threshold,
                                transfer_percentage=args.transfer_pct)
            data_cfg = DataConfig(seed=args.seed)

            df_occ, df_avail, used_synth = load_data(data_cfg)
            _, rate, rate_scaled, scaler = prepare_rate_series(df_occ, df_avail)
            X_train, y_train, X_val, y_val = train_validation_split(rate_scaled, lstm_cfg.lookback)

            model = LSTMForecaster(lstm_cfg.input_dim, lstm_cfg.hidden_dim,
                                   lstm_cfg.num_layers, lstm_cfg.dropout)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lstm_cfg.learning_rate)

            train_losses, val_losses = [], []
            for ep in range(lstm_cfg.epochs):
                model.train()
                optimizer.zero_grad()
                pred = model(X_train)
                loss = criterion(pred, y_train)
                loss.backward(); optimizer.step()
                train_losses.append(float(loss.item()))
                # val
                model.eval()
                with torch.no_grad():
                    v = model(X_val)
                    vloss = criterion(v, y_val).item()
                val_losses.append(float(vloss))
                if (ep+1) % 20 == 0:
                    print(f"Epoch {ep+1}/{lstm_cfg.epochs} | Train {train_losses[-1]:.6f} | Val {val_losses[-1]:.6f}")

            print("
Validation metrics:")
            m = evaluate(model, X_val, y_val, criterion)
            for k in ['mse','rmse','mae','r2','pearson','cosine']:
                print(f"  {k}: {m[k]:.6f}" if isinstance(m[k], float) else f"  {k}: {m[k]}")

            # demo agents
            agents = [
                TrustAgent('ORG_A', 'London', {
                    'General & Acute': {'avail': 100, 'occ': 93},
                    'Maternity': {'avail': 30, 'occ': 15},
                }),
                TrustAgent('ORG_B', 'London', {
                    'General & Acute': {'avail': 100, 'occ': 70},
                    'Maternity': {'avail': 30, 'occ': 15},
                }),
                TrustAgent('ORG_C', 'Midlands', {
                    'General & Acute': {'avail': 100, 'occ': 70},
                    'Maternity': {'avail': 30, 'occ': 28},
                }),
                TrustAgent('ORG_D', 'Midlands', {
                    'General & Acute': {'avail': 100, 'occ': 70},
                    'Maternity': {'avail': 30, 'occ': 15},
                }),
            ]

            results = run_simulation(agents, model, scaler, mas_cfg)
            results.to_csv('simulation_output.csv', index=False)
            print("
Simulation results saved to simulation_output.csv")
            print(results)

        if __name__ == '__main__':
            main()
