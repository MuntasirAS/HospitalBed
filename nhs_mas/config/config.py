
from dataclasses import dataclass

@dataclass
class LSTMConfig:
    input_dim: int = 1
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.005
    epochs: int = 100
    lookback: int = 4

@dataclass
class MASConfig:
    risk_threshold: float = 0.90
    provider_capacity_threshold: float = 0.75
    transfer_percentage: float = 0.07

@dataclass
class DataConfig:
    occupied_csv: str = 'KH03-Occupied-Day-only.csv'
    available_csv: str = 'KH03-Available-Day-only.csv'
    use_synthetic_if_missing: bool = True
    freq: str = 'Q-DEC'  # explicit quarterly anchor
    seed: int = 42
