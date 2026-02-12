
import numpy as np
import torch

def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().astype(float)
    return np.asarray(t, dtype=float)

def evaluate(model, X, y, criterion):
    model.eval()
    with torch.no_grad():
        preds = model(X)
    y_np = _to_numpy(y).flatten()
    p_np = _to_numpy(preds).flatten()

    mse = float(np.mean((p_np - y_np)**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(p_np - y_np)))
    # R2
    ss_res = float(np.sum((y_np - p_np)**2))
    ss_tot = float(np.sum((y_np - np.mean(y_np))**2))
    r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else float('nan')
    # Pearson
    if np.std(y_np)>0 and np.std(p_np)>0:
        pearson = float(np.corrcoef(y_np, p_np)[0,1])
    else:
        pearson = float('nan')
    # Cosine similarity
    denom = (np.linalg.norm(y_np) * np.linalg.norm(p_np))
    cosine = float(np.dot(y_np, p_np)/denom) if denom>0 else float('nan')

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson,
        'cosine': cosine,
        'y_true': y_np,
        'y_pred': p_np,
    }
