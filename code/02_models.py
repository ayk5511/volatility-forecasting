"""
Paper 2: Volatility Forecasting — Model Definitions and Training

Implements all forecasting models compared in the paper:
  1. GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1)
  2. HAR-RV (Corsi 2009)
  3. LSTM, GRU
  4. LightGBM, XGBoost
  5. Simple ensemble (average of top models)

All models use strictly backward-looking features.
Evaluation uses expanding-window (no look-ahead bias).

Usage: python 02_models.py
Output: ../results/model_results.parquet, ../results/metrics.csv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import warnings
import json
from datetime import datetime

# Force unbuffered output so we can track progress
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Forecast horizons
HORIZONS = {"1d": "rv_1d_fwd", "5d": "rv_5d_fwd", "22d": "rv_22d_fwd"}

# Train/validation/test split dates
TRAIN_END = "2018-12-31"
VAL_END = "2021-12-31"
# Test: 2022-01-01 onwards (covers post-pandemic, rate hikes, 2022 bear market)


def load_data() -> pd.DataFrame:
    """Load the combined feature dataset."""
    df = pd.read_parquet(DATA_DIR / "combined.parquet")
    print(f"Loaded {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


# ============================================================
# Model 1: GARCH Family
# ============================================================

def fit_garch_family(returns: pd.Series, train_end: str, test_start: str) -> Dict[str, pd.Series]:
    """
    Fit GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1) using arch package.
    Returns one-step-ahead conditional volatility forecasts.
    Uses expanding window: refit every 22 days.
    """
    from arch import arch_model

    results = {"GARCH": [], "EGARCH": [], "GJR-GARCH": []}
    dates = []

    test_returns = returns[test_start:]
    refit_interval = 22
    n_test = len(test_returns)

    scaled_returns = returns * 100  # GARCH works better with percentage returns

    import signal

    class TimeoutError(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutError("GARCH fit timed out")

    for i in range(0, n_test, refit_interval):
        end_idx = min(i + refit_interval, n_test)
        batch_dates = test_returns.index[i:end_idx]
        train_data = scaled_returns[:batch_dates[0]]

        for model_name, vol_model in [
            ("GARCH", "Garch"),
            ("EGARCH", "EGARCH"),
            ("GJR-GARCH", "GARCH"),
        ]:
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout per fit

                if model_name == "GJR-GARCH":
                    am = arch_model(train_data, vol=vol_model, p=1, o=1, q=1, dist="Normal")
                elif model_name == "EGARCH":
                    am = arch_model(train_data, vol=vol_model, p=1, q=1, dist="Normal")
                else:
                    am = arch_model(train_data, vol=vol_model, p=1, q=1, dist="Normal")

                res = am.fit(disp="off", show_warning=False, options={"maxiter": 200})
                forecasts = res.forecast(horizon=1, start=batch_dates[0])
                cond_var = forecasts.variance.loc[batch_dates]

                signal.alarm(0)  # cancel timeout

                # Convert from percentage variance to annualized vol
                vol_forecast = np.sqrt(cond_var.values.flatten() * 252) / 100
                results[model_name].extend(vol_forecast[:len(batch_dates)])
            except Exception:
                signal.alarm(0)
                results[model_name].extend([np.nan] * len(batch_dates))

        dates.extend(batch_dates)

        if i % 220 == 0:
            print(f"  GARCH: {i}/{n_test} test obs processed")

    return {k: pd.Series(v, index=dates[:len(v)], name=k) for k, v in results.items()}


# ============================================================
# Model 2: HAR-RV (Corsi 2009)
# ============================================================

def fit_har_rv(df: pd.DataFrame, target: str, train_end: str, test_start: str) -> pd.Series:
    """
    Heterogeneous Autoregressive model for Realized Volatility.
    RV_t = c + b_d * RV_{t-1} + b_w * RV_{t-1:t-5} + b_m * RV_{t-1:t-22} + e_t
    Uses expanding-window OLS.
    """
    from sklearn.linear_model import LinearRegression

    har_features = ["har_daily", "har_weekly", "har_monthly"]
    test_data = df[test_start:]
    predictions = []
    dates = []

    refit_interval = 66  # refit quarterly

    for i in range(0, len(test_data), refit_interval):
        end_idx = min(i + refit_interval, len(test_data))
        batch = test_data.iloc[i:end_idx]
        train = df[:batch.index[0]]

        X_train = train[har_features].values
        y_train = train[target].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        X_test = batch[har_features].values
        preds = model.predict(X_test)
        predictions.extend(preds)
        dates.extend(batch.index)

    return pd.Series(predictions, index=dates[:len(predictions)], name="HAR-RV")


# ============================================================
# Model 3: LSTM and GRU
# ============================================================

def fit_lstm_gru(df: pd.DataFrame, target: str, train_end: str, val_end: str,
                 test_start: str, model_type: str = "LSTM") -> pd.Series:
    """
    LSTM or GRU for volatility forecasting.
    Input: 22-day lookback window of features.
    Uses train/val split for early stopping, then predicts test set.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Force CPU — MPS (Metal) hangs on certain LSTM/GRU operations in PyTorch
    device = torch.device("cpu")

    # Select features for neural net
    feature_cols = [c for c in df.columns if c.startswith(("rv_5d_lag", "abs_ret_lag",
                    "har_", "vix_lag", "vix_change", "neg_return", "pos_return"))]
    lookback = 22

    def make_sequences(data, features, tgt, lb):
        X, y = [], []
        vals = data[features].values
        targets = data[tgt].values
        for i in range(lb, len(data)):
            X.append(vals[i - lb:i])
            y.append(targets[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[test_start:]

    # Normalize using train statistics only — work on a copy
    norm_df = df[feature_cols + [target]].copy()
    means = train_df[feature_cols].mean()
    stds = train_df[feature_cols].std().replace(0, 1)
    for c in feature_cols:
        norm_df[c] = (norm_df[c] - means[c]) / stds[c]

    # Replace any inf/nan from normalization
    norm_df = norm_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    X_train, y_train = make_sequences(norm_df[:train_end], feature_cols, target, lookback)
    X_val, y_val = make_sequences(norm_df[train_end:val_end], feature_cols, target, lookback)
    X_test, y_test = make_sequences(norm_df[test_start:], feature_cols, target, lookback)

    print(f"  {model_type} data shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    sys.stdout.flush()

    class VolModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, rnn_type="LSTM"):
            super().__init__()
            if rnn_type == "LSTM":
                self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                                   batch_first=True, dropout=0.2)
            else:
                self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                                  batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = VolModel(len(feature_cols), rnn_type=model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    X_val_t = torch.tensor(X_val).to(device)
    y_val_t = torch.tensor(y_val).to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 10

    for epoch in range(50):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"  {model_type}: early stopping at epoch {epoch}, best val loss: {best_val_loss:.6f}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test).to(device)
        test_preds = model(X_test_t).cpu().numpy()

    test_dates = df[test_start:].index[lookback:]
    return pd.Series(test_preds[:len(test_dates)], index=test_dates[:len(test_preds)], name=model_type)


# ============================================================
# Model 4: LightGBM and XGBoost
# ============================================================

def fit_tree_models(df: pd.DataFrame, target: str, train_end: str, val_end: str,
                    test_start: str) -> Dict[str, pd.Series]:
    """
    LightGBM and XGBoost for volatility forecasting.
    Uses expanding window with quarterly refitting.
    """
    import lightgbm as lgb
    import xgboost as xgb

    feature_cols = [c for c in df.columns if not c.startswith("rv_") or c.startswith("rv_5d_lag")
                    or c.startswith("rv_22d") or c.startswith("rv_66d")]
    # More careful feature selection
    feature_cols = [c for c in df.columns if c.startswith(("rv_5d_lag", "abs_ret_lag",
                    "har_", "vix_", "neg_return", "pos_return", "range_vol",
                    "volume", "dow_", "log_return")) and "_fwd" not in c]

    results = {"LightGBM": [], "XGBoost": []}
    dates = []

    test_data = df[test_start:]
    refit_interval = 66

    for i in range(0, len(test_data), refit_interval):
        end_idx = min(i + refit_interval, len(test_data))
        batch = test_data.iloc[i:end_idx]

        train = df[:batch.index[0]]
        val = df[train_end:val_end]

        X_train = np.nan_to_num(train[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(train[target].values, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(val[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = np.nan_to_num(val[target].values, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(batch[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)

        # LightGBM
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        lgb_params = {
            "objective": "regression",
            "metric": "mse",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": -1,
        }
        lgb_model = lgb.train(
            lgb_params, lgb_train, num_boost_round=500,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        lgb_preds = lgb_model.predict(X_test)
        results["LightGBM"].extend(lgb_preds)

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=30, eval_metric="rmse",
            verbosity=0, n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_preds = xgb_model.predict(X_test)
        results["XGBoost"].extend(xgb_preds)

        dates.extend(batch.index)

        if i % 220 == 0:
            print(f"  Trees: {i}/{len(test_data)} test obs processed")

    return {
        k: pd.Series(v, index=dates[:len(v)], name=k)
        for k, v in results.items()
    }, feature_cols


# ============================================================
# Evaluation Metrics
# ============================================================

def compute_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """Compute standard volatility forecasting metrics."""
    # Align indices
    common = actual.index.intersection(predicted.index)
    a = actual.loc[common].values
    p = predicted.loc[common].values

    # Remove NaN
    mask = ~(np.isnan(a) | np.isnan(p))
    a, p = a[mask], p[mask]

    if len(a) == 0:
        return {"MSE": np.nan, "MAE": np.nan, "RMSE": np.nan, "QLIKE": np.nan, "R2": np.nan}

    mse = np.mean((a - p) ** 2)
    mae = np.mean(np.abs(a - p))
    rmse = np.sqrt(mse)

    # QLIKE loss (Patton 2011) — standard for volatility forecast evaluation
    # QLIKE = mean(sigma^2 / h - log(sigma^2 / h) - 1) where h is forecast
    eps = 1e-8
    ratio = (a ** 2) / (p ** 2 + eps)
    qlike = np.mean(ratio - np.log(ratio + eps) - 1)

    # R-squared
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Mincer-Zarnowitz R² (regression of actual on forecast)
    from scipy.stats import pearsonr
    corr, _ = pearsonr(a, p)
    mz_r2 = corr ** 2

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "QLIKE": qlike,
        "R2": r2,
        "MZ_R2": mz_r2,
        "N_obs": len(a),
    }


# ============================================================
# Main Pipeline
# ============================================================

def main():
    print("=" * 60)
    print("Paper 2: Volatility Forecasting — Model Comparison")
    print("=" * 60)

    df = load_data()

    target = "rv_5d_fwd"  # Primary target: 5-day forward realized vol
    test_start = "2022-01-01"

    all_forecasts = {}
    all_metrics = {}

    # --- GARCH Family ---
    print("\n[1/5] Fitting GARCH family...")
    garch_results = fit_garch_family(df["log_return"], TRAIN_END, test_start)
    all_forecasts.update(garch_results)

    # --- HAR-RV ---
    print("\n[2/5] Fitting HAR-RV...")
    har_pred = fit_har_rv(df, target, TRAIN_END, test_start)
    all_forecasts["HAR-RV"] = har_pred

    # --- Tree models ---
    print("\n[3/5] Fitting LightGBM and XGBoost...")
    tree_results, feature_names = fit_tree_models(df, target, TRAIN_END, VAL_END, test_start)
    all_forecasts.update(tree_results)

    # --- LSTM ---
    print("\n[4/5] Fitting LSTM...")
    df_copy = df.copy()
    lstm_pred = fit_lstm_gru(df_copy, target, TRAIN_END, VAL_END, test_start, "LSTM")
    all_forecasts["LSTM"] = lstm_pred

    # --- GRU ---
    print("\n[5/5] Fitting GRU...")
    df_copy2 = df.copy()
    gru_pred = fit_lstm_gru(df_copy2, target, TRAIN_END, VAL_END, test_start, "GRU")
    all_forecasts["GRU"] = gru_pred

    # --- Ensemble (average of LightGBM, HAR-RV, GARCH) ---
    print("\nBuilding ensemble...")
    common_idx = all_forecasts["LightGBM"].index.intersection(
        all_forecasts["HAR-RV"].index
    ).intersection(all_forecasts["GARCH"].index)

    ensemble = (
        all_forecasts["LightGBM"].loc[common_idx]
        + all_forecasts["HAR-RV"].loc[common_idx]
        + all_forecasts["GARCH"].loc[common_idx]
    ) / 3
    ensemble.name = "Ensemble"
    all_forecasts["Ensemble"] = ensemble

    # --- Evaluate all models ---
    print("\n" + "=" * 60)
    print("Results (target: 5-day forward realized volatility)")
    print("Test period: 2022-01-01 onwards")
    print("=" * 60)

    actual = df[test_start:][target]

    for name, preds in all_forecasts.items():
        metrics = compute_metrics(actual, preds)
        all_metrics[name] = metrics
        print(f"\n{name}:")
        for k, v in metrics.items():
            if k == "N_obs":
                print(f"  {k}: {int(v)}")
            else:
                print(f"  {k}: {v:.6f}")

    # Save results
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(RESULTS_DIR / "metrics_5d.csv")

    forecasts_df = pd.DataFrame(all_forecasts)
    forecasts_df["actual"] = actual
    forecasts_df.to_parquet(RESULTS_DIR / "forecasts_5d.parquet")

    # Save as JSON for easy reference
    metrics_json = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_metrics.items()}
    with open(RESULTS_DIR / "metrics_5d.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}")
    print(f"  metrics_5d.csv")
    print(f"  forecasts_5d.parquet")
    print(f"  metrics_5d.json")

    # Print ranking
    print("\n" + "=" * 60)
    print("MODEL RANKING (by QLIKE, lower is better)")
    print("=" * 60)
    ranking = metrics_df.sort_values("QLIKE")
    for i, (name, row) in enumerate(ranking.iterrows()):
        print(f"  {i+1}. {name:<15} QLIKE={row['QLIKE']:.6f}  RMSE={row['RMSE']:.6f}  MZ_R²={row['MZ_R2']:.4f}")


if __name__ == "__main__":
    main()
