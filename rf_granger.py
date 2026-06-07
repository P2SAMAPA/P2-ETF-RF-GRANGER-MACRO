import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def rf_granger_causality(etf_returns, macro_df, lag=1, n_estimators=50, max_depth=5):
    """
    Test non‑linear Granger causality from macro variables to ETF returns.
    Returns a score = improvement in R² (or reduction in MSE) when macro variables are added.
    """
    # Align lengths
    min_len = min(len(etf_returns), len(macro_df))
    rets = etf_returns[:min_len]
    macro = macro_df.iloc[:min_len]
    if len(rets) < lag + 2:
        return 0.0
    # Build features: lagged ETF returns (and possibly lagged macro)
    # We'll create a dataset of X (lagged returns + lagged macro) and y (current return)
    # For the restricted model: only lagged ETF returns
    # For the full model: lagged ETF returns + lagged macro variables
    X_restricted = []
    X_full = []
    y = []
    for t in range(lag, len(rets)):
        # y: current ETF return
        y.append(rets[t])
        # X_restricted: lagged ETF returns (lag 1,2,...,lag)
        restricted = [rets[t - i] for i in range(1, lag+1)]
        X_restricted.append(restricted)
        # X_full: restricted + lagged macro (all macro variables at lag 1..lag)
        full = restricted.copy()
        for l in range(1, lag+1):
            macro_at_lag = macro.iloc[t - l].values
            full.extend(macro_at_lag)
        X_full.append(full)
    X_restricted = np.array(X_restricted)
    X_full = np.array(X_full)
    y = np.array(y)
    # Remove rows with any NaN
    valid = ~(np.isnan(X_restricted).any(axis=1) | np.isnan(X_full).any(axis=1) | np.isnan(y))
    X_restricted = X_restricted[valid]
    X_full = X_full[valid]
    y = y[valid]
    if len(y) < 10:
        return 0.0
    # Train random forest models
    rf_restricted = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_full = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_restricted.fit(X_restricted, y)
    rf_full.fit(X_full, y)
    # Compute R² (or MSE improvement)
    y_pred_restricted = rf_restricted.predict(X_restricted)
    y_pred_full = rf_full.predict(X_full)
    r2_restricted = 1 - np.mean((y - y_pred_restricted)**2) / np.var(y)
    r2_full = 1 - np.mean((y - y_pred_full)**2) / np.var(y)
    improvement = max(0.0, r2_full - r2_restricted)
    return improvement

def rf_granger_score(etf_returns, macro_df, lag=1, n_estimators=50, max_depth=5):
    """
    Wrapper: compute Granger causality score for a single ETF.
    """
    return rf_granger_causality(etf_returns, macro_df, lag, n_estimators, max_depth)
