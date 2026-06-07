# Random Forest Granger Causality (Macro → ETF)

Tests non‑linear Granger causality from macro variables (VIX, DXY, yields) to ETF returns using random forests. The score is the improvement in out‑of‑sample R² when adding lagged macro variables to a model that uses only lagged ETF returns.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Uses all available macro variables
- Configurable lag (default 1), number of trees, max depth
- Score = R² improvement (0 to 1)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-rf-granger-macro-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast, O(n * trees * features))
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High Granger score → macro variables non‑linearly help predict ETF returns.
- Low score → macro does not improve prediction.

## Requirements

See `requirements.txt`.
