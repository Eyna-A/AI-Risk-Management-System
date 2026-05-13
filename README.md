```markdown
# AI-Powered Multi-Source Risk Management System

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end AI-driven risk management platform that integrates financial, geopolitical, and macroeconomic data to predict market regimes, forecast volatility, price options, run Monte Carlo simulations, and optimize portfolios. The system uses state‑of‑the‑art deep learning techniques including attention mechanisms, gated fusion, and LSTM‑based temporal encoding.

> **Note:** The codebase is written primarily in English; comments and UI strings are in Persian for local domain understanding.

---

## Table of Contents
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [AI Model Components](#ai-model-components)
- [Installation](#installation)
- [Usage](#usage)
  - [Command‑Line Interface](#command-line-interface)
  - [Web Dashboard](#web-dashboard)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Example Outputs](#example-outputs)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [License](#license)

---

## Key Features

- **Multi‑Source Data Integration**  
  Fuses financial (price, volume, liquidity), geopolitical (EPU index, sanctions, events), and macroeconomic (forex, inflation, CDS/EMBI) data.

- **Intelligent Feature Extraction**  
  Separate neural networks for each data source followed by a **cross‑modal attention** and **gating mechanism** to learn the importance of each source dynamically.

- **Market Regime Detection**  
  Automatically identifies up to three hidden regimes (normal, transition, crisis) without rule‑based thresholds.

- **Volatility Forecasting**  
  LSTM with temporal attention predicts realised volatility over 1‑day, 5‑day, and 20‑day horizons.

- **Sentiment Analysis** (optional news input)  
  Transformer‑based sentiment scoring with uncertainty estimation and a derived EPU signal.

- **AI‑Enhanced Quantitative Tools**  
  - Black‑Scholes option pricing with AI‑adjusted volatility and sentiment‑driven risk premia.  
  - Monte Carlo simulation using regime‑dependent drift and volatility.  
  - Portfolio optimisation (maximum Sharpe or minimum variance) with regime‑sensitive adjustments.

- **Interactive Web Dashboard**  
  Flask‑based real‑time dashboard with Plotly charts and REST API for option pricing and scenario analysis.

---

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Financial      │     │  Geopolitical   │     │  Macroeconomic  │
│  Data Loader    │     │  Data Loader    │     │  Data Loader    │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MultiSourceFeatureExtractor                   │
│   (Separate MLPs for each source → shared feature dimension)    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Fusion Layer                             │
│          Cross‑Modal Attention + Gating Mechanism               │
│              → weighted combination of sources                   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
     ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
     │RegimeClassifier│   │Volatility     │   │Sentiment      │
     │ (FC + softmax) │   │ Predictor     │   │ Analyzer      │
     │  → regime probs│   │(LSTM+Attention)│   │(Transformer)  │
     └───────────────┘   └───────────────┘   └───────────────┘
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │  Risk Metrics & Output  │
                    │  • Regime probabilities│
                    │  • Volatility forecasts│
                    │  • Source importance   │
                    │  • Sentiment scores    │
                    └────────────────────────┘
```

The core AI system (`AIPoweredRiskSystem`) orchestrates all modules. A `QuantitativeIntegration` class wraps the AI outputs into practical financial analytics (options, Monte Carlo, portfolio optimisation). The Flask server exposes these functions via a clean REST API and a real‑time dashboard.

---

## AI Model Components

| Component | Description |
|-----------|-------------|
| `MultiSourceFeatureExtractor` | Separate MLPs for financial, geopolitical, and macroeconomic features. |
| `CrossModalAttention` | Multi‑head self‑attention across source embeddings to capture inter‑source relationships. |
| `GatingMechanism` | Learns a soft weight for each source to produce a fused representation. |
| `TemporalEncoder` | Bidirectional LSTM + temporal attention for sequence modelling. |
| `RegimeClassifier` | Fully connected layers that output regime probabilities and expected volatility per regime. |
| `VolatilityPredictor` | Uses the temporal encoder to forecast volatility at multiple horizons. |
| `SentimentAnalyzer` | Multi‑head attention over news embeddings to produce sentiment score (-1 to +1) and uncertainty. |

Training is handled by `RiskSystemTrainer` with a multi‑objective loss (volatility MSE, regime cross‑entropy, sentiment MSE, and source diversity regularisation).

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-risk-management.git
   cd ai-risk-management
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *If `requirements.txt` is not provided, manually install:*  
   `pip install flask flask_cors numpy pandas torch scikit-learn scipy plotly`

4. **Verify PyTorch installation**  
   The system will automatically use CUDA if available, otherwise CPU.

---

## Usage

### Command‑Line Interface

Run the main script to see a demo with synthetic data:

```bash
python risk_system.py   # or the name of the file containing the main() function
```

You will see:
- Data loading progress
- Model initialisation
- Printed risk metrics (regime probabilities, volatility forecasts, source importance)
- Interactive option pricing prompt (optional)

### Web Dashboard

Launch the Flask application:

```bash
python app.py
```

Open your browser at `http://localhost:5000`. The dashboard shows:
- Current market regime (normal/transition/crisis)
- Probability distribution over regimes
- Volatility forecast for 1, 5, and 20 days
- Source importance (financial / geopolitical / macro)
- Interactive price and volatility chart

The dashboard communicates with the backend REST API (see below).

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/risk-metrics` | GET | Returns current regime probabilities, volatility forecasts, source importance, and sentiment (if available). |
| `/api/option-pricing` | POST | Accepts `S`, `K`, `T`, `r` (JSON body) and returns AI‑adjusted option prices (call/put), implied volatility, and Greeks. |
| `/api/monte-carlo` | GET | Query parameters: `S` (spot price), `days` (horizon). Returns distribution metrics (VaR, expected shortfall, profit probability). |
| `/api/chart-data` | GET | Provides last 100 days of price and realised volatility for plotting. |

**Example POST request for option pricing:**
```bash
curl -X POST http://localhost:5000/api/option-pricing \
  -H "Content-Type: application/json" \
  -d '{"S": 10000, "K": 11000, "T": 30, "r": 0.20}'
```

**Example response:**
```json
{
  "call_price": 245.32,
  "put_price": 1250.67,
  "implied_vol": 0.385,
  "greeks": {"delta": 0.42, "gamma": 0.00012, "vega": 18.5, "theta": -2.3}
}
```

---

## Configuration

All hyperparameters are stored in a dictionary passed to `AIPoweredRiskSystem`. Default values (as in `_default_config()`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fin_dim` | 10 | Number of financial features |
| `geo_dim` | 7 | Number of geopolitical features |
| `macro_dim` | 6 | Number of macroeconomic features |
| `hidden_dim` | 64 | Hidden layer size for extractors & LSTM |
| `output_dim` | 16 | Dimension of fused representation |
| `sequence_length` | 20 | Look‑back window for temporal models |
| `num_regimes` | 3 | Number of latent regimes |
| `forecast_horizons` | [1,5,20] | Volatility prediction horizons (days) |
| `learning_rate` | 0.001 | Optimiser learning rate |

You can change these when creating the system, e.g.:
```python
custom_config = {'hidden_dim': 128, 'sequence_length': 30, ...}
system = AIPoweredRiskSystem(custom_config)
```

---

## Example Outputs

**Regime probabilities (crisis mode):**
```
Normal:      ████████░░░░░░░░░░░░ 25.3%
Transition:  ██████████████░░░░░░ 48.1%
Crisis:      ██████████████████░░ 26.6%
```

**Volatility forecasts:**
```
1 day:  24.5%
5 days: 31.2%
20 days: 42.8%
```

**Source importance (gating weights):**
```
Financial:      ████████████████████ 68%
Geopolitical:   ████████░░░░░░░░░░░░ 22%
Macroeconomic:  ████░░░░░░░░░░░░░░░░ 10%
```

**Option pricing output:**
```
Call price: 245 Toman
Put price: 1250 Toman
Implied volatility: 38.5%
```

For a complete report including portfolio optimisation and Monte Carlo, run the `generate_comprehensive_report()` function.

---

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- Flask & Flask‑CORS
- Pandas, NumPy, SciPy
- Scikit‑learn
- Plotly (for charts)

All packages can be installed via `pip install -r requirements.txt`. A sample `requirements.txt`:
```
flask==2.3.2
flask-cors==4.0.0
numpy==1.24.3
pandas==2.0.1
torch==2.0.1
scikit-learn==1.2.2
scipy==1.10.1
plotly==5.14.1
```

---

## Project Structure

```
.
├── app.py                 # Flask web application & REST API
├── risk_system.py         # Core AI models, data loaders, quant integration
├── templates/
│   └── dashboard.html     # Frontend dashboard (Plotly + JS)
├── requirements.txt
└── README.md
```

*Note: The code in `risk_system.py` contains all classes (`MultiSourceDataLoader`, `AIPoweredRiskSystem`, `QuantitativeIntegration`, etc.) and the `main()` demo.*

---

## License

This project is released under the **MIT License**. You are free to use, modify, and distribute it as long as the original copyright notice is included.

---

**Built with PyTorch & Flask** – for research and practical risk management applications.  
Contributions, issues, and feature requests are welcome!
```
