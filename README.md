# 🎯 VoLens - Stock Volatility Predictor

> AI-powered stock volatility analysis and prediction tool to help you make smarter trading decisions in 2026.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

### 📊 Multi-Method Volatility Analysis
- **Historical Volatility** - Classic close-to-close rolling volatility
- **Parkinson Volatility** - Uses high/low prices for better accuracy
- **Garman-Klass Volatility** - Incorporates OHLC data
- **Rogers-Satchell Volatility** - Handles trending markets
- **Yang-Zhang Volatility** - Most efficient estimator (handles gaps)
- **EWMA Volatility** - Exponentially weighted, emphasizes recent data

### 🤖 Machine Learning Predictions
- **GARCH Model** - Industry-standard econometric model for volatility forecasting
- **LSTM Neural Network** - Deep learning for capturing complex patterns
- **Ensemble Model** - Combines multiple models for improved accuracy

### 🎨 Beautiful Dashboard
- Real-time volatility charts
- Forecast visualization with confidence intervals
- Volatility regime detection (Low/Normal/High/Extreme)
- Expected price range calculations
- Actionable trading recommendations

## 🚀 Quick Start

### Installation

```bash
# Navigate to the project directory
cd stock-volatility-predictor

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
python dashboard.py
```

Then open your browser to **http://localhost:8050**

### Command Line Usage

```python
from prediction_models import VolatilityPredictor

# Initialize predictor
predictor = VolatilityPredictor("AAPL", period="2y")

# Fit the model
predictor.fit()

# Get trading signals
signals = predictor.get_trading_signals(days=30)

print(f"Current Price: ${signals['current_price']:.2f}")
print(f"Current Volatility: {signals['current_volatility']:.1%}")
print(f"Regime: {signals['volatility_regime']}")
print(f"\nRecommendations:")
for rec in signals['recommendations']:
    print(f"  {rec}")
```

## 📈 Understanding Volatility

### What is Volatility?
Volatility measures how much a stock's price fluctuates over time. Higher volatility means larger price swings (more risk but also more opportunity).

### Annualized Volatility Scale
| Volatility | Interpretation |
|------------|----------------|
| < 15% | Very Low - Stable, lower risk |
| 15-25% | Low - Relatively stable |
| 25-40% | Moderate - Average risk/reward |
| 40-60% | High - Significant swings expected |
| > 60% | Very High - Extreme, speculative |

### Volatility Regimes
- **🟢 Low** - Good for buy-and-hold, selling options
- **🔵 Normal** - Standard position sizing
- **🟠 High** - Reduce positions, use hedges
- **🔴 Extreme** - Maximum caution, avoid leverage

## 📊 How It Works

### 1. Data Collection
The tool fetches historical OHLC (Open, High, Low, Close) data from Yahoo Finance for any stock ticker.

### 2. Volatility Calculation
Multiple volatility estimators are calculated, each with different strengths:
- **Yang-Zhang** is the primary metric (most accurate)
- **EWMA** adapts quickly to recent changes
- **Historical** provides a baseline comparison

### 3. Prediction
The GARCH model forecasts future volatility by capturing:
- **Volatility Clustering** - High volatility tends to follow high volatility
- **Mean Reversion** - Volatility eventually returns to long-term average

### 4. Trading Signals
Based on the analysis, the tool provides:
- Expected daily and monthly price ranges
- Volatility regime classification
- Actionable trading recommendations

## 💡 Trading Applications

### Options Trading
- **Low Volatility**: Good time to BUY options (cheap premium)
- **High Volatility**: Good time to SELL options (expensive premium)

### Position Sizing
- **Low Volatility**: Can take larger positions
- **High Volatility**: Reduce position sizes

### Risk Management
- Use expected price ranges for stop-loss placement
- Adjust expectations based on volatility regime

## ⚠️ Disclaimer

This tool is for **educational and informational purposes only**. It is NOT financial advice.

- Past performance does not guarantee future results
- Volatility predictions are estimates, not certainties
- Always do your own research before trading
- Consider consulting a financial advisor

## 🛠️ Project Structure

```
stock-volatility-predictor/
├── dashboard.py           # Web dashboard (Dash/Plotly)
├── volatility_engine.py   # Core volatility calculations
├── prediction_models.py   # ML models (GARCH, LSTM)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 📚 Technical Details

### Volatility Formulas

**Historical (Close-to-Close):**
$$\sigma = \sqrt{\frac{252}{n} \sum_{i=1}^{n} (r_i - \bar{r})^2}$$

**Yang-Zhang:**
Combines overnight, open-to-close, and Rogers-Satchell components for the most efficient estimate.

### GARCH(1,1) Model

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

Where:
- $\omega$ = long-term variance weight
- $\alpha$ = reaction to recent shocks
- $\beta$ = persistence of volatility

## 🔮 Future Improvements

- [ ] Add more stocks comparison view
- [ ] Integrate VIX for market-wide context
- [ ] Add options Greeks calculator
- [ ] Portfolio volatility analysis
- [ ] Email alerts for regime changes

---

**Happy Trading! 📈**

*Remember: The best trade is the one you're prepared for.*

