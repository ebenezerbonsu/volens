"""
Stock Volatility Predictor Dashboard
A beautiful, modern dashboard for volatility analysis and prediction
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from volatility_engine import VolatilityEngine, get_volatility_interpretation
from prediction_models import VolatilityPredictor
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')

# Extended stock universe for comprehensive scanning (150+ stocks)
STOCK_UNIVERSE = [
    # Tech Giants & Software
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC',
    'CRM', 'ORCL', 'ADBE', 'NOW', 'SNOW', 'PLTR', 'UBER', 'LYFT', 'DASH', 'ABNB',
    'SQ', 'SHOP', 'TWLO', 'ZM', 'DOCU', 'OKTA', 'CRWD', 'ZS', 'NET', 'DDOG',
    'MDB', 'ESTC', 'PATH', 'U', 'RBLX', 'TTWO', 'EA', 'ATVI',
    
    # Semiconductors
    'AVGO', 'QCOM', 'TXN', 'MU', 'LRCX', 'AMAT', 'KLAC', 'MRVL', 'ON', 'SWKS',
    'MCHP', 'ADI', 'NXPI', 'MPWR', 'ENTG', 'ARM', 'SMCI',
    
    # Finance & Banking
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'USB', 'PNC', 'TFC', 'SCHW',
    'BLK', 'BX', 'KKR', 'APO', 'COIN', 'HOOD', 'SOFI', 'AFRM', 'UPST',
    
    # Payments & Fintech
    'V', 'MA', 'PYPL', 'AXP', 'COF', 'DFS', 'SYF',
    
    # Healthcare & Biotech
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
    'AMGN', 'GILD', 'VRTX', 'REGN', 'BIIB', 'MRNA', 'BNTX', 'ISRG', 'MDT', 'SYK',
    
    # Consumer & Retail
    'WMT', 'COST', 'TGT', 'HD', 'LOW', 'TJX', 'ROST', 'DG', 'DLTR',
    'NKE', 'LULU', 'GPS', 'ANF', 'DECK',
    'SBUX', 'MCD', 'CMG', 'DPZ', 'YUM', 'QSR',
    'KO', 'PEP', 'MNST', 'KDP',
    
    # E-commerce & Internet
    'BABA', 'JD', 'PDD', 'BIDU', 'NTES', 'SE', 'MELI', 'ETSY', 'W', 'CHWY',
    
    # Streaming & Entertainment
    'NFLX', 'DIS', 'WBD', 'PARA', 'SPOT', 'LYV', 'MSGS',
    
    # Social Media & Advertising
    'SNAP', 'PINS', 'TWTR', 'TTD', 'MGNI', 'PUBM',
    
    # Energy
    'XOM', 'CVX', 'COP', 'OXY', 'SLB', 'HAL', 'EOG', 'PXD', 'DVN', 'FANG',
    'MPC', 'VLO', 'PSX',
    
    # Clean Energy & EV
    'ENPH', 'SEDG', 'FSLR', 'RUN', 'PLUG', 'BE', 'CHPT', 'LCID', 'RIVN', 'NIO',
    'XPEV', 'LI', 'FSR',
    
    # Industrial & Aerospace
    'CAT', 'DE', 'BA', 'RTX', 'LMT', 'NOC', 'GD', 'GE', 'HON', 'MMM',
    'EMR', 'ETN', 'ROK', 'CMI', 'PCAR', 'URI',
    
    # Materials & Mining
    'LIN', 'APD', 'ECL', 'SHW', 'NEM', 'FCX', 'GOLD', 'AA', 'CLF', 'X',
    
    # Real Estate
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'AVB', 'EQR',
    
    # Telecom
    'T', 'VZ', 'TMUS', 'CMCSA', 'CHTR',
    
    # ETFs (for reference/comparison)
    'SPY', 'QQQ', 'IWM', 'DIA', 'ARKK', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI',
    'GLD', 'SLV', 'USO', 'TLT', 'HYG', 'VXX', 'SOXL', 'TQQQ',
    
    # High Volatility / Meme / Crypto-adjacent
    'GME', 'AMC', 'BBBY', 'BB', 'MARA', 'RIOT', 'CLSK', 'BITF', 'HUT',
    'MSTR', 'CIFR', 'IREN',
    
    # SPACs & Recent IPOs
    'DWAC', 'IONQ', 'JOBY', 'LILM', 'RUM', 'DNA',
]


def analyze_stock_quick(ticker: str) -> dict:
    """Quickly analyze a single stock for screening"""
    try:
        engine = VolatilityEngine(ticker, period='6mo')
        engine.fetch_data()
        
        if len(engine.data) < 30:
            return None
            
        vol_df = engine.calculate_all_volatilities(window=21)
        
        current_vol = vol_df['Yang_Zhang'].iloc[-1]
        avg_vol = vol_df['Yang_Zhang'].mean()
        vol_trend = (current_vol - avg_vol) / avg_vol  # Is vol rising or falling?
        
        # Get price data
        current_price = vol_df['Close'].iloc[-1]
        price_change_5d = (vol_df['Close'].iloc[-1] / vol_df['Close'].iloc[-5] - 1) if len(vol_df) >= 5 else 0
        price_change_20d = (vol_df['Close'].iloc[-1] / vol_df['Close'].iloc[-20] - 1) if len(vol_df) >= 20 else 0
        
        # Daily average volume (for liquidity)
        avg_dollar_volume = (engine.data['Close'] * engine.data['Volume']).tail(20).mean()
        
        regime = engine.detect_volatility_regime()
        
        return {
            'ticker': ticker,
            'price': current_price,
            'volatility': current_vol,
            'avg_volatility': avg_vol,
            'vol_trend': vol_trend,
            'regime': regime,
            'price_change_5d': price_change_5d,
            'price_change_20d': price_change_20d,
            'dollar_volume': avg_dollar_volume,
            'score_daytrade': 0,
            'score_swing': 0,
            'score_longterm': 0
        }
    except Exception as e:
        return None


def screen_stocks() -> dict:
    """Screen stocks and categorize for different trading styles"""
    results = []
    
    # Use ThreadPoolExecutor for parallel analysis (10 workers for faster scanning)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(analyze_stock_quick, ticker): ticker 
                          for ticker in STOCK_UNIVERSE}
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            result = future.result()
            if result:
                results.append(result)
    
    if not results:
        return {'daytrade': [], 'swing': [], 'longterm': []}
    
    # Score stocks for each category
    for stock in results:
        # Day Trade Score: High volatility + High liquidity + Recent momentum
        if stock['regime'] in ['High', 'Extreme']:
            stock['score_daytrade'] += 30
        elif stock['regime'] == 'Normal':
            stock['score_daytrade'] += 15
        
        if stock['volatility'] > 0.40:
            stock['score_daytrade'] += 25
        elif stock['volatility'] > 0.25:
            stock['score_daytrade'] += 15
        
        if stock['dollar_volume'] > 1e9:  # >$1B daily volume
            stock['score_daytrade'] += 25
        elif stock['dollar_volume'] > 500e6:
            stock['score_daytrade'] += 15
        
        if abs(stock['price_change_5d']) > 0.05:  # Moving stocks
            stock['score_daytrade'] += 20
        
        # Swing Trade Score: Moderate volatility + Trend + Good liquidity
        if stock['regime'] in ['Normal', 'High']:
            stock['score_swing'] += 25
        elif stock['regime'] == 'Low':
            stock['score_swing'] += 10
        
        if 0.20 < stock['volatility'] < 0.50:
            stock['score_swing'] += 25
        
        if stock['dollar_volume'] > 200e6:
            stock['score_swing'] += 20
        
        # Trending stocks are better for swing
        if stock['price_change_20d'] > 0.05:  # Uptrend
            stock['score_swing'] += 30
            stock['swing_direction'] = '📈 Bullish'
        elif stock['price_change_20d'] < -0.05:  # Downtrend (short opportunity)
            stock['score_swing'] += 20
            stock['swing_direction'] = '📉 Bearish'
        else:
            stock['swing_direction'] = '➡️ Neutral'
        
        # Long Term Score: Low volatility + Stable + Quality (using price as proxy)
        if stock['regime'] == 'Low':
            stock['score_longterm'] += 35
        elif stock['regime'] == 'Normal':
            stock['score_longterm'] += 20
        
        if stock['volatility'] < 0.25:
            stock['score_longterm'] += 30
        elif stock['volatility'] < 0.35:
            stock['score_longterm'] += 15
        
        # Favor larger, more stable companies (price > $50 often indicates established)
        if stock['price'] > 100:
            stock['score_longterm'] += 20
        elif stock['price'] > 50:
            stock['score_longterm'] += 10
        
        # Stable recent performance
        if abs(stock['price_change_20d']) < 0.10:
            stock['score_longterm'] += 15
    
    # Sort and get top 50 picks for each category
    daytrade_picks = sorted(results, key=lambda x: x['score_daytrade'], reverse=True)[:50]
    swing_picks = sorted(results, key=lambda x: x['score_swing'], reverse=True)[:50]
    longterm_picks = sorted(results, key=lambda x: x['score_longterm'], reverse=True)[:50]
    
    return {
        'daytrade': daytrade_picks,
        'swing': swing_picks,
        'longterm': longterm_picks,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
    }


def analyze_stock_for_investment(ticker: str) -> dict:
    """Analyze a stock for long-term investment potential"""
    try:
        engine = VolatilityEngine(ticker, period='2y')
        engine.fetch_data()
        
        if len(engine.data) < 252:  # Need at least 1 year of data
            return None
        
        vol_df = engine.calculate_all_volatilities(window=21)
        
        # Calculate key metrics
        current_price = vol_df['Close'].iloc[-1]
        current_vol = vol_df['Yang_Zhang'].iloc[-1]
        
        # 1-year return
        price_1y_ago = vol_df['Close'].iloc[-252] if len(vol_df) >= 252 else vol_df['Close'].iloc[0]
        return_1y = (current_price / price_1y_ago - 1)
        
        # 6-month return
        price_6m_ago = vol_df['Close'].iloc[-126] if len(vol_df) >= 126 else vol_df['Close'].iloc[0]
        return_6m = (current_price / price_6m_ago - 1)
        
        # Calculate Sharpe-like ratio (return / volatility)
        sharpe_approx = return_1y / current_vol if current_vol > 0 else 0
        
        # Calculate max drawdown in last year
        rolling_max = vol_df['Close'].tail(252).cummax()
        drawdown = (vol_df['Close'].tail(252) - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trend strength (linear regression slope)
        prices = vol_df['Close'].tail(60).values
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        trend_strength = slope / current_price * 252  # Annualized trend
        
        regime = engine.detect_volatility_regime()
        
        return {
            'ticker': ticker,
            'price': current_price,
            'volatility': current_vol,
            'return_1y': return_1y,
            'return_6m': return_6m,
            'sharpe_approx': sharpe_approx,
            'max_drawdown': max_drawdown,
            'trend_strength': trend_strength,
            'regime': regime,
            'score': 0
        }
    except Exception as e:
        return None


def build_investment_portfolio(initial_investment: float, target_return: float, 
                                time_horizon_years: int = 2) -> dict:
    """Build an investment portfolio to achieve target returns"""
    
    # Required annual return
    required_annual_return = (1 + target_return) ** (1/time_horizon_years) - 1
    
    # Analyze stocks
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Focus on growth and quality stocks for investment
        investment_universe = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 
            'CRM', 'ADBE', 'NOW', 'SNOW', 'PLTR', 'UBER', 'ABNB',
            'AVGO', 'QCOM', 'MU', 'MRVL', 'AMAT',
            'JPM', 'GS', 'V', 'MA', 'BLK',
            'JNJ', 'UNH', 'LLY', 'ABBV', 'MRK', 'PFE',
            'WMT', 'COST', 'HD', 'NKE', 'SBUX', 'MCD',
            'XOM', 'CVX', 'NEE',
            'CAT', 'BA', 'HON', 'GE', 'RTX',
            'NFLX', 'DIS',
            'SPY', 'QQQ', 'VTI', 'VOO',
            # High growth potential
            'COIN', 'MARA', 'ENPH', 'SEDG', 'RIVN', 'LCID',
            'CRWD', 'ZS', 'NET', 'DDOG', 'MDB',
            'ARM', 'SMCI', 'IONQ'
        ]
        
        future_to_ticker = {executor.submit(analyze_stock_for_investment, ticker): ticker 
                          for ticker in investment_universe}
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            result = future.result()
            if result:
                results.append(result)
    
    if not results:
        return None
    
    # Score stocks for investment
    for stock in results:
        score = 0
        
        # Reward strong 1-year returns
        if stock['return_1y'] > 0.50:
            score += 30
        elif stock['return_1y'] > 0.25:
            score += 20
        elif stock['return_1y'] > 0.10:
            score += 10
        elif stock['return_1y'] < -0.10:
            score -= 10
        
        # Reward positive 6-month momentum
        if stock['return_6m'] > 0.20:
            score += 20
        elif stock['return_6m'] > 0.10:
            score += 15
        elif stock['return_6m'] > 0:
            score += 10
        
        # Reward good risk-adjusted returns (Sharpe)
        if stock['sharpe_approx'] > 1.5:
            score += 25
        elif stock['sharpe_approx'] > 1.0:
            score += 15
        elif stock['sharpe_approx'] > 0.5:
            score += 10
        
        # Penalize high drawdowns
        if stock['max_drawdown'] > -0.15:
            score += 15
        elif stock['max_drawdown'] > -0.25:
            score += 5
        elif stock['max_drawdown'] < -0.40:
            score -= 15
        
        # Reward positive trend
        if stock['trend_strength'] > 0.30:
            score += 20
        elif stock['trend_strength'] > 0.15:
            score += 10
        elif stock['trend_strength'] < -0.10:
            score -= 10
        
        # Moderate volatility is preferred for investment
        if 0.15 < stock['volatility'] < 0.35:
            score += 15
        elif stock['volatility'] > 0.60:
            score -= 10
        
        stock['score'] = score
    
    # Sort by score
    ranked_stocks = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # Build three portfolio options
    portfolios = {}
    
    # Conservative Portfolio (Target: 40% return / $70k)
    conservative_picks = [s for s in ranked_stocks if s['volatility'] < 0.35 and s['return_1y'] > 0][:8]
    portfolios['conservative'] = {
        'name': 'Conservative Growth',
        'target': initial_investment * 1.4,
        'target_return': 0.40,
        'annual_return': 0.183,
        'stocks': conservative_picks[:6],
        'allocation': _calculate_allocation(conservative_picks[:6], initial_investment, 'conservative'),
        'description': 'Lower risk, steady growth with established companies'
    }
    
    # Moderate Portfolio (Target: 70% return / $85k)
    moderate_picks = [s for s in ranked_stocks if s['score'] > 20][:10]
    portfolios['moderate'] = {
        'name': 'Balanced Growth',
        'target': initial_investment * 1.7,
        'target_return': 0.70,
        'annual_return': 0.304,
        'stocks': moderate_picks[:8],
        'allocation': _calculate_allocation(moderate_picks[:8], initial_investment, 'moderate'),
        'description': 'Mix of growth and stability'
    }
    
    # Aggressive Portfolio (Target: 100% return / $100k)
    aggressive_picks = [s for s in ranked_stocks if s['return_1y'] > 0.20 or s['trend_strength'] > 0.20][:12]
    portfolios['aggressive'] = {
        'name': 'Aggressive Growth',
        'target': initial_investment * 2.0,
        'target_return': 1.00,
        'annual_return': 0.414,
        'stocks': aggressive_picks[:10],
        'allocation': _calculate_allocation(aggressive_picks[:10], initial_investment, 'aggressive'),
        'description': 'Higher risk, maximum growth potential'
    }
    
    return {
        'initial_investment': initial_investment,
        'portfolios': portfolios,
        'all_analyzed': ranked_stocks[:20],
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
    }


def _calculate_allocation(stocks: list, total_investment: float, risk_profile: str) -> list:
    """Calculate dollar allocation for each stock"""
    if not stocks:
        return []
    
    allocations = []
    n = len(stocks)
    
    if risk_profile == 'conservative':
        # More equal weighting for conservative
        weights = [1/n] * n
    elif risk_profile == 'moderate':
        # Slight tilt toward top picks
        weights = [0.20, 0.18, 0.15, 0.12, 0.10, 0.10, 0.08, 0.07][:n]
        weights = weights + [1/n] * (n - len(weights))
    else:  # aggressive
        # More concentrated in top picks
        weights = [0.25, 0.20, 0.15, 0.12, 0.08, 0.07, 0.05, 0.04, 0.02, 0.02][:n]
        weights = weights + [1/n] * (n - len(weights))
    
    # Normalize weights
    total_weight = sum(weights[:n])
    weights = [w/total_weight for w in weights[:n]]
    
    for i, stock in enumerate(stocks):
        dollar_amount = total_investment * weights[i]
        shares = int(dollar_amount / stock['price'])
        allocations.append({
            'ticker': stock['ticker'],
            'price': stock['price'],
            'weight': weights[i],
            'dollars': dollar_amount,
            'shares': shares,
            'return_1y': stock['return_1y'],
            'volatility': stock['volatility'],
            'score': stock['score']
        })
    
    return allocations

# Initialize the Dash app with a dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap"
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

app.title = "VoLens - Stock Volatility Predictor"

# Expose server for gunicorn
server = app.server

# Inject custom CSS via index_string
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --bg-primary: #0a0a0f;
                --bg-secondary: #12121a;
                --bg-card: #1a1a24;
                --accent-cyan: #00d4ff;
                --accent-magenta: #ff00aa;
                --accent-lime: #00ff88;
                --accent-orange: #ff6b35;
                --text-primary: #ffffff;
                --text-secondary: #8888aa;
                --gradient-1: linear-gradient(135deg, #00d4ff 0%, #ff00aa 100%);
                --gradient-2: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
            }
            body {
                background: var(--bg-primary) !important;
                font-family: 'Space Grotesk', sans-serif !important;
                color: var(--text-primary);
            }
            .dashboard-title {
                font-family: 'JetBrains Mono', monospace;
                font-size: 2.5rem;
                font-weight: 700;
                background: var(--gradient-1);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                letter-spacing: -0.02em;
            }
            .metric-card {
                background: var(--bg-card);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 16px;
                padding: 1.5rem;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            .metric-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: var(--gradient-1);
            }
            .metric-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 20px 40px rgba(0, 212, 255, 0.1);
            }
            .metric-value {
                font-family: 'JetBrains Mono', monospace;
                font-size: 2rem;
                font-weight: 600;
                color: var(--accent-cyan);
            }
            .metric-label {
                font-size: 0.85rem;
                color: var(--text-secondary);
                text-transform: uppercase;
                letter-spacing: 0.1em;
            }
            .regime-badge {
                display: inline-block;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-family: 'JetBrains Mono', monospace;
                font-weight: 600;
                font-size: 0.9rem;
            }
            .regime-low { background: rgba(0, 255, 136, 0.2); color: var(--accent-lime); border: 1px solid var(--accent-lime); }
            .regime-normal { background: rgba(0, 212, 255, 0.2); color: var(--accent-cyan); border: 1px solid var(--accent-cyan); }
            .regime-high { background: rgba(255, 107, 53, 0.2); color: var(--accent-orange); border: 1px solid var(--accent-orange); }
            .regime-extreme { background: rgba(255, 0, 170, 0.2); color: var(--accent-magenta); border: 1px solid var(--accent-magenta); }
            .recommendation-item {
                background: rgba(255, 255, 255, 0.02);
                border-left: 3px solid var(--accent-cyan);
                padding: 0.75rem 1rem;
                margin-bottom: 0.5rem;
                border-radius: 0 8px 8px 0;
                font-size: 0.95rem;
            }
            .chart-container {
                background: var(--bg-card);
                border-radius: 16px;
                padding: 1rem;
                border: 1px solid rgba(255, 255, 255, 0.05);
            }
            .input-group {
                background: var(--bg-card);
                border-radius: 12px;
                padding: 1rem;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


# Color scheme for charts
CHART_COLORS = {
    'bg': '#1a1a24',
    'grid': '#2a2a3a',
    'text': '#8888aa',
    'cyan': '#00d4ff',
    'magenta': '#ff00aa',
    'lime': '#00ff88',
    'orange': '#ff6b35',
    'yellow': '#ffd93d'
}


def create_price_volatility_chart(vol_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create combined price and volatility chart"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=[f'{ticker} Price', 'Volatility Metrics']
    )
    
    # Price chart with gradient fill
    fig.add_trace(
        go.Scatter(
            x=vol_df.index, y=vol_df['Close'],
            name='Price',
            line=dict(color=CHART_COLORS['cyan'], width=2),
            fill='tonexty',
            fillcolor='rgba(0, 212, 255, 0.1)'
        ),
        row=1, col=1
    )
    
    # Volatility metrics
    fig.add_trace(
        go.Scatter(
            x=vol_df.index, y=vol_df['Yang_Zhang'],
            name='Yang-Zhang',
            line=dict(color=CHART_COLORS['magenta'], width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=vol_df.index, y=vol_df['EWMA'],
            name='EWMA',
            line=dict(color=CHART_COLORS['lime'], width=1.5, dash='dot')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=vol_df.index, y=vol_df['Historical'],
            name='Historical',
            line=dict(color=CHART_COLORS['orange'], width=1.5, dash='dash')
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        plot_bgcolor=CHART_COLORS['bg'],
        paper_bgcolor=CHART_COLORS['bg'],
        font=dict(family='JetBrains Mono, monospace', color=CHART_COLORS['text']),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=60, r=20, t=60, b=40),
        hovermode='x unified'
    )
    
    fig.update_xaxes(
        gridcolor=CHART_COLORS['grid'],
        showgrid=True,
        zeroline=False
    )
    fig.update_yaxes(
        gridcolor=CHART_COLORS['grid'],
        showgrid=True,
        zeroline=False
    )
    
    # Format y-axes
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, tickformat='$,.0f')
    fig.update_yaxes(title_text="Volatility", row=2, col=1, tickformat='.0%')
    
    return fig


def create_price_forecast_chart(forecast_df: pd.DataFrame, current_price: float, 
                                historical_prices: pd.Series, ticker: str) -> go.Figure:
    """Create price forecast chart with volatility-based confidence cones"""
    import numpy as np
    
    fig = go.Figure()
    
    # Historical prices (last 60 days)
    recent_prices = historical_prices.tail(60)
    fig.add_trace(
        go.Scatter(
            x=recent_prices.index, y=recent_prices,
            name='Historical Price',
            line=dict(color=CHART_COLORS['cyan'], width=2)
        )
    )
    
    # Calculate price forecast based on volatility
    # Using the forecasted volatility to create confidence cones
    days = np.arange(1, len(forecast_df) + 1)
    daily_vol = forecast_df['Volatility'].values / np.sqrt(252)  # Convert to daily
    
    # Cumulative volatility for each forecast day
    cumulative_vol = np.sqrt(np.cumsum(daily_vol**2))
    
    # Expected price (assuming no drift for simplicity)
    expected_price = np.full(len(days), current_price)
    
    # 1 standard deviation bounds (~68% confidence)
    upper_1sd = current_price * (1 + cumulative_vol)
    lower_1sd = current_price * (1 - cumulative_vol)
    
    # 2 standard deviation bounds (~95% confidence)  
    upper_2sd = current_price * (1 + 2 * cumulative_vol)
    lower_2sd = current_price * (1 - 2 * cumulative_vol)
    
    # Add 2SD confidence cone (outer)
    fig.add_trace(
        go.Scatter(
            x=list(forecast_df.index) + list(forecast_df.index[::-1]),
            y=list(upper_2sd) + list(lower_2sd[::-1]),
            fill='toself',
            fillcolor='rgba(255, 0, 170, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Range (2σ)',
            showlegend=True
        )
    )
    
    # Add 1SD confidence cone (inner)
    fig.add_trace(
        go.Scatter(
            x=list(forecast_df.index) + list(forecast_df.index[::-1]),
            y=list(upper_1sd) + list(lower_1sd[::-1]),
            fill='toself',
            fillcolor='rgba(0, 212, 255, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='68% Range (1σ)',
            showlegend=True
        )
    )
    
    # Expected price line
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index, y=expected_price,
            name='Current Price',
            line=dict(color=CHART_COLORS['lime'], width=2, dash='dot')
        )
    )
    
    # Upper bound line
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index, y=upper_1sd,
            name='Upper 1σ',
            line=dict(color=CHART_COLORS['magenta'], width=1.5),
            visible='legendonly'
        )
    )
    
    # Lower bound line
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index, y=lower_1sd,
            name='Lower 1σ',
            line=dict(color=CHART_COLORS['orange'], width=1.5),
            visible='legendonly'
        )
    )
    
    # Add vertical line at forecast start
    last_hist_date = recent_prices.index[-1]
    fig.add_shape(
        type="line",
        x0=last_hist_date, x1=last_hist_date,
        y0=0, y1=1,
        yref="paper",
        line=dict(color=CHART_COLORS['text'], width=1, dash='dash')
    )
    fig.add_annotation(
        x=last_hist_date,
        y=1.05,
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        font=dict(size=10, color=CHART_COLORS['text'])
    )
    
    # Add price targets annotation
    final_upper = upper_1sd[-1]
    final_lower = lower_1sd[-1]
    fig.add_annotation(
        x=forecast_df.index[-1],
        y=final_upper,
        text=f"${final_upper:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowcolor=CHART_COLORS['magenta'],
        font=dict(size=11, color=CHART_COLORS['magenta'])
    )
    fig.add_annotation(
        x=forecast_df.index[-1],
        y=final_lower,
        text=f"${final_lower:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowcolor=CHART_COLORS['orange'],
        font=dict(size=11, color=CHART_COLORS['orange'])
    )
    
    fig.update_layout(
        title=dict(
            text=f'{ticker} Price Forecast (Volatility-Based)',
            font=dict(size=18)
        ),
        plot_bgcolor=CHART_COLORS['bg'],
        paper_bgcolor=CHART_COLORS['bg'],
        font=dict(family='JetBrains Mono, monospace', color=CHART_COLORS['text']),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=60, r=20, t=80, b=40),
        yaxis=dict(tickformat='$,.0f', title='Price'),
        xaxis=dict(title='Date'),
        hovermode='x unified'
    )
    
    fig.update_xaxes(gridcolor=CHART_COLORS['grid'], showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor=CHART_COLORS['grid'], showgrid=True, zeroline=False)
    
    return fig


def create_forecast_chart(forecast_df: pd.DataFrame, historical_vol: pd.Series, ticker: str,
                          current_price: float = None, historical_prices: pd.Series = None) -> go.Figure:
    """Create volatility forecast chart with price projections"""
    import numpy as np
    from plotly.subplots import make_subplots
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
        subplot_titles=[f'{ticker} Volatility Forecast', f'{ticker} Price Projection']
    )
    
    # Historical volatility (last 60 days)
    recent_hist = historical_vol.tail(60)
    fig.add_trace(
        go.Scatter(
            x=recent_hist.index, y=recent_hist,
            name='Historical Vol',
            line=dict(color=CHART_COLORS['cyan'], width=2)
        ),
        row=1, col=1
    )
    
    # Forecast volatility
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index, y=forecast_df['Volatility'],
            name='Forecast Vol',
            line=dict(color=CHART_COLORS['magenta'], width=2, dash='dot')
        ),
        row=1, col=1
    )
    
    # Volatility confidence interval
    fig.add_trace(
        go.Scatter(
            x=list(forecast_df.index) + list(forecast_df.index[::-1]),
            y=list(forecast_df['Upper_CI']) + list(forecast_df['Lower_CI'][::-1]),
            fill='toself',
            fillcolor='rgba(255, 0, 170, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Vol 95% CI',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Add price projections if price data is provided
    if current_price is not None and historical_prices is not None:
        recent_prices = historical_prices.tail(60)
        
        # Historical prices
        fig.add_trace(
            go.Scatter(
                x=recent_prices.index, y=recent_prices,
                name='Historical Price',
                line=dict(color=CHART_COLORS['cyan'], width=2)
            ),
            row=2, col=1
        )
        
        # Calculate price projections based on volatility
        days = np.arange(1, len(forecast_df) + 1)
        daily_vol = forecast_df['Volatility'].values / np.sqrt(252)
        cumulative_vol = np.sqrt(np.cumsum(daily_vol**2))
        
        # Price bounds (1 standard deviation)
        upper_price = current_price * (1 + cumulative_vol)
        lower_price = current_price * (1 - cumulative_vol)
        expected_price = np.full(len(days), current_price)
        
        # Price confidence cone
        fig.add_trace(
            go.Scatter(
                x=list(forecast_df.index) + list(forecast_df.index[::-1]),
                y=list(upper_price) + list(lower_price[::-1]),
                fill='toself',
                fillcolor='rgba(0, 255, 136, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Price Range (1σ)',
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Expected price line (current price)
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index, y=expected_price,
                name='Current Price',
                line=dict(color=CHART_COLORS['lime'], width=2, dash='dot')
            ),
            row=2, col=1
        )
        
        # Upper/Lower bounds
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index, y=upper_price,
                name=f'Upper: ${upper_price[-1]:.2f}',
                line=dict(color=CHART_COLORS['lime'], width=1.5)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index, y=lower_price,
                name=f'Lower: ${lower_price[-1]:.2f}',
                line=dict(color=CHART_COLORS['orange'], width=1.5)
            ),
            row=2, col=1
        )
        
        # Add price annotations at end
        fig.add_annotation(
            x=forecast_df.index[-1],
            y=upper_price[-1],
            text=f"${upper_price[-1]:.2f}",
            showarrow=False,
            xanchor='left',
            font=dict(size=11, color=CHART_COLORS['lime']),
            row=2, col=1
        )
        fig.add_annotation(
            x=forecast_df.index[-1],
            y=lower_price[-1],
            text=f"${lower_price[-1]:.2f}",
            showarrow=False,
            xanchor='left',
            font=dict(size=11, color=CHART_COLORS['orange']),
            row=2, col=1
        )
    
    # Add vertical line at forecast start
    last_hist_date = recent_hist.index[-1]
    for row in [1, 2]:
        fig.add_shape(
            type="line",
            x0=last_hist_date, x1=last_hist_date,
            y0=0, y1=1,
            yref=f"y{row} domain" if row > 1 else "y domain",
            line=dict(color=CHART_COLORS['text'], width=1, dash='dash'),
            row=row, col=1
        )
    
    fig.update_layout(
        plot_bgcolor=CHART_COLORS['bg'],
        paper_bgcolor=CHART_COLORS['bg'],
        font=dict(family='JetBrains Mono, monospace', color=CHART_COLORS['text']),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=60, r=60, t=80, b=40),
        hovermode='x unified'
    )
    
    # Format axes
    fig.update_yaxes(tickformat='.0%', title_text='Volatility', row=1, col=1,
                     gridcolor=CHART_COLORS['grid'], showgrid=True, zeroline=False)
    fig.update_yaxes(tickformat='$,.0f', title_text='Price', row=2, col=1,
                     gridcolor=CHART_COLORS['grid'], showgrid=True, zeroline=False)
    fig.update_xaxes(gridcolor=CHART_COLORS['grid'], showgrid=True, zeroline=False)
    
    return fig


def create_volatility_distribution(vol_df: pd.DataFrame, current_vol: float) -> go.Figure:
    """Create volatility distribution chart"""
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(
            x=vol_df['Yang_Zhang'],
            nbinsx=50,
            name='Volatility Distribution',
            marker=dict(
                color='rgba(0, 212, 255, 0.5)',
                line=dict(color=CHART_COLORS['cyan'], width=1)
            )
        )
    )
    
    # Add vertical line for current volatility
    fig.add_vline(
        x=current_vol,
        line=dict(color=CHART_COLORS['magenta'], width=3),
        annotation_text=f"Current: {current_vol:.1%}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=dict(text='Volatility Distribution', font=dict(size=16)),
        plot_bgcolor=CHART_COLORS['bg'],
        paper_bgcolor=CHART_COLORS['bg'],
        font=dict(family='JetBrains Mono, monospace', color=CHART_COLORS['text']),
        margin=dict(l=60, r=20, t=60, b=40),
        xaxis=dict(title='Annualized Volatility', tickformat='.0%'),
        yaxis=dict(title='Frequency'),
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor=CHART_COLORS['grid'], showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor=CHART_COLORS['grid'], showgrid=True, zeroline=False)
    
    return fig


def create_price_range_chart(current_price: float, daily_range: dict, monthly_range: dict, ticker: str) -> go.Figure:
    """Create expected price range chart"""
    fig = go.Figure()
    
    categories = ['Daily Range', '30-Day Range']
    
    # Current price line
    fig.add_trace(
        go.Scatter(
            x=categories,
            y=[current_price, current_price],
            mode='lines+markers',
            name='Current Price',
            line=dict(color=CHART_COLORS['cyan'], width=3),
            marker=dict(size=12)
        )
    )
    
    # Low range
    fig.add_trace(
        go.Scatter(
            x=categories,
            y=[daily_range['low'], monthly_range['low']],
            mode='markers',
            name='Lower Bound',
            marker=dict(size=15, symbol='triangle-down', color=CHART_COLORS['orange'])
        )
    )
    
    # High range
    fig.add_trace(
        go.Scatter(
            x=categories,
            y=[daily_range['high'], monthly_range['high']],
            mode='markers',
            name='Upper Bound',
            marker=dict(size=15, symbol='triangle-up', color=CHART_COLORS['lime'])
        )
    )
    
    # Range bars
    for i, (cat, low, high) in enumerate(zip(categories, 
                                              [daily_range['low'], monthly_range['low']],
                                              [daily_range['high'], monthly_range['high']])):
        fig.add_trace(
            go.Scatter(
                x=[cat, cat],
                y=[low, high],
                mode='lines',
                line=dict(color='rgba(255,255,255,0.3)', width=8),
                showlegend=False
            )
        )
    
    fig.update_layout(
        title=dict(text=f'{ticker} Expected Price Range', font=dict(size=16)),
        plot_bgcolor=CHART_COLORS['bg'],
        paper_bgcolor=CHART_COLORS['bg'],
        font=dict(family='JetBrains Mono, monospace', color=CHART_COLORS['text']),
        margin=dict(l=60, r=20, t=60, b=40),
        yaxis=dict(title='Price ($)', tickformat='$,.0f'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    fig.update_xaxes(gridcolor=CHART_COLORS['grid'], showgrid=False)
    fig.update_yaxes(gridcolor=CHART_COLORS['grid'], showgrid=True, zeroline=False)
    
    return fig


# App Layout
app.layout = html.Div([
    
    # Header
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("VoLens", className="dashboard-title mt-4 mb-0"),
                html.P("Stock Volatility Predictor • AI-Powered Market Analysis", 
                       className="text-muted mb-4", 
                       style={'fontFamily': 'JetBrains Mono, monospace', 'fontSize': '0.9rem'})
            ], width=12)
        ]),
        
        # Input Section
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Stock Ticker", className="text-muted small"),
                            dbc.Input(
                                id="ticker-input",
                                type="text",
                                value="AAPL",
                                placeholder="Enter ticker (e.g., AAPL, TSLA)",
                                className="bg-dark text-light border-secondary",
                                style={'fontFamily': 'JetBrains Mono'}
                            )
                        ], md=4),
                        dbc.Col([
                            dbc.Label("Time Period", className="text-muted small"),
                            dbc.Select(
                                id="period-select",
                                options=[
                                    {"label": "1 Year", "value": "1y"},
                                    {"label": "2 Years", "value": "2y"},
                                    {"label": "5 Years", "value": "5y"},
                                ],
                                value="2y",
                                className="bg-dark text-light border-secondary"
                            )
                        ], md=3),
                        dbc.Col([
                            dbc.Label("Forecast Days", className="text-muted small"),
                            dbc.Input(
                                id="forecast-days",
                                type="number",
                                value=30,
                                min=7,
                                max=90,
                                className="bg-dark text-light border-secondary",
                                style={'fontFamily': 'JetBrains Mono'}
                            )
                        ], md=2),
                        dbc.Col([
                            dbc.Label(" ", className="small"),
                            dbc.Button(
                                "Analyze",
                                id="analyze-btn",
                                color="info",
                                className="w-100",
                                style={'fontFamily': 'JetBrains Mono', 'fontWeight': '600'}
                            )
                        ], md=3, className="d-flex align-items-end")
                    ])
                ], className="input-group mb-4")
            ], width=12)
        ]),
        
        # Loading indicator
        dcc.Loading(
            id="loading",
            type="circle",
            color="#00d4ff",
            children=[
                # Metric Cards
                html.Div(id="metrics-container"),
                
                # Charts Row 1
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dcc.Graph(id="price-volatility-chart", config={'displayModeBar': False})
                        ], className="chart-container mb-4")
                    ], width=12)
                ]),
                
                # Charts Row 2 - Forecasts
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dcc.Graph(id="price-forecast-chart", config={'displayModeBar': False})
                        ], className="chart-container mb-4")
                    ], md=6),
                    dbc.Col([
                        html.Div([
                            dcc.Graph(id="forecast-chart", config={'displayModeBar': False})
                        ], className="chart-container mb-4")
                    ], md=6)
                ]),
                
                # Charts Row 3 - Distribution & Range
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dcc.Graph(id="distribution-chart", config={'displayModeBar': False})
                        ], className="chart-container mb-4")
                    ], md=4),
                    dbc.Col([
                        html.Div([
                            dcc.Graph(id="range-chart", config={'displayModeBar': False})
                        ], className="chart-container mb-4")
                    ], md=4),
                    dbc.Col([
                        html.Div(id="recommendations-container", className="chart-container mb-4")
                    ], md=4)
                ]),
            ]
        ),
        
        # Stock Picks Section
        html.Hr(className="border-secondary my-4"),
        dbc.Row([
            dbc.Col([
                html.H3("🎯 Daily Stock Picks", className="mb-3", 
                       style={'fontFamily': 'JetBrains Mono', 
                              'background': 'linear-gradient(135deg, #00d4ff 0%, #ff00aa 100%)',
                              '-webkit-background-clip': 'text',
                              '-webkit-text-fill-color': 'transparent'}),
                html.P("Stocks analyzed based on volatility, liquidity, and momentum", 
                       className="text-muted small mb-3"),
                dbc.Button(
                    "🔄 Scan Market",
                    id="scan-btn",
                    color="info",
                    className="mb-4",
                    style={'fontFamily': 'JetBrains Mono', 'fontWeight': '600'}
                ),
            ], width=12)
        ]),
        
        dcc.Loading(
            id="loading-picks",
            type="circle",
            color="#00d4ff",
            children=[
                dbc.Row([
                    # Day Trade Picks
                    dbc.Col([
                        html.Div([
                            html.H5("⚡ Day Trade (50)", className="mb-2", 
                                   style={'color': '#ff6b35', 'fontFamily': 'JetBrains Mono'}),
                            html.P("High volatility, liquid stocks for intraday moves", 
                                   className="text-muted small mb-2"),
                            html.Div(id="daytrade-picks", style={
                                'maxHeight': '500px',
                                'overflowY': 'auto',
                                'paddingRight': '10px'
                            })
                        ], className="chart-container mb-4")
                    ], md=4),
                    
                    # Swing Trade Picks
                    dbc.Col([
                        html.Div([
                            html.H5("🌊 Swing Trade (50)", className="mb-2",
                                   style={'color': '#00d4ff', 'fontFamily': 'JetBrains Mono'}),
                            html.P("Trending stocks for multi-day holds (2-10 days)", 
                                   className="text-muted small mb-2"),
                            html.Div(id="swing-picks", style={
                                'maxHeight': '500px',
                                'overflowY': 'auto',
                                'paddingRight': '10px'
                            })
                        ], className="chart-container mb-4")
                    ], md=4),
                    
                    # Long Term Picks
                    dbc.Col([
                        html.Div([
                            html.H5("🏦 Long Term (50)", className="mb-2",
                                   style={'color': '#00ff88', 'fontFamily': 'JetBrains Mono'}),
                            html.P("Stable, lower volatility stocks for investing", 
                                   className="text-muted small mb-2"),
                            html.Div(id="longterm-picks", style={
                                'maxHeight': '500px',
                                'overflowY': 'auto',
                                'paddingRight': '10px'
                            })
                        ], className="chart-container mb-4")
                    ], md=4),
                ])
            ]
        ),
        
        # Investment Portfolio Builder Section
        html.Hr(className="border-secondary my-4"),
        dbc.Row([
            dbc.Col([
                html.H3("💰 Investment Portfolio Builder", className="mb-2", 
                       style={'fontFamily': 'JetBrains Mono', 
                              'background': 'linear-gradient(135deg, #00ff88 0%, #00d4ff 100%)',
                              '-webkit-background-clip': 'text',
                              '-webkit-text-fill-color': 'transparent'}),
                html.P("AI-powered portfolio recommendations to grow your investment", 
                       className="text-muted small mb-3"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Starting Investment ($)", className="text-muted small"),
                            dbc.Input(
                                id="investment-amount",
                                type="number",
                                value=50000,
                                min=1000,
                                step=1000,
                                className="bg-dark text-light border-secondary",
                                style={'fontFamily': 'JetBrains Mono'}
                            )
                        ], md=4),
                        dbc.Col([
                            dbc.Label("Target Amount ($)", className="text-muted small"),
                            dbc.Input(
                                id="target-amount",
                                type="number",
                                value=100000,
                                min=1000,
                                step=1000,
                                className="bg-dark text-light border-secondary",
                                style={'fontFamily': 'JetBrains Mono'}
                            )
                        ], md=3),
                        dbc.Col([
                            dbc.Label("Investment Timeline", className="text-muted small"),
                            dcc.Dropdown(
                                id="investment-timeline",
                                options=[
                                    {'label': '1 Year', 'value': 1},
                                    {'label': '2 Years', 'value': 2},
                                    {'label': '3 Years', 'value': 3},
                                    {'label': '4 Years', 'value': 4},
                                    {'label': '5 Years', 'value': 5},
                                    {'label': '6 Years', 'value': 6},
                                    {'label': '7 Years', 'value': 7},
                                    {'label': '8 Years', 'value': 8},
                                    {'label': '9 Years', 'value': 9},
                                    {'label': '10 Years', 'value': 10},
                                ],
                                value=2,
                                clearable=False,
                                style={
                                    'backgroundColor': '#1a1a2e',
                                    'fontFamily': 'JetBrains Mono',
                                },
                                className="dash-dropdown-dark"
                            )
                        ], md=2),
                        dbc.Col([
                            dbc.Label(" ", className="small"),
                            dbc.Button(
                                "🔮 Build Portfolio",
                                id="build-portfolio-btn",
                                color="success",
                                className="w-100",
                                style={'fontFamily': 'JetBrains Mono', 'fontWeight': '600'}
                            )
                        ], md=3, className="d-flex align-items-end")
                    ])
                ], className="input-group mb-4")
            ], width=12)
        ]),
        
        dcc.Loading(
            id="loading-portfolio",
            type="circle",
            color="#00ff88",
            children=[
                html.Div(id="portfolio-results")
            ]
        ),
        
        # Disclaimer
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.P([
                        "⚠️ ",
                        html.Strong("DISCLAIMER: "),
                        "This is NOT financial advice. Past performance does not guarantee future results. ",
                        "Investing involves risk and you may lose money. Always do your own research and consider ",
                        "consulting a licensed financial advisor before making investment decisions."
                    ], className="small", style={'color': '#ff6b35', 'padding': '10px', 
                                                   'background': 'rgba(255, 107, 53, 0.1)',
                                                   'borderRadius': '8px', 'marginTop': '20px'})
                ])
            ], width=12)
        ]),
        
        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(className="border-secondary my-4"),
                html.P([
                    "VoLens © 2025 • Built for smarter trading decisions • ",
                    html.Span("⚠️ Not financial advice", className="text-warning")
                ], className="text-muted text-center small mb-4")
            ])
        ])
        
    ], fluid=True, style={'maxWidth': '1400px'})
])


@app.callback(
    [Output("metrics-container", "children"),
     Output("price-volatility-chart", "figure"),
     Output("price-forecast-chart", "figure"),
     Output("forecast-chart", "figure"),
     Output("distribution-chart", "figure"),
     Output("range-chart", "figure"),
     Output("recommendations-container", "children")],
    [Input("analyze-btn", "n_clicks")],
    [State("ticker-input", "value"),
     State("period-select", "value"),
     State("forecast-days", "value")],
    prevent_initial_call=False
)
def update_dashboard(n_clicks, ticker, period, forecast_days):
    """Main callback to update all dashboard components"""
    import traceback
    import sys
    
    # Log to file for debugging
    def log(msg):
        with open('/tmp/volens_debug.log', 'a') as f:
            f.write(f"{msg}\n")
        print(msg)
        sys.stdout.flush()
    
    if not ticker:
        ticker = "AAPL"
    
    ticker = ticker.upper().strip()
    forecast_days = int(forecast_days) if forecast_days else 30
    
    try:
        log(f"[DEBUG] Starting analysis for {ticker}, period={period}, days={forecast_days}")
        
        # Initialize predictor and get data
        log("[DEBUG] Step 1: Initialize predictor")
        predictor = VolatilityPredictor(ticker, period)
        
        log("[DEBUG] Step 2: Fit model")
        fit_results = predictor.fit()
        
        log("[DEBUG] Step 3: Get trading signals")
        signals = predictor.get_trading_signals(forecast_days)
        
        log("[DEBUG] Step 4: Get predictions")
        prediction = predictor.predict(forecast_days)
        
        log("[DEBUG] Step 5: Extract vol_df")
        vol_df = prediction['historical_data']
        
        log(f"[DEBUG] vol_df shape: {vol_df.shape}, index dtype: {vol_df.index.dtype}")
        
        log("[DEBUG] Step 6: Extract forecast_df")
        forecast_df = prediction['forecast']
        log(f"[DEBUG] forecast_df shape: {forecast_df.shape}, index dtype: {forecast_df.index.dtype}")
        
        # Metric Cards
        log("[DEBUG] Step 7: Create metric cards")
        regime = signals['volatility_regime']
        regime_class = f"regime-{regime.lower()}"
        
        metrics = dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div(f"${signals['current_price']:.2f}", className="metric-value"),
                    html.Div("Current Price", className="metric-label")
                ], className="metric-card")
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Div(f"{signals['current_volatility']:.1%}", className="metric-value"),
                    html.Div("Current Volatility", className="metric-label")
                ], className="metric-card")
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Div(f"{signals['forecast_volatility']:.1%}", className="metric-value"),
                    html.Div("Forecast Volatility", className="metric-label")
                ], className="metric-card")
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Span(regime, className=f"regime-badge {regime_class}"),
                    html.Div("Volatility Regime", className="metric-label mt-2")
                ], className="metric-card")
            ], md=3),
        ], className="mb-4")
        
        # Create charts
        log("[DEBUG] Step 8: Create price_volatility chart")
        price_vol_fig = create_price_volatility_chart(vol_df, ticker)
        
        log("[DEBUG] Step 9: Create price forecast chart")
        price_forecast_fig = create_price_forecast_chart(
            forecast_df,
            signals['current_price'],
            vol_df['Close'],
            ticker
        )
        
        log("[DEBUG] Step 10: Create volatility forecast chart")
        forecast_fig = create_forecast_chart(
            forecast_df, 
            vol_df['Yang_Zhang'], 
            ticker,
            current_price=signals['current_price'],
            historical_prices=vol_df['Close']
        )
        
        log("[DEBUG] Step 11: Create distribution chart")
        dist_fig = create_volatility_distribution(vol_df, signals['current_volatility'])
        
        log("[DEBUG] Step 12: Create range chart")
        range_fig = create_price_range_chart(
            signals['current_price'],
            signals['expected_daily_range'],
            signals['expected_30day_range'],
            ticker
        )
        
        # Recommendations
        recommendations_content = html.Div([
            html.H5("📋 Trading Recommendations", className="mb-3", 
                   style={'fontFamily': 'JetBrains Mono'}),
            html.Div([
                html.Div(rec, className="recommendation-item")
                for rec in signals['recommendations']
            ]),
            html.Hr(className="border-secondary my-3"),
            html.Div([
                html.H6("📊 Interpretation", className="mb-2"),
                html.P(
                    get_volatility_interpretation(signals['current_volatility']),
                    className="text-muted small"
                )
            ])
        ])
        
        return metrics, price_vol_fig, price_forecast_fig, forecast_fig, dist_fig, range_fig, recommendations_content
        
    except Exception as e:
        tb = traceback.format_exc()
        log(f"[ERROR] Exception occurred: {e}")
        log(f"[ERROR] Full traceback:\n{tb}")
        
        error_msg = html.Div([
            dbc.Alert([
                html.H5("⚠️ Error", className="alert-heading"),
                html.P(f"Could not analyze {ticker}: {str(e)}"),
                html.Hr(),
                html.P("Please check the ticker symbol and try again.", className="mb-0")
            ], color="danger")
        ])
        
        empty_fig = go.Figure()
        empty_fig.update_layout(
            plot_bgcolor=CHART_COLORS['bg'],
            paper_bgcolor=CHART_COLORS['bg'],
            font=dict(color=CHART_COLORS['text']),
            annotations=[dict(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )]
        )
        
        return error_msg, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, html.Div()


def create_stock_card(stock: dict, style: str) -> html.Div:
    """Create a styled card for a stock pick"""
    colors = {
        'daytrade': {'bg': 'rgba(255, 107, 53, 0.1)', 'border': '#ff6b35'},
        'swing': {'bg': 'rgba(0, 212, 255, 0.1)', 'border': '#00d4ff'},
        'longterm': {'bg': 'rgba(0, 255, 136, 0.1)', 'border': '#00ff88'}
    }
    
    color = colors.get(style, colors['swing'])
    
    # Direction indicator for swing trades
    direction = stock.get('swing_direction', '')
    
    return html.Div([
        html.Div([
            html.Span(stock['ticker'], style={
                'fontFamily': 'JetBrains Mono',
                'fontWeight': '700',
                'fontSize': '1.1rem',
                'color': color['border']
            }),
            html.Span(f"${stock['price']:.2f}", style={
                'fontFamily': 'JetBrains Mono',
                'fontSize': '0.9rem',
                'color': '#ffffff',
                'marginLeft': '10px'
            }),
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Span(f"Vol: {stock['volatility']:.1%}", style={
                'fontSize': '0.8rem',
                'color': '#8888aa',
                'marginRight': '10px'
            }),
            html.Span(stock['regime'], className=f"regime-badge regime-{stock['regime'].lower()}", 
                     style={'fontSize': '0.7rem', 'padding': '2px 8px'}),
            html.Span(direction, style={
                'fontSize': '0.8rem',
                'marginLeft': '10px'
            }) if direction else None,
        ]),
        html.Div([
            html.Span(f"5d: {stock['price_change_5d']:+.1%}", style={
                'fontSize': '0.75rem',
                'color': '#00ff88' if stock['price_change_5d'] > 0 else '#ff6b35',
                'marginRight': '10px'
            }),
            html.Span(f"20d: {stock['price_change_20d']:+.1%}", style={
                'fontSize': '0.75rem',
                'color': '#00ff88' if stock['price_change_20d'] > 0 else '#ff6b35'
            }),
        ], style={'marginTop': '5px'}),
    ], style={
        'background': color['bg'],
        'borderLeft': f"3px solid {color['border']}",
        'padding': '10px 12px',
        'marginBottom': '8px',
        'borderRadius': '0 8px 8px 0'
    })


@app.callback(
    [Output("daytrade-picks", "children"),
     Output("swing-picks", "children"),
     Output("longterm-picks", "children")],
    [Input("scan-btn", "n_clicks")],
    prevent_initial_call=True
)
def update_stock_picks(n_clicks):
    """Scan market and update stock picks"""
    if not n_clicks:
        return [html.P("Click 'Scan Market' to analyze 150+ stocks", className="text-muted")] * 3
    
    try:
        # Screen stocks
        picks = screen_stocks()
        
        # Create cards for each category
        daytrade_cards = [create_stock_card(s, 'daytrade') for s in picks['daytrade']]
        swing_cards = [create_stock_card(s, 'swing') for s in picks['swing']]
        longterm_cards = [create_stock_card(s, 'longterm') for s in picks['longterm']]
        
        # Add count and timestamp header
        def make_header(count, updated):
            return html.Div([
                html.Span(f"Found {count} stocks", style={
                    'color': '#00d4ff',
                    'fontFamily': 'JetBrains Mono',
                    'fontSize': '0.8rem'
                }),
                html.Span(f" • {updated}", style={
                    'color': '#8888aa',
                    'fontSize': '0.7rem',
                    'marginLeft': '10px'
                })
            ], style={'marginBottom': '10px'})
        
        return (
            html.Div([make_header(len(picks['daytrade']), picks['last_updated'])] + daytrade_cards) if daytrade_cards else html.P("No picks found", className="text-muted"),
            html.Div([make_header(len(picks['swing']), picks['last_updated'])] + swing_cards) if swing_cards else html.P("No picks found", className="text-muted"),
            html.Div([make_header(len(picks['longterm']), picks['last_updated'])] + longterm_cards) if longterm_cards else html.P("No picks found", className="text-muted")
        )
        
    except Exception as e:
        error_msg = html.P(f"Error scanning: {str(e)}", className="text-danger")
        return error_msg, error_msg, error_msg


@app.callback(
    Output("portfolio-results", "children"),
    [Input("build-portfolio-btn", "n_clicks")],
    [State("investment-amount", "value"),
     State("target-amount", "value"),
     State("investment-timeline", "value")],
    prevent_initial_call=True
)
def update_portfolio(n_clicks, investment, target, timeline):
    """Build and display investment portfolio recommendations"""
    if not n_clicks:
        return html.P("Enter your investment details and click 'Build Portfolio'", className="text-muted")
    
    try:
        investment = float(investment) if investment else 50000
        target = float(target) if target else 100000
        timeline = int(timeline) if timeline else 2
        target_return = (target / investment) - 1
        
        # Build portfolios
        result = build_investment_portfolio(investment, target_return, timeline)
        
        if not result:
            return html.P("Could not analyze stocks. Please try again.", className="text-danger")
        
        # Create portfolio cards
        def create_portfolio_card(portfolio_data, color):
            stocks = portfolio_data['allocation']
            
            stock_rows = []
            for s in stocks:
                stock_rows.append(
                    html.Tr([
                        html.Td(s['ticker'], style={'fontWeight': '600', 'color': color}),
                        html.Td(f"${s['price']:.2f}"),
                        html.Td(f"{s['weight']:.0%}"),
                        html.Td(f"${s['dollars']:,.0f}"),
                        html.Td(f"{s['shares']} shares"),
                        html.Td(f"{s['return_1y']:+.1%}", style={
                            'color': '#00ff88' if s['return_1y'] > 0 else '#ff6b35'
                        }),
                    ])
                )
            
            return html.Div([
                html.H5(portfolio_data['name'], style={'color': color, 'fontFamily': 'JetBrains Mono'}),
                html.P(portfolio_data['description'], className="text-muted small"),
                html.Div([
                    html.Span(f"Target: ${portfolio_data['target']:,.0f}", style={
                        'background': f'rgba({",".join(str(int(color.lstrip("#")[i:i+2], 16)) for i in (0, 2, 4))}, 0.2)',
                        'padding': '5px 10px',
                        'borderRadius': '15px',
                        'marginRight': '10px',
                        'fontSize': '0.9rem'
                    }),
                    html.Span(f"Required Return: {portfolio_data['target_return']:.0%} ({portfolio_data['annual_return']:.1%}/yr)", 
                             className="text-muted small"),
                ], className="mb-3"),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Stock"), html.Th("Price"), html.Th("Weight"), 
                        html.Th("Amount"), html.Th("Shares"), html.Th("1Y Return")
                    ])),
                    html.Tbody(stock_rows)
                ], className="table table-dark table-sm", style={'fontSize': '0.85rem'})
            ], className="chart-container mb-4", style={
                'borderLeft': f'4px solid {color}'
            })
        
        portfolios_ui = dbc.Row([
            dbc.Col([
                create_portfolio_card(result['portfolios']['conservative'], '#00ff88')
            ], md=4),
            dbc.Col([
                create_portfolio_card(result['portfolios']['moderate'], '#00d4ff')
            ], md=4),
            dbc.Col([
                create_portfolio_card(result['portfolios']['aggressive'], '#ff00aa')
            ], md=4),
        ])
        
        # Summary section
        summary = html.Div([
            html.H5("📊 Portfolio Summary", className="mb-3", style={'fontFamily': 'JetBrains Mono'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div(f"${investment:,.0f}", className="metric-value", 
                                style={'color': '#00d4ff', 'fontSize': '1.5rem'}),
                        html.Div("Starting Amount", className="metric-label")
                    ], className="text-center")
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.Div("→", style={'fontSize': '2rem', 'color': '#8888aa'})
                    ], className="text-center", style={'paddingTop': '10px'})
                ], md=1),
                dbc.Col([
                    html.Div([
                        html.Div(f"${result['portfolios']['conservative']['target']:,.0f}", 
                                className="metric-value", style={'color': '#00ff88', 'fontSize': '1.5rem'}),
                        html.Div("Conservative", className="metric-label")
                    ], className="text-center")
                ], md=2),
                dbc.Col([
                    html.Div([
                        html.Div(f"${result['portfolios']['moderate']['target']:,.0f}", 
                                className="metric-value", style={'color': '#00d4ff', 'fontSize': '1.5rem'}),
                        html.Div("Moderate", className="metric-label")
                    ], className="text-center")
                ], md=2),
                dbc.Col([
                    html.Div([
                        html.Div(f"${result['portfolios']['aggressive']['target']:,.0f}", 
                                className="metric-value", style={'color': '#ff00aa', 'fontSize': '1.5rem'}),
                        html.Div("Aggressive", className="metric-label")
                    ], className="text-center")
                ], md=2),
                dbc.Col([
                    html.Div([
                        html.Div(f"{timeline} {'Year' if timeline == 1 else 'Years'}", className="metric-value", 
                                style={'color': '#ffd93d', 'fontSize': '1.5rem'}),
                        html.Div("Time Horizon", className="metric-label")
                    ], className="text-center")
                ], md=2),
            ], className="mb-4 p-3", style={'background': 'rgba(255,255,255,0.02)', 'borderRadius': '12px'}),
        ])
        
        # Timestamp
        timestamp = html.P(f"Analysis generated: {result['last_updated']}", 
                          className="text-muted small text-end mt-3")
        
        return html.Div([summary, portfolios_ui, timestamp])
        
    except Exception as e:
        import traceback
        return html.Div([
            html.P(f"Error building portfolio: {str(e)}", className="text-danger"),
            html.Pre(traceback.format_exc(), className="text-muted small")
        ])


# Run the app
if __name__ == "__main__":
    import socket
    
    # Get local IP address for network access
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "Unable to detect"
    
    local_ip = get_local_ip()
    
    print("\n" + "="*60)
    print("🎯 VoLens - Stock Volatility Predictor")
    print("="*60)
    print("\n📊 Starting dashboard...")
    print("\n🌐 Access the dashboard at:")
    print(f"   • Local:   http://localhost:8050")
    print(f"   • Network: http://{local_ip}:8050")
    print("\n💡 Share the Network URL with other computers on your network!")
    print("="*60 + "\n")
    
    # Run on 0.0.0.0 to accept connections from any IP
    app.run(debug=False, host='0.0.0.0', port=8050)

