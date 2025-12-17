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
import yfinance as yf

from volatility_engine import VolatilityEngine, get_volatility_interpretation
from prediction_models import VolatilityPredictor
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')


# ============================================
# NEWS FETCHING FUNCTIONS
# ============================================

def fetch_stock_news(tickers: list = None, max_news: int = 50) -> list:
    """
    Fetch latest stock news from ALL major stocks across sectors.
    Returns list of news items sorted by publish time.
    """
    import sys
    
    def log(msg):
        print(f"[NEWS] {msg}")
        sys.stdout.flush()
    
    if tickers is None:
        # Comprehensive list of stocks across ALL sectors for broad market news
        tickers = [
            # Tech Giants
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'ORCL', 'CRM', 'ADBE',
            # Semiconductors
            'AMD', 'INTC', 'AVGO', 'QCOM', 'MU', 'AMAT', 'ARM', 'SMCI',
            # Finance & Banks
            'JPM', 'GS', 'BAC', 'MS', 'WFC', 'C', 'BLK', 'SCHW',
            # Fintech & Payments
            'V', 'MA', 'PYPL', 'SQ', 'COIN', 'SOFI',
            # Healthcare & Pharma
            'JNJ', 'UNH', 'PFE', 'LLY', 'ABBV', 'MRK', 'MRNA', 'BMY', 'GILD',
            # Biotech
            'AMGN', 'BIIB', 'REGN', 'VRTX', 'ISRG',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'EOG',
            # Clean Energy & EV
            'ENPH', 'FSLR', 'RIVN', 'LCID', 'NIO',
            # Aerospace & Defense
            'BA', 'RTX', 'LMT', 'NOC', 'GD',
            # Industrial
            'CAT', 'DE', 'HON', 'GE', 'MMM', 'UPS', 'FDX',
            # Consumer & Retail
            'WMT', 'COST', 'HD', 'TGT', 'LOW', 'AMZN',
            # Food & Beverage
            'KO', 'PEP', 'MCD', 'SBUX', 'NKE',
            # Entertainment & Media
            'DIS', 'NFLX', 'WBD', 'PARA', 'CMCSA',
            # Social Media & Internet
            'SNAP', 'PINS', 'UBER', 'LYFT', 'ABNB', 'DASH',
            # Telecom
            'T', 'VZ', 'TMUS',
            # Real Estate
            'AMT', 'PLD', 'EQIX',
            # Crypto-related
            'MARA', 'RIOT', 'MSTR',
            # ETFs for broad market news
            'SPY', 'QQQ', 'DIA', 'IWM', 'VTI',
        ]
    
    all_news = []
    seen_titles = set()  # Avoid duplicates
    
    log(f"Fetching news from {len(tickers)} tickers...")
    
    for ticker in tickers[:40]:  # Scan 40 stocks for comprehensive coverage
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if news:
                for item in news[:5]:  # Top 5 news per ticker
                    # Handle new yfinance structure where data is nested under 'content'
                    content = item.get('content', item)  # Fallback to item if no content key
                    
                    title = content.get('title', '')
                    
                    # Skip if empty title or we've seen this title (dedupe)
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)
                    
                    # Parse publisher - handle nested structure
                    publisher = 'Unknown'
                    if 'provider' in content:
                        publisher = content['provider'].get('displayName', 'Unknown')
                    elif 'publisher' in content:
                        publisher = content['publisher']
                    
                    # Parse link - handle nested structure
                    link = '#'
                    if 'canonicalUrl' in content:
                        link = content['canonicalUrl'].get('url', '#')
                    elif 'clickThroughUrl' in content:
                        link = content['clickThroughUrl'].get('url', '#')
                    elif 'link' in content:
                        link = content['link']
                    
                    # Parse publish time - handle both formats
                    published = 0
                    if 'pubDate' in content:
                        # Convert ISO format to timestamp
                        try:
                            from datetime import datetime
                            pub_date = content['pubDate'].replace('Z', '+00:00')
                            dt = datetime.fromisoformat(pub_date)
                            published = int(dt.timestamp())
                        except:
                            published = 0
                    elif 'providerPublishTime' in content:
                        published = content['providerPublishTime']
                    
                    # Parse thumbnail
                    thumbnail = ''
                    if 'thumbnail' in content and content['thumbnail']:
                        resolutions = content['thumbnail'].get('resolutions', [])
                        if resolutions:
                            thumbnail = resolutions[0].get('url', '')
                    
                    # Parse the news item
                    news_item = {
                        'title': title,
                        'publisher': publisher,
                        'link': link,
                        'published': published,
                        'ticker': ticker,
                        'type': content.get('contentType', 'STORY'),
                        'thumbnail': thumbnail,
                        'related_tickers': [ticker]  # yfinance doesn't provide related tickers in new format
                    }
                    
                    # Categorize news based on keywords
                    news_item['category'] = categorize_news(title)
                    
                    all_news.append(news_item)
                    
        except Exception as e:
            log(f"Error fetching news for {ticker}: {e}")
            continue
    
    # Sort by publish time (newest first)
    all_news.sort(key=lambda x: x['published'], reverse=True)
    
    log(f"Found {len(all_news)} unique news items")
    
    return all_news[:max_news]


def categorize_news(title: str) -> str:
    """Categorize news based on title keywords"""
    title_lower = title.lower()
    
    # Mergers & Acquisitions
    if any(word in title_lower for word in ['merger', 'acquire', 'acquisition', 'buyout', 'takeover', 'deal', 'bid']):
        return '🤝 M&A'
    
    # IPO & Listings
    if any(word in title_lower for word in ['ipo', 'public offering', 'listing', 'debut', 'goes public']):
        return '🎉 IPO'
    
    # Government & Regulatory
    if any(word in title_lower for word in ['fda', 'sec', 'federal', 'regulation', 'approval', 'approved', 'antitrust', 'investigation', 'lawsuit', 'congress', 'government', 'ban', 'fine', 'penalty']):
        return '🏛️ Regulatory'
    
    # Earnings & Financials
    if any(word in title_lower for word in ['earnings', 'revenue', 'profit', 'loss', 'quarter', 'q1', 'q2', 'q3', 'q4', 'beat', 'miss', 'guidance', 'forecast']):
        return '📊 Earnings'
    
    # Analyst & Ratings
    if any(word in title_lower for word in ['upgrade', 'downgrade', 'rating', 'analyst', 'price target', 'buy rating', 'sell rating', 'outperform', 'underperform']):
        return '📈 Analyst'
    
    # Leadership & Management
    if any(word in title_lower for word in ['ceo', 'cfo', 'executive', 'resign', 'appoint', 'hire', 'fire', 'layoff', 'job cut', 'workforce']):
        return '👔 Leadership'
    
    # Products & Innovation
    if any(word in title_lower for word in ['launch', 'product', 'release', 'unveil', 'announce', 'innovation', 'patent', 'breakthrough', 'ai', 'technology']):
        return '🚀 Product'
    
    # Market Movement
    if any(word in title_lower for word in ['surge', 'plunge', 'soar', 'crash', 'rally', 'drop', 'spike', 'jump', 'fall', 'rise', 'gain', 'lose']):
        return '📉 Market'
    
    # Dividends & Buybacks
    if any(word in title_lower for word in ['dividend', 'buyback', 'repurchase', 'payout', 'shareholder']):
        return '💰 Dividend'
    
    # Default
    return '📰 News'


def get_time_ago(timestamp: int) -> str:
    """Convert Unix timestamp to human-readable time ago"""
    if not timestamp:
        return "Recently"
    
    now = datetime.now()
    published = datetime.fromtimestamp(timestamp)
    diff = now - published
    
    if diff.days > 0:
        return f"{diff.days}d ago"
    elif diff.seconds > 3600:
        return f"{diff.seconds // 3600}h ago"
    elif diff.seconds > 60:
        return f"{diff.seconds // 60}m ago"
    else:
        return "Just now"


# ============================================
# MARKET MOVERS FUNCTIONS
# ============================================

def generate_buy_recommendations(num_picks: int = 10) -> list:
    """
    Generate daily stock buy recommendations with detailed reasoning.
    Analyzes technical indicators, momentum, volatility, and fundamentals.
    """
    import sys
    
    def log(msg):
        print(f"[RECOMMEND] {msg}")
        sys.stdout.flush()
    
    # Stocks to analyze for recommendations - diverse price ranges
    recommendation_universe = [
        # Premium ($500+)
        'AVGO', 'META', 'NFLX', 'COST', 'LLY', 'ISRG', 'MSTR', 'BLK', 'ORLY', 'CMG',
        
        # High-priced ($100-$500)
        'NVDA', 'AMD', 'SMCI', 'ARM', 'PLTR', 'CRWD', 'NET', 'DDOG', 'SNOW',
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'CRM', 'ORCL', 'ADBE',
        'JPM', 'GS', 'V', 'MA', 'COIN',
        'UNH', 'ABBV', 'MRK', 'MRNA',
        'WMT', 'HD', 'NKE', 'SBUX', 'MCD',
        'CAT', 'GE', 'HON', 'BA', 'RTX',
        'XOM', 'CVX', 'COP', 'SLB',
        'ENPH', 'FSLR', 'NEE',
        'DIS', 'SPOT', 'UBER', 'ABNB',
        
        # Mid-priced ($10-$100)
        'INTC', 'MU', 'QCOM', 'MRVL', 'ON',
        'BAC', 'C', 'WFC', 'SCHW', 'SOFI', 'HOOD',
        'PFE', 'BMY', 'GILD', 'TEVA',
        'T', 'VZ', 'TMUS',
        'F', 'GM', 'RIVN', 'LCID', 'NIO',
        'SNAP', 'PINS', 'RBLX',
        'VALE', 'CLF', 'X', 'AA',
        'CCL', 'RCL', 'DAL', 'UAL', 'AAL',
        'DKNG', 'PENN',
        
        # Lower-priced ($10 and below)
        'MARA', 'RIOT', 'BITF', 'HUT',
        'AMC', 'GME',
        'SIRI', 'WBD',
        'PLUG', 'FCEL', 'BE',
        'DNA', 'GEVO', 'NKLA',
        'BB', 'NOK', 'ERIC',
        'GRAB', 'PATH',
        'OPEN', 'WISH', 'CLOV',
        'TLRY', 'CGC', 'ACB',
    ]
    
    candidates = []
    log(f"Analyzing {len(recommendation_universe)} stocks for buy recommendations...")
    
    def analyze_for_recommendation(ticker):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='3mo')
            
            if hist.empty or len(hist) < 20:
                return None
            
            info = stock.info
            
            current_price = hist['Close'].iloc[-1]
            
            # Calculate key metrics
            # 1. Price momentum (5-day, 20-day, 60-day returns)
            ret_5d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) * 100 if len(hist) >= 5 else 0
            ret_20d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) * 100 if len(hist) >= 20 else 0
            ret_60d = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
            
            # 2. Volatility
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized %
            
            # 3. Volume trend
            avg_volume_recent = hist['Volume'].tail(5).mean()
            avg_volume_older = hist['Volume'].tail(20).mean()
            volume_surge = (avg_volume_recent / avg_volume_older - 1) * 100 if avg_volume_older > 0 else 0
            
            # 4. 52-week position
            week_52_high = info.get('fiftyTwoWeekHigh', current_price)
            week_52_low = info.get('fiftyTwoWeekLow', current_price)
            pct_from_high = ((current_price - week_52_high) / week_52_high * 100) if week_52_high else 0
            pct_from_low = ((current_price - week_52_low) / week_52_low * 100) if week_52_low else 0
            
            # 5. Moving averages
            ma_20 = hist['Close'].tail(20).mean()
            ma_50 = hist['Close'].tail(50).mean() if len(hist) >= 50 else ma_20
            above_ma_20 = current_price > ma_20
            above_ma_50 = current_price > ma_50
            
            # 6. RSI (simplified)
            gains = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            losses = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.001
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))
            
            # 7. Fundamentals
            pe_ratio = info.get('trailingPE', 0) or 0
            market_cap = info.get('marketCap', 0) or 0
            revenue_growth = info.get('revenueGrowth', 0) or 0
            
            # Calculate buy score (0-100)
            score = 50  # Base score
            reasons = []
            
            # Momentum scoring
            if ret_5d > 3:
                score += 10
                reasons.append(f"Strong 5-day momentum (+{ret_5d:.1f}%)")
            elif ret_5d > 0:
                score += 5
                reasons.append(f"Positive short-term trend (+{ret_5d:.1f}%)")
            
            if ret_20d > 10:
                score += 15
                reasons.append(f"Excellent monthly performance (+{ret_20d:.1f}%)")
            elif ret_20d > 5:
                score += 10
                reasons.append(f"Solid 20-day gains (+{ret_20d:.1f}%)")
            elif ret_20d < -10:
                score -= 10
                reasons.append(f"Potential bounce play (down {ret_20d:.1f}%)")
            
            # Trend scoring
            if above_ma_20 and above_ma_50:
                score += 15
                reasons.append("Trading above 20 & 50-day moving averages (bullish trend)")
            elif above_ma_20:
                score += 8
                reasons.append("Above 20-day moving average")
            
            # Volume analysis
            if volume_surge > 50:
                score += 10
                reasons.append(f"High volume surge (+{volume_surge:.0f}% above average) signals strong interest")
            elif volume_surge > 20:
                score += 5
                reasons.append("Increasing trading volume")
            
            # RSI analysis
            if 30 < rsi < 50:
                score += 10
                reasons.append(f"RSI at {rsi:.0f} - potential oversold bounce opportunity")
            elif 50 < rsi < 70:
                score += 5
                reasons.append(f"RSI at {rsi:.0f} - healthy momentum")
            elif rsi < 30:
                score += 8
                reasons.append(f"RSI at {rsi:.0f} - oversold, potential reversal candidate")
            
            # 52-week analysis
            if pct_from_high > -5:
                score += 10
                reasons.append(f"Near 52-week high ({pct_from_high:+.1f}%) - breakout potential")
            elif pct_from_low < 15:
                score += 8
                reasons.append(f"Near 52-week low - potential value entry point")
            
            # Volatility (prefer moderate volatility for swing trades)
            if 20 < volatility < 50:
                score += 5
                reasons.append(f"Moderate volatility ({volatility:.0f}%) offers good risk/reward")
            elif volatility > 60:
                reasons.append(f"⚠️ High volatility ({volatility:.0f}%) - higher risk/reward")
            
            # Fundamentals
            if revenue_growth and revenue_growth > 0.15:
                score += 10
                reasons.append(f"Strong revenue growth ({revenue_growth*100:.0f}% YoY)")
            
            if 10 < pe_ratio < 30:
                score += 5
                reasons.append(f"Reasonable P/E ratio ({pe_ratio:.1f})")
            elif pe_ratio > 50:
                reasons.append(f"High P/E ({pe_ratio:.0f}) - priced for growth")
            
            # Market cap preference (larger = more stable)
            if market_cap > 100e9:
                score += 5
                reasons.append("Large-cap stability")
            elif market_cap > 10e9:
                score += 3
            
            return {
                'ticker': ticker,
                'price': current_price,
                'score': min(score, 100),
                'ret_5d': ret_5d,
                'ret_20d': ret_20d,
                'volatility': volatility,
                'rsi': rsi,
                'volume_surge': volume_surge,
                'pct_from_high': pct_from_high,
                'reasons': reasons[:5],  # Top 5 reasons
                'name': info.get('shortName', ticker)[:30],
                'sector': info.get('sector', 'Unknown'),
                'signal': 'STRONG BUY' if score >= 80 else 'BUY' if score >= 65 else 'WATCH'
            }
        except Exception as e:
            return None
    
    # Parallel analysis
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_for_recommendation, ticker): ticker 
                   for ticker in recommendation_universe}
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result and result['score'] >= 55:  # Only include decent candidates
                candidates.append(result)
    
    log(f"Found {len(candidates)} buy candidates")
    
    # Sort by score and return top picks
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return candidates[:num_picks]


def fetch_market_movers() -> dict:
    """
    Fetch market movers: most active, gainers, losers, 52-week highs/lows, dividends.
    Returns categorized stock data.
    """
    import sys
    
    def log(msg):
        print(f"[MOVERS] {msg}")
        sys.stdout.flush()
    
    # Extended universe for market movers
    movers_universe = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
        'ORCL', 'ADBE', 'NOW', 'SNOW', 'PLTR', 'UBER', 'ABNB', 'SHOP', 'NET', 'CRWD',
        # Semiconductors
        'AVGO', 'QCOM', 'MU', 'AMAT', 'MRVL', 'ARM', 'SMCI', 'KLAC', 'LRCX',
        # Finance
        'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK', 'SCHW', 'AXP',
        'V', 'MA', 'PYPL', 'SQ', 'COIN', 'SOFI',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'LLY', 'ABBV', 'MRK', 'MRNA', 'BMY', 'GILD', 'AMGN',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'EOG', 'PSX', 'VLO',
        # Consumer
        'WMT', 'COST', 'HD', 'TGT', 'LOW', 'NKE', 'SBUX', 'MCD',
        # Industrial
        'CAT', 'DE', 'HON', 'GE', 'BA', 'RTX', 'LMT', 'UPS', 'FDX',
        # Entertainment
        'DIS', 'NFLX', 'WBD', 'CMCSA', 'T', 'VZ',
        # REITs & Dividends
        'O', 'VICI', 'AMT', 'PLD', 'SPG',
        # Utilities (typically dividend payers)
        'NEE', 'DUK', 'SO', 'D', 'AEP',
        # High volatility
        'GME', 'AMC', 'MARA', 'RIOT', 'MSTR', 'RIVN', 'LCID', 'NIO',
        # ETFs
        'SPY', 'QQQ', 'IWM', 'DIA',
    ]
    
    results = {
        'most_active': [],
        'top_gainers': [],
        'top_losers': [],
        'week_52_high': [],
        'week_52_low': [],
        'dividend_stocks': [],
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    
    all_stocks = []
    log(f"Scanning {len(movers_universe)} stocks for market movers...")
    
    def fetch_stock_data(ticker):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='5d')
            
            if hist.empty or len(hist) < 2:
                return None
            
            info = stock.info
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change_pct = ((current_price - prev_close) / prev_close) * 100
            
            volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            
            # Get 52-week data from info
            week_52_high = info.get('fiftyTwoWeekHigh', 0)
            week_52_low = info.get('fiftyTwoWeekLow', 0)
            
            # Calculate how close to 52-week high/low
            pct_from_high = ((current_price - week_52_high) / week_52_high * 100) if week_52_high else -999
            pct_from_low = ((current_price - week_52_low) / week_52_low * 100) if week_52_low else 999
            
            # Dividend info
            dividend_yield = info.get('dividendYield', 0) or 0
            dividend_rate = info.get('dividendRate', 0) or 0
            
            return {
                'ticker': ticker,
                'price': current_price,
                'change_pct': change_pct,
                'volume': volume,
                'avg_volume': avg_volume,
                'dollar_volume': current_price * volume,
                'week_52_high': week_52_high,
                'week_52_low': week_52_low,
                'pct_from_high': pct_from_high,
                'pct_from_low': pct_from_low,
                'dividend_yield': dividend_yield * 100 if dividend_yield < 1 else dividend_yield,  # Convert to percentage
                'dividend_rate': dividend_rate,
                'name': info.get('shortName', ticker)[:25]
            }
        except Exception as e:
            return None
    
    # Parallel fetch
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_stock_data, ticker): ticker for ticker in movers_universe}
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                all_stocks.append(result)
            
            if completed % 20 == 0:
                log(f"Progress: {completed}/{len(movers_universe)}")
    
    log(f"Successfully fetched {len(all_stocks)} stocks")
    
    if not all_stocks:
        return results
    
    # Sort and categorize
    
    # Most Active (by dollar volume)
    results['most_active'] = sorted(all_stocks, key=lambda x: x['dollar_volume'], reverse=True)[:15]
    
    # Top Gainers (positive change %)
    gainers = [s for s in all_stocks if s['change_pct'] > 0]
    results['top_gainers'] = sorted(gainers, key=lambda x: x['change_pct'], reverse=True)[:15]
    
    # Top Losers (negative change %)
    losers = [s for s in all_stocks if s['change_pct'] < 0]
    results['top_losers'] = sorted(losers, key=lambda x: x['change_pct'])[:15]
    
    # 52-Week Highs (within 2% of high)
    near_highs = [s for s in all_stocks if s['pct_from_high'] > -2 and s['week_52_high'] > 0]
    results['week_52_high'] = sorted(near_highs, key=lambda x: x['pct_from_high'], reverse=True)[:15]
    
    # 52-Week Lows (within 5% of low)
    near_lows = [s for s in all_stocks if s['pct_from_low'] < 5 and s['week_52_low'] > 0]
    results['week_52_low'] = sorted(near_lows, key=lambda x: x['pct_from_low'])[:15]
    
    # Dividend Stocks (yield > 1%)
    dividend_payers = [s for s in all_stocks if s['dividend_yield'] > 1]
    results['dividend_stocks'] = sorted(dividend_payers, key=lambda x: x['dividend_yield'], reverse=True)[:15]
    
    return results

# Optimized stock universe - Top 80 most liquid stocks for faster scanning
STOCK_UNIVERSE = [
    # Tech Giants (most liquid)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC',
    'CRM', 'ORCL', 'ADBE', 'PLTR', 'UBER', 'ABNB', 'SHOP', 'CRWD', 'NET', 'DDOG',
    
    # Semiconductors
    'AVGO', 'QCOM', 'MU', 'AMAT', 'MRVL', 'ARM', 'SMCI',
    
    # Finance & Fintech
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'BLK', 'COIN', 'SOFI',
    'V', 'MA', 'PYPL', 'AXP',
    
    # Healthcare & Biotech
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'MRNA', 'ISRG',
    
    # Consumer & Retail
    'WMT', 'COST', 'HD', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP',
    
    # Streaming & Entertainment
    'NFLX', 'DIS', 'SPOT',
    
    # Social Media
    'SNAP', 'PINS', 'TTD',
    
    # Energy
    'XOM', 'CVX', 'COP', 'OXY', 'SLB',
    
    # Clean Energy & EV
    'ENPH', 'FSLR', 'RIVN', 'NIO', 'LCID',
    
    # Industrial & Aerospace
    'CAT', 'BA', 'RTX', 'GE', 'HON',
    
    # ETFs
    'SPY', 'QQQ', 'IWM', 'ARKK', 'SOXL', 'TQQQ',
    
    # High Volatility / Meme / Crypto
    'GME', 'AMC', 'MARA', 'RIOT', 'MSTR',
]


def analyze_stock_quick(ticker: str) -> dict:
    """Quickly analyze a single stock for screening"""
    try:
        engine = VolatilityEngine(ticker, period='6mo')
        engine.fetch_data()
        
        if engine.data is None or len(engine.data) < 30:
            return None
            
        vol_df = engine.calculate_all_volatilities(window=21)
        
        if vol_df is None or len(vol_df) < 20:
            return None
        
        current_vol = vol_df['Yang_Zhang'].iloc[-1]
        avg_vol = vol_df['Yang_Zhang'].mean()
        vol_trend = (current_vol - avg_vol) / avg_vol if avg_vol > 0 else 0  # Is vol rising or falling?
        
        # Get price data
        current_price = vol_df['Close'].iloc[-1]
        price_change_5d = (vol_df['Close'].iloc[-1] / vol_df['Close'].iloc[-5] - 1) if len(vol_df) >= 5 else 0
        price_change_20d = (vol_df['Close'].iloc[-1] / vol_df['Close'].iloc[-20] - 1) if len(vol_df) >= 20 else 0
        
        # Daily average volume (for liquidity)
        avg_dollar_volume = (engine.data['Close'] * engine.data['Volume']).tail(20).mean()
        
        regime = engine.detect_volatility_regime()
        
        return {
            'ticker': ticker,
            'price': float(current_price),
            'volatility': float(current_vol),
            'avg_volatility': float(avg_vol),
            'vol_trend': float(vol_trend),
            'regime': regime,
            'price_change_5d': float(price_change_5d),
            'price_change_20d': float(price_change_20d),
            'dollar_volume': float(avg_dollar_volume),
            'score_daytrade': 0,
            'score_swing': 0,
            'score_longterm': 0
        }
    except Exception:
        return None


def screen_stocks() -> dict:
    """Screen stocks and categorize for different trading styles"""
    import sys
    
    def log(msg):
        print(f"[SCREEN] {msg}")
        sys.stdout.flush()
    
    results = []
    failed = 0
    total = len(STOCK_UNIVERSE)
    
    log(f"Starting to screen {total} stocks...")
    
    # Use ThreadPoolExecutor for parallel analysis (5 workers for stability)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {executor.submit(analyze_stock_quick, ticker): ticker 
                              for ticker in STOCK_UNIVERSE}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                completed += 1
                try:
                    result = future.result(timeout=15)  # 15 second timeout per stock
                    if result:
                        results.append(result)
                    else:
                        failed += 1
                except concurrent.futures.TimeoutError:
                    failed += 1
                    log(f"Timeout analyzing {ticker}")
                except Exception as e:
                    failed += 1
                    # Only log every 10th error to avoid spam
                    if failed % 10 == 0:
                        log(f"Error analyzing {ticker}: {str(e)}")
                
                # Log progress every 25 stocks
                if completed % 25 == 0:
                    log(f"Progress: {completed}/{total} stocks analyzed, {len(results)} successful, {failed} failed")
    except Exception as e:
        log(f"ThreadPool error: {str(e)}")
    
    log(f"Screening complete: {len(results)} successful out of {total} ({failed} failed)")
    
    if not results:
        log("WARNING: No stocks were successfully analyzed!")
        return {'daytrade': [], 'swing': [], 'longterm': [], 'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')}
    
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
    
    # Get top picks by score, then sort by price (highest to lowest)
    daytrade_picks = sorted(results, key=lambda x: x['score_daytrade'], reverse=True)[:25]
    daytrade_picks = sorted(daytrade_picks, key=lambda x: x['price'], reverse=True)
    
    swing_picks = sorted(results, key=lambda x: x['score_swing'], reverse=True)[:25]
    swing_picks = sorted(swing_picks, key=lambda x: x['price'], reverse=True)
    
    longterm_picks = sorted(results, key=lambda x: x['score_longterm'], reverse=True)[:25]
    longterm_picks = sorted(longterm_picks, key=lambda x: x['price'], reverse=True)
    
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
            .news-card:hover {
                background: rgba(255, 255, 255, 0.05) !important;
                transform: translateX(5px);
            }
            .news-title-link:hover {
                color: var(--accent-cyan) !important;
                text-decoration: underline !important;
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
                html.P("Top 80 liquid stocks analyzed for volatility, liquidity, and momentum", 
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
                            html.H5("⚡ Day Trade", className="mb-2", 
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
                            html.H5("🌊 Swing Trade", className="mb-2",
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
                            html.H5("🏦 Long Term", className="mb-2",
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
                            dbc.Select(
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
                                className="bg-dark text-light border-secondary",
                                style={'fontFamily': 'JetBrains Mono'}
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
        
        # Live Stock News Section
        html.Hr(className="border-secondary my-4"),
        dbc.Row([
            dbc.Col([
                html.H3("📰 Live Market News", className="mb-2", 
                       style={'fontFamily': 'JetBrains Mono', 
                              'background': 'linear-gradient(135deg, #ffd93d 0%, #ff6b35 100%)',
                              '-webkit-background-clip': 'text',
                              '-webkit-text-fill-color': 'transparent'}),
                html.P("Real-time news from 80+ stocks across all sectors: mergers, acquisitions, FDA approvals, earnings & more", 
                       className="text-muted small mb-3"),
            ], width=9),
            dbc.Col([
                dbc.Button(
                    "🔄 Refresh News",
                    id="refresh-news-btn",
                    color="warning",
                    className="mb-3",
                    style={'fontFamily': 'JetBrains Mono', 'fontWeight': '600'}
                ),
            ], width=3, className="text-end")
        ]),
        
        # News category filters
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("Filter: ", className="text-muted me-2", style={'fontSize': '0.85rem'}),
                    dbc.ButtonGroup([
                        dbc.Button("All", id="filter-all", color="secondary", size="sm", outline=True, className="me-1"),
                        dbc.Button("🤝 M&A", id="filter-ma", color="secondary", size="sm", outline=True, className="me-1"),
                        dbc.Button("🏛️ Regulatory", id="filter-reg", color="secondary", size="sm", outline=True, className="me-1"),
                        dbc.Button("📊 Earnings", id="filter-earn", color="secondary", size="sm", outline=True, className="me-1"),
                        dbc.Button("📈 Analyst", id="filter-analyst", color="secondary", size="sm", outline=True, className="me-1"),
                        dbc.Button("🚀 Product", id="filter-prod", color="secondary", size="sm", outline=True, className="me-1"),
                        dbc.Button("📉 Market", id="filter-market", color="secondary", size="sm", outline=True, className="me-1"),
                        dbc.Button("👔 Leadership", id="filter-leader", color="secondary", size="sm", outline=True),
                    ], size="sm")
                ], className="mb-3", style={'overflowX': 'auto', 'whiteSpace': 'nowrap'})
            ], width=12)
        ]),
        
        dcc.Loading(
            id="loading-news",
            type="circle",
            color="#ffd93d",
            children=[
                html.Div(id="news-container", style={
                    'maxHeight': '800px',
                    'overflowY': 'auto',
                    'paddingRight': '10px'
                })
            ]
        ),
        
        # Store for news data (for filtering)
        dcc.Store(id='news-store'),
        
        # Auto-refresh interval (every 5 minutes)
        dcc.Interval(
            id='news-interval',
            interval=5*60*1000,  # 5 minutes in milliseconds
            n_intervals=0
        ),
        
        # Market Movers Section
        html.Hr(className="border-secondary my-4"),
        dbc.Row([
            dbc.Col([
                html.H3("🔥 Market Movers", className="mb-2", 
                       style={'fontFamily': 'JetBrains Mono', 
                              'background': 'linear-gradient(135deg, #ff6b35 0%, #ff00aa 100%)',
                              '-webkit-background-clip': 'text',
                              '-webkit-text-fill-color': 'transparent'}),
                html.P("Real-time most active, gainers, losers, 52-week highs/lows, and dividend stocks", 
                       className="text-muted small mb-3"),
            ], width=9),
            dbc.Col([
                dbc.Button(
                    "🔄 Refresh Movers",
                    id="refresh-movers-btn",
                    color="danger",
                    className="mb-3",
                    style={'fontFamily': 'JetBrains Mono', 'fontWeight': '600'}
                ),
            ], width=3, className="text-end")
        ]),
        
        dcc.Loading(
            id="loading-movers",
            type="circle",
            color="#ff6b35",
            children=[
                dbc.Row([
                    # Most Active
                    dbc.Col([
                        html.Div([
                            html.H5("📊 Most Active", className="mb-2", 
                                   style={'color': '#00d4ff', 'fontFamily': 'JetBrains Mono'}),
                            html.P("Highest trading volume today", className="text-muted small mb-2"),
                            html.Div(id="most-active-list", style={
                                'maxHeight': '400px', 'overflowY': 'auto'
                            })
                        ], className="chart-container mb-3")
                    ], md=4),
                    
                    # Top Gainers
                    dbc.Col([
                        html.Div([
                            html.H5("🚀 Top Gainers", className="mb-2", 
                                   style={'color': '#00ff88', 'fontFamily': 'JetBrains Mono'}),
                            html.P("Biggest price increases today", className="text-muted small mb-2"),
                            html.Div(id="top-gainers-list", style={
                                'maxHeight': '400px', 'overflowY': 'auto'
                            })
                        ], className="chart-container mb-3")
                    ], md=4),
                    
                    # Top Losers
                    dbc.Col([
                        html.Div([
                            html.H5("📉 Top Losers", className="mb-2", 
                                   style={'color': '#ff6b35', 'fontFamily': 'JetBrains Mono'}),
                            html.P("Biggest price decreases today", className="text-muted small mb-2"),
                            html.Div(id="top-losers-list", style={
                                'maxHeight': '400px', 'overflowY': 'auto'
                            })
                        ], className="chart-container mb-3")
                    ], md=4),
                ]),
                
                dbc.Row([
                    # 52-Week Highs
                    dbc.Col([
                        html.Div([
                            html.H5("⬆️ 52-Week Highs", className="mb-2", 
                                   style={'color': '#00ff88', 'fontFamily': 'JetBrains Mono'}),
                            html.P("Near or at yearly highs", className="text-muted small mb-2"),
                            html.Div(id="week52-high-list", style={
                                'maxHeight': '400px', 'overflowY': 'auto'
                            })
                        ], className="chart-container mb-3")
                    ], md=4),
                    
                    # 52-Week Lows
                    dbc.Col([
                        html.Div([
                            html.H5("⬇️ 52-Week Lows", className="mb-2", 
                                   style={'color': '#ff6b35', 'fontFamily': 'JetBrains Mono'}),
                            html.P("Near or at yearly lows", className="text-muted small mb-2"),
                            html.Div(id="week52-low-list", style={
                                'maxHeight': '400px', 'overflowY': 'auto'
                            })
                        ], className="chart-container mb-3")
                    ], md=4),
                    
                    # Dividend Stocks
                    dbc.Col([
                        html.Div([
                            html.H5("💰 Top Dividends", className="mb-2", 
                                   style={'color': '#ffd93d', 'fontFamily': 'JetBrains Mono'}),
                            html.P("Highest dividend yields", className="text-muted small mb-2"),
                            html.Div(id="dividend-list", style={
                                'maxHeight': '400px', 'overflowY': 'auto'
                            })
                        ], className="chart-container mb-3")
                    ], md=4),
                ]),
            ]
        ),
        
        # Auto-refresh interval for market movers (every 5 minutes)
        dcc.Interval(
            id='movers-interval',
            interval=5*60*1000,  # 5 minutes in milliseconds
            n_intervals=0
        ),
        
        # Daily Buy Recommendations Section
        html.Hr(className="border-secondary my-4"),
        dbc.Row([
            dbc.Col([
                html.H3("🎯 Daily Buy Recommendations", className="mb-2", 
                       style={'fontFamily': 'JetBrains Mono', 
                              'background': 'linear-gradient(135deg, #00ff88 0%, #00d4ff 100%)',
                              '-webkit-background-clip': 'text',
                              '-webkit-text-fill-color': 'transparent'}),
                html.P("AI-analyzed stock picks with detailed buy reasons based on momentum, technicals & fundamentals", 
                       className="text-muted small mb-3"),
            ], width=12)
        ]),
        
        # Price filter row
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Label("💰 Filter by Price Range:", className="text-muted small me-3"),
                    dbc.Select(
                        id="price-range-filter",
                        options=[
                            {'label': 'All Prices', 'value': 'all'},
                            {'label': '$500 - $1000+', 'value': '500-1000'},
                            {'label': '$100 - $500', 'value': '100-500'},
                            {'label': '$10 - $100', 'value': '10-100'},
                            {'label': '$10 or Below', 'value': '0-10'},
                        ],
                        value='all',
                        className="bg-dark text-light border-secondary",
                        style={'width': '200px', 'display': 'inline-block', 'fontFamily': 'JetBrains Mono'}
                    ),
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], width=9),
            dbc.Col([
                dbc.Button(
                    "🔄 Get Picks",
                    id="refresh-recommendations-btn",
                    color="success",
                    style={'fontFamily': 'JetBrains Mono', 'fontWeight': '600'}
                ),
            ], width=3, className="text-end")
        ], className="mb-3"),
        
        dcc.Loading(
            id="loading-recommendations",
            type="circle",
            color="#00ff88",
            children=[
                html.Div(id="recommendations-list")
            ]
        ),
        
        # Auto-refresh interval for recommendations (every 10 minutes)
        dcc.Interval(
            id='recommendations-interval',
            interval=10*60*1000,  # 10 minutes in milliseconds
            n_intervals=0
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
    import traceback
    import sys
    
    def log(msg):
        print(f"[SCAN] {msg}")
        sys.stdout.flush()
    
    log(f"Scan button clicked, n_clicks={n_clicks}")
    
    if not n_clicks:
        return (
            html.P("Click 'Scan Market' to find picks", className="text-muted"),
            html.P("Click 'Scan Market' to find picks", className="text-muted"),
            html.P("Click 'Scan Market' to find picks", className="text-muted")
        )
    
    try:
        log("Starting stock screening...")
        
        # Screen stocks
        picks = screen_stocks()
        
        log(f"Screening complete. Found: daytrade={len(picks.get('daytrade', []))}, swing={len(picks.get('swing', []))}, longterm={len(picks.get('longterm', []))}")
        
        if not picks or (not picks.get('daytrade') and not picks.get('swing') and not picks.get('longterm')):
            no_data_msg = html.Div([
                html.P("⚠️ No stocks found. This may be due to:", className="text-warning"),
                html.Ul([
                    html.Li("Market data temporarily unavailable"),
                    html.Li("Network connectivity issues"),
                    html.Li("Try again in a few moments")
                ], className="text-muted small")
            ])
            return no_data_msg, no_data_msg, no_data_msg
        
        # Create cards for each category
        daytrade_cards = [create_stock_card(s, 'daytrade') for s in picks.get('daytrade', [])]
        swing_cards = [create_stock_card(s, 'swing') for s in picks.get('swing', [])]
        longterm_cards = [create_stock_card(s, 'longterm') for s in picks.get('longterm', [])]
        
        log(f"Created cards: daytrade={len(daytrade_cards)}, swing={len(swing_cards)}, longterm={len(longterm_cards)}")
        
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
        
        last_updated = picks.get('last_updated', datetime.now().strftime('%Y-%m-%d %H:%M'))
        
        daytrade_content = html.Div([make_header(len(daytrade_cards), last_updated)] + daytrade_cards) if daytrade_cards else html.P("No day trade picks found", className="text-muted")
        swing_content = html.Div([make_header(len(swing_cards), last_updated)] + swing_cards) if swing_cards else html.P("No swing trade picks found", className="text-muted")
        longterm_content = html.Div([make_header(len(longterm_cards), last_updated)] + longterm_cards) if longterm_cards else html.P("No long-term picks found", className="text-muted")
        
        log("Returning results to UI")
        return daytrade_content, swing_content, longterm_content
        
    except Exception as e:
        tb = traceback.format_exc()
        log(f"Error during scan: {str(e)}")
        log(f"Traceback: {tb}")
        
        error_msg = html.Div([
            html.P(f"❌ Error scanning market: {str(e)}", className="text-danger"),
            html.Details([
                html.Summary("Technical details", className="text-muted small"),
                html.Pre(tb, className="text-muted small", style={'fontSize': '0.7rem', 'maxHeight': '150px', 'overflow': 'auto'})
            ])
        ])
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


def create_news_card(news_item: dict) -> html.Div:
    """Create a styled news card"""
    # Category colors
    category_colors = {
        '🤝 M&A': '#ff00aa',
        '🎉 IPO': '#00ff88',
        '🏛️ Regulatory': '#ffd93d',
        '📊 Earnings': '#00d4ff',
        '📈 Analyst': '#ff6b35',
        '👔 Leadership': '#9966ff',
        '🚀 Product': '#00ff88',
        '📉 Market': '#ff6b35',
        '💰 Dividend': '#00d4ff',
        '📰 News': '#8888aa'
    }
    
    category = news_item.get('category', '📰 News')
    color = category_colors.get(category, '#8888aa')
    time_ago = get_time_ago(news_item.get('published', 0))
    
    # Related tickers
    related = news_item.get('related_tickers', [])
    ticker_badges = [
        html.Span(t, style={
            'background': 'rgba(0, 212, 255, 0.2)',
            'color': '#00d4ff',
            'padding': '2px 6px',
            'borderRadius': '4px',
            'fontSize': '0.7rem',
            'marginRight': '4px',
            'fontFamily': 'JetBrains Mono'
        }) for t in related[:5]  # Limit to 5 tickers
    ]
    
    return html.Div([
        # Header row: Category + Time
        html.Div([
            html.Span(category, style={
                'background': f'{color}22',
                'color': color,
                'padding': '3px 10px',
                'borderRadius': '12px',
                'fontSize': '0.75rem',
                'fontWeight': '600'
            }),
            html.Span(time_ago, style={
                'color': '#8888aa',
                'fontSize': '0.75rem',
                'marginLeft': 'auto'
            })
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),
        
        # Title (link)
        html.A(
            news_item.get('title', 'Untitled'),
            href=news_item.get('link', '#'),
            target='_blank',
            style={
                'color': '#ffffff',
                'textDecoration': 'none',
                'fontWeight': '500',
                'fontSize': '0.95rem',
                'lineHeight': '1.4',
                'display': 'block',
                'marginBottom': '8px'
            },
            className='news-title-link'
        ),
        
        # Footer: Publisher + Related tickers
        html.Div([
            html.Span(news_item.get('publisher', 'Unknown'), style={
                'color': '#666688',
                'fontSize': '0.75rem',
                'marginRight': '10px'
            }),
            html.Div(ticker_badges, style={'display': 'inline-flex', 'flexWrap': 'wrap', 'gap': '2px'})
        ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap'})
        
    ], style={
        'background': 'rgba(255, 255, 255, 0.02)',
        'borderLeft': f'3px solid {color}',
        'padding': '12px 15px',
        'marginBottom': '10px',
        'borderRadius': '0 10px 10px 0',
        'transition': 'all 0.2s ease'
    }, className='news-card')


@app.callback(
    [Output("news-container", "children"),
     Output("news-store", "data")],
    [Input("refresh-news-btn", "n_clicks"),
     Input("news-interval", "n_intervals")],
    prevent_initial_call=False  # Auto-load on page open
)
def update_news(n_clicks, n_intervals):
    """Fetch and display latest news"""
    import sys
    
    def log(msg):
        print(f"[NEWS-CB] {msg}")
        sys.stdout.flush()
    
    log(f"Fetching news (clicks={n_clicks}, intervals={n_intervals})")
    
    try:
        # Fetch news
        news_items = fetch_stock_news(max_news=30)
        
        if not news_items:
            empty_msg = html.Div([
                html.P("📭 No news available at the moment.", className="text-muted"),
                html.P("Click 'Refresh News' to try again.", className="text-muted small")
            ], style={'padding': '20px', 'textAlign': 'center'})
            return empty_msg, []
        
        # Create news cards
        news_cards = [create_news_card(item) for item in news_items]
        
        # Add header with count and timestamp
        header = html.Div([
            html.Span(f"📰 {len(news_items)} Latest Headlines", style={
                'color': '#ffd93d',
                'fontFamily': 'JetBrains Mono',
                'fontSize': '0.9rem',
                'fontWeight': '600'
            }),
            html.Span(f" • Updated {datetime.now().strftime('%H:%M:%S')}", style={
                'color': '#8888aa',
                'fontSize': '0.75rem',
                'marginLeft': '10px'
            })
        ], style={'marginBottom': '15px'})
        
        log(f"Displaying {len(news_items)} news items")
        
        return html.Div([header] + news_cards), news_items
        
    except Exception as e:
        import traceback
        log(f"Error fetching news: {e}")
        error_msg = html.Div([
            html.P(f"❌ Error loading news: {str(e)}", className="text-danger"),
            html.P("Try refreshing in a moment.", className="text-muted small")
        ])
        return error_msg, []


@app.callback(
    Output("news-container", "children", allow_duplicate=True),
    [Input("filter-all", "n_clicks"),
     Input("filter-ma", "n_clicks"),
     Input("filter-reg", "n_clicks"),
     Input("filter-earn", "n_clicks"),
     Input("filter-analyst", "n_clicks"),
     Input("filter-prod", "n_clicks"),
     Input("filter-market", "n_clicks"),
     Input("filter-leader", "n_clicks")],
    [State("news-store", "data")],
    prevent_initial_call=True
)
def filter_news(all_clicks, ma_clicks, reg_clicks, earn_clicks, analyst_clicks, prod_clicks, market_clicks, leader_clicks, news_data):
    """Filter news by category"""
    from dash import ctx
    
    if not news_data:
        return html.P("No news to filter. Click 'Refresh News' first.", className="text-muted")
    
    # Determine which filter was clicked
    triggered = ctx.triggered_id
    
    category_map = {
        'filter-all': None,  # Show all
        'filter-ma': '🤝 M&A',
        'filter-reg': '🏛️ Regulatory',
        'filter-earn': '📊 Earnings',
        'filter-analyst': '📈 Analyst',
        'filter-prod': '🚀 Product',
        'filter-market': '📉 Market',
        'filter-leader': '👔 Leadership'
    }
    
    filter_category = category_map.get(triggered)
    
    # Filter news
    if filter_category is None:
        filtered = news_data
    else:
        filtered = [item for item in news_data if item.get('category') == filter_category]
    
    if not filtered:
        return html.Div([
            html.P(f"No {filter_category or 'news'} found.", className="text-muted"),
            html.P("Try another category or refresh news.", className="text-muted small")
        ])
    
    # Create cards
    news_cards = [create_news_card(item) for item in filtered]
    
    # Header
    header = html.Div([
        html.Span(f"📰 {len(filtered)} Headlines", style={
            'color': '#ffd93d',
            'fontFamily': 'JetBrains Mono',
            'fontSize': '0.9rem',
            'fontWeight': '600'
        }),
        html.Span(f" • Filter: {filter_category or 'All'}", style={
            'color': '#8888aa',
            'fontSize': '0.75rem',
            'marginLeft': '10px'
        })
    ], style={'marginBottom': '15px'})
    
    return html.Div([header] + news_cards)


def create_mover_card(stock: dict, card_type: str) -> html.Div:
    """Create a styled card for market mover stock"""
    
    # Color scheme based on card type
    colors = {
        'active': {'accent': '#00d4ff', 'bg': 'rgba(0, 212, 255, 0.1)'},
        'gainer': {'accent': '#00ff88', 'bg': 'rgba(0, 255, 136, 0.1)'},
        'loser': {'accent': '#ff6b35', 'bg': 'rgba(255, 107, 53, 0.1)'},
        'high': {'accent': '#00ff88', 'bg': 'rgba(0, 255, 136, 0.1)'},
        'low': {'accent': '#ff6b35', 'bg': 'rgba(255, 107, 53, 0.1)'},
        'dividend': {'accent': '#ffd93d', 'bg': 'rgba(255, 217, 61, 0.1)'}
    }
    
    color = colors.get(card_type, colors['active'])
    change_color = '#00ff88' if stock['change_pct'] >= 0 else '#ff6b35'
    
    # Build content based on card type
    if card_type == 'active':
        # Show volume info
        vol_str = f"{stock['volume']/1e6:.1f}M" if stock['volume'] >= 1e6 else f"{stock['volume']/1e3:.0f}K"
        detail = html.Span(f"Vol: {vol_str}", style={'color': '#8888aa', 'fontSize': '0.75rem'})
    elif card_type in ['gainer', 'loser']:
        # Show change %
        detail = html.Span(f"{stock['change_pct']:+.2f}%", style={
            'color': change_color, 'fontSize': '0.85rem', 'fontWeight': '600'
        })
    elif card_type == 'high':
        # Show % from high
        detail = html.Span(f"52W High: ${stock['week_52_high']:.2f}", style={
            'color': '#8888aa', 'fontSize': '0.75rem'
        })
    elif card_type == 'low':
        # Show % from low
        detail = html.Span(f"52W Low: ${stock['week_52_low']:.2f}", style={
            'color': '#8888aa', 'fontSize': '0.75rem'
        })
    elif card_type == 'dividend':
        # Show dividend yield
        detail = html.Span(f"Yield: {stock['dividend_yield']:.2f}%", style={
            'color': '#ffd93d', 'fontSize': '0.85rem', 'fontWeight': '600'
        })
    else:
        detail = None
    
    return html.Div([
        html.Div([
            html.Span(stock['ticker'], style={
                'fontFamily': 'JetBrains Mono',
                'fontWeight': '700',
                'fontSize': '1rem',
                'color': color['accent']
            }),
            html.Span(f"${stock['price']:.2f}", style={
                'fontFamily': 'JetBrains Mono',
                'fontSize': '0.9rem',
                'color': '#ffffff',
                'marginLeft': 'auto'
            }),
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '4px'}),
        html.Div([
            html.Span(stock.get('name', '')[:20], style={
                'color': '#666688',
                'fontSize': '0.7rem',
                'marginRight': '10px'
            }),
            detail
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
    ], style={
        'background': color['bg'],
        'borderLeft': f"3px solid {color['accent']}",
        'padding': '8px 12px',
        'marginBottom': '6px',
        'borderRadius': '0 8px 8px 0'
    })


@app.callback(
    [Output("most-active-list", "children"),
     Output("top-gainers-list", "children"),
     Output("top-losers-list", "children"),
     Output("week52-high-list", "children"),
     Output("week52-low-list", "children"),
     Output("dividend-list", "children")],
    [Input("refresh-movers-btn", "n_clicks"),
     Input("movers-interval", "n_intervals")],
    prevent_initial_call=False  # Auto-load on page open
)
def update_market_movers(n_clicks, n_intervals):
    """Fetch and display market movers"""
    import sys
    
    def log(msg):
        print(f"[MOVERS-CB] {msg}")
        sys.stdout.flush()
    
    log(f"Loading market movers (n_clicks={n_clicks})")
    
    try:
        log("Fetching market movers...")
        data = fetch_market_movers()
        
        log(f"Found: active={len(data['most_active'])}, gainers={len(data['top_gainers'])}, losers={len(data['top_losers'])}")
        
        # Create cards for each category
        def make_list(stocks, card_type, empty_msg):
            if not stocks:
                return html.P(empty_msg, className="text-muted small")
            
            header = html.Div([
                html.Span(f"{len(stocks)} stocks", style={
                    'color': '#00d4ff',
                    'fontFamily': 'JetBrains Mono',
                    'fontSize': '0.75rem'
                }),
                html.Span(f" • {data['last_updated']}", style={
                    'color': '#8888aa',
                    'fontSize': '0.7rem'
                })
            ], style={'marginBottom': '8px'})
            
            cards = [create_mover_card(s, card_type) for s in stocks]
            return html.Div([header] + cards)
        
        most_active = make_list(data['most_active'], 'active', "No active stocks found")
        top_gainers = make_list(data['top_gainers'], 'gainer', "No gainers today")
        top_losers = make_list(data['top_losers'], 'loser', "No losers today")
        week52_high = make_list(data['week_52_high'], 'high', "No stocks near 52-week highs")
        week52_low = make_list(data['week_52_low'], 'low', "No stocks near 52-week lows")
        dividend = make_list(data['dividend_stocks'], 'dividend', "No dividend stocks found")
        
        log("Returning market movers to UI")
        return most_active, top_gainers, top_losers, week52_high, week52_low, dividend
        
    except Exception as e:
        import traceback
        log(f"Error: {e}")
        log(traceback.format_exc())
        error_msg = html.P(f"Error loading: {str(e)[:50]}", className="text-danger small")
        return [error_msg] * 6


def create_recommendation_card(stock: dict) -> html.Div:
    """Create a detailed recommendation card with buy reasons"""
    
    # Signal colors
    signal_colors = {
        'STRONG BUY': {'bg': 'rgba(0, 255, 136, 0.2)', 'text': '#00ff88', 'border': '#00ff88'},
        'BUY': {'bg': 'rgba(0, 212, 255, 0.15)', 'text': '#00d4ff', 'border': '#00d4ff'},
        'WATCH': {'bg': 'rgba(255, 217, 61, 0.15)', 'text': '#ffd93d', 'border': '#ffd93d'}
    }
    
    signal = stock.get('signal', 'WATCH')
    colors = signal_colors.get(signal, signal_colors['WATCH'])
    
    # Build reason list
    reason_items = [
        html.Li(reason, style={
            'color': '#ccccdd',
            'fontSize': '0.8rem',
            'marginBottom': '4px',
            'lineHeight': '1.4'
        }) for reason in stock.get('reasons', [])
    ]
    
    return html.Div([
        # Header: Ticker, Signal, Price
        html.Div([
            html.Div([
                html.Span(stock['ticker'], style={
                    'fontFamily': 'JetBrains Mono',
                    'fontWeight': '700',
                    'fontSize': '1.3rem',
                    'color': colors['text']
                }),
                html.Span(f" • {stock.get('name', '')}", style={
                    'color': '#8888aa',
                    'fontSize': '0.8rem',
                    'marginLeft': '8px'
                }),
            ]),
            html.Div([
                html.Span(signal, style={
                    'background': colors['bg'],
                    'color': colors['text'],
                    'padding': '4px 12px',
                    'borderRadius': '15px',
                    'fontSize': '0.75rem',
                    'fontWeight': '700',
                    'fontFamily': 'JetBrains Mono',
                    'border': f"1px solid {colors['border']}"
                }),
            ])
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '10px'}),
        
        # Price and metrics row
        html.Div([
            html.Div([
                html.Span(f"${stock['price']:.2f}", style={
                    'fontFamily': 'JetBrains Mono',
                    'fontSize': '1.4rem',
                    'fontWeight': '600',
                    'color': '#ffffff'
                }),
            ]),
            html.Div([
                html.Span(f"Score: {stock['score']}/100", style={
                    'background': 'rgba(255,255,255,0.1)',
                    'padding': '3px 10px',
                    'borderRadius': '10px',
                    'fontSize': '0.75rem',
                    'color': colors['text'],
                    'fontFamily': 'JetBrains Mono'
                }),
            ])
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '12px'}),
        
        # Quick stats
        html.Div([
            html.Span(f"5D: {stock['ret_5d']:+.1f}%", style={
                'color': '#00ff88' if stock['ret_5d'] >= 0 else '#ff6b35',
                'fontSize': '0.75rem',
                'marginRight': '12px',
                'fontFamily': 'JetBrains Mono'
            }),
            html.Span(f"20D: {stock['ret_20d']:+.1f}%", style={
                'color': '#00ff88' if stock['ret_20d'] >= 0 else '#ff6b35',
                'fontSize': '0.75rem',
                'marginRight': '12px',
                'fontFamily': 'JetBrains Mono'
            }),
            html.Span(f"RSI: {stock['rsi']:.0f}", style={
                'color': '#8888aa',
                'fontSize': '0.75rem',
                'fontFamily': 'JetBrains Mono'
            }),
        ], style={'marginBottom': '12px', 'paddingBottom': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
        
        # Why Buy section
        html.Div([
            html.Span("💡 Why Buy:", style={
                'color': '#ffd93d',
                'fontWeight': '600',
                'fontSize': '0.85rem',
                'display': 'block',
                'marginBottom': '8px'
            }),
            html.Ul(reason_items, style={
                'margin': '0',
                'paddingLeft': '20px'
            })
        ]),
        
    ], style={
        'background': 'linear-gradient(135deg, rgba(26,26,36,1) 0%, rgba(30,30,45,1) 100%)',
        'border': f'1px solid {colors["border"]}40',
        'borderLeft': f'4px solid {colors["border"]}',
        'borderRadius': '12px',
        'padding': '16px 20px',
        'marginBottom': '15px',
        'boxShadow': '0 4px 15px rgba(0,0,0,0.2)'
    })


@app.callback(
    Output("recommendations-list", "children"),
    [Input("refresh-recommendations-btn", "n_clicks"),
     Input("recommendations-interval", "n_intervals"),
     Input("price-range-filter", "value")],
    prevent_initial_call=False  # Auto-load on page open
)
def update_recommendations(n_clicks, n_intervals, price_range):
    """Generate and display buy recommendations filtered by price range"""
    import sys
    
    def log(msg):
        print(f"[RECO-CB] {msg}")
        sys.stdout.flush()
    
    log(f"Loading recommendations (n_clicks={n_clicks}, price_range={price_range})")
    
    try:
        log("Generating buy recommendations...")
        recommendations = generate_buy_recommendations(num_picks=20)  # Get more to filter
        
        # Apply price filter
        if price_range and price_range != 'all':
            if price_range == '500-1000':
                recommendations = [r for r in recommendations if r['price'] >= 500]
            elif price_range == '100-500':
                recommendations = [r for r in recommendations if 100 <= r['price'] < 500]
            elif price_range == '10-100':
                recommendations = [r for r in recommendations if 10 <= r['price'] < 100]
            elif price_range == '0-10':
                recommendations = [r for r in recommendations if r['price'] < 10]
        
        # Limit to top 10 after filtering
        recommendations = recommendations[:10]
        
        if not recommendations:
            # Show helpful message based on filter
            price_msg = {
                'all': 'any price range',
                '500-1000': '$500-$1000+',
                '100-500': '$100-$500',
                '10-100': '$10-$100',
                '0-10': '$10 or below'
            }.get(price_range, 'selected range')
            
            return html.Div([
                html.P(f"📊 No strong buy signals found for stocks in {price_msg}.", className="text-muted"),
                html.P("Try a different price range or check back later!", className="text-muted small")
            ], style={'padding': '20px', 'textAlign': 'center'})
        
        log(f"Found {len(recommendations)} recommendations after filtering")
        
        # Price range label
        price_label = {
            'all': 'All Prices',
            '500-1000': '$500-$1000+',
            '100-500': '$100-$500',
            '10-100': '$10-$100',
            '0-10': '$10 or Below'
        }.get(price_range, 'All Prices')
        
        # Header
        header = html.Div([
            html.Div([
                html.Span(f"🎯 Top {len(recommendations)} Buy Picks", style={
                    'color': '#00ff88',
                    'fontFamily': 'JetBrains Mono',
                    'fontSize': '1rem',
                    'fontWeight': '600'
                }),
                html.Span(f" • {price_label}", style={
                    'color': '#ffd93d',
                    'fontSize': '0.8rem',
                    'marginLeft': '10px',
                    'background': 'rgba(255, 217, 61, 0.1)',
                    'padding': '2px 8px',
                    'borderRadius': '8px'
                }),
                html.Span(f" • Updated {datetime.now().strftime('%H:%M:%S')}", style={
                    'color': '#8888aa',
                    'fontSize': '0.75rem',
                    'marginLeft': '10px'
                })
            ]),
            html.P("Based on momentum, technicals, volume & fundamentals analysis", 
                   className="text-muted small mb-0 mt-1")
        ], style={'marginBottom': '20px'})
        
        # Create cards
        cards = [create_recommendation_card(rec) for rec in recommendations]
        
        # Split into columns for better layout
        left_cards = cards[:5]
        right_cards = cards[5:]
        
        content = dbc.Row([
            dbc.Col(left_cards, md=6),
            dbc.Col(right_cards, md=6)
        ])
        
        return html.Div([header, content])
        
    except Exception as e:
        import traceback
        log(f"Error: {e}")
        log(traceback.format_exc())
        return html.Div([
            html.P(f"❌ Error generating recommendations: {str(e)}", className="text-danger"),
            html.P("Try refreshing in a moment.", className="text-muted small")
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

