"""
Stock Volatility Engine
Fetches historical data and calculates various volatility metrics
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class VolatilityEngine:
    """Core engine for calculating stock volatility metrics"""
    
    def __init__(self, ticker: str, period: str = "2y"):
        """
        Initialize the volatility engine
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Time period for historical data (e.g., '1y', '2y', '5y')
        """
        self.ticker = ticker.upper()
        self.period = period
        self.data = None
        self.stock_info = None
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(self.ticker)
            self.stock_info = stock.info
            self.data = stock.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"No data found for ticker: {self.ticker}")
            
            # Convert timezone-aware index to timezone-naive to avoid pandas compatibility issues
            if self.data.index.tz is not None:
                self.data.index = self.data.index.tz_localize(None)
            
            # Calculate returns
            self.data['Returns'] = self.data['Close'].pct_change()
            self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
            
            return self.data
        except Exception as e:
            raise Exception(f"Error fetching data for {self.ticker}: {str(e)}")
    
    def calculate_historical_volatility(self, window: int = 21) -> pd.Series:
        """
        Calculate rolling historical volatility (annualized)
        
        Args:
            window: Rolling window size in trading days (21 = ~1 month)
        """
        if self.data is None:
            self.fetch_data()
            
        # Annualized volatility (252 trading days)
        return self.data['Returns'].rolling(window=window).std() * np.sqrt(252)
    
    def calculate_parkinson_volatility(self, window: int = 21) -> pd.Series:
        """
        Parkinson volatility - uses high/low prices for better accuracy
        More efficient than close-to-close volatility
        """
        if self.data is None:
            self.fetch_data()
            
        log_hl = np.log(self.data['High'] / self.data['Low']) ** 2
        factor = 1 / (4 * np.log(2))
        return np.sqrt(factor * log_hl.rolling(window=window).mean() * 252)
    
    def calculate_garman_klass_volatility(self, window: int = 21) -> pd.Series:
        """
        Garman-Klass volatility - uses OHLC prices
        More efficient estimator that includes overnight jumps
        """
        if self.data is None:
            self.fetch_data()
            
        log_hl = np.log(self.data['High'] / self.data['Low']) ** 2
        log_co = np.log(self.data['Close'] / self.data['Open']) ** 2
        
        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return np.sqrt(gk.rolling(window=window).mean() * 252)
    
    def calculate_rogers_satchell_volatility(self, window: int = 21) -> pd.Series:
        """
        Rogers-Satchell volatility - handles trending markets better
        Does not assume zero mean returns
        """
        if self.data is None:
            self.fetch_data()
            
        log_hc = np.log(self.data['High'] / self.data['Close'])
        log_ho = np.log(self.data['High'] / self.data['Open'])
        log_lc = np.log(self.data['Low'] / self.data['Close'])
        log_lo = np.log(self.data['Low'] / self.data['Open'])
        
        rs = log_hc * log_ho + log_lc * log_lo
        return np.sqrt(rs.rolling(window=window).mean() * 252)
    
    def calculate_yang_zhang_volatility(self, window: int = 21) -> pd.Series:
        """
        Yang-Zhang volatility - most efficient estimator
        Handles both overnight jumps and opening jumps
        """
        if self.data is None:
            self.fetch_data()
        
        # Overnight volatility
        log_oc = np.log(self.data['Open'] / self.data['Close'].shift(1))
        overnight_var = log_oc.rolling(window=window).var()
        
        # Open-to-close volatility
        log_co = np.log(self.data['Close'] / self.data['Open'])
        open_close_var = log_co.rolling(window=window).var()
        
        # Rogers-Satchell component
        log_hc = np.log(self.data['High'] / self.data['Close'])
        log_ho = np.log(self.data['High'] / self.data['Open'])
        log_lc = np.log(self.data['Low'] / self.data['Close'])
        log_lo = np.log(self.data['Low'] / self.data['Open'])
        rs = (log_hc * log_ho + log_lc * log_lo).rolling(window=window).mean()
        
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yang_zhang = overnight_var + k * open_close_var + (1 - k) * rs
        
        return np.sqrt(yang_zhang * 252)
    
    def calculate_ewma_volatility(self, span: int = 21) -> pd.Series:
        """
        Exponentially Weighted Moving Average volatility
        Gives more weight to recent observations
        """
        if self.data is None:
            self.fetch_data()
            
        return self.data['Returns'].ewm(span=span).std() * np.sqrt(252)
    
    def calculate_all_volatilities(self, window: int = 21) -> pd.DataFrame:
        """Calculate all volatility metrics and return as DataFrame"""
        if self.data is None:
            self.fetch_data()
            
        vol_df = pd.DataFrame(index=self.data.index)
        vol_df['Close'] = self.data['Close']
        vol_df['Returns'] = self.data['Returns']
        vol_df['Historical'] = self.calculate_historical_volatility(window)
        vol_df['Parkinson'] = self.calculate_parkinson_volatility(window)
        vol_df['Garman_Klass'] = self.calculate_garman_klass_volatility(window)
        vol_df['Rogers_Satchell'] = self.calculate_rogers_satchell_volatility(window)
        vol_df['Yang_Zhang'] = self.calculate_yang_zhang_volatility(window)
        vol_df['EWMA'] = self.calculate_ewma_volatility(window)
        
        return vol_df.dropna()
    
    def get_volatility_summary(self, window: int = 21) -> Dict:
        """Get summary statistics for all volatility metrics"""
        vol_df = self.calculate_all_volatilities(window)
        
        vol_columns = ['Historical', 'Parkinson', 'Garman_Klass', 
                       'Rogers_Satchell', 'Yang_Zhang', 'EWMA']
        
        summary = {
            'ticker': self.ticker,
            'period': self.period,
            'data_points': len(vol_df),
            'current_price': vol_df['Close'].iloc[-1],
            'current_volatility': {},
            'average_volatility': {},
            'min_volatility': {},
            'max_volatility': {},
            'volatility_percentile': {},
        }
        
        for col in vol_columns:
            current = vol_df[col].iloc[-1]
            summary['current_volatility'][col] = current
            summary['average_volatility'][col] = vol_df[col].mean()
            summary['min_volatility'][col] = vol_df[col].min()
            summary['max_volatility'][col] = vol_df[col].max()
            # What percentile is current volatility?
            summary['volatility_percentile'][col] = (
                (vol_df[col] < current).sum() / len(vol_df) * 100
            )
        
        return summary
    
    def detect_volatility_regime(self, window: int = 21) -> str:
        """
        Detect current volatility regime
        Returns: 'Low', 'Normal', 'High', or 'Extreme'
        """
        vol_df = self.calculate_all_volatilities(window)
        current_vol = vol_df['Yang_Zhang'].iloc[-1]  # Use Yang-Zhang as primary
        
        # Calculate percentile
        percentile = (vol_df['Yang_Zhang'] < current_vol).sum() / len(vol_df) * 100
        
        if percentile < 25:
            return 'Low'
        elif percentile < 75:
            return 'Normal'
        elif percentile < 90:
            return 'High'
        else:
            return 'Extreme'
    
    def calculate_volatility_forecast_simple(self, days: int = 30) -> Tuple[pd.Series, float]:
        """
        Simple volatility forecast using exponential smoothing
        Returns forecasted volatility values and confidence
        """
        if self.data is None:
            self.fetch_data()
        
        # Use EWMA volatility for simple forecast
        ewma_vol = self.calculate_ewma_volatility(span=21)
        current_vol = ewma_vol.iloc[-1]
        
        # Simple mean reversion forecast
        long_term_mean = ewma_vol.mean()
        reversion_speed = 0.03  # Daily mean reversion rate
        
        # Create forecast dates safely
        last_date = pd.Timestamp(self.data.index[-1])
        forecast_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days
        )
        
        forecasts = []
        vol = current_vol
        for _ in range(days):
            vol = vol + reversion_speed * (long_term_mean - vol)
            forecasts.append(vol)
        
        # Confidence based on recent volatility stability
        recent_vol_std = ewma_vol.tail(21).std()
        confidence = max(0.3, 1 - (recent_vol_std / current_vol))
        
        return pd.Series(forecasts, index=forecast_dates), confidence


def get_volatility_interpretation(volatility: float) -> str:
    """Interpret annualized volatility value"""
    if volatility < 0.15:
        return "Very Low - Stock is quite stable, lower risk but potentially lower returns"
    elif volatility < 0.25:
        return "Low - Relatively stable stock, moderate risk profile"
    elif volatility < 0.40:
        return "Moderate - Average volatility, balanced risk/reward"
    elif volatility < 0.60:
        return "High - Significant price swings expected, higher risk"
    else:
        return "Very High - Extreme volatility, very high risk, speculative"


if __name__ == "__main__":
    # Example usage
    engine = VolatilityEngine("AAPL", period="2y")
    engine.fetch_data()
    
    summary = engine.get_volatility_summary()
    print(f"\n📊 Volatility Summary for {summary['ticker']}")
    print(f"Current Price: ${summary['current_price']:.2f}")
    print(f"Volatility Regime: {engine.detect_volatility_regime()}")
    print("\nCurrent Volatility Metrics (Annualized):")
    for metric, value in summary['current_volatility'].items():
        print(f"  {metric}: {value:.2%}")

