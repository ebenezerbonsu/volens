"""
Stock Volatility Prediction Models
GARCH, LSTM, and Ensemble models for volatility forecasting
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# TensorFlow/LSTM imports - optional for cloud deployment
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

from volatility_engine import VolatilityEngine


class GARCHPredictor:
    """
    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) Model
    Industry standard for volatility forecasting
    """
    
    def __init__(self, p: int = 1, q: int = 1, vol: str = 'Garch'):
        """
        Args:
            p: Order of the symmetric innovation
            q: Order of lagged volatility
            vol: Volatility model type ('Garch', 'EGarch', 'GJR-GARCH')
        """
        self.p = p
        self.q = q
        self.vol = vol
        self.model = None
        self.fitted_model = None
        
    def fit(self, returns: pd.Series) -> Dict:
        """
        Fit GARCH model to return series
        
        Args:
            returns: Series of stock returns (not percentage)
        """
        # Scale returns for numerical stability
        returns_scaled = returns.dropna() * 100
        
        self.model = arch_model(
            returns_scaled,
            vol=self.vol,
            p=self.p,
            q=self.q,
            mean='Constant',
            dist='Normal'
        )
        
        self.fitted_model = self.model.fit(disp='off')
        
        return {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'log_likelihood': self.fitted_model.loglikelihood,
            'params': dict(self.fitted_model.params)
        }
    
    def forecast(self, horizon: int = 30) -> pd.DataFrame:
        """
        Forecast volatility for specified horizon
        
        Args:
            horizon: Number of days to forecast
        
        Returns:
            DataFrame with variance and volatility forecasts
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecasts = self.fitted_model.forecast(horizon=horizon)
        
        # Convert variance to annualized volatility
        variance_forecast = forecasts.variance.iloc[-1]
        # Convert to numpy array to avoid index issues
        variance_values = np.array(variance_forecast.values)
        volatility_values = np.sqrt(variance_values * 252) / 100  # Unscale
        
        # Create forecast dates safely
        last_date = pd.Timestamp(self.fitted_model.resid.index[-1])
        forecast_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon
        )
        
        return pd.DataFrame({
            'Variance': variance_values,
            'Volatility': volatility_values,
            'Lower_CI': volatility_values * 0.8,  # Approximate CI
            'Upper_CI': volatility_values * 1.2
        }, index=forecast_dates)
    
    def get_conditional_volatility(self) -> pd.Series:
        """Get the fitted conditional volatility series"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        return np.sqrt(self.fitted_model.conditional_volatility ** 2 * 252) / 100


class LSTMPredictor:
    """
    LSTM (Long Short-Term Memory) Neural Network for volatility prediction
    Captures complex non-linear patterns in volatility dynamics
    Note: Requires TensorFlow - falls back gracefully if not available
    """
    
    def __init__(self, lookback: int = 60, units: int = 50):
        """
        Args:
            lookback: Number of past days to use for prediction
            units: Number of LSTM units
        """
        if not LSTM_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM predictions but is not installed")
        self.lookback = lookback
        self.units = units
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.volatility_scaler = MinMaxScaler(feature_range=(0, 1))
        
    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple):
        """Build LSTM model architecture"""
        if not LSTM_AVAILABLE:
            return None
        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(self.units, return_sequences=True),
            Dropout(0.2),
            LSTM(self.units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def fit(self, vol_df: pd.DataFrame, epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Fit LSTM model on volatility data
        
        Args:
            vol_df: DataFrame with volatility metrics from VolatilityEngine
            epochs: Training epochs
            batch_size: Training batch size
        """
        # Prepare features
        features = vol_df[['Returns', 'Historical', 'EWMA']].copy()
        target = vol_df['Yang_Zhang'].values  # Predict Yang-Zhang volatility
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        target_scaled = self.volatility_scaler.fit_transform(target.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self._create_sequences(features_scaled, target_scaled)
        
        # Split train/validation
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # Build and train model
        self.model = self._build_model((X.shape[1], X.shape[2]))
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        val_predictions = self.model.predict(X_val, verbose=0)
        val_predictions = self.volatility_scaler.inverse_transform(val_predictions).flatten()
        y_val_actual = self.volatility_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        
        return {
            'train_loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1],
            'val_rmse': np.sqrt(mean_squared_error(y_val_actual, val_predictions)),
            'val_mae': mean_absolute_error(y_val_actual, val_predictions),
            'epochs_trained': len(history.history['loss'])
        }
    
    def forecast(self, vol_df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
        """
        Forecast volatility for specified horizon
        
        Args:
            vol_df: Recent volatility DataFrame
            horizon: Number of days to forecast
        """
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        features = vol_df[['Returns', 'Historical', 'EWMA']].copy()
        features_scaled = self.scaler.transform(features)
        
        # Start with last lookback period
        current_sequence = features_scaled[-self.lookback:].copy()
        
        predictions = []
        for _ in range(horizon):
            # Predict next volatility
            pred = self.model.predict(current_sequence.reshape(1, self.lookback, -1), verbose=0)[0, 0]
            pred_unscaled = self.volatility_scaler.inverse_transform([[pred]])[0, 0]
            predictions.append(pred_unscaled)
            
            # Update sequence (simplified - uses predicted vol as proxy for features)
            new_row = current_sequence[-1].copy()
            new_row[1] = pred  # Update historical vol
            new_row[2] = pred  # Update EWMA
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Create forecast dates safely
        last_date = pd.Timestamp(vol_df.index[-1])
        forecast_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon
        )
        
        predictions = np.array(predictions)
        
        return pd.DataFrame({
            'Volatility': predictions,
            'Lower_CI': predictions * 0.85,
            'Upper_CI': predictions * 1.15
        }, index=forecast_dates)


class EnsemblePredictor:
    """
    Ensemble model combining GARCH and LSTM predictions
    Weighted average based on recent performance
    Falls back to GARCH-only if LSTM is not available
    """
    
    def __init__(self, garch_weight: float = 0.5):
        self.garch = GARCHPredictor(p=1, q=1)
        self.lstm = None
        self.lstm_available = LSTM_AVAILABLE
        if LSTM_AVAILABLE:
            try:
                self.lstm = LSTMPredictor(lookback=30, units=32)
            except:
                self.lstm_available = False
        self.garch_weight = 1.0 if not self.lstm_available else garch_weight
        self.lstm_weight = 0.0 if not self.lstm_available else (1 - garch_weight)
        
    def fit(self, engine: VolatilityEngine, epochs: int = 30) -> Dict:
        """Fit both models"""
        vol_df = engine.calculate_all_volatilities()
        
        # Fit GARCH
        garch_results = self.garch.fit(engine.data['Returns'])
        
        results = {'garch': garch_results}
        
        # Fit LSTM if available
        if self.lstm_available and self.lstm:
            lstm_results = self.lstm.fit(vol_df, epochs=epochs)
            results['lstm'] = lstm_results
        
        return results
    
    def forecast(self, engine: VolatilityEngine, horizon: int = 30) -> pd.DataFrame:
        """Generate ensemble forecast"""
        vol_df = engine.calculate_all_volatilities()
        
        garch_forecast = self.garch.forecast(horizon)
        
        if self.lstm_available and self.lstm:
            lstm_forecast = self.lstm.forecast(vol_df, horizon)
            # Weighted ensemble
            ensemble_vol = (
                self.garch_weight * garch_forecast['Volatility'].values +
                self.lstm_weight * lstm_forecast['Volatility'].values
            )
            return pd.DataFrame({
                'GARCH': garch_forecast['Volatility'].values,
                'LSTM': lstm_forecast['Volatility'].values,
                'Ensemble': ensemble_vol,
                'Lower_CI': ensemble_vol * 0.8,
                'Upper_CI': ensemble_vol * 1.2
            }, index=garch_forecast.index)
        else:
            # GARCH only
            return pd.DataFrame({
                'GARCH': garch_forecast['Volatility'].values,
                'Ensemble': garch_forecast['Volatility'].values,
                'Lower_CI': garch_forecast['Lower_CI'].values,
                'Upper_CI': garch_forecast['Upper_CI'].values
            }, index=garch_forecast.index)


class VolatilityPredictor:
    """
    Main prediction interface - simplified usage
    """
    
    def __init__(self, ticker: str, period: str = "2y"):
        self.ticker = ticker
        self.engine = VolatilityEngine(ticker, period)
        self.garch = GARCHPredictor()
        self.is_fitted = False
        
    def fit(self) -> Dict:
        """Fetch data and fit prediction model"""
        self.engine.fetch_data()
        vol_df = self.engine.calculate_all_volatilities()
        
        # Fit GARCH (fast and reliable)
        results = self.garch.fit(self.engine.data['Returns'])
        self.is_fitted = True
        
        return {
            'ticker': self.ticker,
            'data_points': len(vol_df),
            'model_fit': results,
            'current_regime': self.engine.detect_volatility_regime()
        }
    
    def predict(self, days: int = 30) -> Dict:
        """Generate volatility predictions"""
        if not self.is_fitted:
            self.fit()
        
        vol_df = self.engine.calculate_all_volatilities()
        summary = self.engine.get_volatility_summary()
        
        # GARCH forecast
        forecast = self.garch.forecast(days)
        
        # Simple forecast for comparison
        simple_forecast, confidence = self.engine.calculate_volatility_forecast_simple(days)
        
        return {
            'ticker': self.ticker,
            'current_price': summary['current_price'],
            'current_volatility': summary['current_volatility']['Yang_Zhang'],
            'regime': self.engine.detect_volatility_regime(),
            'forecast': forecast,
            'simple_forecast': simple_forecast,
            'confidence': confidence,
            'historical_data': vol_df
        }
    
    def get_trading_signals(self, days: int = 30) -> Dict:
        """
        Generate trading signals based on volatility analysis
        """
        prediction = self.predict(days)
        current_vol = prediction['current_volatility']
        avg_forecast_vol = prediction['forecast']['Volatility'].mean()
        
        # Calculate expected price range
        current_price = prediction['current_price']
        daily_vol = current_vol / np.sqrt(252)
        
        signals = {
            'ticker': self.ticker,
            'current_price': current_price,
            'current_volatility': current_vol,
            'volatility_regime': prediction['regime'],
            'forecast_volatility': avg_forecast_vol,
            
            # Expected daily price range (1 std dev)
            'expected_daily_range': {
                'low': current_price * (1 - daily_vol),
                'high': current_price * (1 + daily_vol)
            },
            
            # Expected 30-day price range (1 std dev)
            'expected_30day_range': {
                'low': current_price * (1 - avg_forecast_vol * np.sqrt(30/252)),
                'high': current_price * (1 + avg_forecast_vol * np.sqrt(30/252))
            },
            
            # Trading recommendations
            'recommendations': self._generate_recommendations(prediction)
        }
        
        return signals
    
    def _generate_recommendations(self, prediction: Dict) -> List[str]:
        """Generate trading recommendations based on volatility"""
        recommendations = []
        regime = prediction['regime']
        current_vol = prediction['current_volatility']
        forecast_vol = prediction['forecast']['Volatility'].mean()
        
        # Regime-based recommendations
        if regime == 'Low':
            recommendations.append("📉 Low volatility environment - consider selling options (premium is cheap)")
            recommendations.append("💡 Good time for buy-and-hold strategies")
        elif regime == 'Normal':
            recommendations.append("📊 Normal volatility - balanced approach recommended")
            recommendations.append("💡 Standard position sizing appropriate")
        elif regime == 'High':
            recommendations.append("📈 High volatility - reduce position sizes")
            recommendations.append("💡 Consider buying protective puts")
            recommendations.append("⚠️ Wider stop-losses may be needed")
        else:  # Extreme
            recommendations.append("🔴 EXTREME volatility - highest caution required")
            recommendations.append("💡 Consider hedging or reducing exposure")
            recommendations.append("⚠️ Avoid leverage and speculative positions")
        
        # Trend-based recommendations
        if forecast_vol > current_vol * 1.1:
            recommendations.append("⬆️ Volatility expected to INCREASE - consider buying options")
        elif forecast_vol < current_vol * 0.9:
            recommendations.append("⬇️ Volatility expected to DECREASE - consider selling options")
        else:
            recommendations.append("➡️ Volatility expected to remain STABLE")
        
        return recommendations


if __name__ == "__main__":
    # Example usage
    predictor = VolatilityPredictor("TSLA", period="2y")
    
    print("Fitting model...")
    fit_results = predictor.fit()
    print(f"Model fitted. Current regime: {fit_results['current_regime']}")
    
    print("\nGenerating predictions...")
    signals = predictor.get_trading_signals(30)
    
    print(f"\n🎯 Trading Signals for {signals['ticker']}")
    print(f"Current Price: ${signals['current_price']:.2f}")
    print(f"Current Volatility: {signals['current_volatility']:.2%}")
    print(f"Regime: {signals['volatility_regime']}")
    print(f"\nExpected Daily Range: ${signals['expected_daily_range']['low']:.2f} - ${signals['expected_daily_range']['high']:.2f}")
    print(f"Expected 30-Day Range: ${signals['expected_30day_range']['low']:.2f} - ${signals['expected_30day_range']['high']:.2f}")
    print("\n📋 Recommendations:")
    for rec in signals['recommendations']:
        print(f"  {rec}")

