import pandas as pd
import numpy as np
import talib
from typing import Tuple, Dict, Optional
from utils.logger import logger

class TechnicalIndicators:
    """
    Comprehensive technical analysis indicators based on top crypto trading books:
    - "Technical Analysis of the Financial Markets" by John J. Murphy
    - "Crypto Technical Analysis" by David Lifchitz
    - "The New Trading for a Living" by Alexander Elder
    """
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        Based on Wilder's formula from "New Concepts in Technical Trading Systems"
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        Based on John Bollinger's methodology
        """
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        Based on Gerald Appel's original formula
        """
        ema_fast = TechnicalIndicators.calculate_ema(data, fast_period)
        ema_slow = TechnicalIndicators.calculate_ema(data, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator
        Based on George Lane's methodology
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        Based on J. Welles Wilder's methodology
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Volume Simple Moving Average"""
        return volume.rolling(window=period).mean()
    
    @staticmethod
    def detect_volume_spike(volume: pd.Series, volume_sma: pd.Series, threshold: float = 1.5) -> pd.Series:
        """
        Detect volume spikes
        Volume spike when current volume > threshold * average volume
        """
        return volume > (volume_sma * threshold)
    
    @staticmethod
    def calculate_support_resistance(data: pd.DataFrame, window: int = 20) -> Dict:
        """
        Calculate support and resistance levels using pivot points
        Based on classical pivot point methodology
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate pivot points
        pivot = (high + low + close) / 3
        
        # Calculate support and resistance levels
        support1 = 2 * pivot - high
        support2 = pivot - (high - low)
        resistance1 = 2 * pivot - low
        resistance2 = pivot + (high - low)
        
        return {
            'pivot': pivot.iloc[-1],
            'support1': support1.iloc[-1],
            'support2': support2.iloc[-1],
            'resistance1': resistance1.iloc[-1],
            'resistance2': resistance2.iloc[-1]
        }
    
    @staticmethod
    def calculate_fibonacci_retracement(high_price: float, low_price: float) -> Dict:
        """
        Calculate Fibonacci retracement levels
        Based on Leonardo Fibonacci's sequence
        """
        diff = high_price - low_price
        
        levels = {
            'level_0': high_price,
            'level_236': high_price - 0.236 * diff,
            'level_382': high_price - 0.382 * diff,
            'level_500': high_price - 0.500 * diff,
            'level_618': high_price - 0.618 * diff,
            'level_786': high_price - 0.786 * diff,
            'level_100': low_price
        }
        
        return levels
    
    @staticmethod
    def calculate_ichimoku(data: pd.DataFrame) -> Dict:
        """
        Calculate Ichimoku Cloud components
        Based on Goichi Hosoda's methodology
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = high.rolling(window=9).max()
        period9_low = low.rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = high.rolling(window=26).max()
        period26_low = low.rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = high.rolling(window=52).max()
        period52_low = low.rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close shifted back 26 periods
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Williams %R
        Based on Larry Williams' methodology
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r
    
    @staticmethod
    def calculate_commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI)
        Based on Donald Lambert's methodology
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    @staticmethod
    def calculate_money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI)
        Volume-weighted version of RSI
        """
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
        
        money_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi

class TrendAnalysis:
    """Advanced trend analysis methods"""
    
    @staticmethod
    def detect_ema_crossover(ema_fast: pd.Series, ema_slow: pd.Series) -> pd.Series:
        """
        Detect EMA crossover signals
        Returns: 1 for bullish crossover, -1 for bearish crossover, 0 for no signal
        """
        crossover = pd.Series(0, index=ema_fast.index)
        
        # Bullish crossover: fast EMA crosses above slow EMA
        bullish_cross = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        crossover[bullish_cross] = 1
        
        # Bearish crossover: fast EMA crosses below slow EMA
        bearish_cross = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
        crossover[bearish_cross] = -1
        
        return crossover
    
    @staticmethod
    def detect_trend_strength(ema_fast: pd.Series, ema_slow: pd.Series, atr: pd.Series) -> pd.Series:
        """
        Calculate trend strength based on EMA separation and volatility
        Higher values indicate stronger trends
        """
        ema_separation = abs(ema_fast - ema_slow)
        normalized_separation = ema_separation / ema_slow
        volatility_adjusted_strength = normalized_separation / atr
        return volatility_adjusted_strength
    
    @staticmethod
    def calculate_trend_score(data: pd.DataFrame) -> float:
        """
        Calculate overall trend score combining multiple indicators
        Score ranges from -100 (strong bearish) to +100 (strong bullish)
        """
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Calculate indicators
        ema_50 = TechnicalIndicators.calculate_ema(close, 50)
        ema_200 = TechnicalIndicators.calculate_ema(close, 200)
        rsi = TechnicalIndicators.calculate_rsi(close)
        macd, signal, _ = TechnicalIndicators.calculate_macd(close)
        
        # Calculate individual scores
        ema_score = 20 if ema_50.iloc[-1] > ema_200.iloc[-1] else -20
        rsi_score = 0
        if rsi.iloc[-1] > 70:
            rsi_score = -15  # Overbought
        elif rsi.iloc[-1] < 30:
            rsi_score = 15   # Oversold
        
        macd_score = 15 if macd.iloc[-1] > signal.iloc[-1] else -15
        
        # Price momentum score
        price_change = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
        momentum_score = min(max(price_change * 100, -30), 30)
        
        # Volume confirmation
        volume_sma = TechnicalIndicators.calculate_volume_sma(volume)
        volume_score = 10 if volume.iloc[-1] > volume_sma.iloc[-1] else -5
        
        total_score = ema_score + rsi_score + macd_score + momentum_score + volume_score
        return min(max(total_score, -100), 100)