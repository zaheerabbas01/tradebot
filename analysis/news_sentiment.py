import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re
from textblob import TextBlob
from config import Config
from utils.logger import logger

class NewsSentimentAnalyzer:
    """
    Advanced news sentiment analysis for crypto trading
    
    Features:
    - CryptoPanic API integration
    - Sentiment scoring using multiple methods
    - Impact assessment based on source credibility
    - FUD (Fear, Uncertainty, Doubt) detection
    - Social media sentiment tracking
    
    Based on:
    - "Sentiment Analysis and Opinion Mining" by Bing Liu
    - "The Wisdom of Crowds" by James Surowiecki
    - "Behavioral Finance" by Shefrin
    """
    
    def __init__(self):
        self.api_key = Config.CRYPTOPANIC_API_KEY
        self.base_url = "https://cryptopanic.com/api/v1/posts/"
        self.sentiment_cache = {}
        self.impact_weights = {
            'high': 1.0,
            'medium': 0.6,
            'low': 0.3
        }
        
        # High-impact keywords for crypto markets
        self.bullish_keywords = [
            'adoption', 'institutional', 'etf', 'approval', 'partnership', 
            'integration', 'upgrade', 'bullish', 'pump', 'moon', 'breakout',
            'rally', 'surge', 'breakthrough', 'positive', 'growth', 'expansion'
        ]
        
        self.bearish_keywords = [
            'regulation', 'ban', 'crash', 'dump', 'bearish', 'decline', 
            'investigation', 'hack', 'security', 'fud', 'panic', 'sell-off',
            'correction', 'dip', 'liquidation', 'negative', 'concern', 'risk'
        ]
        
        self.high_impact_sources = [
            'coindesk', 'cointelegraph', 'bloomberg', 'reuters', 'wsj',
            'cnbc', 'sec', 'fed', 'treasury', 'binance', 'coinbase'
        ]
    
    def get_news_sentiment(self, symbol: str, hours_back: int = 24) -> Dict:
        """
        Get aggregated news sentiment for a cryptocurrency
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            hours_back: Number of hours to look back for news
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Convert symbol format (BTC/USDT -> BTC)
            clean_symbol = symbol.split('/')[0].lower()
            
            # Check cache first
            cache_key = f"{clean_symbol}_{hours_back}"
            if cache_key in self.sentiment_cache:
                cached_data = self.sentiment_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < 300:  # 5 min cache
                    return cached_data['data']
            
            # Fetch news from CryptoPanic
            news_data = self._fetch_cryptopanic_news(clean_symbol, hours_back)
            
            if not news_data:
                return self._get_neutral_sentiment()
            
            # Analyze sentiment
            sentiment_analysis = self._analyze_news_sentiment(news_data)
            
            # Cache results
            self.sentiment_cache[cache_key] = {
                'data': sentiment_analysis,
                'timestamp': datetime.now()
            }
            
            logger.news(
                f"Sentiment analysis for {symbol}: Score {sentiment_analysis['sentiment_score']:.2f}",
                sentiment_analysis
            )
            
            return sentiment_analysis
            
        except Exception as e:
            logger.error(f"Error in news sentiment analysis for {symbol}", e)
            return self._get_neutral_sentiment()
    
    def _fetch_cryptopanic_news(self, symbol: str, hours_back: int) -> List[Dict]:
        """Fetch news from CryptoPanic API"""
        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            params = {
                'auth_token': self.api_key,
                'currencies': symbol.upper(),
                'filter': 'hot',
                'public': 'true',
                'kind': 'news'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'results' not in data:
                return []
            
            # Filter by time range
            filtered_news = []
            for item in data['results']:
                created_at = datetime.fromisoformat(item['created_at'].replace('Z', '+00:00'))
                if start_time <= created_at.replace(tzinfo=None) <= end_time:
                    filtered_news.append(item)
            
            return filtered_news[:50]  # Limit to 50 most recent articles
            
        except Exception as e:
            logger.error(f"Failed to fetch CryptoPanic news for {symbol}", e)
            return []
    
    def _analyze_news_sentiment(self, news_data: List[Dict]) -> Dict:
        """Analyze sentiment from news articles"""
        if not news_data:
            return self._get_neutral_sentiment()
        
        total_sentiment = 0
        total_weight = 0
        article_count = len(news_data)
        
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        high_impact_articles = 0
        social_mentions = 0
        
        sentiment_scores = []
        
        for article in news_data:
            try:
                # Extract text for analysis
                title = article.get('title', '')
                
                # Calculate base sentiment using TextBlob
                blob = TextBlob(title)
                base_sentiment = blob.sentiment.polarity  # -1 to 1
                
                # Apply keyword analysis
                keyword_sentiment = self._analyze_keywords(title.lower())
                
                # Combine sentiments
                combined_sentiment = (base_sentiment + keyword_sentiment) / 2
                
                # Determine impact weight
                impact_weight = self._calculate_impact_weight(article)
                
                # Apply weight
                weighted_sentiment = combined_sentiment * impact_weight
                
                total_sentiment += weighted_sentiment
                total_weight += impact_weight
                
                sentiment_scores.append(combined_sentiment)
                
                # Count sentiment categories
                if combined_sentiment > 0.1:
                    bullish_count += 1
                elif combined_sentiment < -0.1:
                    bearish_count += 1
                else:
                    neutral_count += 1
                
                # Track high impact articles
                if impact_weight > 0.8:
                    high_impact_articles += 1
                
                # Track social media mentions
                if 'votes' in article and article['votes'].get('liked', 0) > 10:
                    social_mentions += 1
                    
            except Exception as e:
                logger.error(f"Error analyzing article sentiment: {str(e)}")
                continue
        
        # Calculate final sentiment score
        if total_weight > 0:
            final_sentiment = total_sentiment / total_weight
        else:
            final_sentiment = 0
        
        # Calculate additional metrics
        sentiment_volatility = np.std(sentiment_scores) if sentiment_scores else 0
        sentiment_momentum = self._calculate_sentiment_momentum(sentiment_scores)
        
        # Detect FUD or FOMO
        fud_score = self._detect_fud(news_data)
        fomo_score = self._detect_fomo(news_data)
        
        return {
            'sentiment_score': final_sentiment,
            'sentiment_category': self._categorize_sentiment(final_sentiment),
            'confidence': min(total_weight / article_count, 1.0) if article_count > 0 else 0,
            'article_count': article_count,
            'bullish_articles': bullish_count,
            'bearish_articles': bearish_count,
            'neutral_articles': neutral_count,
            'high_impact_articles': high_impact_articles,
            'social_mentions': social_mentions,
            'sentiment_volatility': sentiment_volatility,
            'sentiment_momentum': sentiment_momentum,
            'fud_score': fud_score,
            'fomo_score': fomo_score,
            'timestamp': datetime.now()
        }
    
    def _analyze_keywords(self, text: str) -> float:
        """Analyze sentiment based on predefined keywords"""
        bullish_score = 0
        bearish_score = 0
        
        # Count bullish keywords
        for keyword in self.bullish_keywords:
            if keyword in text:
                bullish_score += 1
        
        # Count bearish keywords
        for keyword in self.bearish_keywords:
            if keyword in text:
                bearish_score += 1
        
        # Calculate net sentiment
        total_keywords = bullish_score + bearish_score
        if total_keywords == 0:
            return 0
        
        net_sentiment = (bullish_score - bearish_score) / total_keywords
        return net_sentiment
    
    def _calculate_impact_weight(self, article: Dict) -> float:
        """Calculate impact weight based on source and engagement"""
        weight = 0.5  # Base weight
        
        # Source credibility
        source = article.get('source', {}).get('domain', '').lower()
        for high_impact_source in self.high_impact_sources:
            if high_impact_source in source:
                weight += 0.3
                break
        
        # Engagement metrics
        votes = article.get('votes', {})
        liked = votes.get('liked', 0)
        disliked = votes.get('disliked', 0)
        
        if liked > 50:
            weight += 0.2
        elif liked > 20:
            weight += 0.1
        
        # Recency boost
        created_at = datetime.fromisoformat(article['created_at'].replace('Z', '+00:00'))
        hours_ago = (datetime.now() - created_at.replace(tzinfo=None)).total_seconds() / 3600
        
        if hours_ago < 2:
            weight += 0.2
        elif hours_ago < 6:
            weight += 0.1
        
        return min(weight, 1.0)
    
    def _calculate_sentiment_momentum(self, sentiment_scores: List[float]) -> float:
        """Calculate sentiment momentum (trend direction)"""
        if len(sentiment_scores) < 3:
            return 0
        
        # Split into early and recent periods
        mid_point = len(sentiment_scores) // 2
        early_sentiment = np.mean(sentiment_scores[:mid_point])
        recent_sentiment = np.mean(sentiment_scores[mid_point:])
        
        # Calculate momentum
        momentum = recent_sentiment - early_sentiment
        return momentum
    
    def _detect_fud(self, news_data: List[Dict]) -> float:
        """Detect Fear, Uncertainty, and Doubt patterns"""
        fud_indicators = [
            'regulation', 'ban', 'investigation', 'lawsuit', 'sec', 'cftc',
            'hack', 'security breach', 'scam', 'ponzi', 'bubble', 'crash',
            'manipulation', 'whale dump', 'liquidation', 'margin call'
        ]
        
        fud_score = 0
        for article in news_data:
            title = article.get('title', '').lower()
            for indicator in fud_indicators:
                if indicator in title:
                    impact_weight = self._calculate_impact_weight(article)
                    fud_score += impact_weight
        
        # Normalize by article count
        return fud_score / len(news_data) if news_data else 0
    
    def _detect_fomo(self, news_data: List[Dict]) -> float:
        """Detect Fear of Missing Out patterns"""
        fomo_indicators = [
            'all-time high', 'ath', 'new high', 'breakout', 'rally', 'surge',
            'institutional adoption', 'etf approval', 'mainstream', 'mass adoption',
            'partnership', 'integration', 'upgrade', 'moon', 'to the moon'
        ]
        
        fomo_score = 0
        for article in news_data:
            title = article.get('title', '').lower()
            for indicator in fomo_indicators:
                if indicator in title:
                    impact_weight = self._calculate_impact_weight(article)
                    fomo_score += impact_weight
        
        # Normalize by article count
        return fomo_score / len(news_data) if news_data else 0
    
    def _categorize_sentiment(self, sentiment_score: float) -> str:
        """Categorize sentiment score into human-readable labels"""
        if sentiment_score >= Config.BULLISH_SENTIMENT_THRESHOLD:
            return 'VERY_BULLISH'
        elif sentiment_score >= 0.25:
            return 'BULLISH'
        elif sentiment_score >= -0.25:
            return 'NEUTRAL'
        elif sentiment_score >= Config.BEARISH_SENTIMENT_THRESHOLD:
            return 'BEARISH'
        else:
            return 'VERY_BEARISH'
    
    def _get_neutral_sentiment(self) -> Dict:
        """Return neutral sentiment when no data is available"""
        return {
            'sentiment_score': 0.0,
            'sentiment_category': 'NEUTRAL',
            'confidence': 0.0,
            'article_count': 0,
            'bullish_articles': 0,
            'bearish_articles': 0,
            'neutral_articles': 0,
            'high_impact_articles': 0,
            'social_mentions': 0,
            'sentiment_volatility': 0.0,
            'sentiment_momentum': 0.0,
            'fud_score': 0.0,
            'fomo_score': 0.0,
            'timestamp': datetime.now()
        }
    
    def should_trade_on_news(self, sentiment_data: Dict, volume_increase: float) -> Dict:
        """
        Determine if we should trade based on news sentiment
        
        Args:
            sentiment_data: Output from get_news_sentiment()
            volume_increase: Current volume increase percentage
            
        Returns:
            Trading recommendation based on news
        """
        sentiment_score = sentiment_data['sentiment_score']
        confidence = sentiment_data['confidence']
        high_impact_count = sentiment_data['high_impact_articles']
        fud_score = sentiment_data['fud_score']
        fomo_score = sentiment_data['fomo_score']
        
        recommendation = {
            'action': 'HOLD',
            'confidence': 0.0,
            'reason': 'No significant news impact'
        }
        
        # Bullish news trading
        if (sentiment_score > Config.BULLISH_SENTIMENT_THRESHOLD and 
            confidence > 0.6 and 
            volume_increase > Config.VOLUME_SPIKE_THRESHOLD):
            
            recommendation['action'] = 'BUY'
            recommendation['confidence'] = min(confidence * (1 + fomo_score), 1.0)
            recommendation['reason'] = f"Bullish news sentiment ({sentiment_score:.2f}) with volume spike"
            
            # Extra confidence for high-impact articles
            if high_impact_count > 0:
                recommendation['confidence'] *= 1.2
                recommendation['reason'] += f", {high_impact_count} high-impact articles"
        
        # Bearish news trading
        elif (sentiment_score < Config.BEARISH_SENTIMENT_THRESHOLD and 
              confidence > 0.6 and 
              fud_score > 0.3):
            
            recommendation['action'] = 'SELL'
            recommendation['confidence'] = min(confidence * (1 + fud_score), 1.0)
            recommendation['reason'] = f"Bearish news sentiment ({sentiment_score:.2f}) with FUD detection"
            
            # Extra confidence for high-impact articles
            if high_impact_count > 0:
                recommendation['confidence'] *= 1.2
                recommendation['reason'] += f", {high_impact_count} high-impact articles"
        
        # Normalize confidence
        recommendation['confidence'] = min(recommendation['confidence'], 1.0)
        
        return recommendation
    
    def get_market_fear_greed_index(self) -> float:
        """
        Calculate a simple Fear & Greed index based on recent news
        Returns value from 0 (Extreme Fear) to 100 (Extreme Greed)
        """
        try:
            # Get sentiment for major cryptocurrencies
            major_cryptos = ['BTC', 'ETH', 'BNB']
            sentiments = []
            
            for crypto in major_cryptos:
                sentiment_data = self.get_news_sentiment(crypto, hours_back=48)
                if sentiment_data['article_count'] > 0:
                    sentiments.append(sentiment_data['sentiment_score'])
            
            if not sentiments:
                return 50  # Neutral
            
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiments)
            
            # Convert to 0-100 scale (Fear & Greed Index style)
            # -1 sentiment = 0 (Extreme Fear)
            #  0 sentiment = 50 (Neutral)
            # +1 sentiment = 100 (Extreme Greed)
            fear_greed_index = (avg_sentiment + 1) * 50
            
            return max(0, min(100, fear_greed_index))
            
        except Exception as e:
            logger.error("Error calculating Fear & Greed index", e)
            return 50  # Return neutral on error