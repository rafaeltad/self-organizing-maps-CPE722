"""
Twitter data preprocessor for SOM analysis.

This module handles text preprocessing, feature extraction, and data preparation
for use with Self-Organizing Maps.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import nltk
from textblob import TextBlob


import nltk

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("vader_lexicon", quiet=True)
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

NLTK_AVAILABLE = True

from .models import TwitterData, TwitterDataCollection, SOMTrainingConfig


class TwitterPreprocessor:
    """
    Preprocessor for Twitter data to extract features for SOM analysis.

    This class handles:
    - Text cleaning and normalization
    - Feature extraction (temporal, engagement, text, network)
    - Data scaling and normalization
    - Dimensionality reduction (PCA)
    """

    def __init__(self, config: SOMTrainingConfig):
        """
        Initialize the preprocessor with configuration.

        Args:
            config: SOM training configuration
        """
        self.config = config
        self.scaler = StandardScaler() if config.normalize_features else None
        self.pca = PCA(n_components=config.pca_components) if config.use_pca else None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,  # Limit features for SOM
            stop_words=stopwords.words("portuguese"),
            ngram_range=(1, 2),
            min_df=2,
        )

        # Initialize sentiment analyzer if available
        self.sentiment_analyzer = None
        if NLTK_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Feature names for tracking
        self.feature_names: List[str] = []
        self.is_fitted = False

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize tweet text.

        Args:
            text: Raw tweet text

        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)

        # Remove mentions and hashtags (keep the text content)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag text

        # Remove special characters but keep emoticons
        text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_temporal_features(self, tweets: List[TwitterData]) -> np.ndarray:
        """
        Extract temporal features from tweets.

        Args:
            tweets: List of tweet data

        Returns:
            Temporal features array
        """
        features = []

        for tweet in tweets:
            tweet_features = []

            # Hour of day (0-23)
            tweet_features.append(tweet.created_at.hour)

            # Day of week (0-6)
            tweet_features.append(tweet.created_at.weekday())

            # Day of month (1-31)
            tweet_features.append(tweet.created_at.day)

            # Month (1-12)
            tweet_features.append(tweet.created_at.month)

            # Day of year (1-366)
            tweet_features.append(tweet.created_at.timetuple().tm_yday)

            features.append(tweet_features)

        return np.array(features)

    def extract_engagement_features(self, tweets: List[TwitterData]) -> np.ndarray:
        """
        Extract engagement features from tweets.

        NOTE: Engagement features have been removed from the TwitterData model.
        This method now returns an empty feature array.

        Args:
            tweets: List of tweet data

        Returns:
            Empty engagement features array
        """
        # Return empty features since engagement metrics have been removed
        return np.array([]).reshape(len(tweets), 0)

    def extract_text_features(self, tweets: List[TwitterData]) -> np.ndarray:
        """
        Extract text-based features from tweets.

        Args:
            tweets: List of tweet data

        Returns:
            Text features array
        """
        # Clean texts
        clean_texts = [self.clean_text(tweet.text) for tweet in tweets]

        # TF-IDF features
        if not self.is_fitted:
            try:
                tfidf_features = self.tfidf_vectorizer.fit_transform(clean_texts)
            except ValueError as e:
                if "empty vocabulary" in str(e) or "no terms remain" in str(e):
                    # Fallback for edge cases - create vectorizer with min_df=1
                    fallback_vectorizer = TfidfVectorizer(
                        max_features=100,
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=1,
                    )
                    tfidf_features = fallback_vectorizer.fit_transform(clean_texts)
                    self.tfidf_vectorizer = fallback_vectorizer
                else:
                    raise e
        else:
            tfidf_features = self.tfidf_vectorizer.transform(clean_texts)

        # Additional text features
        additional_features = []

        for i, tweet in enumerate(tweets):
            text_features = []
            text = tweet.text
            clean_text = clean_texts[i]

            # Text length features
            text_features.extend([
                len(text),  # Character count
                len(text.split()),  # Word count
                len(clean_text),  # Clean character count
                len(clean_text.split())  # Clean word count
            ])

            # Content type features
            text_features.extend([
                len(tweet.hashtags),  # Number of hashtags
                len(tweet.mentions),  # Number of mentions
                len(tweet.urls),  # Number of URLs
                int(tweet.is_retweet),  # Is retweet
                int(tweet.is_reply),  # Is reply
                int(tweet.is_quote)  # Is quote
            ])

            # Sentiment features (if available)
            if self.sentiment_analyzer:
                scores = self.sentiment_analyzer.polarity_scores(text)
                text_features.extend([
                    scores['pos'],
                    scores['neu'],
                    scores['neg'],
                    scores['compound']
                ])
            else:
                # Use TextBlob as fallback
                try:
                    blob = TextBlob(text)
                    sentiment = blob.sentiment
                    polarity_value = sentiment.polarity  # type: ignore
                except Exception:
                    # Fallback if TextBlob fails
                    polarity_value = 0.0

                text_features.extend(
                    [
                        max(0, polarity_value),  # Positive sentiment
                        1 - abs(polarity_value),  # Neutral sentiment
                        max(0, -polarity_value),  # Negative sentiment
                        polarity_value,  # Compound sentiment
                    ]
                )

            # Language diversity (placeholder)
            text_features.append(1.0 if tweet.lang == 'en' else 0.5)

            additional_features.append(text_features)

        # Combine TF-IDF with additional features
        additional_array = np.array(additional_features)
        combined_features = np.hstack(
            [np.asarray(tfidf_features.todense()), additional_array]
        )

        return combined_features

    def extract_network_features(self, tweets: List[TwitterData]) -> np.ndarray:
        """
        Extract network-based features from tweets.

        Args:
            tweets: List of tweet data

        Returns:
            Network features array
        """
        # Build user interaction network
        user_interactions = {}
        user_tweet_counts = {}

        for tweet in tweets:
            user = tweet.username
            user_tweet_counts[user] = user_tweet_counts.get(user, 0) + 1

            if user not in user_interactions:
                user_interactions[user] = set()

            # Add mentions as interactions
            for mention in tweet.mentions:
                user_interactions[user].add(mention)

        features = []

        for tweet in tweets:
            user = tweet.username
            network_features = []

            # User activity features
            network_features.extend([
                user_tweet_counts.get(user, 1),  # User tweet count
                len(user_interactions.get(user, set())),  # User interaction count
            ])

            # Tweet-specific network features
            network_features.extend([
                len(tweet.mentions),  # Mentions in this tweet
                int(len(tweet.mentions) > 0),  # Has mentions
                int(len(tweet.hashtags) > 0),  # Has hashtags
                int(len(tweet.urls) > 0),  # Has URLs
            ])

            # User activity proxy (tweets per user)
            user_activity = user_tweet_counts[user]
            network_features.append(user_activity)

            features.append(network_features)

        return np.array(features)

    def fit_transform(self, collection: TwitterDataCollection) -> Tuple[np.ndarray, List[str]]:
        """
        Fit the preprocessor and transform the data.

        Args:
            collection: Twitter data collection

        Returns:
            Tuple of (transformed features, feature names)
        """
        tweets = collection.tweets
        all_features = []
        feature_names = []

        # Extract different types of features based on configuration
        if self.config.include_temporal_features:
            temporal_features = self.extract_temporal_features(tweets)
            all_features.append(temporal_features)
            feature_names.extend([
                'hour', 'day_of_week', 'day_of_month', 'month', 'day_of_year'
            ])

        if self.config.include_engagement_features:
            engagement_features = self.extract_engagement_features(tweets)
            all_features.append(engagement_features)
            # Only add feature names if there are actual engagement features
            if engagement_features.shape[1] > 0:
                feature_names.extend(
                    [
                        "like_count",
                        "retweet_count",
                        "reply_count",
                        "quote_count",
                        "like_ratio",
                        "retweet_ratio",
                        "reply_ratio",
                        "quote_ratio",
                        "engagement_score",
                        "engagement_velocity",
                    ]
                )

        if self.config.include_text_features:
            text_features = self.extract_text_features(tweets)
            all_features.append(text_features)

            # Add TF-IDF feature names
            tfidf_names = [f'tfidf_{name}' for name in self.tfidf_vectorizer.get_feature_names_out()]
            feature_names.extend(tfidf_names)

            # Add additional text feature names
            feature_names.extend([
                'char_count', 'word_count', 'clean_char_count', 'clean_word_count',
                'hashtag_count', 'mention_count', 'url_count',
                'is_retweet', 'is_reply', 'is_quote',
                'sentiment_pos', 'sentiment_neu', 'sentiment_neg', 'sentiment_compound',
                'language_english'
            ])

        if self.config.include_network_features:
            network_features = self.extract_network_features(tweets)
            all_features.append(network_features)
            feature_names.extend([
                'user_tweet_count', 'user_interaction_count',
                'tweet_mentions', 'has_mentions', 'has_hashtags', 'has_urls',
                'avg_user_engagement'
            ])

        # Combine all features
        if all_features:
            combined_features = np.hstack(all_features)
        else:
            raise ValueError("No features selected for extraction")

        # Handle missing values
        combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=1e6, neginf=-1e6)

        # Apply scaling
        if self.config.normalize_features and self.scaler:
            combined_features = self.scaler.fit_transform(combined_features)

        # Apply PCA if configured
        if self.config.use_pca and self.pca:
            combined_features = self.pca.fit_transform(combined_features)
            # Update feature names for PCA components
            feature_names = [f'pca_component_{i}' for i in range(combined_features.shape[1])]

        self.feature_names = feature_names
        self.is_fitted = True

        return combined_features, feature_names

    def transform(self, collection: TwitterDataCollection) -> np.ndarray:
        """
        Transform new data using the fitted preprocessor.

        Args:
            collection: Twitter data collection

        Returns:
            Transformed features array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming new data")

        tweets = collection.tweets
        all_features = []

        # Extract features in the same order as fit_transform
        if self.config.include_temporal_features:
            all_features.append(self.extract_temporal_features(tweets))

        if self.config.include_engagement_features:
            all_features.append(self.extract_engagement_features(tweets))

        if self.config.include_text_features:
            all_features.append(self.extract_text_features(tweets))

        if self.config.include_network_features:
            all_features.append(self.extract_network_features(tweets))

        # Combine features
        combined_features = np.hstack(all_features)

        # Handle missing values
        combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=1e6, neginf=-1e6)

        # Apply scaling
        if self.config.normalize_features and self.scaler:
            combined_features = self.scaler.transform(combined_features)

        # Apply PCA
        if self.config.use_pca and self.pca:
            combined_features = self.pca.transform(combined_features)

        return combined_features

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (placeholder implementation).

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before getting feature importance")

        # Placeholder implementation - in practice, you might use
        # feature selection methods or analyze SOM weights
        importance = {}
        for i, name in enumerate(self.feature_names):
            # Random importance for demonstration
            importance[name] = np.random.random()

        return importance
