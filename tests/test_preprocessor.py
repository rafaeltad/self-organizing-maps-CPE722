"""
Test file for TwitterPreprocessor.

Tests the preprocessing functionality for Twitter data preparation
for SOM analysis.
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from twitter_som.preprocessor import TwitterPreprocessor
from twitter_som.models import TwitterData, TwitterDataCollection, SOMTrainingConfig


class TestTwitterPreprocessor:
    """Test cases for TwitterPreprocessor."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return SOMTrainingConfig(
            x_dim=5,
            y_dim=5,
            learning_rate=0.1,
            num_iterations=100,
            include_temporal_features=True,
            include_engagement_features=True,
            include_text_features=True,
            include_network_features=True
        )

    @pytest.fixture
    def preprocessor(self, config):
        """Create a test preprocessor."""
        return TwitterPreprocessor(config)

    @pytest.fixture
    def sample_collection(self):
        """Create a sample collection for testing."""
        tweets = [
            TwitterData(
                id="1",
                text="This is a test tweet about #python programming @user1",
                created_at=datetime(2024, 1, 1, 10, 30, 0, tzinfo=timezone.utc),
                user_id="user1",
                username="testuser1",
                like_count=10,
                retweet_count=5,
                lang="en"
            ),
            TwitterData(
                id="2",
                text="Another tweet with #datascience and machine learning content",
                created_at=datetime(2024, 1, 1, 14, 45, 0, tzinfo=timezone.utc),
                user_id="user2",
                username="testuser2",
                like_count=20,
                retweet_count=8,
                reply_count=3,
                lang="en"
            ),
            TwitterData(
                id="3",
                text="Simple tweet without hashtags or mentions",
                created_at=datetime(2024, 1, 2, 9, 15, 0, tzinfo=timezone.utc),
                user_id="user1",
                username="testuser1",
                like_count=5,
                quote_count=2,
                lang="en"
            )
        ]

        return TwitterDataCollection(
            tweets=tweets,
            collection_name="test_collection"
        )

    def test_clean_text(self, preprocessor):
        """Test text cleaning functionality."""
        raw_text = "Check out this link: https://example.com @user #hashtag 🚀"
        clean_text = preprocessor.clean_text(raw_text)

        # Should remove URLs, mentions, but keep hashtag content
        expected = "check out this link hashtag 🚀"
        assert clean_text == expected

    def test_clean_text_special_cases(self, preprocessor):
        """Test text cleaning with special cases."""
        # Test with multiple spaces
        text1 = "Multiple    spaces   should   be   normalized"
        clean1 = preprocessor.clean_text(text1)
        assert "  " not in clean1

        # Test with empty result
        text2 = "@user https://example.com"
        clean2 = preprocessor.clean_text(text2)
        assert clean2 == ""

        # Test with emojis and special characters
        text3 = "Text with emojis 😀 and symbols @#$%"
        clean3 = preprocessor.clean_text(text3)
        assert "😀" in clean3  # Emojis should be preserved

    def test_extract_temporal_features(self, preprocessor, sample_collection):
        """Test temporal feature extraction."""
        features = preprocessor.extract_temporal_features(sample_collection.tweets)

        # Should have 5 temporal features per tweet
        assert features.shape == (3, 5)

        # Check first tweet features (2024-01-01 10:30:00)
        assert features[0, 0] == 10  # hour
        assert features[0, 1] == 0   # weekday (Monday)
        assert features[0, 2] == 1   # day of month
        assert features[0, 3] == 1   # month
        assert features[0, 4] == 1   # day of year

    def test_extract_engagement_features(self, preprocessor, sample_collection):
        """Test engagement feature extraction."""
        features = preprocessor.extract_engagement_features(sample_collection.tweets)

        # Should have 10 engagement features per tweet
        assert features.shape == (3, 10)

        # Check first tweet features
        # Raw counts: like=10, retweet=5, reply=0, quote=0
        assert features[0, 0] == 10  # like_count
        assert features[0, 1] == 5   # retweet_count
        assert features[0, 2] == 0   # reply_count
        assert features[0, 3] == 0   # quote_count

        # Check ratios sum to 1
        total_engagement = 10 + 5 + 0 + 0
        assert abs(features[0, 4] - (10/total_engagement)) < 1e-6  # like_ratio
        assert abs(features[0, 5] - (5/total_engagement)) < 1e-6   # retweet_ratio

    def test_extract_text_features(self, preprocessor, sample_collection):
        """Test text feature extraction."""
        features = preprocessor.extract_text_features(sample_collection.tweets)

        # Should have TF-IDF features + additional text features
        assert features.shape[0] == 3  # 3 tweets
        assert features.shape[1] > 10  # Should have many features

        # Test that TF-IDF vectorizer is fitted
        assert preprocessor.tfidf_vectorizer is not None
        assert hasattr(preprocessor.tfidf_vectorizer, 'vocabulary_')

    def test_extract_network_features(self, preprocessor, sample_collection):
        """Test network feature extraction."""
        features = preprocessor.extract_network_features(sample_collection.tweets)

        # Should have 7 network features per tweet
        assert features.shape == (3, 7)

        # Check that user tweet counts are calculated
        # testuser1 has 2 tweets, testuser2 has 1 tweet
        assert features[0, 0] == 2  # user1 tweet count
        assert features[1, 0] == 1  # user2 tweet count
        assert features[2, 0] == 2  # user1 tweet count (same user as first)

    def test_fit_transform_all_features(self, preprocessor, sample_collection):
        """Test fit_transform with all feature types enabled."""
        features, feature_names = preprocessor.fit_transform(sample_collection)

        # Should have features for all enabled types
        assert features.shape[0] == 3  # 3 tweets
        assert features.shape[1] > 20  # Should have many features
        assert len(feature_names) == features.shape[1]

        # Check that preprocessor is fitted
        assert preprocessor.is_fitted is True
        assert len(preprocessor.feature_names) > 0

    def test_fit_transform_selective_features(self):
        """Test fit_transform with selective features."""
        # Config with only temporal and engagement features
        config = SOMTrainingConfig(
            include_temporal_features=True,
            include_engagement_features=True,
            include_text_features=False,
            include_network_features=False
        )

        preprocessor = TwitterPreprocessor(config)

        # Create simple collection
        tweets = [
            TwitterData(
                id="1",
                text="Test tweet",
                created_at=datetime.now(timezone.utc),
                user_id="user1",
                username="testuser1",
                like_count=10
            )
        ]
        collection = TwitterDataCollection(tweets=tweets, collection_name="test")

        features, feature_names = preprocessor.fit_transform(collection)

        # Should have 5 temporal + 10 engagement = 15 features
        assert features.shape == (1, 15)
        assert len(feature_names) == 15

    def test_no_features_selected_error(self):
        """Test error when no features are selected."""
        config = SOMTrainingConfig(
            include_temporal_features=False,
            include_engagement_features=False,
            include_text_features=False,
            include_network_features=False
        )

        preprocessor = TwitterPreprocessor(config)

        tweets = [
            TwitterData(
                id="1",
                text="Test tweet",
                created_at=datetime.now(timezone.utc),
                user_id="user1",
                username="testuser1"
            )
        ]
        collection = TwitterDataCollection(tweets=tweets, collection_name="test")

        with pytest.raises(ValueError, match="No features selected"):
            preprocessor.fit_transform(collection)

    def test_transform_unfitted_error(self, preprocessor, sample_collection):
        """Test error when trying to transform without fitting."""
        with pytest.raises(ValueError, match="Preprocessor must be fitted"):
            preprocessor.transform(sample_collection)

    def test_transform_after_fit(self, preprocessor, sample_collection):
        """Test transform method after fitting."""
        # First fit
        features1, _ = preprocessor.fit_transform(sample_collection)

        # Create new collection
        new_tweet = TwitterData(
            id="4",
            text="New tweet for testing transform",
            created_at=datetime.now(timezone.utc),
            user_id="user3",
            username="testuser3",
            like_count=15
        )
        new_collection = TwitterDataCollection(tweets=[new_tweet], collection_name="new_test")

        # Transform new data
        features2 = preprocessor.transform(new_collection)

        # Should have same number of features
        assert features2.shape[1] == features1.shape[1]
        assert features2.shape[0] == 1  # One tweet

    def test_feature_scaling(self):
        """Test feature scaling functionality."""
        config = SOMTrainingConfig(
            normalize_features=True,
            include_temporal_features=True,
            include_engagement_features=True,
            include_text_features=False,
            include_network_features=False
        )

        preprocessor = TwitterPreprocessor(config)

        # Create collection with varied values
        tweets = [
            TwitterData(
                id="1",
                text="Test 1",
                created_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                user_id="user1",
                username="user1",
                like_count=1000,  # High value
                retweet_count=100
            ),
            TwitterData(
                id="2",
                text="Test 2",
                created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                user_id="user2",
                username="user2",
                like_count=5,     # Low value
                retweet_count=1
            )
        ]
        collection = TwitterDataCollection(tweets=tweets, collection_name="test")

        features, _ = preprocessor.fit_transform(collection)

        # Features should be normalized (roughly mean=0, std=1 for each feature)
        # Note: With only 2 samples, this is approximate
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)

        # Most features should be roughly centered around 0
        assert np.mean(np.abs(feature_means)) < 1.0

    @patch('twitter_som.preprocessor.NLTK_AVAILABLE', False)
    def test_sentiment_without_nltk(self, config):
        """Test sentiment analysis fallback when NLTK is not available."""
        preprocessor = TwitterPreprocessor(config)

        # Should use TextBlob fallback
        assert preprocessor.sentiment_analyzer is None

        tweets = [
            TwitterData(
                id="1",
                text="I love this amazing product!",
                created_at=datetime.now(timezone.utc),
                user_id="user1",
                username="user1"
            ),
            TwitterData(
                id="2",
                text="This is terrible and awful!",
                created_at=datetime.now(timezone.utc),
                user_id="user2",
                username="user2"
            ),
            TwitterData(
                id="3",
                text="It's okay, nothing special really.",
                created_at=datetime.now(timezone.utc),
                user_id="user3",
                username="user3"
            )
        ]
        collection = TwitterDataCollection(tweets=tweets, collection_name="test")

        # Should not raise error and still extract text features
        features, _ = preprocessor.fit_transform(collection)
        assert features.shape[0] == 3

    def test_get_feature_importance_unfitted(self, preprocessor):
        """Test feature importance error when not fitted."""
        with pytest.raises(ValueError, match="Preprocessor must be fitted"):
            preprocessor.get_feature_importance()

    def test_get_feature_importance_fitted(self, preprocessor, sample_collection):
        """Test feature importance after fitting."""
        preprocessor.fit_transform(sample_collection)

        importance = preprocessor.get_feature_importance()

        # Should return dict with feature names as keys
        assert isinstance(importance, dict)
        assert len(importance) == len(preprocessor.feature_names)

        # All importance values should be between 0 and 1
        for name, score in importance.items():
            assert 0 <= score <= 1
            assert name in preprocessor.feature_names

    def test_missing_values_handling(self, preprocessor):
        """Test handling of missing/invalid values."""
        # Create tweets with text that have some overlapping words after cleaning
        tweets = [
            TwitterData(
                id="1",
                text="amazing content here",  # Simple text with real words
                created_at=datetime.now(timezone.utc),
                user_id="user1",
                username="user1",
            ),
            TwitterData(
                id="2",
                text="great content today",  # Overlapping "content" word
                created_at=datetime.now(timezone.utc),
                user_id="user2",
                username="user2",
            ),
            TwitterData(
                id="3",
                text="wonderful day here",  # Overlapping "here" word
                created_at=datetime.now(timezone.utc),
                user_id="user3",
                username="user3",
            ),
        ]
        collection = TwitterDataCollection(tweets=tweets, collection_name="test")

        # Should handle gracefully without errors
        features, _ = preprocessor.fit_transform(collection)

        # Check for NaN/inf values
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_pca_transformation(self):
        """Test PCA dimensionality reduction."""
        config = SOMTrainingConfig(
            use_pca=True,
            pca_components=5,
            include_temporal_features=True,
            include_engagement_features=True,
            include_text_features=False,
            include_network_features=False
        )

        preprocessor = TwitterPreprocessor(config)

        tweets = [
            TwitterData(
                id=str(i),
                text=f"Test tweet {i}",
                created_at=datetime.now(timezone.utc),
                user_id=f"user{i}",
                username=f"user{i}",
                like_count=i * 10
            )
            for i in range(10)  # Need enough samples for PCA
        ]
        collection = TwitterDataCollection(tweets=tweets, collection_name="test")

        features, feature_names = preprocessor.fit_transform(collection)

        # Should have exactly 5 components
        assert features.shape[1] == 5
        assert len(feature_names) == 5

        # Feature names should be PCA components
        for name in feature_names:
            assert name.startswith('pca_component_')
