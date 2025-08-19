"""
Test file for Twitter data models.

Following Test-Driven Development (TDD) principles, these tests define
the expected behavior of our Twitter data models before implementation.
"""

import pytest
from datetime import datetime, timezone
from typing import List, Dict, Any

from twitter_som.models import TwitterData, TwitterDataCollection, SOMTrainingConfig


class TestTwitterData:
    """Test cases for TwitterData model."""

    def test_create_valid_twitter_data(self):
        """Test creating a valid TwitterData instance."""
        tweet_data = {
            "id": "1234567890",
            "text": "This is a test tweet #testing @mention",
            "created_at": datetime.now(timezone.utc),
            "user_id": "user123",
            "username": "testuser",
            "retweet_count": 5,
            "like_count": 10,
            "reply_count": 2,
            "quote_count": 1,
            "lang": "en",
            "location": "Test City"
        }

        tweet = TwitterData(**tweet_data)

        assert tweet.id == "1234567890"
        assert tweet.text == "This is a test tweet #testing @mention"
        assert tweet.user_id == "user123"
        assert tweet.username == "testuser"
        assert tweet.retweet_count == 5
        assert tweet.like_count == 10
        assert tweet.reply_count == 2
        assert tweet.quote_count == 1
        assert tweet.lang == "en"
        assert tweet.location == "Test City"

    def test_twitter_data_with_defaults(self):
        """Test TwitterData with only required fields."""
        tweet = TwitterData(
            id="123",
            text="Test tweet",
            created_at=datetime.now(timezone.utc),
            user_id="user1",
            username="testuser"
        )

        assert tweet.retweet_count == 0
        assert tweet.like_count == 0
        assert tweet.reply_count == 0
        assert tweet.quote_count == 0
        assert tweet.hashtags == []
        assert tweet.mentions == []
        assert tweet.urls == []
        assert tweet.is_retweet is False
        assert tweet.is_reply is False
        assert tweet.is_quote is False
        assert tweet.metadata == {}

    def test_empty_text_validation(self):
        """Test that empty text raises validation error."""
        with pytest.raises(ValueError, match="Tweet text cannot be empty"):
            TwitterData(
                id="123",
                text="   ",  # Only whitespace
                created_at=datetime.now(timezone.utc),
                user_id="user1",
                username="testuser"
            )

    def test_negative_engagement_validation(self):
        """Test that negative engagement counts raise validation error."""
        with pytest.raises(ValueError):
            TwitterData(
                id="123",
                text="Test tweet",
                created_at=datetime.now(timezone.utc),
                user_id="user1",
                username="testuser",
                like_count=-1
            )

    def test_hashtag_extraction(self):
        """Test automatic hashtag extraction from text."""
        tweet = TwitterData(
            id="123",
            text="This tweet has #python #datascience #MachineLearning tags",
            created_at=datetime.now(timezone.utc),
            user_id="user1",
            username="testuser"
        )

        expected_hashtags = ["python", "datascience", "machinelearning"]
        assert tweet.hashtags == expected_hashtags

    def test_mention_extraction(self):
        """Test automatic mention extraction from text."""
        tweet = TwitterData(
            id="123",
            text="Hey @alice and @BOB, check this out!",
            created_at=datetime.now(timezone.utc),
            user_id="user1",
            username="testuser"
        )

        expected_mentions = ["alice", "bob"]
        assert tweet.mentions == expected_mentions

    def test_url_extraction(self):
        """Test automatic URL extraction from text."""
        tweet = TwitterData(
            id="123",
            text="Check out this link: https://example.com and http://test.org",
            created_at=datetime.now(timezone.utc),
            user_id="user1",
            username="testuser"
        )

        expected_urls = ["https://example.com", "http://test.org"]
        assert tweet.urls == expected_urls

    def test_engagement_score_calculation(self):
        """Test engagement score calculation."""
        tweet = TwitterData(
            id="123",
            text="Test tweet",
            created_at=datetime.now(timezone.utc),
            user_id="user1",
            username="testuser",
            like_count=10,
            retweet_count=5,
            reply_count=3,
            quote_count=2
        )

        # Expected: 10*1.0 + 5*2.0 + 3*1.5 + 2*2.5 = 10 + 10 + 4.5 + 5 = 29.5
        assert tweet.get_engagement_score() == 29.5

    def test_clean_text_method(self):
        """Test text cleaning functionality."""
        tweet = TwitterData(
            id="123",
            text="Hey @user check https://example.com #hashtag multiple   spaces",
            created_at=datetime.now(timezone.utc),
            user_id="user1",
            username="testuser"
        )

        clean_text = tweet.get_clean_text()
        assert clean_text == "Hey check multiple spaces"

    def test_json_serialization(self):
        """Test that TwitterData can be serialized to JSON."""
        tweet = TwitterData(
            id="123",
            text="Test tweet",
            created_at=datetime.now(timezone.utc),
            user_id="user1",
            username="testuser"
        )

        json_data = tweet.model_dump()
        assert "id" in json_data
        assert "text" in json_data
        assert "created_at" in json_data


class TestTwitterDataCollection:
    """Test cases for TwitterDataCollection model."""

    @pytest.fixture
    def sample_tweets(self) -> List[TwitterData]:
        """Create sample tweets for testing."""
        return [
            TwitterData(
                id=str(i),
                text=f"Test tweet {i} #test",
                created_at=datetime(2024, 1, i, tzinfo=timezone.utc),  # Start from day 1
                user_id=f"user{i}",
                username=f"testuser{i}",
                like_count=i * 2,
                retweet_count=i,
                lang="en"
            )
            for i in range(1, 6)  # 5 tweets, days 1-5
        ]

    def test_create_collection(self, sample_tweets):
        """Test creating a valid TwitterDataCollection."""
        collection = TwitterDataCollection(
            tweets=sample_tweets,
            collection_name="test_collection",
            description="Test collection"
        )

        assert len(collection.tweets) == 5
        assert collection.collection_name == "test_collection"
        assert collection.description == "Test collection"
        assert isinstance(collection.collection_date, datetime)

    def test_empty_collection_validation(self):
        """Test that empty collection raises validation error."""
        with pytest.raises(ValueError, match="Collection must contain at least one tweet"):
            TwitterDataCollection(
                tweets=[],
                collection_name="empty_collection"
            )

    def test_get_date_range(self, sample_tweets):
        """Test date range calculation."""
        collection = TwitterDataCollection(
            tweets=sample_tweets,
            collection_name="test_collection"
        )

        start_date, end_date = collection.get_date_range()
        assert start_date == datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert end_date == datetime(2024, 1, 5, tzinfo=timezone.utc)

    def test_get_unique_users(self, sample_tweets):
        """Test unique users extraction."""
        collection = TwitterDataCollection(
            tweets=sample_tweets,
            collection_name="test_collection"
        )

        users = collection.get_unique_users()
        expected_users = ["testuser1", "testuser2", "testuser3", "testuser4", "testuser5"]
        assert set(users) == set(expected_users)

    def test_get_total_engagement(self, sample_tweets):
        """Test total engagement calculation."""
        collection = TwitterDataCollection(
            tweets=sample_tweets,
            collection_name="test_collection"
        )

        engagement = collection.get_total_engagement()

        # Expected: likes = 2+4+6+8+10 = 30, retweets = 1+2+3+4+5 = 15
        assert engagement["total_likes"] == 30
        assert engagement["total_retweets"] == 15
        assert engagement["total_replies"] == 0
        assert engagement["total_quotes"] == 0

    def test_get_most_common_hashtags(self, sample_tweets):
        """Test hashtag frequency analysis."""
        collection = TwitterDataCollection(
            tweets=sample_tweets,
            collection_name="test_collection"
        )

        common_hashtags = collection.get_most_common_hashtags(top_n=5)
        assert common_hashtags[0] == ("test", 5)  # "test" appears in all 5 tweets

    def test_filter_by_date_range(self, sample_tweets):
        """Test date range filtering."""
        collection = TwitterDataCollection(
            tweets=sample_tweets,
            collection_name="test_collection"
        )

        start_date = datetime(2024, 1, 2, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 4, tzinfo=timezone.utc)

        filtered = collection.filter_by_date_range(start_date, end_date)

        assert len(filtered.tweets) == 3  # tweets 2, 3, 4
        assert filtered.collection_name == "test_collection_filtered"
        assert "filter_applied" in filtered.metadata

    def test_filter_by_language(self, sample_tweets):
        """Test language filtering."""
        # Add a Spanish tweet
        spanish_tweet = TwitterData(
            id="6",
            text="Tweet en español",
            created_at=datetime.now(timezone.utc),
            user_id="user6",
            username="testuser6",
            lang="es"
        )
        sample_tweets.append(spanish_tweet)

        collection = TwitterDataCollection(
            tweets=sample_tweets,
            collection_name="test_collection"
        )

        english_tweets = collection.filter_by_language("en")
        assert len(english_tweets.tweets) == 5

        spanish_tweets = collection.filter_by_language("es")
        assert len(spanish_tweets.tweets) == 1

    def test_sample_method(self, sample_tweets):
        """Test random sampling."""
        collection = TwitterDataCollection(
            tweets=sample_tweets,
            collection_name="test_collection"
        )

        sample = collection.sample(n=3, random_state=42)

        assert len(sample.tweets) == 3
        assert sample.collection_name == "test_collection_sample_3"
        assert sample.metadata["sample_size"] == 3
        assert sample.metadata["random_state"] == 42


class TestSOMTrainingConfig:
    """Test cases for SOMTrainingConfig model."""

    def test_create_default_config(self):
        """Test creating config with default values."""
        config = SOMTrainingConfig()

        assert config.x_dim == 10
        assert config.y_dim == 10
        assert config.learning_rate == 0.1
        assert config.neighborhood_function == 'gaussian'
        assert config.topology == 'rectangular'
        assert config.activation_distance == 'euclidean'
        assert config.num_iterations == 1000
        assert config.normalize_features is True
        assert config.use_pca is False
        assert config.pca_components is None

    def test_create_custom_config(self):
        """Test creating config with custom values."""
        config = SOMTrainingConfig(
            x_dim=15,
            y_dim=20,
            learning_rate=0.05,
            num_iterations=2000,
            use_pca=True,
            pca_components=50
        )

        assert config.x_dim == 15
        assert config.y_dim == 20
        assert config.learning_rate == 0.05
        assert config.num_iterations == 2000
        assert config.use_pca is True
        assert config.pca_components == 50

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise validation errors."""
        with pytest.raises(ValueError):
            SOMTrainingConfig(x_dim=0)

        with pytest.raises(ValueError):
            SOMTrainingConfig(y_dim=-1)

    def test_invalid_learning_rate(self):
        """Test that invalid learning rate raises validation error."""
        with pytest.raises(ValueError):
            SOMTrainingConfig(learning_rate=0)

        with pytest.raises(ValueError):
            SOMTrainingConfig(learning_rate=1.5)

    def test_invalid_iterations(self):
        """Test that invalid iterations raise validation error."""
        with pytest.raises(ValueError):
            SOMTrainingConfig(num_iterations=0)

    def test_feature_flags(self):
        """Test feature inclusion flags."""
        config = SOMTrainingConfig(
            include_temporal_features=False,
            include_engagement_features=True,
            include_text_features=False,
            include_network_features=True
        )

        assert config.include_temporal_features is False
        assert config.include_engagement_features is True
        assert config.include_text_features is False
        assert config.include_network_features is True
