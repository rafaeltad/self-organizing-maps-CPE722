"""
Data models for Twitter data using Pydantic.

This module defines the structure for Twitter data that will be used
with the MiniSOM package for clustering and pattern analysis.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import re


class TwitterData(BaseModel):
    """
    Pydantic model for individual Twitter/X posts.

    This model represents the core data structure for tweets that will be
    analyzed using Self-Organizing Maps.
    """

    # Core tweet identifiers
    id: str = Field(..., description="Unique tweet ID")
    text: str = Field(..., description="Tweet text content", min_length=1)

    # Temporal data
    created_at: datetime = Field(..., description="Tweet creation timestamp")

    # User information
    user_id: str = Field(..., description="User ID who posted the tweet")
    username: str = Field(..., description="Username who posted the tweet")

    # Engagement metrics
    retweet_count: int = Field(default=0, description="Number of retweets", ge=0)
    like_count: int = Field(default=0, description="Number of likes", ge=0)
    reply_count: int = Field(default=0, description="Number of replies", ge=0)
    quote_count: int = Field(default=0, description="Number of quotes", ge=0)

    # Content metadata
    hashtags: List[str] = Field(default_factory=list, description="List of hashtags")
    mentions: List[str] = Field(default_factory=list, description="List of user mentions")
    urls: List[str] = Field(default_factory=list, description="List of URLs in tweet")

    # Geographic and language data
    lang: Optional[str] = Field(default=None, description="Language code")
    location: Optional[str] = Field(default=None, description="User location")

    # Tweet type indicators
    is_retweet: bool = Field(default=False, description="Whether this is a retweet")
    is_reply: bool = Field(default=False, description="Whether this is a reply")
    is_quote: bool = Field(default=False, description="Whether this is a quote tweet")

    # Additional metadata
    source: Optional[str] = Field(default=None, description="Source application")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('text')
    @classmethod
    def validate_text_content(cls, v: str) -> str:
        """Validate that text is not empty after stripping."""
        if not v.strip():
            raise ValueError("Tweet text cannot be empty")
        return v.strip()

    @model_validator(mode='after')
    def extract_content_features(self) -> 'TwitterData':
        """Extract hashtags, mentions, and URLs from text if not provided."""
        # Extract hashtags if not provided
        if not self.hashtags:
            hashtags = re.findall(r'#(\w+)', self.text, re.IGNORECASE)
            self.hashtags = [tag.lower() for tag in hashtags]
        else:
            self.hashtags = [tag.lower() for tag in self.hashtags]

        # Extract mentions if not provided
        if not self.mentions:
            mentions = re.findall(r'@(\w+)', self.text, re.IGNORECASE)
            self.mentions = [mention.lower() for mention in mentions]
        else:
            self.mentions = [mention.lower() for mention in self.mentions]

        # Extract URLs if not provided
        if not self.urls:
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, self.text)
            self.urls = urls

        return self

    def get_engagement_score(self) -> float:
        """Calculate a normalized engagement score."""
        return (
            self.like_count * 1.0 +
            self.retweet_count * 2.0 +
            self.reply_count * 1.5 +
            self.quote_count * 2.5
        )

    def get_clean_text(self) -> str:
        """Return text with URLs, mentions, and hashtags removed."""
        text = self.text
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        json_schema_extra={
            "example": {
                "id": "1234567890",
                "text": "This is an example tweet with #hashtag and @mention!",
                "created_at": "2024-01-15T10:30:00Z",
                "user_id": "user123",
                "username": "example_user",
                "retweet_count": 5,
                "like_count": 10,
                "reply_count": 2,
                "quote_count": 1,
                "hashtags": ["hashtag"],
                "mentions": ["mention"],
                "urls": [],
                "lang": "en",
                "location": "New York",
                "is_retweet": False,
                "is_reply": False,
                "is_quote": False,
                "source": "Twitter Web App"
            }
        }
    )


class TwitterDataCollection(BaseModel):
    """
    Collection of Twitter data with metadata.

    This model represents a dataset of tweets that can be used for
    SOM analysis, including collection metadata and utilities.
    """

    tweets: List[TwitterData] = Field(..., description="List of tweets")
    collection_name: str = Field(..., description="Name of the collection")
    collection_date: datetime = Field(default_factory=datetime.now, description="Collection timestamp")
    description: Optional[str] = Field(default=None, description="Collection description")
    source_query: Optional[str] = Field(default=None, description="Query used to collect data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Collection metadata")

    @field_validator('tweets')
    @classmethod
    def validate_non_empty_tweets(cls, v: List[TwitterData]) -> List[TwitterData]:
        """Ensure collection has at least one tweet."""
        if not v:
            raise ValueError("Collection must contain at least one tweet")
        return v

    def get_date_range(self) -> tuple[datetime, datetime]:
        """Get the date range of tweets in the collection."""
        dates = [tweet.created_at for tweet in self.tweets]
        return min(dates), max(dates)

    def get_unique_users(self) -> List[str]:
        """Get list of unique users in the collection."""
        return list(set(tweet.username for tweet in self.tweets))

    def get_total_engagement(self) -> Dict[str, int]:
        """Get total engagement metrics for the collection."""
        return {
            "total_likes": sum(tweet.like_count for tweet in self.tweets),
            "total_retweets": sum(tweet.retweet_count for tweet in self.tweets),
            "total_replies": sum(tweet.reply_count for tweet in self.tweets),
            "total_quotes": sum(tweet.quote_count for tweet in self.tweets),
        }

    def get_most_common_hashtags(self, top_n: int = 10) -> List[tuple[str, int]]:
        """Get most common hashtags in the collection."""
        from collections import Counter
        all_hashtags = []
        for tweet in self.tweets:
            all_hashtags.extend(tweet.hashtags)
        return Counter(all_hashtags).most_common(top_n)

    def filter_by_date_range(self, start_date: datetime, end_date: datetime) -> 'TwitterDataCollection':
        """Filter tweets by date range."""
        filtered_tweets = [
            tweet for tweet in self.tweets
            if start_date <= tweet.created_at <= end_date
        ]
        return TwitterDataCollection(
            tweets=filtered_tweets,
            collection_name=f"{self.collection_name}_filtered",
            description=f"Filtered from {start_date} to {end_date}",
            source_query=self.source_query,
            metadata={**self.metadata, "filter_applied": f"date_range_{start_date}_{end_date}"}
        )

    def filter_by_language(self, language: str) -> 'TwitterDataCollection':
        """Filter tweets by language."""
        filtered_tweets = [
            tweet for tweet in self.tweets
            if tweet.lang == language
        ]
        return TwitterDataCollection(
            tweets=filtered_tweets,
            collection_name=f"{self.collection_name}_{language}",
            description=f"Filtered by language: {language}",
            source_query=self.source_query,
            metadata={**self.metadata, "filter_applied": f"language_{language}"}
        )

    def sample(self, n: int, random_state: Optional[int] = None) -> 'TwitterDataCollection':
        """Get a random sample of tweets."""
        import random
        if random_state:
            random.seed(random_state)

        sampled_tweets = random.sample(self.tweets, min(n, len(self.tweets)))
        return TwitterDataCollection(
            tweets=sampled_tweets,
            collection_name=f"{self.collection_name}_sample_{n}",
            description=f"Random sample of {n} tweets",
            source_query=self.source_query,
            metadata={**self.metadata, "sample_size": n, "random_state": random_state}
        )

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class SOMTrainingConfig(BaseModel):
    """Configuration for SOM training parameters."""

    # SOM dimensions
    x_dim: int = Field(default=10, description="SOM width", gt=0)
    y_dim: int = Field(default=10, description="SOM height", gt=0)

    # Training parameters
    learning_rate: float = Field(default=0.1, description="Initial learning rate", gt=0, le=1)
    neighborhood_function: str = Field(default='gaussian', description="Neighborhood function")
    topology: str = Field(default='rectangular', description="SOM topology")
    activation_distance: str = Field(default='euclidean', description="Distance metric")

    # Training iterations
    num_iterations: int = Field(default=1000, description="Number of training iterations", gt=0)

    # Preprocessing options
    normalize_features: bool = Field(default=True, description="Whether to normalize features")
    use_pca: bool = Field(default=False, description="Whether to apply PCA before SOM")
    pca_components: Optional[int] = Field(default=None, description="Number of PCA components")

    # Feature selection
    include_temporal_features: bool = Field(default=True, description="Include time-based features")
    include_engagement_features: bool = Field(default=True, description="Include engagement metrics")
    include_text_features: bool = Field(default=True, description="Include text-based features")
    include_network_features: bool = Field(default=True, description="Include network-based features")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "x_dim": 15,
                "y_dim": 15,
                "learning_rate": 0.1,
                "neighborhood_function": "gaussian",
                "topology": "rectangular",
                "activation_distance": "euclidean",
                "num_iterations": 1500,
                "normalize_features": True,
                "use_pca": False,
                "include_temporal_features": True,
                "include_engagement_features": True,
                "include_text_features": True,
                "include_network_features": True
            }
        }
    )
