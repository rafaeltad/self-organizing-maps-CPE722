"""
Data models for Twitter data using Pydantic.

This module defines the structure for Twitter data that will be used
with the MiniSOM package for clustering and pattern analysis.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import re
import pandas as pd


class TwitterData(BaseModel):
    """
    Pydantic model for individual Twitter/X posts.

    This model represents the core data structure for tweets that will be
    analyzed using Self-Organizing Maps.
    """

    # Core tweet identifiers (matching actual data columns)
    id_str: str = Field(..., description="Unique tweet ID")
    text: str = Field(..., description="Tweet text content", min_length=1)

    # Temporal data
    created_at: datetime = Field(..., description="Tweet creation timestamp")

    # User information
    user_id: str = Field(..., description="User ID who posted the tweet")
    screen_name: str = Field(
        ..., description="Screen name who posted the tweet"
    )
    verified: bool = Field(
        default=False, description="Whether user is verified"
    )

    # Content metadata (from actual data format)
    hashtags: List[str] = Field(default_factory=list, description="List of hashtags")
    mentions: List[str] = Field(default_factory=list, description="List of user mentions")
    urls: List[str] = Field(default_factory=list, description="List of URLs in tweet")

    # Geographic and language data
    lang: Optional[str] = Field(default=None, description="Language code")
    location: Optional[str] = Field(default=None, description="User location")

    # Tweet type indicators
    is_rt: bool = Field(default=False, description="Whether this is a retweet")
    is_reply: bool = Field(default=False, description="Whether this is a reply")
    is_quote: bool = Field(default=False, description="Whether this is a quote tweet")

    # Retweet information (from actual data)
    retweeted_status_id_str: Optional[str] = Field(
        default=None, description="Retweeted tweet ID"
    )
    retweeted_status_user_id_str: Optional[str] = Field(
        default=None, description="Retweeted user ID"
    )
    retweeted_status_user_screen_name: Optional[str] = Field(
        default=None, description="Retweeted user screen name"
    )
    retweeted_status_created_at: Optional[datetime] = Field(
        default=None, description="Retweeted tweet timestamp"
    )

    # Additional metadata
    source: Optional[str] = Field(default=None, description="Source application")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Legacy field mapping for backward compatibility
    @property
    def id(self) -> str:
        """Legacy support for id field."""
        return self.id_str

    @property
    def username(self) -> str:
        """Legacy support for username field."""
        return self.screen_name

    @property
    def is_retweet(self) -> bool:
        """Legacy support for is_retweet field."""
        return self.is_rt

    @field_validator("hashtags", mode="before")
    @classmethod
    def validate_hashtags(cls, v) -> List[str]:
        """Convert hashtags from string format to list."""
        if v is None:
            return []
        if isinstance(v, str) and v.strip():
            return [tag.strip().lower() for tag in v.split(";") if tag.strip()]
        if isinstance(v, list):
            return [tag.lower() for tag in v if tag]
        return []

    @field_validator("mentions", mode="before")
    @classmethod
    def validate_mentions(cls, v) -> List[str]:
        """Convert mentions from string format to list."""
        if v is None:
            return []
        if isinstance(v, str) and v.strip():
            return [
                mention.strip().lower()
                for mention in v.split(";")
                if mention.strip()
            ]
        if isinstance(v, list):
            return [mention.lower() for mention in v if mention]
        return []

    @field_validator('text')
    @classmethod
    def validate_text_content(cls, v: str) -> str:
        """Validate that text is not empty after stripping."""
        if not v.strip():
            raise ValueError("Tweet text cannot be empty")
        return v.strip()

    @field_validator("verified", mode="before")
    @classmethod
    def validate_verified(cls, v) -> bool:
        """Convert string representation of boolean to actual boolean."""
        if isinstance(v, str):
            return v.lower() == "true"
        return bool(v)

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_created_at(cls, v) -> datetime:
        """Parse datetime from string format."""
        if isinstance(v, str):
            # Handle the format from the dataset: "2020-03-15T22:36:54"
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                # Try parsing without timezone info
                return datetime.fromisoformat(v)
        return v

    @field_validator("retweeted_status_created_at", mode="before")
    @classmethod
    def parse_retweeted_status_created_at(cls, v) -> Optional[datetime]:
        """Parse retweeted status datetime from string format."""
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return None
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                # Try parsing without timezone info
                return datetime.fromisoformat(v)
        return v

    @model_validator(mode='after')
    def extract_content_features(self) -> 'TwitterData':
        """Extract hashtags, mentions, and URLs from text if not already provided."""
        # Extract hashtags from text if not provided by field validator
        if not self.hashtags:
            hashtags = re.findall(r'#(\w+)', self.text, re.IGNORECASE)
            self.hashtags = [tag.lower() for tag in hashtags]

        # Extract mentions from text if not provided by field validator
        if not self.mentions:
            mentions = re.findall(r'@(\w+)', self.text, re.IGNORECASE)
            self.mentions = [mention.lower() for mention in mentions]

        # Extract URLs if not provided
        if not self.urls:
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, self.text)
            self.urls = urls

        return self

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
        json_encoders={datetime: lambda v: v.isoformat()},
        json_schema_extra={
            "example": {
                "id": "1234567890",
                "text": "This is an example tweet with #hashtag and @mention!",
                "created_at": "2024-01-15T10:30:00Z",
                "user_id": "user123",
                "username": "example_user",
                "hashtags": ["hashtag"],
                "mentions": ["mention"],
                "urls": [],
                "lang": "en",
                "location": "New York",
                "is_retweet": False,
                "is_reply": False,
                "is_quote": False,
                "source": "Twitter Web App",
            }
        },
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
        return list(set(tweet.screen_name for tweet in self.tweets))

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
    include_engagement_features: bool = Field(
        default=False, description="Include engagement metrics"
    )
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
                "include_engagement_features": False,
                "include_text_features": True,
                "include_network_features": True,
            }
        }
    )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "SOMTrainingConfig":
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            SOMTrainingConfig instance loaded from the YAML file

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
            ValueError: If the configuration values are invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with open(config_path, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file)

        # Flatten the nested structure from YAML
        flattened_config = {}

        # Extract SOM parameters
        if "som" in config_data:
            som_config = config_data["som"]
            flattened_config.update(
                {
                    "x_dim": som_config.get("x_dim"),
                    "y_dim": som_config.get("y_dim"),
                    "learning_rate": som_config.get("learning_rate"),
                    "neighborhood_function": som_config.get(
                        "neighborhood_function"
                    ),
                    "topology": som_config.get("topology"),
                    "activation_distance": som_config.get(
                        "activation_distance"
                    ),
                    "num_iterations": som_config.get("num_iterations"),
                }
            )

        # Extract preprocessing parameters
        if "preprocessing" in config_data:
            preprocessing_config = config_data["preprocessing"]
            flattened_config.update(
                {
                    "normalize_features": preprocessing_config.get(
                        "normalize_features"
                    ),
                    "use_pca": preprocessing_config.get("use_pca"),
                    "pca_components": preprocessing_config.get(
                        "pca_components"
                    ),
                }
            )

        # Extract feature selection parameters
        if "features" in config_data:
            features_config = config_data["features"]
            flattened_config.update(
                {
                    "include_temporal_features": features_config.get(
                        "include_temporal_features"
                    ),
                    "include_engagement_features": features_config.get(
                        "include_engagement_features"
                    ),
                    "include_text_features": features_config.get(
                        "include_text_features"
                    ),
                    "include_network_features": features_config.get(
                        "include_network_features"
                    ),
                }
            )

        # Remove None values to use defaults
        flattened_config = {
            k: v for k, v in flattened_config.items() if v is not None
        }

        return cls(**flattened_config)

    def to_yaml(self, config_path: str | Path) -> None:
        """
        Save configuration to a YAML file.

        Args:
            config_path: Path where to save the YAML configuration file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Structure the data in a nested format for better readability
        config_data = {
            "som": {
                "x_dim": self.x_dim,
                "y_dim": self.y_dim,
                "learning_rate": self.learning_rate,
                "neighborhood_function": self.neighborhood_function,
                "topology": self.topology,
                "activation_distance": self.activation_distance,
                "num_iterations": self.num_iterations,
            },
            "preprocessing": {
                "normalize_features": self.normalize_features,
                "use_pca": self.use_pca,
                "pca_components": self.pca_components,
            },
            "features": {
                "include_temporal_features": self.include_temporal_features,
                "include_engagement_features": self.include_engagement_features,
                "include_text_features": self.include_text_features,
                "include_network_features": self.include_network_features,
            },
        }

        with open(config_path, "w", encoding="utf-8") as file:
            yaml.dump(
                config_data, file, default_flow_style=False, sort_keys=False
            )


def load_twitter_data_from_parquet(
    file_path: str | Path,
    max_rows: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    random_state: Optional[int] = None,
) -> TwitterDataCollection:
    """
    Load Twitter data from a parquet file.

    Args:
        file_path: Path to the parquet file
        max_rows: Maximum number of rows to load (None for all)
        sample_fraction: Fraction of data to sample (0.0 to 1.0, None for all)
        random_state: Random seed for sampling

    Returns:
        TwitterDataCollection with the loaded tweets
    """
    import pandas as pd

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    # Load the parquet file
    df = pd.read_parquet(file_path)

    # Sample data if requested
    if sample_fraction is not None:
        df = df.sample(frac=sample_fraction, random_state=random_state)
    elif max_rows is not None:
        df = df.head(max_rows)

    # Convert DataFrame to list of TwitterData objects
    tweets = []
    for _, row in df.iterrows():
        try:
            # Convert row to dict and handle NaN values
            tweet_data = row.to_dict()

            # Replace NaN values with None or appropriate defaults
            for key, value in tweet_data.items():
                if pd.isna(value):
                    tweet_data[key] = None

            # Create TwitterData object
            tweet = TwitterData(**tweet_data)
            tweets.append(tweet)

        except Exception as e:
            # Log the error but continue processing other tweets
            print(f"Warning: Skipping tweet due to validation error: {e}")
            continue

    if not tweets:
        raise ValueError("No valid tweets found in the dataset")

    # Create collection
    collection = TwitterDataCollection(
        tweets=tweets,
        collection_name=f"twitter_data_{file_path.stem}",
        description=f"Loaded from {file_path}",
        metadata={
            "original_file": str(file_path),
            "total_rows_in_file": len(df),
            "loaded_tweets": len(tweets),
            "sample_fraction": sample_fraction,
            "max_rows": max_rows,
            "random_state": random_state,
        },
    )

    return collection
