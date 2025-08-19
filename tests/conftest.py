"""
Test configuration for pytest.
"""

import pytest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return os.path.join(os.path.dirname(__file__), 'test_data')

@pytest.fixture
def sample_tweet_dict():
    """Provide sample tweet data as dictionary."""
    from datetime import datetime, timezone

    return {
        "id": "1234567890",
        "text": "This is a sample tweet for testing #python #testing @testuser",
        "created_at": datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        "user_id": "user123",
        "username": "sample_user",
        "retweet_count": 5,
        "like_count": 10,
        "reply_count": 2,
        "quote_count": 1,
        "hashtags": ["python", "testing"],
        "mentions": ["testuser"],
        "urls": [],
        "lang": "en",
        "location": "Test City",
        "is_retweet": False,
        "is_reply": False,
        "is_quote": False,
        "source": "Twitter Web App"
    }
