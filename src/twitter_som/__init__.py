"""
Twitter SOM - Self-Organizing Maps for Twitter Data Analysis

A Python package for analyzing Twitter data using Self-Organizing Maps (SOM)
to discover patterns, cluster similar tweets, and visualize data relationships.
"""

__version__ = "0.1.0"
__author__ = "Rafael Tadeu"

from .models import (
    TwitterData,
    TwitterDataCollection,
    SOMTrainingConfig,
    load_twitter_data_from_parquet,
)
from .som_analyzer import TwitterSOMAnalyzer
from .preprocessor import TwitterPreprocessor
from .visualizer import SOMVisualizer

__all__ = [
    "TwitterData",
    "TwitterDataCollection",
    "SOMTrainingConfig",
    "TwitterSOMAnalyzer",
    "TwitterPreprocessor",
    "SOMVisualizer",
    "load_twitter_data_from_parquet",
]
