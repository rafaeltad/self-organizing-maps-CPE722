"""
Self-Organizing Map analyzer for Twitter data.

This module implements the main SOM analysis functionality using MiniSOM
for clustering and pattern discovery in Twitter data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pickle
import json

try:
    from minisom import MiniSom
    MINISOM_AVAILABLE = True
except ImportError:
    MINISOM_AVAILABLE = False

from .models import TwitterData, TwitterDataCollection, SOMTrainingConfig
from .preprocessor import TwitterPreprocessor


class TwitterSOMAnalyzer:
    """
    Main class for Twitter data analysis using Self-Organizing Maps.

    This class provides functionality to:
    - Train SOM on Twitter data
    - Analyze clusters and patterns
    - Generate insights and predictions
    - Save/load trained models
    """

    def __init__(self, config: SOMTrainingConfig):
        """
        Initialize the SOM analyzer.

        Args:
            config: SOM training configuration
        """
        if not MINISOM_AVAILABLE:
            raise ImportError("MiniSOM is required but not installed. Install with: pip install minisom")

        self.config = config
        self.preprocessor = TwitterPreprocessor(config)
        self.som: Optional[MiniSom] = None
        self.feature_names: List[str] = []
        self.training_data: Optional[np.ndarray] = None
        self.training_collection: Optional[TwitterDataCollection] = None
        self.is_trained = False

        # Analysis results
        self.cluster_assignments: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.cluster_stats: Dict[str, Any] = {}

    def train(self, collection: TwitterDataCollection, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the SOM on Twitter data.

        Args:
            collection: Twitter data collection for training
            verbose: Whether to print training progress

        Returns:
            Training statistics and metrics
        """
        if verbose:
            print(f"Preprocessing {len(collection.tweets)} tweets...")

        # Preprocess the data
        features, feature_names = self.preprocessor.fit_transform(collection)
        self.training_data = features
        self.training_collection = collection
        self.feature_names = feature_names

        if verbose:
            print(f"Extracted {features.shape[1]} features")

        # Initialize SOM
        self.som = MiniSom(
            x=self.config.x_dim,
            y=self.config.y_dim,
            input_len=features.shape[1],
            learning_rate=self.config.learning_rate,
            neighborhood_function=self.config.neighborhood_function,
            topology=self.config.topology,
            activation_distance=self.config.activation_distance
        )

        if verbose:
            print(f"Initializing SOM with dimensions {self.config.x_dim}x{self.config.y_dim}")

        # Initialize weights
        self.som.pca_weights_init(features)

        if verbose:
            print(f"Training SOM for {self.config.num_iterations} iterations...")

        # Train the SOM
        self.som.train(
            features,
            num_iteration=self.config.num_iterations,
            verbose=verbose
        )

        self.is_trained = True

        # Analyze the results
        self._analyze_clusters()

        # Calculate training statistics
        training_stats = self._calculate_training_stats()

        if verbose:
            print("Training completed!")
            print(f"Found {len(self.cluster_stats)} clusters")
            print(f"Quantization error: {training_stats['quantization_error']:.4f}")
            print(f"Topographic error: {training_stats['topographic_error']:.4f}")

        return training_stats

    def _analyze_clusters(self) -> None:
        """Analyze the trained SOM to identify clusters and patterns."""
        if not self.is_trained or self.som is None:
            raise ValueError("SOM must be trained before analyzing clusters")

        # Get winner neurons for each data point
        winners = []
        for i in range(len(self.training_data)):
            winner = self.som.winner(self.training_data[i])
            winners.append(winner)

        self.cluster_assignments = np.array(winners)

        # Calculate cluster statistics
        unique_clusters = np.unique(self.cluster_assignments, axis=0)
        self.cluster_centers = unique_clusters

        cluster_stats = {}

        for i, cluster in enumerate(unique_clusters):
            cluster_key = f"cluster_{cluster[0]}_{cluster[1]}"

            # Find tweets in this cluster
            mask = np.all(self.cluster_assignments == cluster, axis=1)
            cluster_tweets = [self.training_collection.tweets[j] for j in np.where(mask)[0]]
            cluster_features = self.training_data[mask]

            # Calculate cluster statistics
            stats = {
                'size': len(cluster_tweets),
                'coordinates': tuple(cluster),
                'mean_features': np.mean(cluster_features, axis=0).tolist(),
                'std_features': np.std(cluster_features, axis=0).tolist(),
                'sample_tweets': [tweet.text for tweet in cluster_tweets[:5]],  # Sample tweets
                'avg_engagement': np.mean([tweet.get_engagement_score() for tweet in cluster_tweets]),
                'dominant_hashtags': self._get_dominant_hashtags(cluster_tweets),
                'time_distribution': self._analyze_time_distribution(cluster_tweets),
                'language_distribution': self._analyze_language_distribution(cluster_tweets)
            }

            cluster_stats[cluster_key] = stats

        self.cluster_stats = cluster_stats

    def _get_dominant_hashtags(self, tweets: List[TwitterData], top_n: int = 5) -> List[Tuple[str, int]]:
        """Get dominant hashtags in a cluster."""
        from collections import Counter

        all_hashtags = []
        for tweet in tweets:
            all_hashtags.extend(tweet.hashtags)

        return Counter(all_hashtags).most_common(top_n)

    def _analyze_time_distribution(self, tweets: List[TwitterData]) -> Dict[str, Any]:
        """Analyze temporal distribution of tweets in a cluster."""
        hours = [tweet.created_at.hour for tweet in tweets]
        days = [tweet.created_at.weekday() for tweet in tweets]

        return {
            'peak_hour': max(set(hours), key=hours.count),
            'peak_day': max(set(days), key=days.count),
            'hour_distribution': {str(h): hours.count(h) for h in set(hours)},
            'day_distribution': {str(d): days.count(d) for d in set(days)}
        }

    def _analyze_language_distribution(self, tweets: List[TwitterData]) -> Dict[str, int]:
        """Analyze language distribution of tweets in a cluster."""
        languages = [tweet.lang for tweet in tweets if tweet.lang]
        from collections import Counter
        return dict(Counter(languages))

    def _calculate_training_stats(self) -> Dict[str, Any]:
        """Calculate training statistics and quality metrics."""
        if not self.is_trained or self.som is None:
            raise ValueError("SOM must be trained before calculating stats")

        # Quantization error
        quantization_error = self.som.quantization_error(self.training_data)

        # Topographic error
        topographic_error = self.som.topographic_error(self.training_data)

        # Additional statistics
        stats = {
            'quantization_error': float(quantization_error),
            'topographic_error': float(topographic_error),
            'num_clusters': len(self.cluster_stats),
            'num_features': len(self.feature_names),
            'num_samples': len(self.training_data),
            'som_dimensions': (self.config.x_dim, self.config.y_dim),
            'largest_cluster_size': max(stats['size'] for stats in self.cluster_stats.values()) if self.cluster_stats else 0,
            'smallest_cluster_size': min(stats['size'] for stats in self.cluster_stats.values()) if self.cluster_stats else 0,
            'training_config': self.config.dict()
        }

        return stats

    def predict_cluster(self, collection: TwitterDataCollection) -> List[Tuple[int, int]]:
        """
        Predict cluster assignments for new tweets.

        Args:
            collection: New Twitter data to classify

        Returns:
            List of (x, y) cluster coordinates for each tweet
        """
        if not self.is_trained or self.som is None:
            raise ValueError("SOM must be trained before making predictions")

        # Preprocess new data
        features = self.preprocessor.transform(collection)

        # Get winner neurons
        winners = []
        for feature_vector in features:
            winner = self.som.winner(feature_vector)
            winners.append(winner)

        return winners

    def get_cluster_insights(self, cluster_id: str) -> Dict[str, Any]:
        """
        Get detailed insights for a specific cluster.

        Args:
            cluster_id: Cluster identifier (e.g., "cluster_5_3")

        Returns:
            Detailed cluster analysis
        """
        if cluster_id not in self.cluster_stats:
            raise ValueError(f"Cluster {cluster_id} not found")

        stats = self.cluster_stats[cluster_id]

        # Add additional insights
        insights = {
            **stats,
            'feature_importance': self._get_cluster_feature_importance(cluster_id),
            'similarity_to_other_clusters': self._calculate_cluster_similarities(cluster_id),
            'temporal_patterns': stats['time_distribution'],
            'content_themes': self._extract_content_themes(cluster_id)
        }

        return insights

    def _get_cluster_feature_importance(self, cluster_id: str) -> Dict[str, float]:
        """Calculate feature importance for a specific cluster."""
        if cluster_id not in self.cluster_stats:
            return {}

        cluster_stats = self.cluster_stats[cluster_id]
        mean_features = np.array(cluster_stats['mean_features'])

        # Simple feature importance based on deviation from global mean
        global_mean = np.mean(self.training_data, axis=0)
        importance = np.abs(mean_features - global_mean)

        # Normalize to 0-1 range
        if np.max(importance) > 0:
            importance = importance / np.max(importance)

        return {name: float(importance[i]) for i, name in enumerate(self.feature_names)}

    def _calculate_cluster_similarities(self, cluster_id: str) -> Dict[str, float]:
        """Calculate similarity to other clusters."""
        if cluster_id not in self.cluster_stats:
            return {}

        cluster_features = np.array(self.cluster_stats[cluster_id]['mean_features'])
        similarities = {}

        for other_id, other_stats in self.cluster_stats.items():
            if other_id != cluster_id:
                other_features = np.array(other_stats['mean_features'])

                # Calculate cosine similarity
                dot_product = np.dot(cluster_features, other_features)
                norm_product = np.linalg.norm(cluster_features) * np.linalg.norm(other_features)

                if norm_product > 0:
                    similarity = dot_product / norm_product
                    similarities[other_id] = float(similarity)

        return similarities

    def _extract_content_themes(self, cluster_id: str) -> List[str]:
        """Extract content themes from cluster tweets."""
        if cluster_id not in self.cluster_stats:
            return []

        # Use dominant hashtags as simple theme indicators
        hashtags = self.cluster_stats[cluster_id]['dominant_hashtags']
        return [hashtag for hashtag, count in hashtags]

    def get_all_clusters_summary(self) -> Dict[str, Any]:
        """Get summary of all clusters."""
        if not self.cluster_stats:
            return {}

        summary = {
            'total_clusters': len(self.cluster_stats),
            'total_tweets': sum(stats['size'] for stats in self.cluster_stats.values()),
            'cluster_sizes': {cluster_id: stats['size'] for cluster_id, stats in self.cluster_stats.items()},
            'dominant_themes': self._get_global_themes(),
            'temporal_overview': self._get_global_temporal_patterns(),
            'engagement_distribution': self._get_engagement_distribution()
        }

        return summary

    def _get_global_themes(self) -> List[Tuple[str, int]]:
        """Get global themes across all clusters."""
        from collections import Counter

        all_hashtags = []
        for stats in self.cluster_stats.values():
            for hashtag, count in stats['dominant_hashtags']:
                all_hashtags.extend([hashtag] * count)

        return Counter(all_hashtags).most_common(10)

    def _get_global_temporal_patterns(self) -> Dict[str, Any]:
        """Get global temporal patterns."""
        all_hours = []
        all_days = []

        for tweet in self.training_collection.tweets:
            all_hours.append(tweet.created_at.hour)
            all_days.append(tweet.created_at.weekday())

        from collections import Counter

        return {
            'peak_hours': Counter(all_hours).most_common(5),
            'peak_days': Counter(all_days).most_common(7)
        }

    def _get_engagement_distribution(self) -> Dict[str, float]:
        """Get engagement distribution across clusters."""
        engagements = [stats['avg_engagement'] for stats in self.cluster_stats.values()]

        return {
            'mean': float(np.mean(engagements)),
            'std': float(np.std(engagements)),
            'min': float(np.min(engagements)),
            'max': float(np.max(engagements)),
            'median': float(np.median(engagements))
        }

    def save_model(self, filepath: str) -> None:
        """
        Save the trained SOM model and preprocessor.

        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            'som_weights': self.som.get_weights(),
            'config': self.config.dict(),
            'feature_names': self.feature_names,
            'cluster_stats': self.cluster_stats,
            'preprocessor': self.preprocessor,
            'training_stats': self._calculate_training_stats()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str) -> None:
        """
        Load a trained SOM model.

        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Restore configuration
        self.config = SOMTrainingConfig(**model_data['config'])

        # Restore SOM
        weights = model_data['som_weights']
        self.som = MiniSom(
            x=weights.shape[0],
            y=weights.shape[1],
            input_len=weights.shape[2],
            learning_rate=self.config.learning_rate,
            neighborhood_function=self.config.neighborhood_function,
            topology=self.config.topology,
            activation_distance=self.config.activation_distance
        )
        self.som._weights = weights

        # Restore other components
        self.feature_names = model_data['feature_names']
        self.cluster_stats = model_data['cluster_stats']
        self.preprocessor = model_data['preprocessor']
        self.is_trained = True

    def export_results(self, filepath: str) -> None:
        """
        Export analysis results to JSON.

        Args:
            filepath: Path to save the results
        """
        if not self.is_trained:
            raise ValueError("Cannot export results from untrained model")

        results = {
            'training_stats': self._calculate_training_stats(),
            'cluster_summary': self.get_all_clusters_summary(),
            'feature_names': self.feature_names,
            'cluster_details': self.cluster_stats
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
