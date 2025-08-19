"""
Test file for SOMVisualizer.

Tests the visualization functionality for SOM analysis results.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from twitter_som.visualizer import SOMVisualizer
from twitter_som.som_analyzer import TwitterSOMAnalyzer
from twitter_som.models import TwitterData, TwitterDataCollection, SOMTrainingConfig


class TestSOMVisualizer:
    """Test cases for SOMVisualizer."""

    @pytest.fixture
    def mock_analyzer(self):
        """Create a mock trained analyzer."""
        analyzer = Mock(spec=TwitterSOMAnalyzer)
        analyzer.is_trained = True
        analyzer.som = Mock()
        analyzer.som._weights = np.random.random((5, 5, 10))
        analyzer.som.winner.return_value = (2, 3)
        analyzer.som.quantization_error.return_value = 0.3
        analyzer.som.topographic_error.return_value = 0.05

        analyzer.training_data = np.random.random((20, 10))
        analyzer.feature_names = [f"feature_{i}" for i in range(10)]

        # Mock cluster stats
        analyzer.cluster_stats = {
            "cluster_1_1": {
                "size": 5,
                "coordinates": (1, 1),
                "mean_features": np.random.random(10).tolist(),
                "std_features": np.random.random(10).tolist(),
                "sample_tweets": ["Sample tweet 1", "Sample tweet 2"],
                "avg_engagement": 15.5,
                "dominant_hashtags": [("python", 3), ("ai", 2)],
                "time_distribution": {
                    "peak_hour": 14,
                    "peak_day": 1,
                    "hour_distribution": {"14": 2, "15": 1},
                    "day_distribution": {"1": 3, "2": 2}
                },
                "language_distribution": {"en": 5}
            },
            "cluster_2_3": {
                "size": 8,
                "coordinates": (2, 3),
                "mean_features": np.random.random(10).tolist(),
                "std_features": np.random.random(10).tolist(),
                "sample_tweets": ["Another tweet", "More sample text"],
                "avg_engagement": 25.2,
                "dominant_hashtags": [("datascience", 4), ("ml", 3)],
                "time_distribution": {
                    "peak_hour": 10,
                    "peak_day": 3,
                    "hour_distribution": {"10": 3, "11": 2},
                    "day_distribution": {"3": 4, "4": 4}
                },
                "language_distribution": {"en": 6, "es": 2}
            }
        }

        # Mock training collection
        tweets = [
            TwitterData(
                id=str(i),
                text=f"Test tweet {i}",
                created_at=datetime(2024, 1, i+1, 10+i, 0, 0, tzinfo=timezone.utc),
                user_id=f"user{i}",
                username=f"user{i}",
                like_count=i*2,
                lang="en"
            )
            for i in range(5)
        ]
        analyzer.training_collection = TwitterDataCollection(
            tweets=tweets,
            collection_name="test"
        )

        # Mock methods
        analyzer._get_cluster_feature_importance.return_value = {
            f"feature_{i}": np.random.random() for i in range(10)
        }
        analyzer._calculate_cluster_similarities.return_value = {
            "cluster_2_3": 0.7
        }
        analyzer.get_all_clusters_summary.return_value = {
            "total_clusters": 2,
            "total_tweets": 13,
            "cluster_sizes": {"cluster_1_1": 5, "cluster_2_3": 8},
            "dominant_themes": [("python", 10), ("ai", 8)],
            "temporal_overview": {
                "peak_hours": [(14, 5), (10, 3)],
                "peak_days": [(1, 7), (3, 6)]
            },
            "engagement_distribution": {
                "mean": 20.35,
                "std": 6.85,
                "min": 15.5,
                "max": 25.2,
                "median": 20.35
            }
        }
        analyzer._calculate_training_stats.return_value = {
            "quantization_error": 0.3,
            "topographic_error": 0.05,
            "num_clusters": 2,
            "largest_cluster_size": 8,
            "smallest_cluster_size": 5
        }

        return analyzer

    def test_visualizer_initialization_trained(self, mock_analyzer):
        """Test visualizer initialization with trained analyzer."""
        visualizer = SOMVisualizer(mock_analyzer)

        assert visualizer.analyzer == mock_analyzer

    def test_visualizer_initialization_untrained(self):
        """Test visualizer initialization with untrained analyzer."""
        untrained_analyzer = Mock(spec=TwitterSOMAnalyzer)
        untrained_analyzer.is_trained = False

        with pytest.warns(UserWarning, match="Analyzer is not trained"):
            visualizer = SOMVisualizer(untrained_analyzer)
            assert visualizer.analyzer == untrained_analyzer

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_som_topology(self, mock_savefig, mock_show, mock_analyzer):
        """Test SOM topology plotting."""
        visualizer = SOMVisualizer(mock_analyzer)

        # Should not raise errors
        visualizer.plot_som_topology(save_path="test_topology.png")

        # Verify save was called
        mock_savefig.assert_called_once_with("test_topology.png", dpi=300, bbox_inches='tight')
        mock_show.assert_called_once()

    def test_plot_som_topology_untrained(self):
        """Test error when plotting with untrained analyzer."""
        untrained_analyzer = Mock(spec=TwitterSOMAnalyzer)
        untrained_analyzer.is_trained = False

        with pytest.warns(UserWarning):
            visualizer = SOMVisualizer(untrained_analyzer)

        with pytest.raises(ValueError, match="Analyzer must be trained"):
            visualizer.plot_som_topology()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_cluster_analysis(self, mock_savefig, mock_show, mock_analyzer):
        """Test cluster analysis plotting."""
        visualizer = SOMVisualizer(mock_analyzer)

        # Should not raise errors
        visualizer.plot_cluster_analysis(save_path="test_cluster.png")

        # Verify save was called
        mock_savefig.assert_called_once_with("test_cluster.png", dpi=300, bbox_inches='tight')
        mock_show.assert_called_once()

    def test_plot_cluster_analysis_no_clusters(self, mock_analyzer):
        """Test error when no clusters are available."""
        mock_analyzer.cluster_stats = {}

        visualizer = SOMVisualizer(mock_analyzer)

        with pytest.raises(ValueError, match="No cluster data available"):
            visualizer.plot_cluster_analysis()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_tweet_distribution(self, mock_savefig, mock_show, mock_analyzer):
        """Test tweet distribution plotting."""
        visualizer = SOMVisualizer(mock_analyzer)

        # Should not raise errors
        visualizer.plot_tweet_distribution(save_path="test_distribution.png")

        # Verify save was called
        mock_savefig.assert_called_once_with("test_distribution.png", dpi=300, bbox_inches='tight')
        mock_show.assert_called_once()

    def test_plot_tweet_distribution_no_clusters(self, mock_analyzer):
        """Test error when plotting distribution with no clusters."""
        mock_analyzer.cluster_stats = {}

        visualizer = SOMVisualizer(mock_analyzer)

        with pytest.raises(ValueError, match="No cluster data available"):
            visualizer.plot_tweet_distribution()

    @patch('twitter_som.visualizer.PLOTLY_AVAILABLE', True)
    @patch('plotly.graph_objects.Figure.write_html')
    def test_create_interactive_visualization(self, mock_write_html, mock_analyzer):
        """Test interactive visualization creation."""
        visualizer = SOMVisualizer(mock_analyzer)

        # Should not raise errors
        visualizer.create_interactive_visualization("test_interactive.html")

        # Verify HTML was written
        mock_write_html.assert_called_once_with("test_interactive.html")

    @patch('twitter_som.visualizer.PLOTLY_AVAILABLE', False)
    def test_create_interactive_visualization_no_plotly(self, mock_analyzer):
        """Test error when Plotly is not available."""
        visualizer = SOMVisualizer(mock_analyzer)

        with pytest.raises(ImportError, match="Plotly is required"):
            visualizer.create_interactive_visualization()

    def test_create_interactive_visualization_untrained(self):
        """Test error when creating interactive viz with untrained analyzer."""
        untrained_analyzer = Mock(spec=TwitterSOMAnalyzer)
        untrained_analyzer.is_trained = False

        with pytest.warns(UserWarning):
            visualizer = SOMVisualizer(untrained_analyzer)

        with patch('twitter_som.visualizer.PLOTLY_AVAILABLE', True):
            with pytest.raises(ValueError, match="Analyzer must be trained"):
                visualizer.create_interactive_visualization()

    @patch('builtins.open', new_callable=MagicMock)
    def test_generate_report(self, mock_open, mock_analyzer):
        """Test report generation."""
        visualizer = SOMVisualizer(mock_analyzer)

        # Should not raise errors
        visualizer.generate_report("test_report.txt")

        # Verify file was opened for writing
        mock_open.assert_called_once_with("test_report.txt", 'w')

        # Verify content was written
        mock_file = mock_open.return_value.__enter__.return_value
        assert mock_file.write.called

    def test_generate_report_untrained(self):
        """Test error when generating report with untrained analyzer."""
        untrained_analyzer = Mock(spec=TwitterSOMAnalyzer)
        untrained_analyzer.is_trained = False

        with pytest.warns(UserWarning):
            visualizer = SOMVisualizer(untrained_analyzer)

        with pytest.raises(ValueError, match="Analyzer must be trained"):
            visualizer.generate_report()

    def test_plot_distance_map(self, mock_analyzer):
        """Test distance map plotting helper method."""
        visualizer = SOMVisualizer(mock_analyzer)

        fig, ax = plt.subplots()

        # Should not raise errors
        visualizer._plot_distance_map(ax)

        plt.close(fig)

    def test_plot_hit_map(self, mock_analyzer):
        """Test hit map plotting helper method."""
        visualizer = SOMVisualizer(mock_analyzer)

        fig, ax = plt.subplots()

        # Should not raise errors
        visualizer._plot_hit_map(ax)

        plt.close(fig)

    def test_plot_cluster_sizes(self, mock_analyzer):
        """Test cluster sizes plotting helper method."""
        visualizer = SOMVisualizer(mock_analyzer)

        fig, ax = plt.subplots()

        # Should not raise errors
        visualizer._plot_cluster_sizes(ax)

        plt.close(fig)

    def test_plot_feature_map(self, mock_analyzer):
        """Test feature map plotting helper method."""
        visualizer = SOMVisualizer(mock_analyzer)

        fig, ax = plt.subplots()

        # Should not raise errors
        visualizer._plot_feature_map(ax)

        plt.close(fig)

    def test_plot_feature_map_no_features(self, mock_analyzer):
        """Test feature map plotting with no features."""
        mock_analyzer.feature_names = []

        visualizer = SOMVisualizer(mock_analyzer)

        fig, ax = plt.subplots()

        # Should handle gracefully
        visualizer._plot_feature_map(ax)

        plt.close(fig)

    def test_plot_engagement_vs_size(self, mock_analyzer):
        """Test engagement vs size plotting helper method."""
        visualizer = SOMVisualizer(mock_analyzer)

        fig, ax = plt.subplots()

        # Should not raise errors
        visualizer._plot_engagement_vs_size(ax)

        plt.close(fig)

    def test_plot_temporal_distribution(self, mock_analyzer):
        """Test temporal distribution plotting helper method."""
        visualizer = SOMVisualizer(mock_analyzer)

        fig, ax = plt.subplots()

        # Should not raise errors
        visualizer._plot_temporal_distribution(ax)

        plt.close(fig)

    def test_plot_top_hashtags(self, mock_analyzer):
        """Test top hashtags plotting helper method."""
        visualizer = SOMVisualizer(mock_analyzer)

        fig, ax = plt.subplots()

        # Should not raise errors
        visualizer._plot_top_hashtags(ax)

        plt.close(fig)

    def test_plot_top_hashtags_no_data(self, mock_analyzer):
        """Test top hashtags plotting with no hashtag data."""
        # Remove hashtag data
        for cluster_stats in mock_analyzer.cluster_stats.values():
            cluster_stats['dominant_hashtags'] = []

        visualizer = SOMVisualizer(mock_analyzer)

        fig, ax = plt.subplots()

        # Should handle gracefully
        visualizer._plot_top_hashtags(ax)

        plt.close(fig)

    def test_plot_language_distribution(self, mock_analyzer):
        """Test language distribution plotting helper method."""
        visualizer = SOMVisualizer(mock_analyzer)

        fig, ax = plt.subplots()

        # Should not raise errors
        visualizer._plot_language_distribution(ax)

        plt.close(fig)

    def test_plot_language_distribution_no_data(self, mock_analyzer):
        """Test language distribution plotting with no language data."""
        # Remove language data
        for tweet in mock_analyzer.training_collection.tweets:
            tweet.lang = None

        visualizer = SOMVisualizer(mock_analyzer)

        fig, ax = plt.subplots()

        # Should handle gracefully
        visualizer._plot_language_distribution(ax)

        plt.close(fig)

    def test_plot_feature_importance_heatmap(self, mock_analyzer):
        """Test feature importance heatmap plotting helper method."""
        visualizer = SOMVisualizer(mock_analyzer)

        fig, ax = plt.subplots()

        # Should not raise errors
        visualizer._plot_feature_importance_heatmap(ax)

        plt.close(fig)

    def test_plot_feature_importance_heatmap_no_features(self, mock_analyzer):
        """Test feature importance heatmap with no features."""
        mock_analyzer.feature_names = []

        visualizer = SOMVisualizer(mock_analyzer)

        fig, ax = plt.subplots()

        # Should handle gracefully
        visualizer._plot_feature_importance_heatmap(ax)

        plt.close(fig)

    def test_plot_cluster_similarity(self, mock_analyzer):
        """Test cluster similarity matrix plotting helper method."""
        visualizer = SOMVisualizer(mock_analyzer)

        fig, ax = plt.subplots()

        # Should not raise errors
        visualizer._plot_cluster_similarity(ax)

        plt.close(fig)

    def test_plot_cluster_similarity_insufficient_clusters(self, mock_analyzer):
        """Test cluster similarity plotting with insufficient clusters."""
        # Keep only one cluster
        mock_analyzer.cluster_stats = {
            "cluster_1_1": mock_analyzer.cluster_stats["cluster_1_1"]
        }

        visualizer = SOMVisualizer(mock_analyzer)

        fig, ax = plt.subplots()

        # Should handle gracefully
        visualizer._plot_cluster_similarity(ax)

        plt.close(fig)

    @patch('twitter_som.visualizer.PLOTLY_AVAILABLE', True)
    def test_plotly_helper_methods(self, mock_analyzer):
        """Test Plotly helper methods."""
        from plotly.subplots import make_subplots

        visualizer = SOMVisualizer(mock_analyzer)

        # Create mock figure
        fig = make_subplots(rows=2, cols=2)

        # Test each Plotly helper method
        visualizer._add_som_topology_plotly(fig, row=1, col=1)
        visualizer._add_engagement_scatter_plotly(fig, row=1, col=2)
        visualizer._add_temporal_bar_plotly(fig, row=2, col=1)
        visualizer._add_feature_importance_plotly(fig, row=2, col=2)

        # Should not raise errors
        assert fig is not None

    def test_visualization_error_handling(self, mock_analyzer):
        """Test that visualization methods handle errors gracefully."""
        # Corrupt the mock data to trigger potential errors
        mock_analyzer.som._weights = np.array([])  # Empty weights

        visualizer = SOMVisualizer(mock_analyzer)

        fig, ax = plt.subplots()

        # Methods should handle errors gracefully
        try:
            visualizer._plot_distance_map(ax)
            visualizer._plot_hit_map(ax)
            visualizer._plot_feature_map(ax)
        except Exception as e:
            # Should not raise unhandled exceptions
            pytest.fail(f"Visualization method raised unhandled exception: {e}")

        plt.close(fig)
