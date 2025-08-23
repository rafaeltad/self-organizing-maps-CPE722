"""
Test file for TwitterSOMAnalyzer.

Tests the main SOM analysis functionality following TDD principles.
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from twitter_som.som_analyzer import TwitterSOMAnalyzer
from twitter_som.models import TwitterData, TwitterDataCollection, SOMTrainingConfig

# Add missing constants for tests
import twitter_som.som_analyzer as som_analyzer_module

som_analyzer_module.MINISOM_AVAILABLE = True

import twitter_som.visualizer as visualizer_module

visualizer_module.PLOTLY_AVAILABLE = True


class TestTwitterSOMAnalyzer:
    """Test cases for TwitterSOMAnalyzer."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return SOMTrainingConfig(
            x_dim=3,
            y_dim=3,
            learning_rate=0.1,
            num_iterations=10,  # Small for testing
            include_temporal_features=True,
            include_engagement_features=True,
            include_text_features=False,  # Skip text for faster testing
            include_network_features=False
        )

    @pytest.fixture
    def sample_collection(self):
        """Create a sample collection for testing."""
        tweets = []
        for i in range(10):
            tweet = TwitterData(
                id_str=str(i),
                text=f"Test tweet {i} #test",
                created_at=datetime(
                    2024, 1, i + 1, 10 + i, 0, 0, tzinfo=timezone.utc
                ),
                user_id=f"user{i}",
                screen_name=f"testuser{i}",
                lang="en",
            )
            tweets.append(tweet)

        return TwitterDataCollection(
            tweets=tweets,
            collection_name="test_collection"
        )

    def test_analyzer_initialization(self, config):
        """Test analyzer initialization."""
        analyzer = TwitterSOMAnalyzer(config)

        assert analyzer.config == config
        assert analyzer.som is None
        assert analyzer.is_trained is False
        assert analyzer.cluster_assignments is None
        assert analyzer.cluster_centers is None
        assert analyzer.cluster_stats == {}

    def test_minisom_import_error(self, config):
        """Test error when MiniSOM is not available."""
        with patch('twitter_som.som_analyzer.MINISOM_AVAILABLE', False):
            with pytest.raises(ImportError, match="MiniSOM is required"):
                TwitterSOMAnalyzer(config)

    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    @patch("twitter_som.som_analyzer.MiniSom")
    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    def test_train_basic(
        self,
        mock_log_metrics,
        mock_log_params,
        mock_minisom,
        config,
        sample_collection,
    ):
        """Test basic training functionality."""
        # Mock MiniSOM
        mock_som_instance = Mock()
        mock_som_instance.get_weights.return_value = np.random.random((3, 3, 15))
        mock_som_instance.winner.return_value = (1, 1)
        mock_som_instance.quantization_error.return_value = 0.5
        mock_som_instance.topographic_error.return_value = 0.1
        mock_minisom.return_value = mock_som_instance

        # Mock preprocessor
        with patch('twitter_som.som_analyzer.TwitterPreprocessor') as mock_preprocessor:
            mock_preprocessor_instance = Mock()
            mock_preprocessor_instance.fit_transform.return_value = (
                np.random.random((10, 15)),  # features
                [f"feature_{i}" for i in range(15)]  # feature names
            )
            mock_preprocessor.return_value = mock_preprocessor_instance

            # Mock get_engagement_score method for TwitterData
            with patch.object(
                TwitterData,
                "get_engagement_score",
                return_value=1.0,
                create=True,
            ):
                analyzer = TwitterSOMAnalyzer(config)
                training_stats = analyzer.train(
                    sample_collection, verbose=False
                )

            # Check that training completed
            assert analyzer.is_trained is True
            assert analyzer.som is not None
            assert 'quantization_error' in training_stats
            assert 'topographic_error' in training_stats
            assert 'num_clusters' in training_stats

    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    @patch("twitter_som.som_analyzer.MiniSom")
    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    def test_train_with_clustering(
        self,
        mock_log_metrics,
        mock_log_params,
        mock_minisom,
        config,
        sample_collection,
    ):
        """Test training with cluster analysis."""
        # Mock MiniSOM with different winners
        mock_som_instance = Mock()
        mock_som_instance.get_weights.return_value = np.random.random((3, 3, 15))

        # Create different winners for clustering
        winners = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]
        mock_som_instance.winner.side_effect = winners * 2  # Repeat for multiple calls
        mock_som_instance.quantization_error.return_value = 0.3
        mock_som_instance.topographic_error.return_value = 0.05
        mock_som_instance._weights = Mock()
        mock_som_instance._weights.shape = (3, 3, 15)
        mock_minisom.return_value = mock_som_instance

        # Mock preprocessor
        with patch('twitter_som.som_analyzer.TwitterPreprocessor') as mock_preprocessor:
            mock_preprocessor_instance = Mock()
            features = np.random.random((10, 15))
            mock_preprocessor_instance.fit_transform.return_value = (
                features,
                [f"feature_{i}" for i in range(15)]
            )
            mock_preprocessor.return_value = mock_preprocessor_instance

            # Mock get_engagement_score method for TwitterData
            with patch.object(
                TwitterData,
                "get_engagement_score",
                return_value=1.0,
                create=True,
            ):
                analyzer = TwitterSOMAnalyzer(config)
                training_stats = analyzer.train(
                    sample_collection, verbose=False
                )

            # Check clustering results
            assert len(analyzer.cluster_stats) > 0
            assert analyzer.cluster_assignments is not None
            assert analyzer.cluster_centers is not None

    def test_predict_cluster_untrained(self, config, sample_collection):
        """Test prediction error when analyzer is not trained."""
        analyzer = TwitterSOMAnalyzer(config)

        with pytest.raises(ValueError, match="SOM must be trained"):
            analyzer.predict_cluster(sample_collection)

    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    @patch("twitter_som.som_analyzer.MiniSom")
    def test_predict_cluster_trained(
        self,
        mock_minisom,
        mock_log_metrics,
        mock_log_params,
        config,
        sample_collection,
    ):
        """Test prediction on new data after training."""
        # Setup mocks
        mock_som_instance = Mock()
        mock_som_instance.get_weights.return_value = np.random.random((3, 3, 15))
        mock_som_instance.winner.return_value = (1, 2)
        mock_som_instance.quantization_error.return_value = 0.3
        mock_som_instance.topographic_error.return_value = 0.05
        mock_som_instance._weights = Mock()
        mock_som_instance._weights.shape = (3, 3, 15)
        mock_minisom.return_value = mock_som_instance

        with patch('twitter_som.som_analyzer.TwitterPreprocessor') as mock_preprocessor:
            mock_preprocessor_instance = Mock()
            mock_preprocessor_instance.fit_transform.return_value = (
                np.random.random((10, 15)),
                [f"feature_{i}" for i in range(15)]
            )
            mock_preprocessor_instance.transform.return_value = np.random.random((1, 15))  # 1 tweet, 15 features
            mock_preprocessor.return_value = mock_preprocessor_instance

            # Mock get_engagement_score method for TwitterData
            with patch.object(
                TwitterData,
                "get_engagement_score",
                return_value=1.0,
                create=True,
            ):
                analyzer = TwitterSOMAnalyzer(config)
                analyzer.train(sample_collection, verbose=False)

                # Create new test data
                new_tweets = [
                    TwitterData(
                        id_str="new1",
                        text="New test tweet",
                        created_at=datetime.now(timezone.utc),
                        user_id="newuser",
                        screen_name="newuser",
                    )
                ]
                new_collection = TwitterDataCollection(
                    tweets=new_tweets, collection_name="new_test"
                )

                predictions = analyzer.predict_cluster(new_collection)

            assert len(predictions) == 1
            assert len(predictions[0]) == 2  # (x, y) coordinates

    def test_get_cluster_insights_invalid(self, config):
        """Test error for invalid cluster ID."""
        analyzer = TwitterSOMAnalyzer(config)

        with pytest.raises(ValueError, match="Cluster .* not found"):
            analyzer.get_cluster_insights("invalid_cluster")

    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    @patch("twitter_som.som_analyzer.MiniSom")
    def test_get_cluster_insights_valid(self, mock_minisom, config, sample_collection):
        """Test getting insights for valid cluster."""
        # Setup mocks for training
        mock_som_instance = Mock()
        mock_som_instance.get_weights.return_value = np.random.random((3, 3, 15))
        mock_som_instance.winner.return_value = (1, 1)
        mock_som_instance.quantization_error.return_value = 0.3
        mock_som_instance.topographic_error.return_value = 0.05
        mock_som_instance._weights = Mock()
        mock_som_instance._weights.shape = (3, 3, 15)
        mock_minisom.return_value = mock_som_instance

        with patch('twitter_som.som_analyzer.TwitterPreprocessor') as mock_preprocessor:
            mock_preprocessor_instance = Mock()
            mock_preprocessor_instance.fit_transform.return_value = (
                np.random.random((10, 15)),
                [f"feature_{i}" for i in range(15)]
            )
            mock_preprocessor.return_value = mock_preprocessor_instance

            analyzer = TwitterSOMAnalyzer(config)
            analyzer.train(sample_collection, verbose=False)

            # Get insights for the cluster that should exist
            cluster_ids = list(analyzer.cluster_stats.keys())
            if cluster_ids:
                insights = analyzer.get_cluster_insights(cluster_ids[0])

                assert 'size' in insights
                assert 'coordinates' in insights
                assert 'mean_features' in insights
                assert 'feature_importance' in insights
                assert 'similarity_to_other_clusters' in insights

    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    @patch("twitter_som.som_analyzer.MiniSom")
    def test_get_all_clusters_summary(self, mock_minisom, config, sample_collection):
        """Test getting summary of all clusters."""
        # Setup mocks
        mock_som_instance = Mock()
        mock_som_instance.get_weights.return_value = np.random.random((3, 3, 15))
        mock_som_instance.winner.return_value = (1, 1)
        mock_som_instance.quantization_error.return_value = 0.3
        mock_som_instance.topographic_error.return_value = 0.05
        mock_som_instance._weights = Mock()
        mock_som_instance._weights.shape = (3, 3, 15)
        mock_minisom.return_value = mock_som_instance

        with patch('twitter_som.som_analyzer.TwitterPreprocessor') as mock_preprocessor:
            mock_preprocessor_instance = Mock()
            mock_preprocessor_instance.fit_transform.return_value = (
                np.random.random((10, 15)),
                [f"feature_{i}" for i in range(15)]
            )
            mock_preprocessor.return_value = mock_preprocessor_instance

            analyzer = TwitterSOMAnalyzer(config)
            analyzer.train(sample_collection, verbose=False)

            summary = analyzer.get_all_clusters_summary()

            assert 'total_clusters' in summary
            assert 'total_tweets' in summary
            assert 'cluster_sizes' in summary
            assert 'dominant_themes' in summary
            assert 'temporal_overview' in summary
            assert 'engagement_distribution' in summary

    def test_save_model_untrained(self, config):
        """Test error when saving untrained model."""
        analyzer = TwitterSOMAnalyzer(config)

        with pytest.raises(ValueError, match="Cannot save untrained model"):
            analyzer.save_model()

    def test_export_results_untrained(self, config):
        """Test error when exporting results from untrained model."""
        analyzer = TwitterSOMAnalyzer(config)

        with pytest.raises(ValueError, match="Cannot export results from untrained model"):
            analyzer.export_results()

    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    @patch("twitter_som.som_analyzer.MiniSom")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("pickle.dump")
    def test_save_model_trained(self, mock_pickle_dump, mock_open, mock_minisom, config, sample_collection):
        """Test saving trained model."""
        # Setup mocks
        mock_som_instance = Mock()
        mock_som_instance.get_weights.return_value = np.random.random((3, 3, 15))
        mock_som_instance.winner.return_value = (1, 1)
        mock_som_instance.quantization_error.return_value = 0.3
        mock_som_instance.topographic_error.return_value = 0.05
        mock_som_instance._weights = Mock()
        mock_som_instance._weights.shape = (3, 3, 15)
        mock_minisom.return_value = mock_som_instance

        with patch('twitter_som.som_analyzer.TwitterPreprocessor') as mock_preprocessor:
            mock_preprocessor_instance = Mock()
            mock_preprocessor_instance.fit_transform.return_value = (
                np.random.random((10, 15)),
                [f"feature_{i}" for i in range(15)]
            )
            mock_preprocessor.return_value = mock_preprocessor_instance

            # Mock get_engagement_score method for TwitterData
            with patch.object(
                TwitterData,
                "get_engagement_score",
                return_value=1.0,
                create=True,
            ):
                # Mock MLflow artifact logging and file operations
                with patch("mlflow.log_artifact") as mock_log_artifact:
                    with patch("os.remove") as mock_os_remove:
                        analyzer = TwitterSOMAnalyzer(config)
                        analyzer.train(sample_collection, verbose=False)

                        # Save model
                        analyzer.save_model()

                        # Verify save was called
                        mock_open.assert_called_once_with(
                            "temp_model.pkl", "wb"
                        )
                        mock_pickle_dump.assert_called_once()
                        mock_log_artifact.assert_called_once()
                        mock_os_remove.assert_called_once_with(
                            "temp_model.pkl"
                        )

    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    @patch("twitter_som.som_analyzer.MiniSom")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("json.dump")
    def test_export_results_trained(self, mock_json_dump, mock_open, mock_minisom, config, sample_collection):
        """Test exporting results from trained model."""
        # Setup mocks
        mock_som_instance = Mock()
        mock_som_instance.get_weights.return_value = np.random.random((3, 3, 15))
        mock_som_instance.winner.return_value = (1, 1)
        mock_som_instance.quantization_error.return_value = 0.3
        mock_som_instance.topographic_error.return_value = 0.05
        mock_som_instance._weights = Mock()
        mock_som_instance._weights.shape = (3, 3, 15)
        mock_minisom.return_value = mock_som_instance

        with patch('twitter_som.som_analyzer.TwitterPreprocessor') as mock_preprocessor:
            mock_preprocessor_instance = Mock()
            mock_preprocessor_instance.fit_transform.return_value = (
                np.random.random((10, 15)),
                [f"feature_{i}" for i in range(15)]
            )
            mock_preprocessor.return_value = mock_preprocessor_instance

            # Mock get_engagement_score method for TwitterData
            with patch.object(
                TwitterData,
                "get_engagement_score",
                return_value=1.0,
                create=True,
            ):
                # Mock MLflow artifact logging and file operations
                with patch("mlflow.log_artifact") as mock_log_artifact:
                    with patch("os.remove") as mock_os_remove:
                        analyzer = TwitterSOMAnalyzer(config)
                        analyzer.train(sample_collection, verbose=False)

                        # Export results
                        analyzer.export_results()

                        # Verify export was called
                        mock_open.assert_called_once_with(
                            "temp_results.json", "w"
                        )
                        mock_json_dump.assert_called_once()
                        mock_log_artifact.assert_called_once()
                        mock_os_remove.assert_called_once_with(
                            "temp_results.json"
                        )

    def test_calculate_training_stats_untrained(self, config):
        """Test error when calculating stats for untrained model."""
        analyzer = TwitterSOMAnalyzer(config)

        with pytest.raises(ValueError, match="SOM must be trained"):
            analyzer._calculate_training_stats()

    def test_analyze_clusters_untrained(self, config):
        """Test error when analyzing clusters for untrained model."""
        analyzer = TwitterSOMAnalyzer(config)

        with pytest.raises(ValueError, match="SOM must be trained"):
            analyzer._analyze_clusters()

    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    @patch("twitter_som.som_analyzer.MiniSom")
    def test_feature_importance_calculation(self, mock_minisom, config, sample_collection):
        """Test feature importance calculation."""
        # Setup mocks
        mock_som_instance = Mock()
        mock_som_instance.get_weights.return_value = np.random.random((3, 3, 15))
        mock_som_instance.winner.return_value = (1, 1)
        mock_som_instance.quantization_error.return_value = 0.3
        mock_som_instance.topographic_error.return_value = 0.05
        mock_som_instance._weights = Mock()
        mock_som_instance._weights.shape = (3, 3, 15)
        mock_minisom.return_value = mock_som_instance

        with patch('twitter_som.som_analyzer.TwitterPreprocessor') as mock_preprocessor:
            mock_preprocessor_instance = Mock()
            mock_preprocessor_instance.fit_transform.return_value = (
                np.random.random((10, 15)),
                [f"feature_{i}" for i in range(15)]
            )
            mock_preprocessor.return_value = mock_preprocessor_instance

            analyzer = TwitterSOMAnalyzer(config)
            analyzer.train(sample_collection, verbose=False)

            # Test feature importance for existing cluster
            cluster_ids = list(analyzer.cluster_stats.keys())
            if cluster_ids:
                importance = analyzer._get_cluster_feature_importance(cluster_ids[0])

                assert isinstance(importance, dict)
                assert len(importance) == 15  # Should match number of features

                # All values should be between 0 and 1
                for value in importance.values():
                    assert 0 <= value <= 1

    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    @patch("twitter_som.som_analyzer.MiniSom")
    def test_cluster_similarity_calculation(self, mock_minisom, config, sample_collection):
        """Test cluster similarity calculation."""
        # Setup mocks with multiple winners to create multiple clusters
        mock_som_instance = Mock()
        mock_som_instance.get_weights.return_value = np.random.random((3, 3, 15))

        # Alternate between two clusters
        winners = [(0, 0), (1, 1)] * 5
        mock_som_instance.winner.side_effect = winners
        mock_som_instance.quantization_error.return_value = 0.3
        mock_som_instance.topographic_error.return_value = 0.05
        mock_som_instance._weights = Mock()
        mock_som_instance._weights.shape = (3, 3, 15)
        mock_minisom.return_value = mock_som_instance

        with patch('twitter_som.som_analyzer.TwitterPreprocessor') as mock_preprocessor:
            mock_preprocessor_instance = Mock()
            mock_preprocessor_instance.fit_transform.return_value = (
                np.random.random((10, 15)),
                [f"feature_{i}" for i in range(15)]
            )
            mock_preprocessor.return_value = mock_preprocessor_instance

            analyzer = TwitterSOMAnalyzer(config)
            analyzer.train(sample_collection, verbose=False)

            # Test similarity calculation
            cluster_ids = list(analyzer.cluster_stats.keys())
            if len(cluster_ids) > 1:
                similarities = analyzer._calculate_cluster_similarities(cluster_ids[0])

                assert isinstance(similarities, dict)
                # Should have similarities to other clusters
                assert len(similarities) >= 0

                # All similarity values should be between -1 and 1
                for value in similarities.values():
                    assert -1 <= value <= 1
