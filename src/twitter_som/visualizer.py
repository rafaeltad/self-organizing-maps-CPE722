"""
Visualization tools for SOM analysis results.

This module provides various visualization capabilities for analyzing
Self-Organizing Map results on Twitter data.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend to prevent figure windows
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .som_analyzer import TwitterSOMAnalyzer
from .models import TwitterDataCollection


class SOMVisualizer:
    """
    Visualization tools for SOM analysis results.

    Provides methods to create various plots and visualizations
    for understanding SOM clustering results and patterns.
    """

    def __init__(self, analyzer: TwitterSOMAnalyzer):
        """
        Initialize the visualizer.

        Args:
            analyzer: Trained TwitterSOMAnalyzer instance
        """
        self.analyzer = analyzer
        if not analyzer.is_trained:
            warnings.warn("Analyzer is not trained. Some visualizations may not work.")

    def plot_som_topology(self, figsize: Tuple[int, int] = (12, 10), save_path: Optional[str] = None) -> None:
        """
        Plot the SOM topology with cluster assignments.

        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        if not self.analyzer.is_trained:
            raise ValueError("Analyzer must be trained before plotting")

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Distance map (U-matrix)
        ax1 = axes[0, 0]
        self._plot_distance_map(ax1)
        ax1.set_title('Distance Map (U-Matrix)')

        # Plot 2: Hit map (frequency of winning neurons)
        ax2 = axes[0, 1]
        self._plot_hit_map(ax2)
        ax2.set_title('Hit Map (Neuron Activation Frequency)')

        # Plot 3: Cluster size distribution
        ax3 = axes[1, 0]
        self._plot_cluster_sizes(ax3)
        ax3.set_title('Cluster Size Distribution')

        # Plot 4: Weight map for most important feature
        ax4 = axes[1, 1]
        self._plot_feature_map(ax4)
        ax4.set_title('Feature Weight Map')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()  # Close figure to free memory

    def _plot_distance_map(self, ax: plt.Axes) -> None:
        """Plot the distance map (U-matrix)."""
        som = self.analyzer.som

        # Calculate distance map
        distance_map = np.zeros((som._weights.shape[0], som._weights.shape[1]))

        for i in range(som._weights.shape[0]):
            for j in range(som._weights.shape[1]):
                neighbors = []

                # Get neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < som._weights.shape[0] and
                            0 <= nj < som._weights.shape[1] and
                            (di != 0 or dj != 0)):
                            neighbors.append(som._weights[ni, nj])

                if neighbors:
                    distances = [np.linalg.norm(som._weights[i, j] - neighbor)
                               for neighbor in neighbors]
                    distance_map[i, j] = np.mean(distances)

        im = ax.imshow(distance_map, cmap='viridis', interpolation='nearest')
        ax.set_xlabel('SOM X')
        ax.set_ylabel('SOM Y')
        plt.colorbar(im, ax=ax, label='Average Distance')

    def _plot_hit_map(self, ax: plt.Axes) -> None:
        """Plot the hit map showing neuron activation frequency."""
        som = self.analyzer.som
        training_data = self.analyzer.training_data

        # Calculate hit frequencies
        hit_map = np.zeros((som._weights.shape[0], som._weights.shape[1]))

        for data_point in training_data:
            winner = som.winner(data_point)
            hit_map[winner] += 1

        im = ax.imshow(hit_map, cmap='Blues', interpolation='nearest')
        ax.set_xlabel('SOM X')
        ax.set_ylabel('SOM Y')
        plt.colorbar(im, ax=ax, label='Hit Count')

        # Add text annotations for non-zero cells
        for i in range(hit_map.shape[0]):
            for j in range(hit_map.shape[1]):
                if hit_map[i, j] > 0:
                    ax.text(j, i, f'{int(hit_map[i, j])}',
                           ha='center', va='center', fontsize=8)

    def _plot_cluster_sizes(self, ax: plt.Axes) -> None:
        """Plot cluster size distribution."""
        cluster_sizes = [stats['size'] for stats in self.analyzer.cluster_stats.values()]

        ax.hist(cluster_sizes, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Cluster Size')
        ax.set_ylabel('Frequency')
        ax.axvline(np.mean(cluster_sizes), color='red', linestyle='--',
                  label=f'Mean: {np.mean(cluster_sizes):.1f}')
        ax.legend()

    def _plot_feature_map(self, ax: plt.Axes) -> None:
        """Plot feature weight map for the most important feature."""
        if not self.analyzer.feature_names:
            ax.text(0.5, 0.5, 'No feature data available',
                   ha='center', va='center', transform=ax.transAxes)
            return

        som = self.analyzer.som

        # For demonstration, plot the first feature
        feature_idx = 0
        feature_map = som._weights[:, :, feature_idx]

        im = ax.imshow(feature_map, cmap='RdYlBu', interpolation='nearest')
        ax.set_xlabel('SOM X')
        ax.set_ylabel('SOM Y')
        plt.colorbar(im, ax=ax, label=f'Weight ({self.analyzer.feature_names[feature_idx]})')

    def plot_cluster_analysis(self, figsize: Tuple[int, int] = (15, 10), save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive cluster analysis.

        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        if not self.analyzer.cluster_stats:
            raise ValueError("No cluster data available")

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Plot 1: Engagement vs cluster size
        ax1 = axes[0, 0]
        self._plot_engagement_vs_size(ax1)
        ax1.set_title('Engagement vs Cluster Size')

        # Plot 2: Temporal distribution
        ax2 = axes[0, 1]
        self._plot_temporal_distribution(ax2)
        ax2.set_title('Temporal Patterns')

        # Plot 3: Top hashtags
        ax3 = axes[0, 2]
        self._plot_top_hashtags(ax3)
        ax3.set_title('Top Hashtags by Cluster')

        # Plot 4: Language distribution
        ax4 = axes[1, 0]
        self._plot_language_distribution(ax4)
        ax4.set_title('Language Distribution')

        # Plot 5: Feature importance heatmap
        ax5 = axes[1, 1]
        self._plot_feature_importance_heatmap(ax5)
        ax5.set_title('Feature Importance by Cluster')

        # Plot 6: Cluster similarity matrix
        ax6 = axes[1, 2]
        self._plot_cluster_similarity(ax6)
        ax6.set_title('Cluster Similarity Matrix')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()  # Close figure to free memory

    def _plot_engagement_vs_size(self, ax: plt.Axes) -> None:
        """Plot engagement vs cluster size."""
        sizes = []
        engagements = []

        for stats in self.analyzer.cluster_stats.values():
            sizes.append(stats['size'])
            engagements.append(stats['avg_engagement'])

        ax.scatter(sizes, engagements, alpha=0.7)
        ax.set_xlabel('Cluster Size')
        ax.set_ylabel('Average Engagement')

        # Add trend line
        if len(sizes) > 1:
            z = np.polyfit(sizes, engagements, 1)
            p = np.poly1d(z)
            ax.plot(sizes, p(sizes), "r--", alpha=0.8)

    def _plot_temporal_distribution(self, ax: plt.Axes) -> None:
        """Plot temporal distribution across clusters."""
        # Aggregate temporal data
        all_hours = []
        for tweet in self.analyzer.training_collection.tweets:
            all_hours.append(tweet.created_at.hour)

        ax.hist(all_hours, bins=24, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Tweet Count')
        ax.set_xticks(range(0, 24, 4))

    def _plot_top_hashtags(self, ax: plt.Axes) -> None:
        """Plot top hashtags across all clusters."""
        hashtag_counts = {}

        for stats in self.analyzer.cluster_stats.values():
            for hashtag, count in stats['dominant_hashtags']:
                hashtag_counts[hashtag] = hashtag_counts.get(hashtag, 0) + count

        # Get top 10 hashtags
        top_hashtags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        if top_hashtags:
            hashtags, counts = zip(*top_hashtags)
            y_pos = np.arange(len(hashtags))

            ax.barh(y_pos, counts)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(hashtags)
            ax.set_xlabel('Frequency')
        else:
            ax.text(0.5, 0.5, 'No hashtag data', ha='center', va='center', transform=ax.transAxes)

    def _plot_language_distribution(self, ax: plt.Axes) -> None:
        """Plot language distribution."""
        lang_counts = {}

        for tweet in self.analyzer.training_collection.tweets:
            if tweet.lang:
                lang_counts[tweet.lang] = lang_counts.get(tweet.lang, 0) + 1

        if lang_counts:
            languages = list(lang_counts.keys())
            counts = list(lang_counts.values())

            ax.pie(counts, labels=languages, autopct='%1.1f%%')
        else:
            ax.text(0.5, 0.5, 'No language data', ha='center', va='center', transform=ax.transAxes)

    def _plot_feature_importance_heatmap(self, ax: plt.Axes) -> None:
        """Plot feature importance heatmap."""
        if not self.analyzer.feature_names:
            ax.text(0.5, 0.5, 'No feature data', ha='center', va='center', transform=ax.transAxes)
            return

        # Get top 10 features for visualization
        feature_importance_matrix = []
        cluster_names = []

        for cluster_id in list(self.analyzer.cluster_stats.keys())[:10]:  # Limit to 10 clusters
            importance = self.analyzer._get_cluster_feature_importance(cluster_id)
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

            cluster_names.append(cluster_id.replace('cluster_', ''))
            feature_importance_matrix.append([imp for _, imp in top_features])

        if feature_importance_matrix:
            feature_names = [name for name, _ in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]]

            sns.heatmap(feature_importance_matrix,
                       xticklabels=feature_names,
                       yticklabels=cluster_names,
                       cmap='viridis',
                       ax=ax)
            ax.set_xlabel('Features')
            ax.set_ylabel('Clusters')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    def _plot_cluster_similarity(self, ax: plt.Axes) -> None:
        """Plot cluster similarity matrix."""
        cluster_ids = list(self.analyzer.cluster_stats.keys())

        if len(cluster_ids) < 2:
            ax.text(0.5, 0.5, 'Not enough clusters', ha='center', va='center', transform=ax.transAxes)
            return

        # Build similarity matrix
        similarity_matrix = np.zeros((len(cluster_ids), len(cluster_ids)))

        for i, cluster_id in enumerate(cluster_ids):
            similarities = self.analyzer._calculate_cluster_similarities(cluster_id)
            for j, other_id in enumerate(cluster_ids):
                if other_id in similarities:
                    similarity_matrix[i, j] = similarities[other_id]
                elif i == j:
                    similarity_matrix[i, j] = 1.0

        # Limit to top 10 clusters for readability
        if len(cluster_ids) > 10:
            cluster_ids = cluster_ids[:10]
            similarity_matrix = similarity_matrix[:10, :10]

        cluster_labels = [cid.replace('cluster_', '') for cid in cluster_ids]

        sns.heatmap(similarity_matrix,
                   xticklabels=cluster_labels,
                   yticklabels=cluster_labels,
                   cmap='coolwarm',
                   center=0,
                   ax=ax)
        ax.set_xlabel('Clusters')
        ax.set_ylabel('Clusters')

    def create_interactive_visualization(self, output_file: str = 'som_analysis.html') -> None:
        """
        Create interactive Plotly visualization.

        Args:
            output_file: Output HTML file path
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive visualizations")

        if not self.analyzer.is_trained:
            raise ValueError("Analyzer must be trained before creating visualizations")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('SOM Topology', 'Cluster Engagement',
                          'Temporal Patterns', 'Feature Importance'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "heatmap"}]]
        )

        # Plot 1: SOM topology
        self._add_som_topology_plotly(fig, row=1, col=1)

        # Plot 2: Cluster engagement scatter
        self._add_engagement_scatter_plotly(fig, row=1, col=2)

        # Plot 3: Temporal patterns
        self._add_temporal_bar_plotly(fig, row=2, col=1)

        # Plot 4: Feature importance
        self._add_feature_importance_plotly(fig, row=2, col=2)

        fig.update_layout(
            title_text="Twitter SOM Analysis Dashboard",
            showlegend=False,
            height=800
        )

        fig.write_html(output_file)
        print(f"Interactive visualization saved to {output_file}")

    def _add_som_topology_plotly(self, fig: go.Figure, row: int, col: int) -> None:
        """Add SOM topology to Plotly figure."""
        som = self.analyzer.som
        hit_map = np.zeros((som._weights.shape[0], som._weights.shape[1]))

        for data_point in self.analyzer.training_data:
            winner = som.winner(data_point)
            hit_map[winner] += 1

        fig.add_trace(
            go.Heatmap(z=hit_map, colorscale='Blues', showscale=False),
            row=row, col=col
        )

    def _add_engagement_scatter_plotly(self, fig: go.Figure, row: int, col: int) -> None:
        """Add engagement scatter plot to Plotly figure."""
        sizes = []
        engagements = []
        cluster_names = []

        for cluster_id, stats in self.analyzer.cluster_stats.items():
            sizes.append(stats['size'])
            engagements.append(stats['avg_engagement'])
            cluster_names.append(cluster_id)

        fig.add_trace(
            go.Scatter(
                x=sizes, y=engagements,
                mode='markers',
                text=cluster_names,
                hovertemplate='Cluster: %{text}<br>Size: %{x}<br>Engagement: %{y:.2f}',
                showlegend=False
            ),
            row=row, col=col
        )

    def _add_temporal_bar_plotly(self, fig: go.Figure, row: int, col: int) -> None:
        """Add temporal patterns bar chart to Plotly figure."""
        hour_counts = {}

        for tweet in self.analyzer.training_collection.tweets:
            hour = tweet.created_at.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        hours = list(range(24))
        counts = [hour_counts.get(hour, 0) for hour in hours]

        fig.add_trace(
            go.Bar(x=hours, y=counts, showlegend=False),
            row=row, col=col
        )

    def _add_feature_importance_plotly(self, fig: go.Figure, row: int, col: int) -> None:
        """Add feature importance heatmap to Plotly figure."""
        if not self.analyzer.feature_names:
            return

        # Simplified version for demonstration
        feature_matrix = []
        cluster_names = []

        for cluster_id in list(self.analyzer.cluster_stats.keys())[:5]:
            importance = self.analyzer._get_cluster_feature_importance(cluster_id)
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

            cluster_names.append(cluster_id.replace('cluster_', ''))
            feature_matrix.append([imp for _, imp in top_features])

        if feature_matrix:
            fig.add_trace(
                go.Heatmap(
                    z=feature_matrix,
                    colorscale='Viridis',
                    showscale=False
                ),
                row=row, col=col
            )

    def plot_tweet_distribution(self, figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None) -> None:
        """
        Plot tweet distribution across clusters with sample tweets.

        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        if not self.analyzer.cluster_stats:
            raise ValueError("No cluster data available")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Cluster size distribution
        cluster_ids = list(self.analyzer.cluster_stats.keys())
        sizes = [self.analyzer.cluster_stats[cid]['size'] for cid in cluster_ids]

        ax1.bar(range(len(cluster_ids)), sizes)
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Number of Tweets')
        ax1.set_title('Tweet Distribution Across Clusters')
        ax1.set_xticks(range(len(cluster_ids)))
        ax1.set_xticklabels([cid.replace('cluster_', '') for cid in cluster_ids], rotation=45)

        # Plot 2: Sample tweets from largest clusters
        # Get top 5 clusters by size
        top_clusters = sorted(self.analyzer.cluster_stats.items(),
                            key=lambda x: x[1]['size'], reverse=True)[:5]

        ax2.axis('off')
        y_pos = 0.9

        for cluster_id, stats in top_clusters:
            ax2.text(0.02, y_pos, f"{cluster_id}:", fontweight='bold', transform=ax2.transAxes)
            y_pos -= 0.05

            for i, tweet_text in enumerate(stats['sample_tweets'][:2]):  # Show 2 sample tweets
                # Truncate long tweets
                if len(tweet_text) > 60:
                    tweet_text = tweet_text[:60] + "..."
                ax2.text(0.05, y_pos, f"• {tweet_text}", fontsize=8, transform=ax2.transAxes)
                y_pos -= 0.04

            y_pos -= 0.05

        ax2.set_title('Sample Tweets from Top Clusters')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()  # Close figure to free memory

    def generate_report(self, output_file: str = 'som_analysis_report.txt') -> None:
        """
        Generate a text report of the SOM analysis.

        Args:
            output_file: Output text file path
        """
        if not self.analyzer.is_trained:
            raise ValueError("Analyzer must be trained before generating report")

        with open(output_file, 'w') as f:
            f.write("TWITTER SOM ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Training statistics
            stats = self.analyzer._calculate_training_stats()
            f.write("TRAINING STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Number of tweets: {stats['num_samples']}\n")
            f.write(f"Number of features: {stats['num_features']}\n")
            f.write(f"SOM dimensions: {stats['som_dimensions']}\n")
            f.write(f"Number of clusters: {stats['num_clusters']}\n")
            f.write(f"Quantization error: {stats['quantization_error']:.4f}\n")
            f.write(f"Topographic error: {stats['topographic_error']:.4f}\n\n")

            # Cluster analysis
            f.write("CLUSTER ANALYSIS\n")
            f.write("-" * 16 + "\n")

            for cluster_id, cluster_stats in self.analyzer.cluster_stats.items():
                f.write(f"\n{cluster_id.upper()}\n")
                f.write(f"Size: {cluster_stats['size']} tweets\n")
                f.write(f"Average engagement: {cluster_stats['avg_engagement']:.2f}\n")
                f.write(f"Dominant hashtags: {cluster_stats['dominant_hashtags'][:3]}\n")
                f.write(f"Peak hour: {cluster_stats['time_distribution']['peak_hour']}\n")
                f.write(f"Sample tweets:\n")
                for tweet in cluster_stats['sample_tweets'][:3]:
                    f.write(f"  • {tweet[:80]}...\n")

            # Global insights
            summary = self.analyzer.get_all_clusters_summary()
            f.write(f"\nGLOBAL INSIGHTS\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total clusters: {summary['total_clusters']}\n")
            f.write(f"Total tweets: {summary['total_tweets']}\n")
            f.write(f"Top global themes: {summary['dominant_themes'][:5]}\n")
            f.write(f"Peak hours: {summary['temporal_overview']['peak_hours'][:3]}\n")

        print(f"Analysis report saved to {output_file}")
