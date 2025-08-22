"""
Main application for Twitter SOM Analysis.

This module provides example usage and demonstrates the capabilities
of the Twitter SOM analysis package.
"""

import os
import yaml
import mlflow
import random
import argparse
import numpy as np

from typing import List, Optional
from datetime import datetime, timezone, timedelta

from src.twitter_som import (
    TwitterData,
    TwitterDataCollection,
    SOMTrainingConfig,
    TwitterSOMAnalyzer,
    TwitterPreprocessor,
    SOMVisualizer,
    load_twitter_data_from_parquet,
)


import logging

LOGGER = logging.getLogger(__name__)


def main():
    """Main application entry point."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Twitter SOM Analysis with MLflow logging support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run without MLflow logging (uses sample data if main data fails)
  python main.py --experiment "twitter-som-exp"    # Run with MLflow experiment
  python main.py --data-file data/twitter_data.parquet --max-rows 1000   # Load first 1000 tweets
  python main.py --data-file data/twitter_data.parquet --sample-fraction 0.01   # Load 1% sample
  python main.py --experiment "prod-experiment"    # Run with custom experiment name
        """,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="MLflow experiment name for logging metrics and artifacts. If not provided, MLflow logging is disabled.",
    )

    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking server URI (optional). If not provided, uses local file-based tracking.",
    )

    parser.add_argument(
        "--data-file",
        type=str,
        default="data/twitter_data.parquet",
        help="Path to the Twitter data parquet file.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to load from the dataset (for testing/development).",
    )

    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Fraction of data to sample (0.0 to 1.0). Useful for large datasets.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )

    args = parser.parse_args()

    # Set MLflow tracking URI if provided
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        LOGGER.info(f"MLflow tracking URI set to: {args.mlflow_tracking_uri}")

    # Load Twitter Data from parquet file
    LOGGER.info(f"Loading Twitter data from: {args.data_file}")
    try:
        collection = load_twitter_data_from_parquet(
            file_path=args.data_file,
            max_rows=args.max_rows,
            sample_fraction=args.sample_fraction,
            random_state=args.random_state,
        )
        LOGGER.info(
            f"Loaded {len(collection.tweets)} tweets from {args.data_file}"
        )

        # Log some basic statistics
        date_range = collection.get_date_range()
        unique_users = len(collection.get_unique_users())
        LOGGER.info(f"Date range: {date_range[0]} to {date_range[1]}")
        LOGGER.info(f"Unique users: {unique_users}")

    except Exception as e:
        LOGGER.error(f"Failed to load data from {args.data_file}: {e}")
        # Fallback to sample data for demonstration
        LOGGER.info("Using sample data for demonstration")
        tweets = [
            TwitterData(
                id_str="1",
                text="Excited about #MachineLearning and #DataScience!",
                created_at=datetime.now(timezone.utc),
                user_id="user1",
                screen_name="datascientist",
            ),
            # ... more tweets
        ]
        collection = TwitterDataCollection(
            tweets=tweets, collection_name="sample_data"
        )

    # Get Config From Yaml
    # Load configuration from YAML file using the new from_yaml method
    config = SOMTrainingConfig.from_yaml("config/config.yaml")

    # Train SOM (with optional MLflow tracking)
    analyzer = TwitterSOMAnalyzer(
        config, mlflow_experiment_name="twitter-analysis"
    )
    training_stats = analyzer.train(collection)

    # Analyze results
    cluster_summary = analyzer.get_all_clusters_summary()
    LOGGER.info(f"Found {cluster_summary['total_clusters']} clusters")

    # Visualize results
    visualizer = SOMVisualizer(analyzer)
    visualizer.plot_som_topology(save_path="som_topology.png")

    # Log artifacts to MLflow (if enabled)
    analyzer.log_visualizations_to_mlflow({"topology": "som_topology.png"})

    # Save model and end MLflow run
    analyzer.save_model("model.pkl")
    analyzer.end_mlflow_run()
    visualizer.plot_cluster_analysis()

    if args.experiment:
        LOGGER.info(
            f"MLflow experiment '{args.experiment}' contains logged metrics and artifacts."
        )

    # Log data usage information
    LOGGER.info(f"Analysis completed using {len(collection.tweets)} tweets")
    if args.max_rows:
        LOGGER.info(f"Limited to first {args.max_rows} rows from dataset")
    elif args.sample_fraction:
        LOGGER.info(
            f"Used {args.sample_fraction*100:.1f}% sample of the dataset"
        )

    return analyzer, visualizer


if __name__ == "__main__":
    main()
