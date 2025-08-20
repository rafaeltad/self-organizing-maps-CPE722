"""
Main application for Twitter SOM Analysis.

This module provides example usage and demonstrates the capabilities
of the Twitter SOM analysis package.
"""

import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import random
import argparse
import os

from src.twitter_som import (
    TwitterData,
    TwitterDataCollection,
    SOMTrainingConfig,
    TwitterSOMAnalyzer,
    TwitterPreprocessor,
    SOMVisualizer
)


def generate_sample_data(num_tweets: int = 100) -> TwitterDataCollection:
    """
    Generate sample Twitter data for demonstration purposes.

    Args:
        num_tweets: Number of sample tweets to generate

    Returns:
        TwitterDataCollection with sample data
    """
    print(f"Generating {num_tweets} sample tweets...")

    # Sample topics and hashtags
    topics = {
        "technology": ["#AI", "#MachineLearning", "#Python", "#DataScience", "#Tech"],
        "sports": ["#Football", "#Soccer", "#Olympics", "#Sports", "#Fitness"],
        "politics": ["#Politics", "#Election", "#Democracy", "#Vote", "#Government"],
        "entertainment": ["#Movies", "#Music", "#TV", "#Entertainment", "#Celebrity"],
        "health": ["#Health", "#Wellness", "#Fitness", "#Mental", "#Medicine"]
    }

    sample_texts = {
        "technology": [
            "Just implemented a neural network using {hashtag}! Amazing results!",
            "The future of {hashtag} is looking bright. Exciting times ahead!",
            "Working on a new project with {hashtag}. Can't wait to share results!",
            "Anyone else excited about the latest developments in {hashtag}?",
            "Learning {hashtag} has been such a rewarding journey!"
        ],
        "sports": [
            "What a game! {hashtag} at its finest!",
            "Training hard for the next competition {hashtag}",
            "The teamwork in today's match was incredible {hashtag}",
            "Nothing beats the excitement of {hashtag}!",
            "Inspired by the athletes in {hashtag}"
        ],
        "politics": [
            "Important discussion about {hashtag} happening now",
            "Everyone should participate in {hashtag}",
            "The impact of {hashtag} on our society is significant",
            "Critical thinking about {hashtag} is essential",
            "Engaging with {hashtag} responsibly"
        ],
        "entertainment": [
            "Just watched an amazing show! {hashtag}",
            "The creativity in {hashtag} never ceases to amaze me",
            "Looking forward to the new releases in {hashtag}",
            "The artistry in {hashtag} is incredible",
            "Can't get enough of {hashtag}"
        ],
        "health": [
            "Taking care of my {hashtag} today",
            "Important reminder about {hashtag}",
            "The science behind {hashtag} is fascinating",
            "Making positive changes for my {hashtag}",
            "Research shows the importance of {hashtag}"
        ]
    }

    usernames = [f"user{i:03d}" for i in range(1, 51)]  # 50 different users
    languages = ["en", "es", "fr", "de", "pt"]
    locations = ["New York", "London", "Paris", "Berlin", "Madrid", "Rome", "Tokyo"]

    tweets = []
    base_date = datetime.now(timezone.utc) - timedelta(days=30)

    for i in range(num_tweets):
        # Choose random topic
        topic = random.choice(list(topics.keys()))
        hashtag = random.choice(topics[topic])
        text_template = random.choice(sample_texts[topic])

        # Generate tweet text
        text = text_template.format(hashtag=hashtag)

        # Add random mentions occasionally
        if random.random() < 0.3:
            mentioned_user = random.choice(usernames)
            text += f" @{mentioned_user}"

        # Add random additional hashtags occasionally
        if random.random() < 0.4:
            additional_hashtag = random.choice([tag for sublist in topics.values() for tag in sublist])
            if additional_hashtag not in text:
                text += f" {additional_hashtag}"

        # Generate engagement with some correlation to topic popularity
        topic_popularity = {
            "technology": 2.0,
            "sports": 1.8,
            "entertainment": 1.5,
            "politics": 1.2,
            "health": 1.0
        }

        base_engagement = topic_popularity[topic] * random.uniform(0.5, 2.0)

        tweet = TwitterData(
            id=str(i + 1),
            text=text,
            created_at=base_date
            + timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            ),
            user_id=f"uid_{random.choice(usernames)}",
            username=random.choice(usernames),
            like_count=max(
                0, int(np.random.exponential(10 * base_engagement))
            ),
            retweet_count=max(
                0, int(np.random.exponential(3 * base_engagement))
            ),
            reply_count=max(
                0, int(np.random.exponential(2 * base_engagement))
            ),
            quote_count=max(
                0, int(np.random.exponential(1 * base_engagement))
            ),
            lang=random.choice(languages),
            location=(
                random.choice(locations) if random.random() < 0.7 else None
            ),
            is_retweet=random.random() < 0.2,
            is_reply=random.random() < 0.15,
            is_quote=random.random() < 0.1,
        )

        tweets.append(tweet)

    return TwitterDataCollection(
        tweets=tweets,
        collection_name="sample_twitter_data",
        description="Generated sample data for SOM analysis demonstration",
        source_query="sample_data_generation"
    )


def demonstrate_som_analysis(experiment_name: Optional[str] = None):
    """Demonstrate the Twitter SOM analysis workflow with optional MLflow logging."""
    print("=" * 60)
    print("TWITTER SOM ANALYSIS DEMONSTRATION")
    print("=" * 60)

    if experiment_name:
        print(f"MLflow experiment: {experiment_name}")
        print("-" * 60)

    # 1. Generate sample data
    print("\n1. GENERATING SAMPLE DATA")
    print("-" * 30)
    collection = generate_sample_data(200)

    print(f"Generated {len(collection.tweets)} tweets")
    print(f"Date range: {collection.get_date_range()}")
    print(f"Unique users: {len(collection.get_unique_users())}")
    print(f"Top hashtags: {collection.get_most_common_hashtags(5)}")

    # 2. Configure and train SOM
    print("\n2. CONFIGURING AND TRAINING SOM")
    print("-" * 35)

    config = SOMTrainingConfig(
        x_dim=8,
        y_dim=8,
        learning_rate=0.1,
        num_iterations=1000,
        normalize_features=True,
        use_pca=False,
        include_temporal_features=True,
        include_engagement_features=True,
        include_text_features=True,
        include_network_features=True
    )

    print(f"SOM Configuration:")
    print(f"  Dimensions: {config.x_dim}x{config.y_dim}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Feature normalization: {config.normalize_features}")

    # Initialize and train analyzer with MLflow support
    analyzer = TwitterSOMAnalyzer(
        config, mlflow_experiment_name=experiment_name
    )
    training_stats = analyzer.train(collection, verbose=True)

    print(f"\nTraining completed!")
    print(f"  Quantization error: {training_stats['quantization_error']:.4f}")
    print(f"  Topographic error: {training_stats['topographic_error']:.4f}")
    print(f"  Number of clusters: {training_stats['num_clusters']}")

    # 3. Analyze results
    print("\n3. ANALYZING RESULTS")
    print("-" * 22)

    cluster_summary = analyzer.get_all_clusters_summary()
    print(f"Total clusters found: {cluster_summary['total_clusters']}")
    print(f"Largest cluster size: {training_stats['largest_cluster_size']}")
    print(f"Smallest cluster size: {training_stats['smallest_cluster_size']}")

    # Show top clusters
    print(f"\nTop 5 clusters by size:")
    sorted_clusters = sorted(
        cluster_summary['cluster_sizes'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    for cluster_id, size in sorted_clusters:
        cluster_insights = analyzer.get_cluster_insights(cluster_id)
        print(f"  {cluster_id}: {size} tweets")
        print(f"    Avg engagement: {cluster_insights['avg_engagement']:.2f}")
        print(f"    Top hashtags: {cluster_insights['dominant_hashtags'][:3]}")
        print(f"    Peak hour: {cluster_insights['time_distribution']['peak_hour']}")
        print(f"    Sample: {cluster_insights['sample_tweets'][0][:60]}...")
        print()

    # 4. Generate visualizations
    print("\n4. GENERATING VISUALIZATIONS")
    print("-" * 32)

    try:
        visualizer = SOMVisualizer(analyzer)
        
        # Use temporary files if MLflow is enabled, otherwise save permanently
        use_temp_files = experiment_name is not None
        
        if use_temp_files:
            # Create temporary files for MLflow logging
            import tempfile
            temp_dir = tempfile.mkdtemp()
            topology_path = f"{temp_dir}/som_topology.png"
            cluster_path = f"{temp_dir}/cluster_analysis.png"
            distribution_path = f"{temp_dir}/tweet_distribution.png"
            report_path = f"{temp_dir}/som_analysis_report.txt"
            interactive_path = f"{temp_dir}/som_interactive.html"
        else:
            # Save files permanently when MLflow is not used
            topology_path = "som_topology.png"
            cluster_path = "cluster_analysis.png"
            distribution_path = "tweet_distribution.png"
            report_path = "som_analysis_report.txt"
            interactive_path = "som_interactive.html"

        print("Creating SOM topology plot...")
        visualizer.plot_som_topology(save_path=topology_path)

        print("Creating cluster analysis plot...")
        visualizer.plot_cluster_analysis(save_path=cluster_path)

        print("Creating tweet distribution plot...")
        visualizer.plot_tweet_distribution(save_path=distribution_path)

        print("Generating analysis report...")
        visualizer.generate_report(report_path)

        # Try interactive visualization if Plotly is available
        try:
            print("Creating interactive visualization...")
            visualizer.create_interactive_visualization(interactive_path)
        except ImportError:
            print("Plotly not available - skipping interactive visualization")

        if use_temp_files:
            print("Visualizations created and will be logged to MLflow only!")
        else:
            print("Visualizations saved successfully!")

        # Log visualizations to MLflow if enabled
        if experiment_name:
            visualization_paths = {
                "som_topology": topology_path,
                "cluster_analysis": cluster_path,
                "tweet_distribution": distribution_path,
                "analysis_report": report_path,
                "interactive_viz": interactive_path,
            }
            analyzer.log_visualizations_to_mlflow(visualization_paths)
            
            # Clean up temporary directory
            if use_temp_files and 'temp_dir' in locals():
                import shutil
                shutil.rmtree(temp_dir)
                print("Temporary files cleaned up.")

    except Exception as e:
        print(f"Visualization error: {e}")
        print("Continuing without visualizations...")

    # 5. Test prediction on new data
    print("\n5. TESTING PREDICTION ON NEW DATA")
    print("-" * 38)

    # Generate a small test set
    test_collection = generate_sample_data(20)
    predictions = analyzer.predict_cluster(test_collection)

    print(f"Predicted clusters for {len(test_collection.tweets)} new tweets:")
    for i, (tweet, prediction) in enumerate(zip(test_collection.tweets[:5], predictions[:5])):
        print(f"  Tweet {i+1}: {tweet.text[:50]}...")
        print(f"    Predicted cluster: ({prediction[0]}, {prediction[1]})")

    # 6. Save and export results
    print("\n6. SAVING RESULTS")
    print("-" * 18)

    try:
        if experiment_name:
            # Only log to MLflow, don't save files locally
            print("Saving trained model to MLflow...")
            analyzer.save_model()  # No filepath = temporary file for MLflow only

            print("Exporting results to MLflow...")
            analyzer.export_results()  # No filepath = temporary file for MLflow only
            
            print("Results logged to MLflow successfully!")
        else:
            # Save files locally when MLflow is not used
            print("Saving trained model...")
            analyzer.save_model("twitter_som_model.pkl")

            print("Exporting results to JSON...")
            analyzer.export_results("som_results.json")
            
            print("Results saved successfully!")

        # Log additional metrics to MLflow if enabled
        if experiment_name:
            cluster_summary = analyzer.get_all_clusters_summary()
            cluster_sizes = (
                [stats["size"] for stats in analyzer.cluster_stats.values()]
                if analyzer.cluster_stats
                else [0]
            )
            additional_metrics = {
                "avg_cluster_size": np.mean(cluster_sizes),
                "cluster_size_variance": np.var(cluster_sizes),
                "unique_users": len(collection.get_unique_users()),
                "total_tweets": len(collection.tweets),
                "avg_engagement_per_tweet": np.mean(
                    [tweet.get_engagement_score() for tweet in collection.tweets]
                ),
            }
            analyzer.log_additional_metrics(additional_metrics)

    except Exception as e:
        print(f"Error saving results: {e}")

    # End MLflow run if it was started
    if experiment_name:
        analyzer.end_mlflow_run()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED!")
    print("=" * 60)

    return analyzer, visualizer if "visualizer" in locals() else None


def demonstrate_use_cases():
    """Demonstrate specific use cases for Twitter SOM analysis."""
    print("\n" + "=" * 60)
    print("TWITTER SOM ANALYSIS USE CASES")
    print("=" * 60)

    use_cases = [
        {
            "title": "1. Content Clustering and Topic Discovery",
            "description": """
Use SOM to automatically discover topics and themes in Twitter data:
- Cluster tweets by content similarity
- Identify emerging topics and trends
- Understand content patterns across different user groups
- Detect viral content characteristics
            """
        },
        {
            "title": "2. User Behavior Analysis",
            "description": """
Analyze user behavior patterns using temporal and engagement features:
- Identify user archetypes based on posting patterns
- Understand engagement strategies and their effectiveness
- Cluster users by activity levels and interaction styles
- Detect bot vs. human user patterns
            """
        },
        {
            "title": "3. Sentiment and Emotion Mapping",
            "description": """
Map emotional landscapes and sentiment patterns:
- Cluster tweets by sentiment and emotional content
- Visualize how emotions spread across topics
- Identify sentiment clusters and their characteristics
- Track emotional responses to events or campaigns
            """
        },
        {
            "title": "4. Campaign and Marketing Analysis",
            "description": """
Analyze marketing campaigns and brand mentions:
- Cluster brand-related tweets to understand perception
- Identify effective hashtag combinations
- Analyze campaign reach and engagement patterns
- Understand audience segments and their preferences
            """
        },
        {
            "title": "5. Crisis and Event Detection",
            "description": """
Detect and analyze crisis situations or major events:
- Identify unusual activity clusters that may indicate events
- Analyze information spread patterns during crises
- Cluster response types to emergency situations
- Track how information evolves during breaking news
            """
        },
        {
            "title": "6. Network Analysis and Influence Mapping",
            "description": """
Understand social network structures and influence patterns:
- Cluster users by their network connections and interactions
- Identify influential users and opinion leaders
- Analyze information flow patterns
- Detect community structures and echo chambers
            """
        },
        {
            "title": "7. Language and Cultural Analysis",
            "description": """
Analyze linguistic and cultural patterns:
- Cluster tweets by language usage and cultural markers
- Understand regional differences in communication styles
- Analyze multilingual communities and code-switching
- Study cultural trends and their digital expressions
            """
        },
        {
            "title": "8. Temporal Pattern Analysis",
            "description": """
Understand time-based patterns in social media activity:
- Cluster tweets by temporal patterns (time of day, day of week)
- Identify seasonal trends and cyclical behaviors
- Analyze event-driven temporal anomalies
- Understand global vs. local time-based patterns
            """
        }
    ]

    for use_case in use_cases:
        print(f"\n{use_case['title']}")
        print("-" * len(use_case['title']))
        print(use_case['description'])

    print(f"\n{'IMPLEMENTATION BENEFITS'}")
    print("-" * 25)
    print("""
Benefits of using SOM for Twitter analysis:

✓ Unsupervised Learning: No need for labeled data
✓ Visualization: 2D map representation of high-dimensional data
✓ Pattern Discovery: Automatic identification of hidden patterns
✓ Scalability: Can handle large datasets efficiently
✓ Interpretability: Results are easy to understand and explain
✓ Flexibility: Can incorporate multiple feature types
✓ Real-time Analysis: Can be used for streaming data analysis
✓ Anomaly Detection: Identifies unusual patterns and outliers
    """)


def main():
    """Main application entry point."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Twitter SOM Analysis with MLflow logging support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run without MLflow logging
  python main.py --experiment "twitter-som-exp"    # Run with MLflow experiment
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

    args = parser.parse_args()

    # Set MLflow tracking URI if provided
    if args.mlflow_tracking_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
            print(f"MLflow tracking URI set to: {args.mlflow_tracking_uri}")
        except ImportError:
            print(
                "Warning: MLflow not available, tracking URI setting ignored"
            )

    try:
        # Run main demonstration with optional MLflow experiment
        analyzer, visualizer = demonstrate_som_analysis(
            experiment_name=args.experiment
        )

        # Show use cases
        demonstrate_use_cases()

        print(f"\n{'SUCCESS!'}")
        print("The Twitter SOM analysis demonstration completed successfully.")
        print("Check the generated files for detailed results and visualizations.")

        if args.experiment:
            print(
                f"MLflow experiment '{args.experiment}' contains logged metrics and artifacts."
            )

        return analyzer, visualizer

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Please check your environment and dependencies.")
        return None, None


if __name__ == "__main__":
    main()
