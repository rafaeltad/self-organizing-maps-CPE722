# Twitter SOM Analysis - CPE722

A comprehensive Python package for analyzing Twitter data using Self-Organizing Maps (SOM) to discover patterns, cluster similar tweets, and visualize data relationships.

## 🎯 Project Overview

This project implements a complete pipeline for Twitter data analysis using Self-Organizing Maps, following Test-Driven Development (TDD) principles. It provides tools for preprocessing Twitter data, training SOM models, analyzing clusters, and generating visualizations.

## 🚀 Features

### Core Functionality
- **Robust Data Models**: Pydantic-based models for Twitter data with validation
- **Advanced Preprocessing**: Text cleaning, feature extraction, and normalization
- **SOM Analysis**: Complete SOM training and clustering using MiniSOM
- **Rich Visualizations**: Static and interactive plots for analysis results
- **Export Capabilities**: Save models and export results in multiple formats
- **MLflow Integration**: Comprehensive experiment tracking and artifact management

### Data Processing
- **Text Features**: TF-IDF, sentiment analysis, content type detection
- **Temporal Features**: Time-based patterns and seasonality analysis
- **Engagement Features**: Like, retweet, reply, and quote metrics
- **Network Features**: User interaction and influence patterns

### Analysis Capabilities
- **Cluster Discovery**: Automatic identification of tweet clusters
- **Pattern Recognition**: Detection of trends and anomalies
- **Topic Analysis**: Hashtag and content theme extraction
- **User Behavior**: Activity patterns and engagement analysis

## 📋 Use Cases

### 1. Content Clustering and Topic Discovery
- Automatically discover topics and themes in Twitter data
- Identify emerging trends and viral content patterns
- Understand content distribution across user groups

### 2. User Behavior Analysis
- Analyze posting patterns and engagement strategies
- Identify user archetypes and behavior clusters
- Detect bot vs. human user patterns

### 3. Sentiment and Emotion Mapping
- Map emotional landscapes across topics
- Track sentiment evolution over time
- Identify emotional response patterns

### 4. Campaign and Marketing Analysis
- Analyze brand mentions and campaign effectiveness
- Understand audience segments and preferences
- Optimize hashtag strategies

### 5. Crisis and Event Detection
- Identify unusual activity indicating breaking news
- Analyze information spread during emergencies
- Track event evolution patterns

### 6. Network Analysis and Influence Mapping
- Understand social network structures
- Identify influential users and opinion leaders
- Detect community structures and echo chambers

### 7. Language and Cultural Analysis
- Analyze linguistic patterns and cultural markers
- Study multilingual communities
- Understand regional communication differences

### 8. Temporal Pattern Analysis
- Identify time-based activity patterns
- Analyze seasonal trends and cyclical behaviors
- Detect temporal anomalies

## 🛠️ Installation

### Prerequisites
- Python 3.13+
- pip package manager

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd self-organizing-maps-cpe722

# Install dependencies
pip install -r requirements.in

# Or install with development dependencies
pip install -e ".[dev]"
```

### Required Packages
- `pydantic>=2.0.0` - Data validation and models
- `minisom>=2.3.0` - Self-Organizing Map implementation
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation
- `scikit-learn>=1.3.0` - Machine learning utilities
- `matplotlib>=3.7.0` - Static visualizations
- `seaborn>=0.12.0` - Statistical visualizations
- `nltk>=3.8.0` - Natural language processing
- `textblob>=0.17.0` - Text analysis
- `plotly>=5.15.0` - Interactive visualizations

## 🎮 Quick Start

### Running the Demo

```bash
# Basic demo without MLflow tracking
uv run python main.py

# With MLflow experiment tracking
uv run python main.py --experiment "my-twitter-som-experiment"

# With custom MLflow tracking server
uv run python main.py --experiment "my-experiment" --mlflow-tracking-uri "http://mlflow-server:5000"
```

### MLflow Integration

This project includes comprehensive MLflow integration for experiment tracking:

- **Automatic Parameter Logging**: SOM configuration, data statistics
- **Metrics Tracking**: Training metrics, cluster analysis, custom metrics
- **Artifact Management**: Models, visualizations, reports
- **Experiment Organization**: Compare different runs and configurations

See [MLflow_Integration.md](MLflow_Integration.md) for detailed documentation.

### Basic Usage

```python
from src.twitter_som import (
    TwitterData,
    TwitterDataCollection,
    SOMTrainingConfig,
    TwitterSOMAnalyzer,
    SOMVisualizer
)

# 1. Create Twitter data
tweets = [
    TwitterData(
        id="1",
        text="Excited about #MachineLearning and #DataScience!",
        created_at=datetime.now(timezone.utc),
        user_id="user1",
        username="datascientist",
        like_count=50,
        retweet_count=10
    ),
    # ... more tweets
]

collection = TwitterDataCollection(
    tweets=tweets,
    collection_name="sample_data"
)

# 2. Configure SOM
config = SOMTrainingConfig(
    x_dim=10,
    y_dim=10,
    learning_rate=0.1,
    num_iterations=1000,
    normalize_features=True
)

# 3. Train SOM (with optional MLflow tracking)
analyzer = TwitterSOMAnalyzer(config, mlflow_experiment_name="twitter-analysis")
training_stats = analyzer.train(collection)

# 4. Analyze results
cluster_summary = analyzer.get_all_clusters_summary()
print(f"Found {cluster_summary['total_clusters']} clusters")

# 5. Visualize results
visualizer = SOMVisualizer(analyzer)
visualizer.plot_som_topology(save_path="som_topology.png")

# 6. Log artifacts to MLflow (if enabled)
analyzer.log_visualizations_to_mlflow({
    "topology": "som_topology.png"
})

# 7. Save model and end MLflow run
analyzer.save_model("model.pkl")
analyzer.end_mlflow_run()
visualizer.plot_cluster_analysis()
```

### Run Complete Demonstration

```bash
python main.py
```

This will generate sample data, train a SOM model, perform analysis, and create visualizations.

## 📊 Data Model

### TwitterData
Core model for individual tweets with validation:

```python
TwitterData(
    id="tweet_id",                    # Unique identifier
    text="Tweet content...",          # Tweet text
    created_at=datetime(...),         # Timestamp
    user_id="user_id",               # User identifier
    username="username",             # Username
    like_count=10,                   # Engagement metrics
    retweet_count=5,
    reply_count=2,
    quote_count=1,
    hashtags=["tag1", "tag2"],       # Extracted hashtags
    mentions=["user1", "user2"],     # User mentions
    urls=["http://..."],             # URLs in tweet
    lang="en",                       # Language code
    location="City, Country",        # User location
    is_retweet=False,               # Tweet type flags
    is_reply=False,
    is_quote=False
)
```

### TwitterDataCollection
Container for multiple tweets with utilities:

```python
collection = TwitterDataCollection(
    tweets=tweet_list,
    collection_name="dataset_name",
    description="Dataset description"
)

# Built-in analysis methods
date_range = collection.get_date_range()
users = collection.get_unique_users()
engagement = collection.get_total_engagement()
hashtags = collection.get_most_common_hashtags()

# Filtering methods
filtered = collection.filter_by_date_range(start, end)
english_tweets = collection.filter_by_language("en")
sample = collection.sample(n=100, random_state=42)
```

## 🔧 Configuration

### YAML Configuration Files

The project now supports YAML configuration files for easy parameter management. Configuration files are located in the `config/` directory:

```python
# Load configuration from YAML file
from src.twitter_som.models import SOMTrainingConfig

config = SOMTrainingConfig.from_yaml("config/config.yaml")
```

#### Available Configurations

- **`config/config.yaml`** - Default balanced settings
- **`config/fast_training_config.yaml`** - Quick prototyping (8x8 grid, 500 iterations)
- **`config/high_performance_config.yaml`** - Detailed analysis (20x20 grid, 3000 iterations)
- **`config/text_only_config.yaml`** - Text-focused analysis with cosine distance

#### Custom Configurations

Create custom configurations by copying and modifying existing files:

```bash
cp config/config.yaml config/my_config.yaml
# Edit my_config.yaml with your settings
```

Save configurations programmatically:

```python
config = SOMTrainingConfig(x_dim=25, y_dim=25, learning_rate=0.08)
config.to_yaml("config/my_saved_config.yaml")
```

### SOM Training Configuration
Comprehensive configuration for SOM training (programmatic approach):

```python
config = SOMTrainingConfig(
    # SOM dimensions
    x_dim=15,                        # Map width
    y_dim=15,                        # Map height

    # Training parameters
    learning_rate=0.1,               # Learning rate
    neighborhood_function='gaussian', # Neighborhood function
    topology='rectangular',          # Map topology
    activation_distance='euclidean', # Distance metric
    num_iterations=1500,             # Training iterations

    # Preprocessing options
    normalize_features=True,         # Feature normalization
    use_pca=False,                  # PCA dimensionality reduction
    pca_components=50,              # PCA components (if enabled)

    # Feature selection
    include_temporal_features=True,  # Time-based features
    include_engagement_features=True,# Engagement metrics
    include_text_features=True,      # Text-based features
    include_network_features=True    # Network-based features
)
```

For detailed configuration documentation, see [`config/README.md`](config/README.md).

## 🧪 Testing (TDD Approach)

This project follows Test-Driven Development principles:

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run specific test
pytest tests/test_models.py::TestTwitterData::test_create_valid_twitter_data -v
```

### Test Structure
```
tests/
├── conftest.py              # Test configuration and fixtures
├── test_models.py           # Data model tests
├── test_preprocessor.py     # Preprocessing tests
├── test_som_analyzer.py     # SOM analysis tests
└── test_visualizer.py       # Visualization tests
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Data Validation Tests**: Pydantic model validation
- **Feature Extraction Tests**: Preprocessing functionality
- **SOM Training Tests**: Analysis pipeline testing

## 📈 Analysis Pipeline

### 1. Data Preprocessing
```python
preprocessor = TwitterPreprocessor(config)

# Extract features
features, feature_names = preprocessor.fit_transform(collection)

# Feature types:
# - Temporal: hour, day_of_week, month, etc.
# - Engagement: likes, retweets, ratios, scores
# - Text: TF-IDF, sentiment, length, hashtags
# - Network: user activity, mentions, interactions
```

### 2. SOM Training
```python
analyzer = TwitterSOMAnalyzer(config)
training_stats = analyzer.train(collection)

# Training metrics:
# - Quantization error
# - Topographic error
# - Cluster statistics
```

### 3. Cluster Analysis
```python
# Get cluster insights
cluster_insights = analyzer.get_cluster_insights("cluster_5_3")

# Cluster information includes:
# - Size and coordinates
# - Feature importance
# - Dominant hashtags
# - Temporal patterns
# - Sample tweets
# - Similarity to other clusters
```

### 4. Visualization
```python
visualizer = SOMVisualizer(analyzer)

# Static plots
visualizer.plot_som_topology()
visualizer.plot_cluster_analysis()
visualizer.plot_tweet_distribution()

# Interactive visualization
visualizer.create_interactive_visualization('dashboard.html')

# Generate report
visualizer.generate_report('analysis_report.txt')
```

## 🎨 Visualizations

### Available Plots
1. **SOM Topology**: Distance map, hit map, feature maps
2. **Cluster Analysis**: Engagement vs size, temporal patterns, hashtags
3. **Tweet Distribution**: Cluster sizes, sample tweets
4. **Feature Importance**: Heatmaps of feature weights
5. **Interactive Dashboard**: Plotly-based interactive visualizations

### Example Outputs
- `som_topology.png` - SOM structure visualization
- `cluster_analysis.png` - Comprehensive cluster analysis
- `tweet_distribution.png` - Tweet distribution across clusters
- `som_interactive.html` - Interactive dashboard
- `som_analysis_report.txt` - Detailed text report

## 💾 Model Persistence

### Save Trained Model
```python
# Save complete model
analyzer.save_model('twitter_som_model.pkl')

# Export results to JSON
analyzer.export_results('results.json')
```

### Load Saved Model
```python
# Load and use saved model
new_analyzer = TwitterSOMAnalyzer(config)
new_analyzer.load_model('twitter_som_model.pkl')

# Make predictions on new data
predictions = new_analyzer.predict_cluster(new_collection)
```

## 🔍 Advanced Features

### Custom Feature Engineering
Extend the preprocessor for domain-specific features:

```python
class CustomPreprocessor(TwitterPreprocessor):
    def extract_custom_features(self, tweets):
        # Add your custom feature extraction logic
        pass
```

### Real-time Analysis
Use the trained model for streaming data:

```python
# Process streaming tweets
for batch in tweet_stream:
    batch_collection = TwitterDataCollection(tweets=batch, collection_name="stream")
    predictions = analyzer.predict_cluster(batch_collection)
    # Process predictions...
```

### Multi-language Support
Handle multilingual datasets:

```python
# Filter by language
english_tweets = collection.filter_by_language("en")
spanish_tweets = collection.filter_by_language("es")

# Train separate models or combined analysis
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/
mypy src/
```

### Adding New Features
1. Write tests first (TDD approach)
2. Implement the feature
3. Update documentation
4. Run full test suite
5. Submit pull request

## 📄 License

This project is developed for academic purposes as part of CPE722 coursework.

## 👥 Authors

- Rafael Tadeu - Initial development

## 🙏 Acknowledgments

- MiniSOM library for SOM implementation
- Pydantic for data validation
- scikit-learn for machine learning utilities
- NLTK and TextBlob for text processing
- Matplotlib, Seaborn, and Plotly for visualizations

## 📞 Support

For questions or issues:
1. Check the test files for usage examples
2. Review the main.py demonstration
3. Create an issue in the repository

## 🔮 Future Enhancements

Potential improvements and extensions:
- Real-time streaming data analysis
- Deep learning feature extraction
- Advanced sentiment analysis models
- Geographic analysis capabilities
- Social network analysis integration
- Anomaly detection algorithms
- Performance optimization for large datasets