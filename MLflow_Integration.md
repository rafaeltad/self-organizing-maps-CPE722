# MLflow Integration Guide

This document describes the MLflow integration added to the Twitter SOM Analysis project for experiment tracking, metrics logging, and artifact management.

## Overview

The project now includes comprehensive MLflow integration that allows you to:
- Track experiment parameters and hyperparameters
- Log training and evaluation metrics
- Store model artifacts and visualizations
- Compare different experimental runs
- Manage model versions

## Features

### 1. Experiment Tracking
- Configurable experiment names via command line
- Automatic parameter logging (SOM configuration, data statistics)
- Training metrics logging (quantization error, topographic error, cluster metrics)
- Custom metrics support

### 2. Artifact Management
- Model artifacts (trained SOM models)
- Visualization artifacts (plots, charts, interactive dashboards)
- Results artifacts (JSON reports, analysis summaries)
- Organized artifact storage by type

### 3. Metrics Logged

#### Training Metrics
- `quantization_error`: SOM quantization error
- `topographic_error`: SOM topographic error
- `num_clusters`: Number of clusters discovered
- `largest_cluster_size`: Size of the largest cluster
- `smallest_cluster_size`: Size of the smallest cluster
- `mean_cluster_size`: Average cluster size
- `cluster_size_std`: Standard deviation of cluster sizes

#### Data Metrics
- `num_samples`: Number of tweets processed
- `num_features`: Number of features extracted
- `unique_users`: Number of unique users in dataset
- `total_tweets`: Total number of tweets
- `avg_engagement_per_tweet`: Average engagement score per tweet
- `cluster_size_variance`: Variance in cluster sizes

#### Configuration Parameters
- `x_dim`, `y_dim`: SOM grid dimensions
- `learning_rate`: SOM learning rate
- `num_iterations`: Number of training iterations
- `normalize_features`: Whether features are normalized
- `use_pca`: Whether PCA is used
- `include_temporal_features`: Whether temporal features are included
- `include_engagement_features`: Whether engagement features are included
- `include_text_features`: Whether text features are included
- `include_network_features`: Whether network features are included
- `neighborhood_function`: SOM neighborhood function
- `topology`: SOM topology
- `activation_distance`: SOM activation distance
- `som_grid_size`: Total number of SOM neurons

### 4. Artifacts Stored

#### Models
- `twitter_som_model.pkl`: Complete trained SOM model

#### Visualizations
- `som_topology.png`: SOM topology visualization
- `cluster_analysis.png`: Cluster analysis charts
- `tweet_distribution.png`: Tweet distribution plots
- `som_interactive.html`: Interactive SOM dashboard
- `som_analysis_report.txt`: Detailed analysis report

#### Results
- `som_results.json`: Complete analysis results in JSON format

## Usage

### Command Line Interface

The main script now supports MLflow configuration via command line arguments:

```bash
# Run without MLflow logging (backward compatible)
uv run python main.py

# Run with MLflow experiment tracking
uv run python main.py --experiment "my-experiment-name"

# Run with custom MLflow tracking server
uv run python main.py --experiment "my-experiment" --mlflow-tracking-uri "http://mlflow-server:5000"
```

### Programmatic Usage

```python
from src.twitter_som import TwitterSOMAnalyzer, SOMTrainingConfig

# Configure SOM
config = SOMTrainingConfig(
    x_dim=8,
    y_dim=8,
    learning_rate=0.1,
    num_iterations=1000
)

# Initialize analyzer with MLflow experiment
analyzer = TwitterSOMAnalyzer(config, mlflow_experiment_name="my-experiment")

# Train model (automatically logs parameters and metrics)
training_stats = analyzer.train(collection, verbose=True)

# Log visualizations and artifacts
visualization_paths = {
    "topology": "som_topology.png",
    "analysis": "cluster_analysis.png"
}
analyzer.log_visualizations_to_mlflow(visualization_paths)

# Log additional custom metrics
custom_metrics = {
    "custom_score": 0.85,
    "validation_accuracy": 0.92
}
analyzer.log_additional_metrics(custom_metrics)

# Save model and results (automatically logged as artifacts)
analyzer.save_model("model.pkl")
analyzer.export_results("results.json")

# End MLflow run
analyzer.end_mlflow_run()
```

### Viewing Results

#### MLflow UI
Start the MLflow UI to view logged experiments:

```bash
cd /path/to/project
uv run mlflow ui --host 127.0.0.1 --port 5000
```

Then navigate to `http://127.0.0.1:5000` to view:
- Experiment runs and comparisons
- Parameter and metric values
- Artifact downloads
- Run details and notes

#### Command Line
List experiments:
```bash
uv run mlflow experiments search
```

## Configuration

### Environment Variables

You can configure MLflow behavior using environment variables:

```bash
# Set tracking URI
export MLFLOW_TRACKING_URI="http://mlflow-server:5000"

# Set default experiment
export MLFLOW_EXPERIMENT_NAME="default-twitter-som"

# Set artifact storage location
export MLFLOW_DEFAULT_ARTIFACT_ROOT="s3://my-bucket/mlflow-artifacts"
```

### MLflow Server Setup

For production use, you may want to set up a dedicated MLflow tracking server:

```bash
# Install MLflow server dependencies
uv add mlflow[extras]

# Start tracking server with database backend
mlflow server \
    --backend-store-uri postgresql://user:password@localhost/mlflow \
    --default-artifact-root s3://my-bucket/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

## Integration Details

### Code Changes

1. **TwitterSOMAnalyzer Class**: Enhanced with MLflow support
   - Added `mlflow_experiment_name` parameter to constructor
   - Automatic parameter and metrics logging in `train()` method
   - Artifact logging in `save_model()` and `export_results()` methods
   - New methods: `log_visualizations_to_mlflow()`, `log_additional_metrics()`, `end_mlflow_run()`

2. **Main Script**: Enhanced with command-line interface
   - Added argument parsing for experiment configuration
   - Configurable MLflow tracking URI
   - Backward compatibility maintained

3. **Dependencies**: Added MLflow to project dependencies
   - `mlflow>=2.8.0` added to `pyproject.toml` and `requirements.in`

### Backward Compatibility

The MLflow integration is completely optional and maintains full backward compatibility:
- Running without the `--experiment` flag disables MLflow logging
- Existing code continues to work without modifications
- No breaking changes to the API

### Error Handling

Robust error handling ensures the system continues to work even if MLflow is unavailable:
- Graceful degradation when MLflow is not installed
- Error handling for network issues with remote tracking servers
- Fallback to local execution when MLflow operations fail

## Best Practices

### Experiment Organization
- Use descriptive experiment names that reflect the research question
- Group related runs in the same experiment
- Use consistent naming conventions

### Parameter Tracking
- Log all relevant hyperparameters for reproducibility
- Include data characteristics as parameters
- Document parameter choices in run descriptions

### Metric Selection
- Focus on metrics that align with your evaluation criteria
- Log both training and validation metrics when applicable
- Include business metrics alongside technical metrics

### Artifact Management
- Organize artifacts by type (models, visualizations, reports)
- Use consistent naming conventions for artifacts
- Consider artifact size and storage costs

### Model Lifecycle
- Tag important runs for easy identification
- Use model registry for production model management
- Implement model versioning strategies

## Troubleshooting

### Common Issues

1. **MLflow UI not starting**
   ```bash
   # Check if port is already in use
   lsof -i :5000

   # Use different port
   uv run mlflow ui --port 5001
   ```

2. **Artifacts not logging**
   - Ensure files exist before logging
   - Check file permissions
   - Verify MLflow run is active

3. **Permission errors**
   ```bash
   # Check MLflow directory permissions
   ls -la mlruns/

   # Fix permissions if needed
   chmod -R 755 mlruns/
   ```

4. **Import errors**
   ```bash
   # Reinstall MLflow
   uv remove mlflow
   uv add mlflow>=2.8.0
   ```

### Performance Considerations

- Artifact logging can be slow for large files
- Consider using artifact storage backends (S3, Azure, GCS) for production
- Monitor disk usage in local MLflow tracking directory
- Use MLflow's built-in cleanup utilities for old runs

## Future Enhancements

Potential improvements to the MLflow integration:

1. **Model Registry Integration**: Automatic model registration and staging
2. **Automated Model Validation**: Compare new models against baselines
3. **Hyperparameter Optimization**: Integration with MLflow's hyperopt
4. **Real-time Monitoring**: Live metric updates during training
5. **Distributed Training**: Support for distributed SOM training with MLflow
6. **Custom Metrics**: Domain-specific evaluation metrics
7. **Data Versioning**: Track dataset versions and lineage
8. **Automated Reporting**: Generate experiment summary reports
