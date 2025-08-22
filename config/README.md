# Configuration Directory

This directory contains YAML configuration files for the SOM (Self-Organizing Map) training parameters.

## Available Configurations

### 1. `config.yaml` (Default)
The main configuration file with balanced settings suitable for most use cases.
- **Grid Size**: 15x15 (moderate resolution)
- **Learning Rate**: 0.1 (balanced learning speed)
- **Iterations**: 1500 (sufficient for convergence)
- **Features**: All feature types enabled
- **PCA**: Disabled (preserves all features)

### 2. `high_performance_config.yaml`
Optimized for detailed analysis with larger datasets.
- **Grid Size**: 20x20 (high resolution for detailed clustering)
- **Learning Rate**: 0.05 (slower, more precise learning)
- **Iterations**: 3000 (extensive training)
- **Topology**: Hexagonal (better neighbor relationships)
- **PCA**: Enabled with 75 components

### 3. `fast_training_config.yaml`
Optimized for quick prototyping and testing.
- **Grid Size**: 8x8 (smaller grid for speed)
- **Learning Rate**: 0.3 (faster convergence)
- **Iterations**: 500 (minimal training time)
- **Features**: Only temporal and engagement features
- **PCA**: Disabled for speed

### 4. `text_only_config.yaml`
Specialized for text-based feature analysis.
- **Grid Size**: 12x12 (moderate resolution)
- **Distance Metric**: Cosine (optimal for text features)
- **Features**: Only text features enabled
- **PCA**: Enabled with 50 components (good for text dimensionality)

## Usage

### Loading Configuration in Python

```python
from src.twitter_som.models import SOMTrainingConfig

# Load default configuration
config = SOMTrainingConfig.from_yaml("config/config.yaml")

# Load specialized configuration
config = SOMTrainingConfig.from_yaml("config/fast_training_config.yaml")

# Use with SOM analyzer
from src.twitter_som.som_analyzer import TwitterSOMAnalyzer
analyzer = TwitterSOMAnalyzer(config)
```

### Creating Custom Configurations

You can create custom configuration files by copying any of the existing files and modifying the parameters:

```bash
cp config/config.yaml config/my_custom_config.yaml
# Edit my_custom_config.yaml with your preferred settings
```

### Saving Configuration from Code

```python
# Create configuration programmatically
config = SOMTrainingConfig(
    x_dim=25,
    y_dim=25,
    learning_rate=0.08,
    num_iterations=2000
)

# Save to YAML file
config.to_yaml("config/my_saved_config.yaml")
```

## Configuration Parameters

### SOM Section
- **x_dim**: Grid width (positive integer)
- **y_dim**: Grid height (positive integer)
- **learning_rate**: Initial learning rate (0 < value <= 1)
- **neighborhood_function**: "gaussian", "bubble", "triangle"
- **topology**: "rectangular", "hexagonal"
- **activation_distance**: "euclidean", "manhattan", "cosine"
- **num_iterations**: Number of training iterations (positive integer)

### Preprocessing Section
- **normalize_features**: Whether to normalize input features (boolean)
- **use_pca**: Whether to apply PCA before SOM training (boolean)
- **pca_components**: Number of PCA components (integer or null)

### Features Section
- **include_temporal_features**: Include time-based features (boolean)
- **include_engagement_features**: Include engagement metrics (boolean)
- **include_text_features**: Include text-based features (boolean)
- **include_network_features**: Include network-based features (boolean)

## Best Practices

1. **Start with default configuration** for initial experiments
2. **Use fast_training_config** for quick prototyping
3. **Use high_performance_config** for final analysis with large datasets
4. **Use text_only_config** when focusing on content analysis
5. **Always backup** working configurations before modifications
6. **Test configurations** with small datasets first
7. **Enable PCA** when working with high-dimensional data
8. **Adjust grid size** based on expected number of clusters

## Validation

All configuration files are automatically validated when loaded. Invalid parameters will raise clear error messages indicating the problem and expected values.
