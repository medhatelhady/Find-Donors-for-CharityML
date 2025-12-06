# End-to-End ML Pipeline

A production-ready machine learning pipeline for income prediction using Census data. Features sklearn Pipeline with ColumnTransformer for preprocessing, Decision Tree classifier, and Docker deployment.

## Project Structure

```
ml-pipeline/
├── data/
│   └── raw/
│       └── census.csv          # Census income dataset
├── src/
│   ├── data/
│   │   ├── loader.py           # Data loading
│   │   └── preprocessor.py     # sklearn Pipeline preprocessing
│   ├── models/
│   │   └── trainer.py          # Decision Tree training
│   └── deployment/
│       └── api.py              # Flask REST API
├── configs/
│   └── config.yaml             # Configuration
├── models/                     # Saved models (generated)
├── notebooks/                  # Exploratory notebooks
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
└── sample_request.json         # Example API request
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
python src/train_pipeline.py
```

This will:
- Load census data from `data/raw/census.csv`
- Build sklearn preprocessing pipeline (ColumnTransformer)
- Train Decision Tree classifier
- Save complete pipeline to `models/best_model.pkl`

### 3. Run API Locally (without Docker)
```bash
python src/deployment/api.py
```

API will be available at: http://localhost:5000

### 4. Deploy with Docker
```bash
# Build image
docker build -t ml-api .

# Run container
docker run -p 5000:5000 ml-api
```

### 5. Make Predictions
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

**Windows PowerShell:**
```powershell
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -InFile sample_request.json
```

## Features

### sklearn Pipeline Architecture
- **ColumnTransformer**: Applies different transformations to different column groups
- **Numerical Pipeline**: Log transform → Imputation → Scaling
- **Categorical Pipeline**: Imputation → One-hot encoding / Binary encoding
- **Complete Pipeline**: Preprocessing + Decision Tree in one object

### Model
- **Algorithm**: Decision Tree Classifier
- **No grid search**: Uses sensible default hyperparameters
- **Fast training**: Simple and efficient

### Deployment
- **Single pipeline file**: Everything in one `.pkl` file
- **REST API**: Flask with CORS support
- **Docker ready**: One command deployment

## API Endpoints

**Base URL**: http://localhost:5000

### GET /health
Health check endpoint

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "pipeline_loaded": true
}
```

### POST /predict
Predict income level (<=50K or >50K)

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

**Request Format:**
```json
{
  "features": {
    "age": 39,
    "workclass": " State-gov",
    "education_level": " Bachelors",
    "education-num": 13.0,
    "marital-status": " Never-married",
    "occupation": " Adm-clerical",
    "relationship": " Not-in-family",
    "race": " White",
    "sex": " Male",
    "capital-gain": 2174.0,
    "capital-loss": 0.0,
    "hours-per-week": 40.0,
    "native-country": " United-States"
  }
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": [0.85, 0.15]
}
```

- `prediction`: 0 (<=50K) or 1 (>50K)
- `probability`: [prob_class_0, prob_class_1]

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
data:
  raw_path: "data/raw/census.csv"
  label: "income"
  test_size: 0.2
  random_state: 42

preprocessing:
  scaling: "minmax"
  handle_missing: "mean"

model:
  output_path: "models/"

deployment:
  host: "0.0.0.0"
  port: 5000
  model_path: "models/best_model.pkl"
```

## Data Pipeline

### Preprocessing Steps
1. **Log Transform**: Applied to skewed features (capital-gain, capital-loss)
2. **Numerical Features**: Imputation → MinMax scaling
3. **Categorical Features**: 
   - One-hot encoding for low cardinality (marital-status, relationship, race, sex)
   - Binary encoding for high cardinality (education_level, occupation, native-country, workclass)

### Column Groups
- **Skewed**: capital-gain, capital-loss
- **Numerical**: age, education-num, capital-gain, capital-loss, hours-per-week
- **Categorical (One-hot)**: marital-status, relationship, race, sex
- **Categorical (Binary)**: education_level, occupation, native-country, workclass

## Docker Commands

```bash
# Build image
docker build -t ml-api .

# Run container
docker run -p 5000:5000 ml-api

# Run in background
docker run -d -p 5000:5000 ml-api

# View logs
docker logs <container_id>

# Stop container
docker stop <container_id>

# Remove container
docker rm <container_id>
```

## Development

### Project Components

- **DataLoader** (`src/data/loader.py`): Loads census data
- **DataPreprocessor** (`src/data/preprocessor.py`): sklearn Pipeline with ColumnTransformer
- **ModelTrainer** (`src/models/trainer.py`): Trains Decision Tree pipeline
- **API** (`src/deployment/api.py`): Flask REST API

### Training Pipeline

The training pipeline (`src/train_pipeline.py`) follows these steps:

1. Load census data
2. Build preprocessing pipeline
3. Split data (train/test)
4. Train Decision Tree pipeline
5. Evaluate on test set
6. Save complete pipeline

### Key Design Decisions

- **sklearn Pipeline**: All preprocessing and model in one object
- **No grid search**: Faster training, good default parameters
- **ColumnTransformer**: Clean separation of numerical/categorical processing
- **Module-level functions**: Ensures pipeline can be pickled/unpickled
- **Single file deployment**: Just load pipeline and predict

## Troubleshooting

### Pickle Error
If you get a pickle error, ensure:
- Lambda functions are replaced with module-level functions
- All custom transformers are defined at module level
- Python path includes project root

### Module Not Found
If you get `ModuleNotFoundError: No module named 'src'`:
- Run from project root directory
- API file adds project root to sys.path automatically

### API Not Starting
- Check if port 5000 is available
- Ensure model file exists at `models/best_model.pkl`
- Train the model first with `python src/train_pipeline.py`

## Future Work

### Autoscaling
- **Kubernetes HPA**: Implement Horizontal Pod Autoscaler to automatically scale API replicas based on request load
- **Load Balancing**: Add load balancer (nginx/HAProxy) to distribute traffic across multiple API instances
- **Resource Optimization**: Profile and optimize memory usage for container efficiency

### Monitoring

#### Model Monitoring
- **Model Performance Tracking**: Track prediction accuracy, precision, recall, and F1-score over time
- **Model Drift Detection**: Monitor input data distribution shifts and alert when model retraining is recommended
- **Prediction Confidence**: Track prediction confidence scores to identify uncertain predictions

#### Data Monitoring
- **Data Quality Checks**: Validate incoming request data against expected schema and ranges
- **Anomaly Detection**: Alert on unusual input patterns that deviate from training data distribution
- **Missing Value Tracking**: Monitor frequency of missing or null values in predictions

#### Infrastructure Monitoring
- **CPU Monitoring**: Track CPU usage with alerts for high utilization (>80%)
- **RAM Monitoring**: Monitor memory consumption and alert on potential memory leaks
- **Request Metrics**: Log response times, throughput, and error rates
- **Health Checks**: Automated health checks and container restart policies

### Implementation Stack (Recommended)
- **Monitoring**: Prometheus + Grafana for metrics and visualization
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana) for centralized logging
- **Alerting**: AlertManager for automated alerts on threshold violations
- **Container Orchestration**: Kubernetes for autoscaling and resource management
