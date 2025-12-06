# Deployment Guide

## Step 1: Train the Model

```bash
python src/train_pipeline.py
```

This generates:
- `models/best_model.pkl` - Trained model
- `models/preprocessor.pkl` - Data preprocessor
- `models/model_metadata.json` - Model info

## Step 2: Build Docker Image

```bash
docker build -t ml-pipeline .
```

## Step 3: Run Container

```bash
docker run -p 5000:5000 ml-pipeline
```

## Step 4: Test the API

### Health Check
```bash
curl http://localhost:5000/health
```

### Make Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

### Example Response
```json
{
  "prediction": 1,
  "probability": [0.23, 0.77]
}
```

## API Endpoint

**URL**: `http://localhost:5000/predict`

**Method**: `POST`

**Request Body**:
```json
{
  "features": {
    "feature_0": 0.5,
    "feature_1": -0.3,
    "feature_2": 1.2,
    ...
  }
}
```

**Response**:
```json
{
  "prediction": 1,
  "probability": [0.23, 0.77]
}
```

That's it! Your model is deployed and ready to serve predictions.
