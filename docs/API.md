# 🔌 API Documentation

The Graph Neural Tutor provides a RESTful API for programmatic access to algebraic reasoning capabilities. This document covers all available endpoints, request/response formats, and usage examples.

## 🌐 Base URL

- **Local Development:** `http://localhost:8000` (run locally for testing)

## 🔑 Authentication

Currently, the API is open and does not require authentication. Rate limiting is applied to prevent abuse.

## 📊 API Endpoints

### Core Analysis

#### Analyze Single Transformation

**POST** `/api/analyze`

Analyze a single algebraic transformation step.

**Request Body:**
```json
{
  "original": "3*x + 5 = 11",
  "transformed": "3*x = 6",
  "model": "main"
}
```

**Parameters:**
- `original` (string, required): Original equation
- `transformed` (string, required): Transformed equation  
- `model` (string, optional): Model type (`main`, `simple`, `minimal`). Default: `main`

**Response:**
```json
{
  "rule_prediction": "sub_const",
  "rule_confidence": [0.05, 0.89, 0.02, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00],
  "validity_score": 0.92,
  "is_valid": true,
  "pointer_ranking": [1, 0, 2],
  "all_rules": ["add_const", "sub_const", "div_coeff", "mul_denom", "expand", "factor", "pow_reduce", "combine_fracs", "sub_var"],
  "model_used": "main",
  "original_equation": "Eq(3*x + 5, 11)",
  "transformed_equation": "Eq(3*x, 6)",
  "processing_time": 0.045,
  "analysis_id": "uuid-string"
}
```

#### Batch Analysis

**POST** `/api/analyze/batch`

Analyze multiple transformations in a single request.

**Request Body:**
```json
{
  "equations": [
    {
      "original": "3*x + 5 = 11",
      "transformed": "3*x = 6",
      "model": "main"
    },
    {
      "original": "2*x = 8",
      "transformed": "x = 4",
      "model": "simple"
    }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "index": 0,
      "result": { /* Analysis result object */ }
    }
  ],
  "errors": [
    {
      "index": 1,
      "error": "Error message"
    }
  ],
  "total_processed": 1,
  "total_errors": 1
}
```

#### Retrieve Cached Analysis

**GET** `/api/analysis/{analysis_id}`

Retrieve a previously cached analysis result.

**Response:**
```json
{
  /* Same as analyze response */
}
```

### Educational Resources

#### Get Example Equations

**GET** `/api/examples`

Get example transformations for demonstration.

**Response:**
```json
[
  {
    "original": "3*x + 5 = 11",
    "transformed": "3*x = 6", 
    "rule": "sub_const",
    "description": "Subtract constant from both sides"
  }
]
```

#### Get Supported Rules

**GET** `/api/rules`

Get information about all supported algebraic rules.

**Response:**
```json
{
  "add_const": {
    "name": "Add/Subtract Constant",
    "description": "Add or subtract the same constant from both sides",
    "example": "3x + 5 = 11 → 3x = 6"
  }
}
```

### Model Information

#### Get Available Models

**GET** `/api/models`

Get information about available model architectures.

**Response:**
```json
{
  "main": {
    "name": "GNT-Main",
    "architecture": "GAT + Transformer + Uncertainty",
    "parameters": 128456,
    "status": "ready",
    "accuracy": 0.724,
    "training_time": "2024-01-15T10:30:00Z"
  }
}
```

#### Get Specific Model Info

**GET** `/api/models/{model_type}`

Get detailed information about a specific model.

**Response:**
```json
{
  "name": "GNT-Main",
  "architecture": "GAT + Transformer + Uncertainty", 
  "parameters": 128456,
  "status": "ready"
}
```

### Data Management

#### Generate Dataset

**POST** `/api/dataset/generate`

Generate a synthetic dataset (background task).

**Request Body:**
```json
{
  "num_per_rule": 250,
  "num_negative": 600,
  "save_dataset": true
}
```

**Response:**
```json
{
  "message": "Dataset generation started",
  "num_per_rule": 250,
  "num_negative": 600,
  "total_expected": 2850
}
```

### System Information

#### Health Check

**GET** `/api/health`

Check system health and status.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda:0",
  "models_loaded": 3,
  "redis_available": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### System Statistics

**GET** `/api/stats`

Get detailed system statistics.

**Response:**
```json
{
  "models_available": 3,
  "rules_supported": 9,
  "device": "cuda:0",
  "python_version": "3.10.0",
  "torch_version": "2.0.0",
  "redis_stats": {
    "connected_clients": 5,
    "used_memory_human": "2.5M",
    "keyspace_hits": 1250
  }
}
```

## 🧪 Code Examples

### Python Client

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

def analyze_transformation(original, transformed, model="main"):
    """Analyze algebraic transformation"""
    url = f"{BASE_URL}/api/analyze"
    
    payload = {
        "original": original,
        "transformed": transformed,
        "model": model
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

# Example usage
result = analyze_transformation("3*x + 5 = 11", "3*x = 6")
print(f"Predicted rule: {result['rule_prediction']}")
print(f"Confidence: {result['rule_confidence'][1]*100:.1f}%")
print(f"Valid: {result['is_valid']}")
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');

const BASE_URL = 'http://localhost:8000';

async function analyzeTransformation(original, transformed, model = 'main') {
    try {
        const response = await axios.post(`${BASE_URL}/api/analyze`, {
            original,
            transformed,
            model
        });
        
        return response.data;
    } catch (error) {
        throw new Error(`API Error: ${error.response.status} - ${error.response.data}`);
    }
}

// Example usage
analyzeTransformation('3*x + 5 = 11', '3*x = 6')
    .then(result => {
        console.log(`Predicted rule: ${result.rule_prediction}`);
        console.log(`Confidence: ${(result.rule_confidence[1] * 100).toFixed(1)}%`);
        console.log(`Valid: ${result.is_valid}`);
    })
    .catch(console.error);
```

### cURL Examples

```bash
# Analyze transformation
curl -X POST "http://localhost:8000/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "original": "3*x + 5 = 11",
    "transformed": "3*x = 6",
    "model": "main"
  }'

# Get examples
curl "http://localhost:8000/api/examples"

# Health check
curl "http://localhost:8000/api/health"
```

## 🚨 Error Handling

### HTTP Status Codes

- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Not Found
- `422` - Validation Error
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error

### Error Response Format

```json
{
  "error": "Error description",
  "detail": "Detailed error message",
  "status_code": 400
}
```

### Common Errors

1. **Invalid Equation Format:**
```json
{
  "error": "Error parsing original equation: invalid syntax"
}
```

2. **Model Not Available:**
```json
{
  "error": "Model 'invalid_model' not available"
}
```

3. **Rate Limit Exceeded:**
```json
{
  "error": "Rate limit exceeded. Please try again later."
}
```

## 🔄 Rate Limiting

- **Default Limit:** 100 requests per minute per IP
- **Burst Capacity:** 20 requests
- **Headers Included:**
  - `X-RateLimit-Limit`: Requests per window
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Window reset time

## 📝 Request Validation

### Equation Format

Equations should be provided as strings in standard mathematical notation:

**Valid Formats:**
- `"3*x + 5 = 11"`
- `"x/4 = 3"`
- `"(x + 2)*(x + 3) = 0"`
- `"x**2 = 16"`

**Invalid Formats:**
- `"3x + 5 = 11"` (missing multiplication operator)
- `"x = "` (incomplete equation)
- `"not_an_equation"` (not mathematical)

### Model Types

Valid model types:
- `"main"` - Full GNN with GAT + Transformer
- `"simple"` - Graph Convolutional Network
- `"minimal"` - Multi-Layer Perceptron baseline

## 🔒 Security

### Input Sanitization

All equation inputs are:
- Parsed using SymPy (safe symbolic computation)
- Validated for mathematical correctness
- Limited in complexity to prevent DoS

### CORS Policy

- Allowed origins: `*` (configurable)
- Allowed methods: `GET`, `POST`, `OPTIONS`
- Allowed headers: `Content-Type`, `Authorization`

## 📊 Performance

### Response Times

- Simple analysis: ~50ms
- Batch analysis: ~200ms per equation
- Model loading: ~2s (cached after first request)

### Throughput

- Single instance: ~20 requests/second
- With load balancing: ~100+ requests/second

## 📞 Support

- **Issues:** Use GitHub Issues for technical questions
- **Documentation:** See the `docs/` directory for full documentation

---

**Happy coding! 🚀**
