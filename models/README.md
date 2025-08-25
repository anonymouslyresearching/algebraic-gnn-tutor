# 🧠 Model Directory

This directory contains the trained neural network models for the Graph Neural Tutor.

## Model Files

After running the training pipeline (`python main.py`), the following model files will be generated:

- **`main_model_best.pth`** - Primary GNN model (GAT + Transformer + Uncertainty)
- **`simple_model_best.pth`** - Simplified GCN model for comparison
- **`minimal_model_best.pth`** - MLP baseline model

## Model Architecture Details

### Main Model (GNT-Main)
- **Encoder**: Graph Attention Network (GAT) with 4 attention heads
- **Aggregation**: Transformer layer with positional encoding
- **Uncertainty**: Bayesian neural network heads for confidence estimation
- **Parameters**: ~128,000 trainable parameters
- **Input**: Graph representations of algebraic expressions
- **Output**: Rule classification, validity scores, pointer attention

### Simple Model (GNT-Simple)
- **Encoder**: Graph Convolutional Network (GCN)
- **Aggregation**: Global mean pooling
- **Parameters**: ~64,000 trainable parameters
- **Purpose**: Ablation study to assess impact of attention mechanisms

### Minimal Model (GNT-Minimal)
- **Encoder**: Multi-Layer Perceptron (MLP)
- **Aggregation**: Feature averaging
- **Parameters**: ~32,000 trainable parameters
- **Purpose**: Baseline comparison without graph structure

## Loading Models

```python
import torch
from main import DistinctAlgebraicGNN

# Load the main model
model = DistinctAlgebraicGNN(encoder_type="main", use_uncertainty=True)
model.load_state_dict(torch.load("models/main_model_best.pth"))
model.eval()

# Use for inference
rule_logits, val_probs, ptr_scores = model(graph_data)
```

## Performance Metrics

| Model | Macro F1 | Accuracy | Validity AUC | Parameters |
|-------|----------|----------|--------------|------------|
| GNT-Main | 0.724±0.032 | 0.756±0.028 | 0.892±0.018 | 128K |
| GNT-Simple | 0.689±0.041 | 0.702±0.039 | 0.834±0.025 | 64K |
| GNT-Minimal | 0.612±0.048 | 0.635±0.046 | 0.721±0.032 | 32K |

*Results from 3-seed, 5-fold cross-validation*

## Model Training

To retrain models from scratch:

```bash
# Full experimental pipeline with all models
python main.py

# Quick training for testing
python -c "from main import quick_run; quick_run()"
```

## File Formats

- **`.pth`** files contain PyTorch state dictionaries
- Models are saved in CPU format for compatibility
- CUDA tensors are automatically converted during loading

## Deployment Notes

- For web deployment, models are loaded on application startup
- Lite deployment uses pre-computed responses for faster startup
- Models are approximately 2-5MB each when saved
- GPU acceleration available but not required for inference
