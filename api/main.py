"""
FastAPI Backend for Graph Neural Tutor
Production-ready API server with advanced features
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, validator
import torch
import sympy as sp
import numpy as np
import json
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
import uuid
import redis
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from main import (
    DistinctAlgebraicGNN, create_ast_graph, ALGEBRAIC_RULES,
    set_seed, evaluate_model, create_robust_dataset
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Graph Neural Tutor API",
    description="API for algebraic reasoning using Graph Neural Networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {}
redis_client = None

# Initialize Redis for caching (optional)
try:
    redis_client = redis.Redis(
        host=os.environ.get('REDIS_HOST', 'localhost'),
        port=int(os.environ.get('REDIS_PORT', 6379)),
        decode_responses=True
    )
    redis_client.ping()
    logger.info("Redis connected successfully")
except:
    logger.warning("Redis not available, caching disabled")
    redis_client = None

# Pydantic models
class EquationAnalysisRequest(BaseModel):
    original: str
    transformed: str
    model: str = "main"
    
    @validator('model')
    def validate_model(cls, v):
        if v not in ['main', 'simple', 'minimal']:
            raise ValueError('Model must be one of: main, simple, minimal')
        return v

class EquationAnalysisResponse(BaseModel):
    rule_prediction: str
    rule_confidence: List[float]
    validity_score: float
    is_valid: bool
    pointer_ranking: List[int]
    all_rules: List[str]
    model_used: str
    original_equation: str
    transformed_equation: str
    processing_time: float
    analysis_id: str

class BatchAnalysisRequest(BaseModel):
    equations: List[EquationAnalysisRequest]
    
class DatasetGenerationRequest(BaseModel):
    num_per_rule: int = 250
    num_negative: int = 600
    save_dataset: bool = True

class TrainingRequest(BaseModel):
    dataset_size: int = 2850
    epochs: int = 10
    model_type: str = "main"

class ModelInfo(BaseModel):
    name: str
    architecture: str
    parameters: int
    status: str
    accuracy: Optional[float] = None
    training_time: Optional[str] = None

# Model management class
class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_info = {}
        self.load_models()
    
    def load_models(self):
        """Load all pre-trained models"""
        model_configs = [
            ("main", "GAT + Transformer + Uncertainty", True),
            ("simple", "Graph Convolutional Network", False),
            ("minimal", "Multi-Layer Perceptron", False)
        ]
        
        for model_type, architecture, use_uncertainty in model_configs:
            try:
                model = DistinctAlgebraicGNN(
                    encoder_type=model_type, 
                    use_uncertainty=use_uncertainty
                )
                
                # Try to load pre-trained weights
                model_path = f"models/{model_type}_model_best.pth"
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    logger.info(f"Loaded {model_type} model from {model_path}")
                else:
                    logger.warning(f"No pre-trained weights found for {model_type}, using random weights")
                
                model.to(device).eval()
                self.models[model_type] = model
                
                # Count parameters
                param_count = sum(p.numel() for p in model.parameters())
                
                self.model_info[model_type] = ModelInfo(
                    name=f"GNT-{model_type.capitalize()}",
                    architecture=architecture,
                    parameters=param_count,
                    status="ready"
                )
                
                logger.info(f"Model {model_type} loaded successfully with {param_count:,} parameters")
                
            except Exception as e:
                logger.error(f"Error loading {model_type} model: {e}")
                self.model_info[model_type] = ModelInfo(
                    name=f"GNT-{model_type.capitalize()}",
                    architecture=architecture,
                    parameters=0,
                    status="error"
                )
    
    def get_model(self, model_type: str):
        """Get model by type"""
        return self.models.get(model_type)
    
    def get_model_info(self, model_type: str):
        """Get model information"""
        return self.model_info.get(model_type)

# Initialize model manager
model_manager = ModelManager()

# Utility functions
def parse_equation(equation_str: str):
    """Parse equation string into SymPy expression"""
    try:
        equation_str = equation_str.replace(' ', '').replace('=', ' = ')
        x = sp.symbols('x')
        equation = sp.sympify(equation_str)
        
        if not isinstance(equation, sp.Eq):
            equation = sp.Eq(equation, 0)
        
        return equation, None
    except Exception as e:
        return None, str(e)

def analyze_transformation(original_eq, transformed_eq, model_type: str = "main"):
    """Analyze algebraic transformation using specified model"""
    start_time = datetime.now()
    
    try:
        model = model_manager.get_model(model_type)
        if model is None:
            raise HTTPException(status_code=500, detail=f"Model {model_type} not available")
        
        # Create graph representation
        graph = create_ast_graph(original_eq)
        graph.batch = torch.zeros(len(graph.x), dtype=torch.long)
        graph = graph.to(device)
        
        # Run inference
        with torch.no_grad():
            rule_logits, val_probs, ptr_scores = model(graph)
            
            # Get predictions
            rule_pred = torch.argmax(rule_logits, dim=1).item()
            rule_confidence = torch.softmax(rule_logits, dim=1)[0].cpu().tolist()
            validity = val_probs.item()
            
            # Get pointer ranking
            ptr_scores_list = ptr_scores.cpu().tolist()
            ptr_ranking = sorted(range(len(ptr_scores_list)), 
                               key=lambda x: ptr_scores_list[x], reverse=True)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        result = EquationAnalysisResponse(
            rule_prediction=ALGEBRAIC_RULES[rule_pred],
            rule_confidence=rule_confidence,
            validity_score=validity,
            is_valid=validity > 0.5,
            pointer_ranking=ptr_ranking[:3],
            all_rules=list(ALGEBRAIC_RULES.values()),
            model_used=model_type,
            original_equation=str(original_eq),
            transformed_equation=str(transformed_eq),
            processing_time=processing_time,
            analysis_id=analysis_id
        )
        
        # Cache result if Redis is available
        if redis_client:
            try:
                redis_client.setex(
                    f"analysis:{analysis_id}",
                    3600,  # 1 hour expiry
                    result.json()
                )
            except:
                pass  # Ignore caching errors
        
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_transformation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# API Routes

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve main page"""
    try:
        with open("web/templates/index.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Graph Neural Tutor API</h1><p>Visit /docs for API documentation</p>",
            status_code=200
        )

@app.post("/api/analyze", response_model=EquationAnalysisResponse)
async def analyze_equation(request: EquationAnalysisRequest):
    """Analyze algebraic transformation"""
    # Parse equations
    orig_eq, orig_error = parse_equation(request.original)
    trans_eq, trans_error = parse_equation(request.transformed)
    
    if orig_error:
        raise HTTPException(status_code=400, detail=f"Error parsing original equation: {orig_error}")
    
    if trans_error:
        raise HTTPException(status_code=400, detail=f"Error parsing transformed equation: {trans_error}")
    
    # Analyze transformation
    result = analyze_transformation(orig_eq, trans_eq, request.model)
    return result

@app.post("/api/analyze/batch")
async def batch_analyze_equations(request: BatchAnalysisRequest):
    """Analyze multiple transformations in batch"""
    results = []
    errors = []
    
    for i, eq_request in enumerate(request.equations):
        try:
            # Parse equations
            orig_eq, orig_error = parse_equation(eq_request.original)
            trans_eq, trans_error = parse_equation(eq_request.transformed)
            
            if orig_error or trans_error:
                errors.append({
                    "index": i,
                    "error": orig_error or trans_error
                })
                continue
            
            # Analyze transformation
            result = analyze_transformation(orig_eq, trans_eq, eq_request.model)
            results.append({
                "index": i,
                "result": result
            })
            
        except Exception as e:
            errors.append({
                "index": i,
                "error": str(e)
            })
    
    return {
        "results": results,
        "errors": errors,
        "total_processed": len(results),
        "total_errors": len(errors)
    }

@app.get("/api/analysis/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """Retrieve cached analysis result"""
    if not redis_client:
        raise HTTPException(status_code=404, detail="Caching not available")
    
    try:
        cached_result = redis_client.get(f"analysis:{analysis_id}")
        if cached_result:
            return json.loads(cached_result)
        else:
            raise HTTPException(status_code=404, detail="Analysis result not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/examples")
async def get_examples():
    """Get example equations for demonstration"""
    examples = [
        {
            "original": "3*x + 5 = 11",
            "transformed": "3*x = 6",
            "rule": "sub_const",
            "description": "Subtract constant from both sides"
        },
        {
            "original": "2*x = 8",
            "transformed": "x = 4",
            "rule": "div_coeff",
            "description": "Divide both sides by coefficient"
        },
        {
            "original": "x/4 = 3",
            "transformed": "x = 12",
            "rule": "mul_denom",
            "description": "Multiply both sides by denominator"
        },
        {
            "original": "(x + 2)*(x + 3) = 0",
            "transformed": "x**2 + 5*x + 6 = 0",
            "rule": "expand",
            "description": "Expand the product"
        },
        {
            "original": "x**2 = 16",
            "transformed": "x = 4",
            "rule": "pow_reduce",
            "description": "Take square root of both sides"
        }
    ]
    return examples

@app.get("/api/rules")
async def get_rules():
    """Get all supported algebraic rules"""
    rules_info = {
        "add_const": {
            "name": "Add/Subtract Constant",
            "description": "Add or subtract the same constant from both sides",
            "example": "3x + 5 = 11 → 3x = 6"
        },
        "sub_const": {
            "name": "Subtract Constant",
            "description": "Subtract a constant from both sides",
            "example": "3x - 2 = 10 → 3x = 12"
        },
        "div_coeff": {
            "name": "Divide by Coefficient",
            "description": "Divide both sides by the coefficient of x",
            "example": "3x = 12 → x = 4"
        },
        "mul_denom": {
            "name": "Multiply by Denominator",
            "description": "Multiply both sides by denominator to clear fractions",
            "example": "x/4 = 3 → x = 12"
        },
        "expand": {
            "name": "Expand Expression",
            "description": "Expand products and distribute terms",
            "example": "(x+2)(x+3) → x² + 5x + 6"
        },
        "factor": {
            "name": "Factor Expression",
            "description": "Factor out common terms or expressions",
            "example": "x² + 5x + 6 → (x+2)(x+3)"
        },
        "pow_reduce": {
            "name": "Apply Root/Power",
            "description": "Apply square root or other inverse operations",
            "example": "x² = 16 → x = ±4"
        },
        "combine_fracs": {
            "name": "Combine Fractions",
            "description": "Add or subtract fractions with common denominators",
            "example": "2/3 + 1/6 → 5/6"
        },
        "sub_var": {
            "name": "Subtract Variable Terms",
            "description": "Subtract variable terms from both sides",
            "example": "3x + 5 = 2x + 8 → x = 3"
        }
    }
    return rules_info

@app.get("/api/models")
async def get_models():
    """Get information about available models"""
    return {model_type: info.dict() for model_type, info in model_manager.model_info.items()}

@app.get("/api/models/{model_type}")
async def get_model_info(model_type: str):
    """Get detailed information about a specific model"""
    if model_type not in model_manager.model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model_manager.model_info[model_type]

@app.post("/api/dataset/generate")
async def generate_dataset(request: DatasetGenerationRequest, background_tasks: BackgroundTasks):
    """Generate synthetic dataset"""
    try:
        # Generate dataset in background
        background_tasks.add_task(
            _generate_dataset_task,
            request.num_per_rule,
            request.num_negative,
            request.save_dataset
        )
        
        return {
            "message": "Dataset generation started",
            "num_per_rule": request.num_per_rule,
            "num_negative": request.num_negative,
            "total_expected": request.num_per_rule * 9 + request.num_negative
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _generate_dataset_task(num_per_rule: int, num_negative: int, save_dataset: bool):
    """Background task for dataset generation"""
    try:
        logger.info("Starting dataset generation...")
        dataset = create_robust_dataset(num_per_rule, num_negative)
        
        if save_dataset:
            os.makedirs("datasets", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save dataset metadata
            metadata = {
                "timestamp": timestamp,
                "num_per_rule": num_per_rule,
                "num_negative": num_negative,
                "total_examples": len(dataset),
                "rules": list(ALGEBRAIC_RULES.values())
            }
            
            with open(f"datasets/dataset_{timestamp}_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Save dataset
            torch.save(dataset, f"datasets/dataset_{timestamp}.pt")
            logger.info(f"Dataset saved to datasets/dataset_{timestamp}.pt")
        
        logger.info("Dataset generation completed")
        
    except Exception as e:
        logger.error(f"Error in dataset generation: {e}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(device),
        "models_loaded": len(model_manager.models),
        "redis_available": redis_client is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stats")
async def get_stats():
    """Get API usage statistics"""
    stats = {
        "models_available": len(model_manager.models),
        "rules_supported": len(ALGEBRAIC_RULES),
        "device": str(device),
        "python_version": sys.version,
        "torch_version": torch.__version__
    }
    
    # Add Redis stats if available
    if redis_client:
        try:
            info = redis_client.info()
            stats["redis_stats"] = {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "N/A"),
                "keyspace_hits": info.get("keyspace_hits", 0)
            }
        except:
            stats["redis_stats"] = "unavailable"
    
    return stats

# Mount static files
if os.path.exists("web/static"):
    app.mount("/static", StaticFiles(directory="web/static"), name="static")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=debug,
        log_level="info"
    )
