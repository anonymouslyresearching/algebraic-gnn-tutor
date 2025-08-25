"""
Flask Web Application for Graph Neural Tutor
Interactive demonstration of algebraic reasoning capabilities
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import sympy as sp
import json
import os
import sys
import random
import logging
from datetime import datetime

# Add parent directory to path to import main modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    DistinctAlgebraicGNN, create_ast_graph, ALGEBRAIC_RULES, 
    NODE_TYPES, UNK_NODE, set_seed
)

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model storage
models = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlgebraicTutor:
    def __init__(self):
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            # Initialize models
            self.main_model = DistinctAlgebraicGNN(encoder_type="main", use_uncertainty=True)
            self.simple_model = DistinctAlgebraicGNN(encoder_type="simple", use_uncertainty=False)
            self.minimal_model = DistinctAlgebraicGNN(encoder_type="minimal", use_uncertainty=False)
            
            # Try to load pre-trained weights if available
            model_paths = {
                'main': 'models/main_model_best.pth',
                'simple': 'models/simple_model_best.pth', 
                'minimal': 'models/minimal_model_best.pth'
            }
            
            for model_name, path in model_paths.items():
                try:
                    if os.path.exists(path):
                        if model_name == 'main':
                            self.main_model.load_state_dict(torch.load(path, map_location=device))
                        elif model_name == 'simple':
                            self.simple_model.load_state_dict(torch.load(path, map_location=device))
                        else:
                            self.minimal_model.load_state_dict(torch.load(path, map_location=device))
                        logger.info(f"Loaded {model_name} model from {path}")
                    else:
                        logger.warning(f"Model file {path} not found. Using random weights.")
                except Exception as e:
                    logger.error(f"Error loading {model_name} model: {e}")
            
            # Move models to device and set to eval mode
            self.main_model.to(device).eval()
            self.simple_model.to(device).eval()
            self.minimal_model.to(device).eval()
            
            self.models_loaded = True
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models_loaded = False
    
    def parse_equation(self, equation_str):
        """Parse equation string into SymPy expression - ROBUST for any algebra"""
        try:
            # Clean and prepare equation string
            equation_str = equation_str.strip()
            
            # Basic validation
            if not equation_str:
                return None, "Empty equation"
            
            # Preprocessing for common input formats
            equation_str = self.preprocess_equation(equation_str)
            
            # Handle different input formats
            if '=' in equation_str:
                # Split by = and parse both sides
                parts = equation_str.split('=')
                if len(parts) != 2:
                    return None, "Equation must have exactly one '=' sign"
                
                left, right = parts[0].strip(), parts[1].strip()
                
                if not left or not right:
                    return None, "Both sides of equation must be non-empty"
                
                # Parse with SymPy - support multiple variables
                symbols = self.extract_variables(equation_str)
                locals_dict = {var: sp.symbols(var) for var in symbols}
                
                left_expr = sp.sympify(left, locals=locals_dict)
                right_expr = sp.sympify(right, locals=locals_dict)
                equation = sp.Eq(left_expr, right_expr)
                
                # Handle special cases like True/False results
                simplified = sp.simplify(equation)
                if simplified in [True, False]:
                    equation = simplified
            else:
                # If no equals sign, treat as expression = 0
                symbols = self.extract_variables(equation_str)
                locals_dict = {var: sp.symbols(var) for var in symbols}
                
                expr = sp.sympify(equation_str, locals=locals_dict)
                equation = sp.Eq(expr, 0)
            
            return equation, None
        except Exception as e:
            return None, f"Failed to parse equation '{equation_str}': {str(e)}"
    
    def preprocess_equation(self, equation_str):
        """Preprocess equation to handle various input formats"""
        import re
        
        # Handle common notation variations
        equation_str = equation_str.replace('^', '**')  # Convert ^ to **
        equation_str = equation_str.replace('√', 'sqrt')  # Convert √ to sqrt
        equation_str = equation_str.replace('÷', '/')   # Convert ÷ to /
        equation_str = equation_str.replace('×', '*')   # Convert × to *
        
        # Handle implicit multiplication, but be careful with function names
        # Don't add * between letters that form function names
        
        # Handle number-variable multiplication: 2x -> 2*x
        equation_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation_str)
        
        # Handle variable-number multiplication: x2 -> x*2  
        equation_str = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', equation_str)
        
        # Handle parentheses multiplication: )( -> )*(
        equation_str = re.sub(r'\)(\()', r')*\1', equation_str)
        
        # Handle variable-parentheses: x( -> x*(, but avoid function calls
        # Only add * if the letter before ( is not part of a function name
        equation_str = re.sub(r'([a-zA-Z])(\()', 
                            lambda m: f"{m.group(1)}*{m.group(2)}" 
                            if m.group(1) not in ['n', 'g', 't', 'p', 's', 'x', 'o', 'r'] 
                            else f"{m.group(1)}{m.group(2)}", 
                            equation_str)
        
        # Handle parentheses-variable: )x -> )*x
        equation_str = re.sub(r'(\))([a-zA-Z])', r'\1*\2', equation_str)
        
        # Handle scientific notation properly (e.g., 1e3 -> 1*10**3)
        # But be careful not to break expressions like e**x
        equation_str = re.sub(r'(\d+)e([+-]?\d+)(?![a-zA-Z])', r'\1*10**\2', equation_str)
        
        return equation_str
    
    def extract_variables(self, equation_str):
        """Extract all variables from equation string"""
        import re
        
        # Find all potential variable names (letters that aren't functions)
        variables = set()
        
        # Common function names to exclude
        functions = {'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'abs', 'factorial', 'pi', 'e'}
        
        # Remove function calls first to avoid extracting variables from inside
        temp_eq = equation_str
        for func in functions:
            temp_eq = re.sub(rf'{func}\([^)]*\)', '', temp_eq)
        
        # Find all letter sequences in the cleaned equation
        for match in re.finditer(r'[a-zA-Z_][a-zA-Z0-9_]*', temp_eq):
            var_name = match.group()
            if var_name.lower() not in functions:
                # Single letters are definitely variables
                if len(var_name) == 1:
                    variables.add(var_name)
                # Multi-letter names: keep if they look like variable names
                elif var_name in ['theta', 'phi', 'alpha', 'beta', 'gamma', 'delta']:
                    variables.add(var_name)
                else:
                    # For other multi-letter sequences, extract individual letters
                    for char in var_name:
                        if char.isalpha() and char not in functions:
                            variables.add(char)
        
        # Add common mathematical variables if not already present
        common_vars = {'x', 'y', 'z'}
        variables.update(common_vars)
        
        # Remove constants and functions
        variables.discard('e')  # Euler's number
        variables.discard('pi') # π
        for func in functions:
            variables.discard(func)
        
        return list(variables)
    
    def analyze_transformation(self, original_eq, transformed_eq, model_type="main"):
        """Analyze algebraic transformation using specified model"""
        try:
            if not self.models_loaded:
                return {"error": "Models not loaded"}
            
            # Select model
            if model_type == "main":
                model = self.main_model
            elif model_type == "simple":
                model = self.simple_model
            else:
                model = self.minimal_model
            
            # Handle special cases (True/False equations)
            if original_eq in [True, False]:
                return {
                    "rule_prediction": "identity" if original_eq else "contradiction",
                    "rule_confidence": [1.0] * 9,
                    "validity_score": 1.0 if original_eq else 0.0,
                    "is_valid": bool(original_eq),
                    "pointer_ranking": [0, 1, 2],
                    "all_rules": list(ALGEBRAIC_RULES.values()),
                    "model_used": model_type,
                    "original_equation": str(original_eq),
                    "transformed_equation": str(transformed_eq),
                    "is_mathematically_valid": bool(original_eq),
                    "symbolic_analysis": "identity" if original_eq else "contradiction"
                }
            
            # Create graph representation
            graph = create_ast_graph(original_eq)
            
            # Add batch dimension
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
            
            # Format results
            result = {
                "rule_prediction": ALGEBRAIC_RULES[rule_pred],
                "rule_confidence": rule_confidence,
                "validity_score": validity,
                "is_valid": validity > 0.5,
                "pointer_ranking": ptr_ranking[:3],  # Top 3 locations
                "all_rules": list(ALGEBRAIC_RULES.values()),
                "model_used": model_type,
                "original_equation": str(original_eq),
                "transformed_equation": str(transformed_eq)
            }
            
            # Enhanced analysis with symbolic validation
            enhanced_result = self.enhance_analysis(result, original_eq, transformed_eq)
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in analyze_transformation: {e}")
            return {"error": str(e)}
    
    def enhance_analysis(self, base_result, original_eq, transformed_eq):
        """Enhance analysis with symbolic validation and better classification"""
        try:
            # Add symbolic analysis
            transformation_type = self.classify_transformation_symbolic(original_eq, transformed_eq)
            is_valid = self.validate_transformation(original_eq, transformed_eq)
            
            # Update the result
            base_result["symbolic_analysis"] = transformation_type
            base_result["is_mathematically_valid"] = is_valid
            
            # If symbolic analysis found a better classification, use it
            if transformation_type and transformation_type != "unknown":
                base_result["rule_prediction"] = transformation_type
            
            return base_result
            
        except Exception as e:
            logger.debug(f"Enhancement failed: {e}")
            return base_result
    
    def classify_transformation_symbolic(self, original_eq, transformed_eq):
        """Use SymPy to classify transformation type symbolically"""
        try:
            orig_expr = original_eq.lhs - original_eq.rhs
            trans_expr = transformed_eq.lhs - transformed_eq.rhs
            
            # Check if they're equivalent
            if sp.simplify(orig_expr - trans_expr) == 0:
                
                # Check for specific transformation types
                orig_expanded = sp.expand(orig_expr)
                if str(orig_expanded) != str(orig_expr) and str(orig_expanded) == str(trans_expr):
                    return "expand"
                
                orig_factored = sp.factor(orig_expr)
                if str(orig_factored) != str(orig_expr) and str(orig_factored) == str(trans_expr):
                    return "factor"
                
                orig_simplified = sp.simplify(orig_expr)
                if str(orig_simplified) != str(orig_expr) and str(orig_simplified) == str(trans_expr):
                    return "simplify"
                
                # Check for variable isolation
                orig_vars = original_eq.free_symbols
                trans_vars = transformed_eq.free_symbols
                
                if len(orig_vars) == 1 and len(trans_vars) == 1:
                    var = list(orig_vars)[0]
                    try:
                        orig_solved = sp.solve(original_eq, var)
                        trans_solved = sp.solve(transformed_eq, var)
                        if orig_solved == trans_solved:
                            return "solve_for_variable"
                    except:
                        pass
                
                return "algebraic_manipulation"
            
            return "unknown_transformation"
            
        except Exception as e:
            logger.debug(f"Symbolic classification failed: {e}")
            return "unknown"
    
    def validate_transformation(self, original_eq, transformed_eq):
        """Validate that the transformation is mathematically correct"""
        try:
            # Method 1: Check algebraic equivalence
            orig_expr = original_eq.lhs - original_eq.rhs
            trans_expr = transformed_eq.lhs - transformed_eq.rhs
            
            difference = sp.simplify(orig_expr - trans_expr)
            if difference == 0:
                return True
            
            # Method 2: Check if they have the same solutions
            try:
                all_vars = original_eq.free_symbols.union(transformed_eq.free_symbols)
                for var in all_vars:
                    orig_solutions = sp.solve(original_eq, var)
                    trans_solutions = sp.solve(transformed_eq, var)
                    
                    # Convert to comparable format
                    if isinstance(orig_solutions, list) and isinstance(trans_solutions, list):
                        orig_set = {sp.simplify(sol) for sol in orig_solutions}
                        trans_set = {sp.simplify(sol) for sol in trans_solutions}
                        if orig_set == trans_set:
                            return True
            except:
                pass
            
            # Method 3: Check equivalence at random points
            try:
                all_vars = list(original_eq.free_symbols.union(transformed_eq.free_symbols))
                if all_vars:
                    for _ in range(5):  # Test 5 random points
                        test_vals = {var: random.randint(-10, 10) for var in all_vars}
                        orig_val = original_eq.subs(test_vals)
                        trans_val = transformed_eq.subs(test_vals)
                        
                        if sp.simplify(orig_val - trans_val) != 0:
                            return False
                    return True
            except:
                pass
            
            return False
            
        except Exception as e:
            logger.debug(f"Validation failed: {e}")
            return False
    
    def get_example_equations(self):
        """Get example equations for demonstration - ROBUST examples"""
        examples = [
            {
                "original": "3*x + 5 = 11",
                "transformed": "3*x = 6",
                "rule": "sub_const",
                "description": "Subtract constant from both sides",
                "explanation": "Subtract 5 from both sides: (3*x + 5) - 5 = 11 - 5"
            },
            {
                "original": "2*x = 8",
                "transformed": "x = 4",
                "rule": "div_coeff",
                "description": "Divide both sides by coefficient",
                "explanation": "Divide both sides by 2: (2*x)/2 = 8/2"
            },
            {
                "original": "x**2 + 5*x + 6 = 0",
                "transformed": "(x + 2)*(x + 3) = 0",
                "rule": "factor",
                "description": "Factor quadratic expression",
                "explanation": "Factor x² + 5x + 6 = (x + 2)(x + 3)"
            },
            {
                "original": "(x + 1)**2 = x**2 + 2*x + 1",
                "transformed": "x**2 + 2*x + 1 = x**2 + 2*x + 1",
                "rule": "expand",
                "description": "Expand perfect square",
                "explanation": "Expand (x + 1)² using (a + b)² = a² + 2ab + b²"
            },
            {
                "original": "sqrt(x) = 4",
                "transformed": "x = 16",
                "rule": "pow_reduce",
                "description": "Square both sides",
                "explanation": "Square both sides: (√x)² = 4²"
            },
            {
                "original": "log(x) + log(y) = log(5)",
                "transformed": "log(x*y) = log(5)",
                "rule": "simplify",
                "description": "Combine logarithms",
                "explanation": "Use log property: log(a) + log(b) = log(ab)"
            },
            {
                "original": "sin(x)**2 + cos(x)**2 = 1",
                "transformed": "1 = 1",
                "rule": "simplify",
                "description": "Apply trigonometric identity",
                "explanation": "Use fundamental trig identity: sin²(x) + cos²(x) = 1"
            },
            {
                "original": "x/3 + x/6 = 5",
                "transformed": "2*x/6 + x/6 = 5",
                "rule": "combine_fracs",
                "description": "Common denominator",
                "explanation": "Convert x/3 to 2x/6 to combine fractions"
            },
            {
                "original": "2*a + 3*b = 7",
                "transformed": "a = (7 - 3*b)/2",
                "rule": "solve_for_variable",
                "description": "Solve for variable a",
                "explanation": "Isolate a: 2a = 7 - 3b, then a = (7 - 3b)/2"
            }
        ]
        return examples

# Initialize tutor
tutor = AlgebraicTutor()

@app.route('/')
def index():
    """Serve main demo page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_equation():
    """API endpoint for equation analysis"""
    try:
        data = request.get_json()
        original = data.get('original', '')
        transformed = data.get('transformed', '')
        model_type = data.get('model', 'main')
        
        # Parse equations
        orig_eq, orig_error = tutor.parse_equation(original)
        trans_eq, trans_error = tutor.parse_equation(transformed)
        
        if orig_error:
            return jsonify({"error": f"Error parsing original equation: {orig_error}"})
        
        if trans_error:
            return jsonify({"error": f"Error parsing transformed equation: {trans_error}"})
        
        # Analyze transformation
        result = tutor.analyze_transformation(orig_eq, trans_eq, model_type)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in analyze_equation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/examples')
def get_examples():
    """Get example equations"""
    return jsonify(tutor.get_example_equations())

@app.route('/api/suggest', methods=['POST'])
def suggest_transformations():
    """Suggest possible transformations for a given equation"""
    try:
        data = request.get_json()
        equation = data.get('equation', '')
        
        # Parse the equation
        parsed_eq, error = tutor.parse_equation(equation)
        if error:
            return jsonify({"error": f"Error parsing equation: {error}"})
        
        # Generate suggestions based on equation structure
        suggestions = []
        eq_str = str(parsed_eq.lhs) + " = " + str(parsed_eq.rhs)
        
        # Add constant operations
        if parsed_eq.lhs.has(sp.Add):
            suggestions.append({
                "rule": "sub_const",
                "description": "Subtract a constant from both sides",
                "example": "If you have +5, subtract 5 from both sides"
            })
        
        # Add coefficient operations
        if parsed_eq.lhs.has(sp.Mul):
            suggestions.append({
                "rule": "div_coeff", 
                "description": "Divide both sides by the coefficient of x",
                "example": "If you have 3*x, divide both sides by 3"
            })
        
        # Add fraction operations
        if parsed_eq.lhs.has(sp.Rational) or str(parsed_eq.lhs).count('/') > 0:
            suggestions.append({
                "rule": "mul_denom",
                "description": "Multiply both sides by the denominator",
                "example": "If you have x/4, multiply both sides by 4"
            })
        
        # Add expansion operations
        if parsed_eq.lhs.has(sp.Mul) and any(isinstance(arg, sp.Add) for arg in parsed_eq.lhs.args):
            suggestions.append({
                "rule": "expand",
                "description": "Expand the product",
                "example": "Expand (x+2)(x+3) using distributive property"
            })
        
        return jsonify({
            "equation": eq_str,
            "suggestions": suggestions
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/rules')
def get_rules():
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
    return jsonify(rules_info)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": tutor.models_loaded,
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Run app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug)
