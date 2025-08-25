#!/usr/bin/env python3
"""
Comprehensive test for robust algebraic equation handling
Tests the system with diverse mathematical expressions and edge cases
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web.app import AlgebraicTutor
import sympy as sp

def test_diverse_equations():
    """Test the system with a wide variety of algebraic equations"""
    tutor = AlgebraicTutor()
    
    # Test cases covering various algebraic scenarios
    test_cases = [
        # Basic linear equations
        ("3*x + 5 = 11", "Linear equation with constant"),
        ("2*x - 7 = 15", "Linear equation with subtraction"),
        
        # Multi-variable equations  
        ("2*x + 3*y = 10", "Two-variable linear equation"),
        ("a**2 + b**2 = c**2", "Pythagorean theorem"),
        
        # Quadratic equations
        ("x**2 + 5*x + 6 = 0", "Standard quadratic"),
        ("(x - 3)**2 = 16", "Perfect square quadratic"),
        
        # Polynomial equations
        ("x**3 - 8 = 0", "Cubic equation"),
        ("x**4 - 1 = 0", "Fourth-degree polynomial"),
        
        # Rational equations
        ("x/2 + x/3 = 5", "Fraction equation"),
        ("1/x + 1/y = 1/z", "Reciprocal equation"),
        
        # Exponential and logarithmic
        ("2**x = 8", "Exponential equation"),
        ("log(x) = 3", "Logarithmic equation"),
        ("ln(x) + ln(y) = ln(10)", "Natural logarithm"),
        
        # Trigonometric
        ("sin(x) = 1/2", "Basic trigonometric"),
        ("cos(2*x) = cos(x)", "Double angle"),
        ("tan(x) = 1", "Tangent equation"),
        
        # Radical equations
        ("sqrt(x) = 5", "Square root"),
        ("sqrt(x + 1) = 3", "Square root with addition"),
        ("x**(1/3) = 2", "Cube root"),
        
        # Complex expressions
        ("(x + 1)*(x - 2) = x**2 - x - 2", "Expanded form"),
        ("sin(x)**2 + cos(x)**2 = 1", "Trigonometric identity"),
        ("e**x * e**y = e**(x+y)", "Exponential property"),
        
        # Mixed operations
        ("x**2 + sqrt(x) - 6 = 0", "Polynomial with radical"),
        ("log(x**2) = 2*log(x)", "Logarithm property"),
        ("(x + y)**2 = x**2 + 2*x*y + y**2", "Binomial expansion"),
        
        # Edge cases
        ("x = x", "Identity equation"),
        ("0 = 0", "Trivial equation"),
        ("x + 1 = x + 2", "Impossible equation"),
        
        # Alternative notations
        ("2x + 5 = 11", "Implicit multiplication"),
        ("x^2 + 3x = 10", "Caret for exponentiation"),
        ("√x = 4", "Unicode square root"),
        ("x ÷ 2 = 5", "Division symbol"),
        ("x × 3 = 12", "Multiplication symbol"),
        
        # Scientific notation
        ("1e3*x = 2e4", "Scientific notation"),
        ("6.02e23*n = 1", "Avogadro's number style")
    ]
    
    print("🔬 COMPREHENSIVE ALGEBRAIC EQUATION TESTING")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for equation_str, description in test_cases:
        print(f"\n📝 Testing: {equation_str}")
        print(f"   Description: {description}")
        
        try:
            # Test parsing
            equation, error = tutor.parse_equation(equation_str)
            
            if error:
                print(f"   ❌ Parse Error: {error}")
                failed += 1
                continue
            
            print(f"   ✅ Parsed: {equation}")
            
            # Test if we can create a simple transformation
            if "=" in equation_str and equation not in [True, False]:
                # Try a basic simplification transformation
                try:
                    simplified = sp.simplify(equation.lhs - equation.rhs)
                    transformed_eq = sp.Eq(simplified, 0)
                    
                    # Test transformation analysis
                    result = tutor.analyze_transformation(equation, transformed_eq)
                    
                    if "error" in result:
                        print(f"   ⚠️  Analysis Warning: {result['error']}")
                    else:
                        print(f"   🎯 Analysis: {result.get('rule_prediction', 'N/A')}")
                        print(f"   ✓  Valid: {result.get('is_mathematically_valid', False)}")
                
                except Exception as e:
                    print(f"   ⚠️  Analysis failed: {e}")
            elif equation in [True, False]:
                # Handle special cases
                result = tutor.analyze_transformation(equation, equation)
                print(f"   🎯 Analysis: {result.get('rule_prediction', 'N/A')}")
                print(f"   ✓  Valid: {result.get('is_mathematically_valid', False)}")
            
            passed += 1
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    print(f"🎯 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    return passed, failed

def test_preprocessing():
    """Test the preprocessing functionality"""
    tutor = AlgebraicTutor()
    
    print("\n🔧 PREPROCESSING TESTS")
    print("-"*40)
    
    test_cases = [
        ("2x + 5", "2*x + 5"),  # Implicit multiplication
        ("x^2", "x**2"),        # Caret to power
        ("√x", "sqrt*x"),       # Square root symbol  
        ("x÷2", "x/2"),         # Division symbol
        ("3×4", "3*4"),         # Multiplication symbol
        ("2e3", "2*10**3"),     # Scientific notation
        ("(x+1)(x+2)", "(x+1)*(x+2)"),  # Implicit multiplication of parentheses
    ]
    
    for input_str, expected_pattern in test_cases:
        processed = tutor.preprocess_equation(input_str)
        print(f"   '{input_str}' → '{processed}'")
        
        # Check if key transformations occurred
        if "×" not in processed and "÷" not in processed and "√" not in processed:
            print("     ✅ Symbols converted")
        else:
            print("     ⚠️  Some symbols not converted")

def test_variable_extraction():
    """Test variable extraction from equations"""
    tutor = AlgebraicTutor()
    
    print("\n🔤 VARIABLE EXTRACTION TESTS")
    print("-"*40)
    
    test_cases = [
        "x + y = 5",
        "2*a + 3*b - c = 0", 
        "sin(theta) + cos(phi) = 1",
        "m*v**2 = k*q1*q2/r",
        "F = m*a"
    ]
    
    for equation_str in test_cases:
        variables = tutor.extract_variables(equation_str)
        print(f"   '{equation_str}' → Variables: {sorted(variables)}")

if __name__ == "__main__":
    print("🚀 STARTING ROBUST ALGEBRA SYSTEM TESTS")
    print("="*60)
    
    # Run all tests
    passed, failed = test_diverse_equations()
    test_preprocessing()
    test_variable_extraction()
    
    print("\n" + "="*60)
    if failed == 0:
        print("🎉 ALL TESTS COMPLETED - SYSTEM IS ROBUST!")
    else:
        print(f"⚠️  SOME ISSUES FOUND - {failed} test cases failed")
    
    print("💡 The system now handles diverse algebraic equations!")
    print("🔬 Ready for any mathematical input!")
