# 🤝 Contributing to Graph Neural Tutor

Thank you for your interest in contributing to the Graph Neural Tutor project! This document provides guidelines for reviewers, researchers, and potential collaborators.

## 📋 Table of Contents

- [For Reviewers](#for-reviewers)
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Submission Process](#submission-process)

## 👥 For Reviewers

### ATCM 2025 Peer Review

This is an **anonymous submission** for ATCM 2025. When reviewing, please focus on:

#### Technical Merit
- [ ] **Novel Architecture**: Evaluate the GAT + Transformer + Uncertainty approach
- [ ] **Experimental Rigor**: Assess the multi-seed, k-fold cross-validation methodology
- [ ] **Statistical Validity**: Review the bootstrap confidence intervals and significance testing
- [ ] **Baseline Comparisons**: Examine the ablation studies (GNN vs GCN vs MLP)

#### Reproducibility
- [ ] **Code Quality**: All implementation details are available
- [ ] **Environment Setup**: Dependencies and setup instructions are clear
- [ ] **Data Generation**: Synthetic dataset creation is fully specified
- [ ] **Model Training**: Training procedures are deterministic and documented

#### Educational Impact
- [ ] **Practical Application**: Web interface demonstrates real-world utility
- [ ] **Learning Integration**: System provides meaningful feedback for students
- [ ] **Teacher Tools**: Analytical capabilities support educators
- [ ] **Scalability**: Architecture supports classroom deployment

#### Deployment Readiness
- [ ] **Production Setup**: Multiple deployment options are provided
- [ ] **Performance**: System meets real-time interaction requirements
- [ ] **Reliability**: Comprehensive testing ensures robustness
- [ ] **Documentation**: Complete setup and usage instructions

### Evaluation Checklist

To thoroughly evaluate this work:

1. **Quick Demo Test**:
   ```bash
   # Start the local server first: python web/app.py
   curl -X POST "http://localhost:5000/api/analyze" \
     -H "Content-Type: application/json" \
     -d '{"original": "3*x + 5 = 11", "transformed": "3*x = 6", "model": "main"}'
   ```

2. **Local Setup Test**:
   ```bash
   git clone https://github.com/anonymouslyresearching/algebraic-gnn-tutor.git
   cd algebraic-gnn-tutor
   ./QUICKSTART.sh  # or QUICKSTART.bat on Windows
   ```

3. **Reproducibility Test**:
   ```bash
   python main.py  # Should reproduce reported results
   python test_system.py  # Should pass all tests
   ```

4. **Educational Use Test**:
   ```bash
   python web/app.py  # Test the interactive interface
   # Navigate to http://localhost:5000
   ```

## 📜 Code of Conduct

### Research Ethics

- **Academic Integrity**: All code and results are original work
- **Open Science**: Full transparency in methodology and implementation
- **Reproducibility**: All experiments can be independently verified
- **Educational Focus**: Primary goal is advancing mathematics education

### Collaboration Standards

- **Respectful Communication**: Professional and constructive feedback
- **Attribution**: Proper credit for all contributions and prior work
- **Inclusive Environment**: Welcome diverse perspectives and backgrounds
- **Quality Focus**: High standards for code, documentation, and research

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **4GB+ RAM** (8GB+ for full model training)
- **Git** for version control
- **Docker** (optional, for containerized development)

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/anonymouslyresearching/algebraic-gnn-tutor.git
cd algebraic-gnn-tutor

# 2. Automated setup
./QUICKSTART.sh  # Linux/macOS
# or
./QUICKSTART.bat  # Windows

# 3. Verify installation
python test_system.py
```

## 🛠️ Development Setup

### Development Environment

```bash
# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### IDE Configuration

#### VS Code Settings
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
}
```

#### PyCharm Configuration
- Set Python interpreter to `./venv/bin/python`
- Enable pytest as test runner
- Configure Black as code formatter
- Enable flake8 linting

### Docker Development

```bash
# Start development environment
docker-compose --profile dev up

# Services available:
# - Main app: http://localhost:8001
# - Jupyter: http://localhost:8888
# - Redis: localhost:6379
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── test_system.py          # Comprehensive system tests
├── test_robust_algebra.py  # Robustness validation
├── test_models.py          # Model architecture tests
├── test_api.py            # API endpoint tests
└── test_deployment.py     # Deployment configuration tests
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python test_system.py       # Core functionality
python test_robust_algebra.py  # Mathematical robustness
python test_models.py       # Model validation

# Run with coverage
python -m pytest --cov=. --cov-report=html

# Run performance tests
python -m pytest tests/ -k "performance" -v
```

### Writing Tests

Follow these patterns:

```python
def test_equation_parsing():
    """Test equation parsing with various input formats"""
    from web.app import AlgebraicTutor
    
    tutor = AlgebraicTutor()
    
    # Test valid equations
    valid_cases = [
        "3*x + 5 = 11",
        "x/2 = 4", 
        "(x+1)*(x+2) = 0"
    ]
    
    for equation in valid_cases:
        result = tutor.parse_equation(equation)
        assert result is not None, f"Failed to parse: {equation}"
    
    # Test invalid equations
    invalid_cases = [
        "not an equation",
        "x = ",
        "3x + 5 = "  # Missing multiplication
    ]
    
    for equation in invalid_cases:
        result = tutor.parse_equation(equation)
        assert result is None, f"Should reject: {equation}"
```

### Test Data

- Use deterministic random seeds for reproducible tests
- Create minimal test datasets for speed
- Mock external dependencies when possible
- Test edge cases and error conditions

## 📚 Documentation Standards

### Code Documentation

```python
def analyze_transformation(self, original_eq, transformed_eq, model_type="main"):
    """
    Analyze an algebraic transformation using the specified model.
    
    Args:
        original_eq (sympy.Eq): The original algebraic equation
        transformed_eq (sympy.Eq): The transformed equation
        model_type (str): Model to use ('main', 'simple', 'minimal')
    
    Returns:
        dict: Analysis results containing:
            - rule: Predicted transformation rule
            - confidence: Confidence scores for each rule
            - validity: Boolean validity assessment
            - explanation: Human-readable explanation
    
    Raises:
        ValueError: If equations cannot be parsed
        RuntimeError: If model inference fails
    
    Example:
        >>> tutor = AlgebraicTutor()
        >>> orig = sp.Eq(3*x + 5, 11)
        >>> trans = sp.Eq(3*x, 6)
        >>> result = tutor.analyze_transformation(orig, trans)
        >>> print(result['rule'])  # 'sub_const'
    """
```

### README Guidelines

- **Clear Structure**: Use consistent heading hierarchy
- **Quick Start**: Provide immediate working examples
- **Visual Aids**: Include diagrams and screenshots where helpful
- **Troubleshooting**: Address common issues and solutions
- **Links**: Reference related documentation and resources

### API Documentation

Use OpenAPI/Swagger standards:

```python
@app.route('/api/analyze', methods=['POST'])
def analyze_equation():
    """
    Analyze algebraic transformation.
    ---
    tags:
      - Analysis
    parameters:
      - in: body
        name: request
        schema:
          type: object
          required:
            - original
            - transformed
          properties:
            original:
              type: string
              example: "3*x + 5 = 11"
            transformed:
              type: string
              example: "3*x = 6"
            model:
              type: string
              enum: [main, simple, minimal]
              default: main
    responses:
      200:
        description: Analysis completed successfully
        schema:
          type: object
          properties:
            rule:
              type: string
              description: Predicted transformation rule
            confidence:
              type: array
              items:
                type: number
              description: Confidence scores for each rule
    """
```

## 📤 Submission Process

### For Academic Contributions

1. **Research Proposals**:
   - Submit detailed research plan
   - Include literature review and methodology
   - Specify expected contributions and timeline

2. **Code Contributions**:
   - Fork the repository
   - Create feature branch with descriptive name
   - Implement changes with comprehensive tests
   - Submit pull request with detailed description

3. **Documentation Improvements**:
   - Identify areas needing clarification
   - Propose specific improvements
   - Test documentation with fresh environment

### Pull Request Guidelines

#### Title Format
```
[TYPE] Brief description of changes

Types: FEAT, FIX, DOCS, TEST, REFACTOR, PERF, STYLE
```

#### Description Template
```markdown
## Summary
Brief description of what this PR accomplishes.

## Motivation
Why is this change needed? What problem does it solve?

## Changes
- Specific change 1
- Specific change 2
- Specific change 3

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Documentation
- [ ] Code comments updated
- [ ] README updated if needed
- [ ] API documentation updated if needed

## Breaking Changes
List any breaking changes and migration steps.

## Related Issues
Closes #XXX
```

### Code Review Process

1. **Automated Checks**: All CI tests must pass
2. **Peer Review**: At least one reviewer approval required
3. **Documentation Review**: Ensure completeness and clarity
4. **Performance Review**: Assess computational impact
5. **Educational Review**: Evaluate pedagogical value

## 🔍 Quality Standards

### Code Quality

- **PEP 8 Compliance**: Follow Python style guidelines
- **Type Hints**: Use type annotations for clarity
- **Error Handling**: Comprehensive exception handling
- **Performance**: Optimize for educational use cases
- **Security**: Validate all user inputs

### Research Standards

- **Reproducibility**: All experiments must be reproducible
- **Statistical Rigor**: Proper experimental design and analysis
- **Baseline Comparisons**: Compare against relevant benchmarks
- **Ablation Studies**: Isolate individual component contributions
- **Error Analysis**: Detailed analysis of failure cases

### Educational Standards

- **Usability**: Interface must be intuitive for students and teachers
- **Accessibility**: Support for diverse learning needs
- **Feedback Quality**: Provide meaningful, actionable feedback
- **Curriculum Alignment**: Support standard educational objectives

## 📞 Communication

### Anonymous Review Period

During the ATCM 2025 review process:
- Use GitHub issues for technical questions
- Reference paper sections in discussions
- Maintain anonymity in all communications
- Focus on technical and educational merit

### Post-Review Communication

After the review process:
- Join the project Discord for real-time discussion
- Participate in monthly research meetings
- Contribute to educational community forums
- Collaborate on follow-up research

## 🎯 Contribution Areas

### High-Priority Areas

1. **Educational Validation**:
   - Classroom studies with real students
   - Teacher feedback and usability studies
   - Curriculum integration guidelines

2. **Model Improvements**:
   - Advanced GNN architectures
   - Multi-modal learning (text + visual)
   - Personalized learning adaptations

3. **Deployment Enhancements**:
   - Mobile application development
   - Offline functionality
   - Integration with LMS systems

4. **Research Extensions**:
   - Additional mathematical domains
   - Multilingual support
   - Advanced pedagogical features

### Medium-Priority Areas

1. **Performance Optimization**:
   - Model compression techniques
   - Inference acceleration
   - Memory usage optimization

2. **User Experience**:
   - Interface improvements
   - Accessibility enhancements
   - Gamification elements

3. **Analytics and Insights**:
   - Learning analytics dashboard
   - Progress tracking systems
   - Difficulty adaptation algorithms

## 🏆 Recognition

### Contributor Recognition

- **Code Contributors**: Listed in CONTRIBUTORS.md
- **Research Contributors**: Co-authorship on publications (after review period)
- **Educational Contributors**: Recognition in educational materials
- **Community Contributors**: Special badges and privileges

### Academic Credit

- **Significant Contributions**: Invitation to co-author papers
- **Dataset Contributions**: Data contributor acknowledgment
- **Evaluation Studies**: Research collaboration opportunities
- **Tool Development**: Software authorship recognition

---

## 📬 Contact Information

During the anonymous review period, please use:
- **GitHub Issues**: For technical questions and bug reports
- **Email**: anonymous-gnn-tutor@protonmail.com (monitored during review)

Thank you for contributing to advancing AI-powered mathematics education! 🚀
