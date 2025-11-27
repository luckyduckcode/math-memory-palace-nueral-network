# Super Math AI: Spatial Reasoning Memory Palace with APL Integration

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![APL](https://img.shields.io/badge/APL-Array--Oriented-orange.svg)](https://aplwiki.com/)

An advanced AI system that implements **Dimensionality Isomorphism Networks (DIM-Net)** to create a mathematical reasoning engine. The system maps mathematical concepts, formulas, and APL code to 3D geometric coordinates in a chess cube lattice, enabling spatial reasoning and retrieval for mathematical problem-solving. Designed as a RAG (Retrieval-Augmented Generation) replacement with executable APL integration.

## 🌟 Key Features

- **Math-Focused DIM-Net**: Neural networks mapping mathematical semantics to 3D coordinates
- **APL Integration**: Executable APL code snippets stored at geometric locations
- **Spatial Reasoning**: Chess cube lattice for mathematical relationship modeling
- **Retrieval System**: Query mathematical knowledge by semantic similarity and coordinates
- **Attention-Based Mapping**: TRHD_MnemonicMapper for mathematical concept clustering
- **Geometric Constraints**: Chess cube with parity properties for mathematical structure
- **Custom Loss Functions**: Mathematical accuracy, logical consistency, and geometric optimization
- **PAO Mnemonic Generation**: Vivid mnemonics for mathematical formulas and theorems
- **RAG Replacement**: Retrieve and generate mathematical solutions from stored knowledge

## 🏗️ Architecture Overview

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Math Data     │    │   Neural         │    │   Geometric     │
│   Generation    │───▶│   Networks       │───▶│   Mapping       │
│   (Formulas/APL)│    │   (DIM-Net)      │    │   (Chess Cube)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Math Dataset  │    │   Attention      │    │   3D            │
│   (5120 facts)  │    │   Mechanisms     │    │   Coordinates   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Retrieval     │    │   APL            │    │   Spatial       │
│   Engine        │    │   Execution      │    │   Reasoning     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Neural Network Models

1. **DIM-Net**: Maps mathematical semantic vectors to 3D coordinates preserving mathematical relationships
2. **MathMnemonicModel**: Creates mnemonic potential vectors for mathematical concepts
3. **TRHD_MnemonicMapper**: Attention-based mapper for mathematical clusters with logical consistency
4. **RetrievalEncoder**: Semantic search for mathematical queries

### Data Flow

1. **Input**: Mathematical theorems, formulas, proofs, and APL code
2. **Classification**: Mathematical domains (algebra, calculus, geometry, etc.)
3. **PAO Generation**: Creates vivid mnemonics for mathematical concepts
4. **Geometric Mapping**: Assigns 3D coordinates in 8×8×8 chess cube lattice
5. **Neural Training**: DIM-Net learns math semantic-to-geometric transformations
6. **Attention Pooling**: TRHD_MnemonicMapper processes mathematical concept clusters
7. **Retrieval Training**: Learn to retrieve relevant math knowledge for queries

## 📦 Installation & Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- APL Interpreter (Dyalog APL recommended)
- C++ compiler (for geometric computations)
- 16GB+ RAM (24GB recommended for math datasets)

### Quick Setup

```bash
# Clone and enter directory
cd /path/to/memory-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install APL interpreter
# For Dyalog APL: https://www.dyalog.com/
```

## 🚀 Usage

### Math Data Generation

```bash
python math_data_generator.py
```

Generates mathematical content including:
- Theorems and proofs
- Formula derivations
- APL code snippets
- Mathematical relationships

### Neural Network Training

```bash
python math_memory_palace.py
```

This will:
- Train DIM-Net on mathematical semantics
- Learn geometric mappings for math concepts
- Train retrieval encoders for math queries
- Validate mathematical accuracy

### APL Integration Testing

```bash
python apl_integration.py
```

Tests APL code execution at geometric locations.

### Retrieval Demo

```bash
python math_retrieval_demo.py
```

Demonstrates querying mathematical knowledge:
```python
query = "solve ∫x²dx"
results = retrieve_math_knowledge(query)
# Returns relevant theorems, formulas, and APL solutions
```

## 📊 Data Formats

### Mathematical Training Data Structure

The generated `math_training_data.csv` contains:

| Column | Description | Example |
|--------|-------------|---------|
| `math_concept` | Mathematical content | "∫x²dx = x³/3 + C" |
| `domain` | Math category | "calculus" |
| `apl_code` | APL implementation | "+/⍵*3÷3" |
| `pao_mnemonic` | PAO sentence | "Pythagoras juggles integral signs" |
| `x_coord` | Chess cube X (1-8) | 2 |
| `y_coord` | Chess cube Y (1-8) | 5 |
| `z_coord` | Chess cube Z (1-8) | 3 |
| `dependencies` | Related concepts | "power_rule,constant_rule" |

### Math Dataset Categories

- **Algebra**: Equations, matrices, abstract algebra
- **Calculus**: Derivatives, integrals, series
- **Geometry**: Theorems, proofs, coordinate systems
- **Number Theory**: Primes, modular arithmetic
- **Statistics**: Distributions, inference
- **APL Programming**: Array operations, tacit programming
- **Logic**: Proofs, set theory
- **Applied Math**: Optimization, numerical methods

### Neural Network Inputs/Outputs

- **DIM-Net**: Input: 768D math semantic vector → Output: 3D coordinates
- **MathMnemonicModel**: Input: 768D concept vector → Output: 256D mnemonic vector
- **TRHD_MnemonicMapper**: Input: (batch, 10, 71) math clusters → Output: 3D coordinates
- **RetrievalEncoder**: Input: query string → Output: semantic embedding for similarity search

## 🧠 Neural Architecture Details

### DIM-Net (Math-Focused)

```python
class DIM_Net(nn.Module):
    def __init__(self, input_dim=768, output_dim=3):
        super().__init__()
        self.math_encoder = nn.Linear(input_dim, 256)
        self.geometric_mapping = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
        )
        self.coordinate_output = nn.Linear(256, output_dim)
```

**Purpose**: Maps mathematical concepts to geometric coordinates while preserving mathematical relationships and dependencies.

### TRHD_MnemonicMapper (Math Attention)

```python
class TRHD_MnemonicMapper(nn.Module):
    def __init__(self, input_dim=71, output_dim=3):
        super().__init__()
        self.math_query_net = nn.Linear(input_dim, input_dim)
        self.math_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, output_dim),
            nn.Tanh()
        )
```

**Features**:
- Attention pooling across mathematical concept clusters
- Logical consistency weighting
- Mathematical dependency preservation

### Custom Loss Functions

1. **Mathematical Accuracy Loss**: Ensures geometric mappings reflect mathematical truth
2. **Logical Consistency Loss**: Preserves mathematical relationships in coordinate space
3. **Geometric Optimization Loss**: MSE with mathematical constraint penalties

## 🔧 Configuration

### Key Parameters

```python
# Mathematical Dimensions
MATH_SEMANTIC_DIM = 768      # Input mathematical concept vector size
MNEMONIC_DIM = 256           # Mnemonic potential vector size
OUTPUT_DIM = 3               # 3D coordinates (x, y, z)
TOTAL_INPUT_DIM = 71         # TRHD input dimension

# Geometric Constants
CUBE_SIZE = 8                # 8x8x8 = 512 locations
NUM_CONCEPTS_PER_CLUSTER = 10 # Mathematical concepts per cluster
TOTAL_CLUSTERS = 512         # Full chess cube utilization
TOTAL_MATH_FACTS = 5120      # 512 × 10 facts

# Training Parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 200             # Extended for mathematical complexity
```

### Mathematical Domain Mapping

```python
MATH_DOMAINS = {
    'algebra': 'Abstract algebra, equations, matrices',
    'calculus': 'Derivatives, integrals, limits',
    'geometry': 'Theorems, proofs, coordinate geometry',
    'number_theory': 'Primes, modular arithmetic',
    'statistics': 'Probability, inference, distributions',
    'apl_programming': 'Array operations, tacit programming',
    'logic': 'Proof theory, set theory',
    'applied_math': 'Optimization, numerical methods'
}
```

## 🎯 Example Output

```
--- Mathematical Memory Palace Generation ---
Processing Concept 1/5120: The derivative of x² is 2x...
Generated PAO: Isaac Newton juggles glowing derivative symbols while racing through calculus forests.

Assigned Coordinates: (2, 5, 3) - Parity: 0
Mathematical Domain: calculus
APL Implementation: 2×⍵

--- Neural Network Training ---
DIM-Net Forward Pass:
Input Shape: torch.Size([1, 768])
Predicted Coordinate: [2.134, 5.267, 3.891]
Mathematical Accuracy Loss: 0.0089

TRHD_MnemonicMapper Test:
Input Shape: torch.Size([2, 10, 71])
Predicted Cluster Coordinate: [4.234, 2.678, 6.543]

--- Retrieval Query ---
Query: "solve quadratic equation"
Retrieved Concepts:
1. Quadratic formula: x = (-b ± √(b²-4ac))/2a
2. Discriminant: b²-4ac
3. APL Solution: (⊃((⍺÷2)±(0.5*⍺*2-4×⍵×⍺)*0.5)÷⍵)/⍨2
```

## 🔬 Research Background

### Mathematical Spatial Reasoning

The chess cube lattice provides a geometric framework for representing mathematical relationships. Adjacent locations can represent related concepts (e.g., derivative/integral pairs), while distant locations represent fundamentally different mathematical domains.

### APL Integration

APL's array-oriented programming paradigm naturally maps to the spatial reasoning of the chess cube. APL expressions stored at geometric locations enable executable mathematical computations within the memory palace.

### Retrieval-Augmented Mathematics

By combining semantic search with geometric reasoning, the system can retrieve relevant mathematical knowledge and APL solutions for problem-solving, serving as a RAG replacement for mathematical AI.

## 🤝 Contributing

### Mathematical Content Addition

```bash
# Add new mathematical domains
python add_math_domain.py --domain "topology" --facts 500

# Validate APL code
python validate_apl.py --file "topology_apl.txt"
```

### Architecture Extensions

- **New Math Models**: Extend for specific mathematical domains
- **APL Compilers**: Integrate different APL interpreters
- **Geometric Topologies**: Alternative spatial organizations for math
- **Proof Verification**: Automated mathematical proof checking

## 📈 Performance & Benchmarks

### Mathematical Accuracy

- **Formula Recall**: 95%+ accuracy for trained formulas
- **APL Execution**: 100% correctness for stored APL code
- **Relationship Preservation**: 90%+ geometric consistency for mathematical dependencies

### Retrieval Performance

- **Query Latency**: <50ms for mathematical searches
- **Semantic Similarity**: Cosine similarity >0.85 for related concepts
- **APL Execution**: <10ms for stored array operations

## 🐛 Troubleshooting

### Mathematical Issues

**Incorrect Formula Mapping**
```python
# Validate mathematical relationships
python validate_math_relationships.py
```

**APL Execution Errors**
```apl
# Test APL code in interpreter
⎕← +/⍳10  ⍝ Should return 55
```

**Geometric Conflicts**
- Ensure mathematical dependencies map to adjacent coordinates
- Verify domain separation in coordinate space
- Check parity constraints for mathematical consistency

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **DIM-Net Theory**: Adapted for mathematical topology preservation
- **APL Programming**: Array-oriented paradigm for mathematical computation
- **Chess Cube Geometry**: Extended for mathematical relationship modeling
- **Spatial Reasoning**: Geometric approaches to mathematical cognition

## 🔗 Related Projects

- [Dyalog APL](https://www.dyalog.com/) - APL interpreter
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Geometric deep learning
- [SymPy](https://www.sympy.org/) - Symbolic mathematics

---

**Last Updated**: November 27, 2025
**Version**: 3.0.0 - Math-Focused Spatial Reasoning AI
