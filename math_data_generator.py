import numpy as np
import pandas as pd
import json
import random
import os
from collections import defaultdict
import spacy
import sympy as sp
import nltk

# --- 1. HIERARCHICAL MATH STRUCTURE ---

# Mathematical hierarchy from elementary to advanced
MATH_HIERARCHY = {
    1: {
        'level_name': 'Elementary Arithmetic',
        'description': 'Basic numbers, operations, and arithmetic tables',
        'domains': ['numbers', 'addition', 'subtraction', 'multiplication', 'division'],
        'difficulty': 'elementary'
    },
    2: {
        'level_name': 'Elementary Algebra',
        'description': 'Variables, equations, and basic algebraic manipulation',
        'domains': ['variables', 'equations', 'inequalities', 'functions'],
        'difficulty': 'elementary'
    },
    3: {
        'level_name': 'Geometry Fundamentals',
        'description': 'Points, lines, shapes, and basic geometric theorems',
        'domains': ['points_lines', 'triangles', 'circles', 'polygons'],
        'difficulty': 'elementary'
    },
    4: {
        'level_name': 'Number Theory',
        'description': 'Primes, divisibility, modular arithmetic',
        'domains': ['primes', 'factors', 'modular_math', 'number_properties'],
        'difficulty': 'intermediate'
    },
    5: {
        'level_name': 'Calculus Foundations',
        'description': 'Limits, derivatives, integrals, and basic theorems',
        'domains': ['limits', 'derivatives', 'integrals', 'series'],
        'difficulty': 'intermediate'
    },
    6: {
        'level_name': 'Linear Algebra',
        'description': 'Vectors, matrices, linear transformations',
        'domains': ['vectors', 'matrices', 'linear_systems', 'eigenvalues'],
        'difficulty': 'intermediate'
    },
    7: {
        'level_name': 'Abstract Algebra',
        'description': 'Groups, rings, fields, and algebraic structures',
        'domains': ['groups', 'rings', 'fields', 'homomorphisms'],
        'difficulty': 'advanced'
    },
    8: {
        'level_name': 'Topology',
        'description': 'Topological spaces, continuity, connectedness',
        'domains': ['metric_spaces', 'topological_spaces', 'continuity', 'compactness'],
        'difficulty': 'advanced'
    },
    9: {
        'level_name': 'Computer Science Fundamentals',
        'description': 'Binary systems, algorithms, data structures, and computation',
        'domains': ['number_systems', 'boolean_logic', 'algorithms', 'data_structures'],
        'difficulty': 'intermediate'
    },
    10: {
        'level_name': 'Mathematical Theory and Proofs',
        'description': 'Set theory, logic, formal proofs, and mathematical foundations',
        'domains': ['set_theory', 'logic', 'proof_techniques', 'formal_systems'],
        'difficulty': 'advanced'
    }
}

# 10 fundamental locations per level (x-coordinate 1-10, y=1 for level foundation)
FOUNDATION_LOCATIONS = list(range(1, 11))  # 1-10 for 10 fundamental concepts per level

# --- 2. ARITHMETIC TABLES AND ELEMENTARY MATH ---

def generate_arithmetic_tables():
    """Generate basic arithmetic tables for foundation"""
    tables = {}

    # Multiplication table (times tables)
    multiplication = {}
    for i in range(1, 13):  # 1-12 times tables
        for j in range(1, 13):
            multiplication[f"{i} × {j}"] = i * j

    # Addition table
    addition = {}
    for i in range(1, 21):  # 1-20 addition
        for j in range(1, 21):
            if i + j <= 20:  # Keep it elementary
                addition[f"{i} + {j}"] = i + j

    # Subtraction table
    subtraction = {}
    for i in range(1, 21):
        for j in range(1, i+1):  # Only positive results
            subtraction[f"{i} - {j}"] = i - j

    # Division table (basic fractions)
    division = {}
    for i in range(1, 13):
        for j in range(1, 13):
            if i % j == 0:  # Only clean divisions
                division[f"{i} ÷ {j}"] = i // j

    tables['multiplication'] = multiplication
    tables['addition'] = addition
    tables['subtraction'] = subtraction
    tables['division'] = division

    return tables

def generate_number_facts():
    """Generate fundamental number facts"""
    number_facts = {}

    # Number properties
    for i in range(1, 101):
        facts = []
        if i > 1:
            facts.append(f"{i} is {'even' if i % 2 == 0 else 'odd'}")
        if i > 3 and all(i % j != 0 for j in range(2, int(i**0.5) + 1)):
            facts.append(f"{i} is prime")
        if i in [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]:
            root = int(i**0.5)
            facts.append(f"√{i} = {root}")

        if facts:
            number_facts[str(i)] = facts

    return number_facts

# --- 3. HIERARCHICAL MATH CONCEPTS ---

def generate_math_concepts_by_level():
    """Generate mathematical concepts organized by hierarchy level"""
    concepts = {}

    # Level 1: Elementary Arithmetic
    concepts[1] = {
        'numbers': [
            "Natural numbers: 1, 2, 3, ...",
            "Whole numbers: 0, 1, 2, 3, ...",
            "Integers: ..., -2, -1, 0, 1, 2, ...",
            "Rational numbers: fractions a/b where b ≠ 0",
            "Irrational numbers: π, e, √2",
            "Real numbers: all rational and irrational numbers",
            "Complex numbers: a + bi where i² = -1",
            "Zero is neither positive nor negative",
            "One is the multiplicative identity",
            "Numbers can be prime or composite"
        ],
        'addition': [
            "Addition is commutative: a + b = b + a",
            "Addition is associative: (a + b) + c = a + (b + c)",
            "Additive identity: a + 0 = a",
            "Additive inverse: a + (-a) = 0",
            "Carrying over in addition",
            "Regrouping in column addition",
            "Adding fractions requires common denominators",
            "Adding decimals aligns decimal points",
            "Word problems involve addition",
            "Addition of negative numbers"
        ],
        'subtraction': [
            "Subtraction is not commutative: a - b ≠ b - a",
            "Subtraction is the inverse of addition",
            "Borrowing in subtraction",
            "Subtracting fractions requires common denominators",
            "Subtracting negatives: a - (-b) = a + b",
            "Regrouping in column subtraction",
            "Word problems with subtraction",
            "Subtracting decimals",
            "Relationship to addition: a - b = a + (-b)",
            "Zero minus anything is negative"
        ],
        'multiplication': [
            "Multiplication is commutative: a × b = b × a",
            "Multiplication is associative: (a × b) × c = a × (b × c)",
            "Multiplicative identity: a × 1 = a",
            "Multiplicative inverse: a × (1/a) = 1 for a ≠ 0",
            "Zero times anything is zero",
            "Distributive property: a × (b + c) = a×b + a×c",
            "Multiplying fractions: (a/b) × (c/d) = (a×c)/(b×d)",
            "Multiplying decimals",
            "Times tables are multiplication facts",
            "Multiplication of negatives"
        ],
        'division': [
            "Division is not commutative: a ÷ b ≠ b ÷ a",
            "Division by zero is undefined",
            "Division is the inverse of multiplication",
            "Dividing fractions: (a/b) ÷ (c/d) = (a/b) × (d/c)",
            "Long division algorithm",
            "Remainder in division",
            "Dividing decimals",
            "Relationship to fractions",
            "Division of negatives",
            "Rational numbers from division"
        ]
    }

    # Level 2: Elementary Algebra
    concepts[2] = {
        'variables': [
            "Variables represent unknown quantities",
            "Variables are usually letters: x, y, z",
            "Variables can represent numbers",
            "Expressions contain variables and operations",
            "Evaluating expressions substitutes values",
            "Simplifying expressions combines like terms",
            "Variables in equations",
            "Variables in inequalities",
            "Domain and range of variables",
            "Variables in functions"
        ],
        'equations': [
            "Equations state equality: left = right",
            "Solving equations finds unknown values",
            "Linear equations: ax + b = c",
            "Quadratic equations: ax² + bx + c = 0",
            "Systems of equations",
            "Solution sets",
            "Equivalent equations",
            "Extraneous solutions",
            "Word problems as equations",
            "Graphing equations"
        ],
        'inequalities': [
            "Inequalities use <, >, ≤, ≥",
            "Solutions are intervals or regions",
            "Linear inequalities: ax + b > c",
            "Quadratic inequalities",
            "Compound inequalities",
            "Absolute value inequalities",
            "Graphing inequalities",
            "Systems of inequalities",
            "Word problems with inequalities",
            "Properties of inequalities"
        ],
        'functions': [
            "Functions map inputs to outputs",
            "f(x) notation for functions",
            "Domain is input values",
            "Range is output values",
            "Linear functions: f(x) = mx + b",
            "Quadratic functions: f(x) = ax² + bx + c",
            "Function composition: f(g(x))",
            "Inverse functions",
            "One-to-one and onto functions",
            "Graphing functions"
        ]
    }

    # Level 3: Geometry Fundamentals
    concepts[3] = {
        'points_lines': [
            "Point: has no dimension, position only",
            "Line: extends infinitely in two directions",
            "Ray: extends infinitely in one direction",
            "Line segment: finite portion of a line",
            "Parallel lines never intersect",
            "Perpendicular lines intersect at 90°",
            "Intersecting lines cross at a point",
            "Collinear points lie on same line",
            "Concurrent lines meet at a point",
            "Transversal crosses parallel lines"
        ],
        'triangles': [
            "Triangle: three sides, three angles",
            "Sum of angles in triangle is 180°",
            "Pythagorean theorem: a² + b² = c²",
            "Equilateral: all sides equal",
            "Isosceles: two sides equal",
            "Scalene: no sides equal",
            "Right triangle: one 90° angle",
            "Acute triangle: all angles < 90°",
            "Obtuse triangle: one angle > 90°",
            "Area of triangle: (1/2) × base × height"
        ],
        'circles': [
            "Circle: set of points equidistant from center",
            "Radius: distance from center to circumference",
            "Diameter: twice the radius",
            "Circumference: 2πr or πd",
            "Area of circle: πr²",
            "Chord: line segment between two points",
            "Arc: portion of circumference",
            "Sector: portion of circle between radii",
            "Tangent touches circle at one point",
            "Central angle at circle center"
        ],
        'polygons': [
            "Polygon: closed shape with straight sides",
            "Regular polygon: equal sides and angles",
            "Triangle: 3 sides",
            "Quadrilateral: 4 sides",
            "Pentagon: 5 sides",
            "Hexagon: 6 sides",
            "Sum of interior angles: (n-2)×180°",
            "Sum of exterior angles: 360°",
            "Area formulas for different polygons",
            "Regular polygon properties",
            "Convex vs concave polygons"
        ]
    }

    # Level 4: Number Theory
    concepts[4] = {
        'primes': [
            "Prime number > 1 with no divisors except 1 and itself",
            "2 is the only even prime",
            "Prime factorization is unique",
            "Sieve of Eratosthenes finds primes",
            "Twin primes differ by 2",
            "Mersenne primes: 2ⁿ - 1",
            "Prime number theorem approximates prime distribution",
            "Goldbach conjecture: even numbers as sum of two primes",
            "Prime gaps and their distribution",
            "Primality testing algorithms"
        ],
        'factors': [
            "Factors divide a number evenly",
            "Prime factors are prime",
            "Greatest common divisor (GCD)",
            "Least common multiple (LCM)",
            "GCD × LCM = product of numbers",
            "Factor trees show prime factorization",
            "Number of factors depends on prime exponents",
            "Perfect numbers equal sum of proper divisors",
            "Abundant numbers: sum of divisors > number",
            "Deficient numbers: sum of divisors < number"
        ],
        'modular_math': [
            "Modular arithmetic: clock arithmetic",
            "a ≡ b (mod m) means m divides (a-b)",
            "Modular addition and multiplication",
            "Multiplicative inverses modulo m",
            "Chinese Remainder Theorem",
            "Fermat's Little Theorem",
            "Euler's Totient Function φ(n)",
            "RSA encryption uses modular arithmetic",
            "Linear congruential generators",
            "Modular exponentiation"
        ],
        'number_properties': [
            "Even numbers divisible by 2",
            "Odd numbers leave remainder 1 when divided by 2",
            "Perfect squares: n²",
            "Perfect cubes: n³",
            "Fibonacci sequence: each term is sum of previous two",
            "Triangular numbers: n(n+1)/2",
            "Square numbers: n²",
            "Cubic numbers: n³",
            "Catalan numbers",
            "Bell numbers count partitions"
        ]
    }

    # Level 5: Calculus Foundations
    concepts[5] = {
        'limits': [
            "Limit describes function behavior near a point",
            "ε-δ definition of limits",
            "One-sided limits: left and right",
            "Infinite limits",
            "Limits at infinity",
            "Continuity requires limit equals function value",
            "Intermediate Value Theorem",
            "Squeeze Theorem (Sandwich Theorem)",
            "L'Hôpital's Rule for indeterminate forms",
            "Limits of sequences and series"
        ],
        'derivatives': [
            "Derivative measures instantaneous rate of change",
            "Power rule: d/dx[xⁿ] = nxⁿ⁻¹",
            "Product rule: d/dx[uv] = u'v + uv'",
            "Quotient rule: d/dx[u/v] = (u'v - uv')/v²",
            "Chain rule: d/dx[f(g(x))] = f'(g(x)) × g'(x)",
            "Implicit differentiation",
            "Higher-order derivatives",
            "Applications: velocity, acceleration, optimization",
            "Mean Value Theorem",
            "Linear approximation and differentials"
        ],
        'integrals': [
            "Integral finds area under curve",
            "Fundamental Theorem connects derivatives and integrals",
            "Indefinite integrals (antiderivatives)",
            "Definite integrals with limits",
            "Substitution method (u-substitution)",
            "Integration by parts: ∫udv = uv - ∫vdu",
            "Partial fractions",
            "Trigonometric integrals",
            "Improper integrals",
            "Applications: area, volume, work, probability"
        ],
        'series': [
            "Infinite series: sum of infinite terms",
            "Geometric series: ∑arⁿ⁻¹",
            "Telescoping series",
            "p-series: ∑1/nᵖ",
            "Integral test for convergence",
            "Comparison test",
            "Ratio test",
            "Root test",
            "Alternating series test",
            "Power series and Taylor series"
        ]
    }

    # Level 6: Linear Algebra
    concepts[6] = {
        'vectors': [
            "Vectors have magnitude and direction",
            "Vector addition and scalar multiplication",
            "Dot product: u·v = |u||v|cosθ",
            "Cross product in 3D space",
            "Vector spaces and subspaces",
            "Linear independence",
            "Basis and dimension",
            "Coordinate systems",
            "Vector norms and distances",
            "Applications in physics and computer graphics"
        ],
        'matrices': [
            "Matrices are rectangular arrays of numbers",
            "Matrix addition and scalar multiplication",
            "Matrix multiplication: AB ≠ BA generally",
            "Identity matrix I",
            "Inverse matrix A⁻¹",
            "Transpose Aᵀ",
            "Determinant det(A)",
            "Matrix rank",
            "Elementary row operations",
            "LU decomposition"
        ],
        'linear_systems': [
            "Systems of linear equations",
            "Gaussian elimination",
            "Cramer's rule",
            "Matrix form: Ax = b",
            "Solution existence and uniqueness",
            "Homogeneous systems",
            "Particular and general solutions",
            "Underdetermined and overdetermined systems",
            "Least squares solutions",
            "Applications in optimization"
        ],
        'eigenvalues': [
            "Eigenvalues λ satisfy Ax = λx",
            "Characteristic equation det(A - λI) = 0",
            "Eigenvectors corresponding to eigenvalues",
            "Diagonalization A = PDP⁻¹",
            "Complex eigenvalues",
            "Geometric multiplicity vs algebraic multiplicity",
            "Applications in differential equations",
            "Principal component analysis (PCA)",
            "Markov chains and steady states",
            "Quantum mechanics"
        ]
    }

    # Level 7: Abstract Algebra
    concepts[7] = {
        'groups': [
            "Group: set with binary operation satisfying axioms",
            "Closure, associativity, identity, inverses",
            "Abelian (commutative) groups",
            "Cyclic groups generated by single element",
            "Subgroups and normal subgroups",
            "Quotient groups G/N",
            "Group homomorphisms",
            "Isomorphism theorem",
            "Symmetric groups Sₙ",
            "Dihedral groups"
        ],
        'rings': [
            "Ring: set with addition and multiplication",
            "Commutative rings",
            "Rings with identity",
            "Integral domains",
            "Fields: commutative rings with multiplicative inverses",
            "Polynomial rings",
            "Ring homomorphisms",
            "Ideals and quotient rings",
            "Principal ideal domains",
            "Unique factorization domains"
        ],
        'fields': [
            "Fields contain additive and multiplicative inverses",
            "Characteristic of a field",
            "Prime fields: ℚ, ℤ/pℤ",
            "Field extensions",
            "Algebraic and transcendental elements",
            "Finite fields (Galois fields)",
            "Field automorphisms",
            "Galois theory",
            "Splitting fields",
            "Algebraic closures"
        ],
        'homomorphisms': [
            "Structure-preserving maps between algebraic objects",
            "Group homomorphisms preserve operation",
            "Ring homomorphisms preserve addition and multiplication",
            "Kernels and images",
            "First isomorphism theorem",
            "Fundamental homomorphism theorems",
            "Representation theory",
            "Group representations",
            "Character theory",
            "Applications in physics and chemistry"
        ]
    }

    # Level 8: Topology
    concepts[8] = {
        'metric_spaces': [
            "Metric spaces generalize distance",
            "Distance function d(x,y) satisfies axioms",
            "Open balls B(x,r) = {y | d(x,y) < r}",
            "Open and closed sets",
            "Interior, closure, boundary",
            "Continuous functions between metric spaces",
            "Isometries preserve distance",
            "Complete metric spaces",
            "Compact metric spaces",
            "Banach fixed point theorem"
        ],
        'topological_spaces': [
            "Topological spaces defined by open sets",
            "Basis for topology",
            "Subbasis and topology generated by it",
            "Order topology",
            "Product topology",
            "Quotient topology",
            "Connectedness and path-connectedness",
            "Compactness in topological spaces",
            "Hausdorff spaces",
            "Separation axioms (T₁, T₂, T₃, T₄)"
        ],
        'continuity': [
            "Continuous functions preserve open sets",
            "ε-δ definition in metric spaces",
            "Sequential continuity",
            "Uniform continuity",
            "Homeomorphisms: continuous bijections with continuous inverse",
            "Topological invariants",
            "Fundamental group π₁(X)",
            "Homotopy and homotopy equivalence",
            "Retractions and deformation retracts",
            "Covering spaces"
        ],
        'compactness': [
            "Compact sets: every open cover has finite subcover",
            "Sequential compactness",
            "Compact metric spaces are complete and totally bounded",
            "Tychonoff's theorem for product spaces",
            "Compactness in Hausdorff spaces",
            "One-point compactification",
            "Stone-Čech compactification",
            "Local compactness",
            "Paracompactness",
            "Compact-open topology"
        ]
    }

    # Level 9: Computer Science Fundamentals
    concepts[9] = {
        'number_systems': [
            "Binary system: base-2 using digits 0 and 1",
            "Decimal system: base-10 using digits 0-9",
            "Hexadecimal system: base-16 using digits 0-9 and A-F",
            "Octal system: base-8 using digits 0-7",
            "Converting between number systems",
            "Binary addition and subtraction",
            "Hexadecimal arithmetic",
            "Bitwise operations: AND, OR, XOR, NOT",
            "Two's complement representation",
            "Floating-point representation (IEEE 754)"
        ],
        'boolean_logic': [
            "Boolean values: true and false (1 and 0)",
            "Logical AND: true only if both inputs true",
            "Logical OR: true if at least one input true",
            "Logical NOT: inverts the input value",
            "Logical XOR: true if inputs are different",
            "Truth tables for logical operations",
            "De Morgan's laws",
            "Boolean algebra identities",
            "Logic gates: AND, OR, NOT, NAND, NOR, XOR",
            "Combinational vs sequential logic"
        ],
        'algorithms': [
            "Algorithm: step-by-step procedure to solve a problem",
            "Time complexity: Big O notation (O(1), O(n), O(n²), O(log n))",
            "Space complexity: memory usage analysis",
            "Sorting algorithms: bubble sort, insertion sort, quicksort, mergesort",
            "Searching algorithms: linear search, binary search",
            "Recursion: function calling itself",
            "Divide and conquer strategy",
            "Greedy algorithms",
            "Dynamic programming",
            "Algorithm correctness and termination"
        ],
        'data_structures': [
            "Arrays: fixed-size contiguous memory blocks",
            "Linked lists: nodes connected by pointers",
            "Stacks: LIFO (Last In, First Out) structure",
            "Queues: FIFO (First In, First Out) structure",
            "Trees: hierarchical data structure with root and leaves",
            "Binary trees: each node has at most two children",
            "Binary search trees: ordered binary trees",
            "Hash tables: key-value storage with fast lookup",
            "Graphs: nodes connected by edges",
            "Heaps: specialized trees for priority queues"
        ]
    }

    # Level 10: Mathematical Theory and Proofs
    concepts[10] = {
        'set_theory': [
            "Sets: collections of distinct objects",
            "Elements belong to sets: x ∈ A",
            "Subsets: A ⊆ B means every element of A is in B",
            "Power set: set of all subsets of a set",
            "Union: A ∪ B contains elements in A or B or both",
            "Intersection: A ∩ B contains elements in both A and B",
            "Set difference: A - B contains elements in A but not B",
            "Cartesian product: A × B is set of all ordered pairs",
            "Russell's paradox and axiomatic set theory",
            "Zermelo-Fraenkel axioms (ZFC)"
        ],
        'logic': [
            "Propositional logic: statements that are true or false",
            "Logical connectives: ∧ (and), ∨ (or), ¬ (not), → (implies), ↔ (iff)",
            "Truth tables determine logical validity",
            "Predicate logic extends propositional logic with quantifiers",
            "Universal quantifier ∀ (for all)",
            "Existential quantifier ∃ (there exists)",
            "Logical equivalence and tautologies",
            "First-order logic vs higher-order logic",
            "Gödel's incompleteness theorems",
            "Model theory: interpretations and satisfaction"
        ],
        'proof_techniques': [
            "Direct proof: assume premise, derive conclusion",
            "Proof by contradiction: assume negation, derive contradiction",
            "Proof by contrapositive: prove ¬Q → ¬P instead of P → Q",
            "Proof by cases: consider all possible cases",
            "Mathematical induction: prove base case and inductive step",
            "Existence proofs: constructive vs non-constructive",
            "Uniqueness proofs: show at most one object satisfies conditions",
            "Proof by exhaustion: check all possibilities",
            "Diagonalization arguments (Cantor's theorem)",
            "Pigeonhole principle"
        ],
        'formal_systems': [
            "Formal systems consist of axioms, rules of inference, theorems",
            "Axioms are assumed true statements",
            "Rules of inference derive new theorems from existing ones",
            "Consistency: no contradictions derivable",
            "Completeness: all true statements are theorems",
            "Decidability: algorithm exists to determine theoremhood",
            "Peano arithmetic formalizes natural numbers",
            "Zermelo-Fraenkel set theory (ZF)",
            "Category theory: objects and morphisms between them",
            "Type theory and dependent types"
        ]
    }

    return concepts

# --- 4. APL INTEGRATION ---

def generate_apl_expressions():
    """Generate APL expressions for mathematical concepts"""
    apl_expressions = {}

    # Level 1: Basic arithmetic
    apl_expressions[1] = {
        'addition': [
            "2 + 3  ⍝ Addition",
            "1 2 3 + 4 5 6  ⍝ Vector addition",
            "+/⍳10  ⍝ Sum of first 10 numbers",
            "2 + ⍳10  ⍝ Add 2 to each of first 10 numbers"
        ],
        'multiplication': [
            "3 × 4  ⍝ Multiplication",
            "2 3 4 × 5 6 7  ⍝ Vector multiplication",
            "×/⍳5  ⍝ Product of first 5 numbers",
            "2 × ⍳10  ⍝ Multiply each by 2"
        ],
        'subtraction': [
            "7 - 3  ⍝ Subtraction",
            "10 9 8 - 1 2 3  ⍝ Vector subtraction",
            "-/⍳5  ⍝ Alternating sum"
        ],
        'division': [
            "8 ÷ 2  ⍝ Division",
            "10 20 30 ÷ 2 4 5  ⍝ Vector division",
            "÷/⍳4  ⍝ Reciprocal product"
        ]
    }

    # Level 2: Elementary algebra
    apl_expressions[2] = {
        'equations': [
            "2 × ⍵  ⍝ Linear function f(x) = 2x",
            "(⍵*2) + (3×⍵) + 2  ⍝ Quadratic f(x) = x² + 3x + 2",
            "⍵ = 5  ⍝ Equation x = 5",
            "(2×⍵) + 3 = 7  ⍝ Linear equation 2x + 3 = 7"
        ],
        'functions': [
            "{⍵ + 1}  ⍝ Anonymous function adding 1",
            "f ← {⍵ × 2}  ⍝ Define function f(x) = 2x",
            "f 5  ⍝ Apply function to 5",
            "g ← {⍵ * 2}  ⍝ g(x) = x²",
            "f g 3  ⍝ Function composition f(g(3))"
        ]
    }

    # Level 3: Geometry
    apl_expressions[3] = {
        'triangles': [
            "0.5 × 3 × 4  ⍝ Area of triangle: ½ × base × height",
            "3 4 5  ⍝ Pythagorean triple",
            "(3*2) + (4*2) = 5*2  ⍝ Pythagorean theorem verification"
        ],
        'circles': [
            "○ 1  ⍝ π × 1² = π",
            "○ 2  ⍝ π × 2² = 4π",
            "2 × ○ 3  ⍝ Circumference: 2πr"
        ]
    }

    # Level 4: Number theory
    apl_expressions[4] = {
        'primes': [
            "{(⍵>1) ∧ (∧/⍵ ≠ ⍸0=⍵|⍨⍳⍵)}  ⍝ Is prime function",
            "{(⍵>1) ∧ (∧/⍵ ≠ ⍸0=⍵|⍨⍳⍵)} ¨ ⍳20  ⍝ Primes up to 20",
            "+/ {(⍵>1) ∧ (∧/⍵ ≠ ⍸0=⍵|⍨⍳⍵)} ¨ ⍳100  ⍝ Count primes ≤ 100"
        ],
        'factors': [
            "{⍸0=⍵|⍨⍳⍵}  ⍝ Find divisors of number",
            "{⍸0=⍵|⍨⍳⍵} 12  ⍝ Divisors of 12",
            "∧/ 2 3 5 7 ∊ {⍸0=⍵|⍨⍳⍵} 210  ⍝ 210 = 2×3×5×7"
        ]
    }

    # Level 5: Calculus
    apl_expressions[5] = {
        'derivatives': [
            "2 × ⍵  ⍝ Derivative of x² is 2x",
            "3 × ⍵*2  ⍝ Derivative of x³ is 3x²",
            "{(⍵+0.001) - ⍵} ÷ 0.001  ⍝ Numerical derivative approximation"
        ],
        'integrals': [
            "+/ 0.1 × ⍳10  ⍝ Approximate integral using rectangles",
            "0.5 × (+/ 2 × ⍳10) × 0.1  ⍝ Trapezoidal rule approximation",
            "{+/ (⍺ ÷ ⍵) × ⍳⍵}  ⍝ Riemann sum approximation"
        ]
    }

    # Level 6: Linear algebra
    apl_expressions[6] = {
        'matrices': [
            "2 2 ⍴ 1 2 3 4  ⍝ Create 2×2 matrix",
            "A ← 2 2 ⍴ 1 2 3 4 ⋄ B ← 2 2 ⍴ 5 6 7 8 ⋄ A +.× B  ⍝ Matrix multiplication",
            "⌹ 2 2 ⍴ 1 2 3 4  ⍝ Matrix inverse"
        ],
        'vectors': [
            "1 2 3 +.× 4 5 6  ⍝ Dot product",
            "1 0 0 +.× 0 1 0  ⍝ Orthogonal vectors",
            "+/ (⍳10) * 2  ⍝ Scalar multiplication"
        ]
    }

    # Level 9: Computer Science
    apl_expressions[9] = {
        'number_systems': [
            "2 ⊥ 1 0 1 0  ⍝ Binary 1010 to decimal",
            "2 ⊥ ⍳4  ⍝ Binary representation of numbers 1-4",
            "16 ⊥ 10 15  ⍝ Hex AF to decimal",
            "2 8 10 16 ⊥¨ ⊂ 1 0 1 0  ⍝ Convert binary to different bases",
            "2 8 16 ⍕¨ 42  ⍝ Convert 42 to binary, octal, hex"
        ],
        'boolean_logic': [
            "1 ∧ 0  ⍝ Logical AND",
            "1 ∨ 0  ⍝ Logical OR",
            "~ 1 0  ⍝ Logical NOT",
            "1 ≠ 0  ⍝ Logical XOR (not equal)",
            "(⍳2) ∧.∧ (⍳2)  ⍝ AND truth table",
            "(⍳2) ∨.∨ (⍳2)  ⍝ OR truth table"
        ],
        'algorithms': [
            "{⍵[⍋⍵]} 3 1 4 1 5  ⍝ Sort array (grade up)",
            "{⍵[⍒⍵]} 3 1 4 1 5  ⍝ Sort descending",
            "{(⍳≢⍵) ⍷ ⍵⍳⍺}  ⍝ Linear search function",
            "{⌊0.5×≢⍵} ∇ ∇ ⍵  ⍝ Binary search (recursive)",
            "+/⍳10  ⍝ Sum 1 to 10 (linear time)",
            "×/⍳5  ⍝ Factorial (recursive equivalent)"
        ],
        'data_structures': [
            "1 2 3 4 5  ⍝ Array creation",
            "⍳10  ⍝ Index array 0-9",
            "10↑⍳100  ⍝ First 10 elements",
            "⌽ 1 2 3 4 5  ⍝ Reverse array (stack-like)",
            "1,2,3,4,5  ⍝ Concatenate (linked list simulation)",
            "2 2 ⍴ ⍳4  ⍝ 2×2 matrix",
            "(⍳10) ∘.× ⍳10  ⍝ Multiplication table (2D array)"
        ]
    }

    # Level 10: Mathematical Theory and Proofs
    apl_expressions[10] = {
        'set_theory': [
            "⍳10  ⍝ Set of first 10 natural numbers",
            "1 2 3 4 5 ∩ 3 4 5 6 7  ⍝ Intersection {1,2,3,4,5} ∩ {3,4,5,6,7} = {3,4,5}",
            "1 2 3 ∪ 3 4 5  ⍝ Union {1,2,3} ∪ {3,4,5} = {1,2,3,4,5}",
            "~ 1 2 3 4 5 ∊ 3 4 5 6 7  ⍝ Set difference {1,2} (elements not in second set)",
            "2 2 ⍴ ⍳4  ⍝ Cartesian product simulation with matrix",
            "⍴ 1 2 3  ⍝ Cardinality of set {1,2,3} is 3"
        ],
        'logic': [
            "1 ∧ 0  ⍝ Logical AND (conjunction)",
            "1 ∨ 0  ⍝ Logical OR (disjunction)",
            "~ 1  ⍝ Logical NOT (negation)",
            "1 → 0  ⍝ Implication (in APL, comparison gives boolean)",
            "∧/ 1 1 1  ⍝ Universal quantifier (all true)",
            "∨/ 0 0 1  ⍝ Existential quantifier (some true)",
            "(⍳2) ∧.∧ (⍳2)  ⍝ Truth table for AND",
            "(⍳2) →.→ (⍳2)  ⍝ Truth table for implication"
        ],
        'proof_techniques': [
            "+/⍳⍵  ⍝ Mathematical induction base: sum 1 to n = n(n+1)/2",
            "{⍵ ≤ 1: 1 ⋄ ⍵ × ∇ ⍵-1} 5  ⍝ Recursive proof by induction (factorial)",
            "~ (∧/ 2 = +/ 2 ÷⍨ ⍳⍵)  ⍝ Proof by contradiction (no perfect numbers?)",
            "{2|⍵: 'even' ⋄ 'odd'} ¨ ⍳10  ⍝ Proof by cases (even/odd)",
            "⌈/ ⍳10  ⍝ Pigeonhole principle example (maximum in set)",
            "⍳10  ⍝ Exhaustive proof base (check all cases up to 10)"
        ],
        'formal_systems': [
            "⍝ Axioms: basic assumptions",
            "1 2 3  ⍝ Axiom: these are natural numbers",
            "⍝ Rules of inference: modus ponens",
            "{⍺ ∧ (⍺ → ⍵)}  ⍝ If P and (P→Q) then Q",
            "⍝ Consistency check: no contradictions",
            "~ 1 ∧ 0  ⍝ Not both true and false",
            "⍝ Completeness: all truths derivable",
            "∧/ 1 1 1  ⍝ All theorems are true",
            "⍝ Peano axioms simulation",
            "0 , 1 + ⍳9  ⍝ Successor function: n → n+1"
        ]
    }

    return apl_expressions

# --- 4.5 WORD PROBLEM GENERATION ---

def parse_word_problem(problem_text):
    """Parse a word problem to extract entities, relationships, and equations using spaCy and sympy"""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        return {"error": "spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm"}
    
    doc = nlp(problem_text)
    entities = {}
    relationships = []
    
    # Extract numbers and potential variables
    for token in doc:
        if token.pos_ == "NUM":
            entities[token.text] = sp.Symbol(f"x{token.text}")  # Symbolic variable for numbers
        elif token.pos_ == "NOUN" and token.dep_ in ["nsubj", "dobj", "pobj"]:
            entities[token.lemma_] = sp.Symbol(token.lemma_)
    
    # Simple relationship extraction
    for token in doc:
        if token.lemma_ in ["twice", "double"]:
            relationships.append("multiplication by 2")
        elif token.lemma_ in ["sum", "total", "add", "plus"]:
            relationships.append("addition")
        elif token.lemma_ in ["difference", "subtract", "minus"]:
            relationships.append("subtraction")
        elif token.lemma_ in ["product", "multiply", "times"]:
            relationships.append("multiplication")
        elif token.lemma_ in ["quotient", "divide", "divided"]:
            relationships.append("division")
    
    # Basic equation generation (simplified for common patterns)
    equation = None
    if "twice" in problem_text.lower() and len(entities) >= 2:
        vars_list = list(entities.values())
        equation = sp.Eq(vars_list[0], 2 * vars_list[1])
    elif "sum" in problem_text.lower() and len(entities) >= 2:
        vars_list = list(entities.values())
        equation = sp.Eq(vars_list[0], vars_list[1] + vars_list[2]) if len(vars_list) > 2 else None
    
    solution = None
    if equation:
        try:
            solution = sp.solve(equation, list(entities.keys())[0])
        except:
            pass
    
    return {
        "entities": {k: str(v) for k, v in entities.items()},
        "relationships": relationships,
        "equation": str(equation) if equation else None,
        "solution": str(solution) if solution else None
    }

def generate_word_problem_templates():
    """Define templates for different problem types, organized by level/domain"""
    templates = {
        1: {  # Elementary Arithmetic
            "addition": [
                ("John has {a} apples. Mary gives him {b} more. How many does he have?", "{a} + {b}"),
                ("There are {a} birds on a tree. {b} more fly in. How many birds are there?", "{a} + {b}")
            ],
            "subtraction": [
                ("John has {a} apples. He gives {b} to Mary. How many does he have left?", "{a} - {b}"),
                ("There are {a} cookies. {b} are eaten. How many remain?", "{a} - {b}")
            ],
            "multiplication": [
                ("Each box has {a} pencils. There are {b} boxes. How many pencils total?", "{a} * {b}"),
                ("A train has {a} cars, each with {b} seats. How many seats total?", "{a} * {b}")
            ],
            "division": [
                ("{a} pencils are shared equally among {b} children. How many each?", "{a} / {b}"),
                ("A pizza is cut into {b} slices. There are {a} pizzas. How many slices total?", "{a} * {b}")
            ]
        },
        2: {  # Elementary Algebra
            "equations": [
                ("{person1} is {a} years older than {person2}. {person2} is {b} years old. How old is {person1}?", "{b} + {a}"),
                ("A number plus {a} equals {b}. What is the number?", "{b} - {a}")
            ],
            "inequalities": [
                ("John has at least {a} marbles. Mary has {b}. Who has more?", "Compare {a} and {b}"),
                ("The temperature is above {a} degrees. It is {b} degrees. Is it warm?", "{b} > {a}")
            ]
        },
        3: {  # Geometry Fundamentals
            "triangles": [
                ("A triangle has base {a} and height {b}. What is its area?", "0.5 * {a} * {b}"),
                ("Two sides of a triangle are {a} and {b}, angle between is 90 degrees. What is the hypotenuse?", "({a}**2 + {b}**2)**0.5")
            ],
            "circles": [
                ("A circle has radius {a}. What is its area?", "3.14159 * {a}**2"),
                ("Circumference of a circle with diameter {a} is?", "3.14159 * {a}")
            ]
        }
        # Extend for higher levels as needed
    }
    return templates

def generate_word_problems_by_level(templates, num_problems_per_template=5):
    """Generate problems from templates, filling with random values"""
    problems = []
    for level, domains in templates.items():
        for domain, template_list in domains.items():
            for template, formula in template_list:
                for _ in range(num_problems_per_template):
                    # Random values based on level
                    if level == 1:
                        a, b = random.randint(1, 20), random.randint(1, 20)
                    elif level == 2:
                        a, b = random.randint(1, 50), random.randint(1, 50)
                        person1, person2 = random.choice(["John", "Mary", "Bob"]), random.choice(["Alice", "Tom", "Sue"])
                    elif level == 3:
                        a, b = random.randint(1, 20), random.randint(1, 20)
                    else:
                        a, b = random.randint(1, 10), random.randint(1, 10)
                    
                    try:
                        problem = template.format(a=a, b=b, person1=person1 if 'person1' in template else "", person2=person2 if 'person2' in template else "")
                        answer = eval(formula.format(a=a, b=b))
                        parsed = parse_word_problem(problem)
                        problems.append({
                            'level': level,
                            'domain': domain,
                            'problem': problem,
                            'formula': formula,
                            'answer': answer,
                            'parsed': parsed
                        })
                    except:
                        continue  # Skip if formatting fails
    return problems

# --- 5. COORDINATE ASSIGNMENT ---

def assign_math_to_coordinates():
    """Assign mathematical concepts to 3D chess cube coordinates"""
    math_assignments = []

    concepts = generate_math_concepts_by_level()
    apl_expr = generate_apl_expressions()
    templates = generate_word_problem_templates()
    word_problems = generate_word_problems_by_level(templates, num_problems_per_template=2)  # Limit for efficiency

    cluster_id = 0

    # Full 8×8×8 chess cube = 512 locations, plus level 9 and 10
    for z in range(1, 11):  # 10 levels (8 chess cube + 2 extended)
        if z <= 8:  # Standard chess cube levels
            y_range = range(1, 9)
            x_range = range(1, 9)
        else:  # Levels 9-10: Extended coordinates
            y_range = range(1, 9)
            x_range = range(1, 9)  # Could extend this if needed

        for y in y_range:
            for x in x_range:
                level_concepts = concepts.get(z, {})
                level_apl = apl_expr.get(z, {})

                # Get domain for this level (cycle through available domains)
                domain_names = list(level_concepts.keys())
                if not domain_names:
                    continue  # Skip if no concepts for this level

                # Use x coordinate to select domain (cycle through domains)
                domain_idx = (x - 1) % len(domain_names)
                domain = domain_names[domain_idx]
                domain_concepts = level_concepts[domain]
                domain_apl = level_apl.get(domain, [])

                # Create 10 facts per location (to reach 5120+ total facts)
                for fact_num in range(10):
                    cluster_id += 1

                    # Get concept for this fact (cycle through available concepts)
                    concept_idx = ((y-1) * 10 + fact_num) % len(domain_concepts)
                    concept = domain_concepts[concept_idx]

                    # Get APL expression if available
                    apl_code = ""
                    if domain_apl:
                        apl_idx = ((y-1) * 10 + fact_num) % len(domain_apl)
                        apl_code = domain_apl[apl_idx]

                    # Calculate parity (handle level 9 specially)
                    if z <= 8:
                        parity = (x + y + z) % 2
                    else:
                        parity = (x + y) % 2  # Level 9 parity

                    math_assignments.append({
                        'cluster_id': f"math_l{z}_x{x}y{y}_f{fact_num+1}",
                        'level': z,
                        'level_name': MATH_HIERARCHY[z]['level_name'],
                        'domain': domain,
                        'x_coord': x,
                        'y_coord': y,
                        'z_coord': z,
                        'color_parity': parity,
                        'math_concept': concept,
                        'apl_code': apl_code,
                        'difficulty': MATH_HIERARCHY[z]['difficulty']
                    })

    # Add word problems as additional assignments
    for idx, wp in enumerate(word_problems):
        z = wp['level']
        # Assign to fixed coordinates for simplicity (can be randomized later)
        x, y = 1, 1
        if z <= 8:
            parity = (x + y + z) % 2
        else:
            parity = (x + y) % 2
        math_assignments.append({
            'cluster_id': f"word_l{z}_p{idx+1}",
            'level': z,
            'level_name': MATH_HIERARCHY[z]['level_name'],
            'domain': wp['domain'],
            'x_coord': x,
            'y_coord': y,
            'z_coord': z,
            'color_parity': parity,
            'math_concept': wp['problem'],  # Problem text as concept
            'apl_code': '',
            'difficulty': MATH_HIERARCHY[z]['difficulty'],
            'word_problem': True,
            'answer': wp['answer'],
            'parsed': wp['parsed']
        })

    return math_assignments

# --- 6. SAVE MATH DATASET ---

def save_math_dataset(math_assignments):
    """Save the mathematical dataset for training"""
    with open('math_concepts.json', 'w') as f:
        json.dump(math_assignments, f, indent=2)

    # Create CSV format
    rows = []
    for item in math_assignments:
        rows.append({
            'cluster_id': item['cluster_id'],
            'level': item['level'],
            'level_name': item['level_name'],
            'domain': item['domain'],
            'math_concept': item['math_concept'],
            'apl_code': item['apl_code'],
            'difficulty': item['difficulty'],
            'x_coord': item['x_coord'],
            'y_coord': item['y_coord'],
            'z_coord': item['z_coord'],
            'color_parity': item['color_parity']
        })

    df = pd.DataFrame(rows)
    df.to_csv('math_training_data.csv', index=False)

    print(f"Created {len(math_assignments)} mathematical concept clusters")
    print(f"Saved to math_concepts.json and math_training_data.csv")

# --- 7. EXECUTION ---

if __name__ == "__main__":
    print("=== GENERATING HIERARCHICAL MATH DATASET ===")

    # Generate arithmetic tables
    print("Generating arithmetic tables...")
    arithmetic_tables = generate_arithmetic_tables()
    print(f"Created tables for: {list(arithmetic_tables.keys())}")

    # Generate number facts
    print("Generating number facts...")
    number_facts = generate_number_facts()
    print(f"Generated facts for {len(number_facts)} numbers")

    # Generate hierarchical concepts
    print("Generating hierarchical math concepts...")
    math_concepts = generate_math_concepts_by_level()
    print(f"Created concepts for {len(math_concepts)} levels")

    # Generate APL expressions
    print("Generating APL expressions...")
    apl_expressions = generate_apl_expressions()
    print(f"Created APL expressions for {len(apl_expressions)} levels")

    # Generate word problem templates
    print("Generating word problem templates...")
    templates = generate_word_problem_templates()
    print("Generating word problems...")
    word_problems = generate_word_problems_by_level(templates, num_problems_per_template=2)
    print(f"Generated {len(word_problems)} word problems")

    # Assign to coordinates
    print("Assigning concepts to 3D coordinates...")
    math_assignments = assign_math_to_coordinates()

    # Save dataset
    print("Saving mathematical dataset...")
    save_math_dataset(math_assignments)

    print("\n=== MATH DATASET GENERATION COMPLETE ===")
    print(f"Total clusters: {len(math_assignments)}")
    print(f"Levels: 1-10 (Elementary Arithmetic → Mathematical Theory and Proofs)")
    print(f"Domains per level: 4-5 mathematical domains")
    print(f"Clusters per foundation location: 10")
    print(f"Foundation locations per level: 10")
    print(f"Word problems included: {len(word_problems)}")
    print("Files: math_concepts.json, math_training_data.csv")