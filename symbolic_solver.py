import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, simplify, expand, factor, solve, limit, series, Matrix
import re

class AdvancedSymbolicSolver:
    """
    Advanced Symbolic Solver with comprehensive mathematical capabilities.
    Supports: calculus, algebra, linear algebra, trigonometry, limits, series, and more.
    """
    def __init__(self):
        self.history = []
        self.custom_formulas = {}  # Store user-defined formulas
        self.x, self.y, self.z, self.t = symbols('x y z t')
        self.n = symbols('n', integer=True)
        
    def execute_path(self, initial_expression_str, path_sequence):
        """
        Executes the symbolic operations defined in the path_sequence.
        """
        self.history = []
        
        try:
            # Special case: Equation solving
            if '=' in initial_expression_str and any(step in ['linear_equations', 'quadratic_equations', 'solve_equation'] for step in path_sequence):
                # Parse equation
                lhs, rhs = initial_expression_str.split('=')
                lhs_expr = self._parse_expression(lhs.strip())
                rhs_expr = self._parse_expression(rhs.strip())
                equation = lhs_expr - rhs_expr
                
                self.history.append(f"Initial Equation: {lhs_expr} = {rhs_expr}")
                
                # Detect variable
                free_symbols = equation.free_symbols
                if self.x in free_symbols:
                    var = self.x
                elif len(free_symbols) == 1:
                    var = list(free_symbols)[0]
                else:
                    var = self.x
                
                # Solve
                solutions = solve(equation, var)
                self.history.append(f"Solving for {var}")
                self.history.append(f"Solutions: {solutions}")
                
                return str(solutions), self.history
            
            # Parse Initial State
            current_expr = self._parse_expression(initial_expression_str)
            self.history.append(f"Initial State: {current_expr}")
            
            # Execute Steps
            for step in path_sequence:
                if step in ['differentiation', 'power_rule', 'chain_rule', 'product_rule']:
                    new_expr = self._differentiate(current_expr)
                    self.history.append(f"Applied {step}: {new_expr}")
                    current_expr = new_expr
                    
                elif step == 'integration':
                    new_expr = self._integrate(current_expr)
                    self.history.append(f"Applied {step}: {new_expr}")
                    current_expr = new_expr
                    
                elif step == 'simplify':
                    new_expr = simplify(current_expr)
                    self.history.append(f"Simplified: {new_expr}")
                    current_expr = new_expr
                    
                elif step == 'expand':
                    new_expr = expand(current_expr)
                    self.history.append(f"Expanded: {new_expr}")
                    current_expr = new_expr
                    
                elif step == 'factor':
                    new_expr = factor(current_expr)
                    self.history.append(f"Factored: {new_expr}")
                    current_expr = new_expr
                    
                elif step in ['calculus', 'algebra', 'trigonometry', 'linear_algebra']:
                    self.history.append(f"Entered Domain: {step}")
                    
                else:
                    self.history.append(f"Traversed Loci: {step} (No Operation)")
            
            return str(current_expr), self.history
            
        except Exception as e:
            error_msg = f"Symbolic Execution Error: {e}"
            self.history.append(error_msg)
            return "ERROR", self.history
    
    def _parse_expression(self, expr_str):
        """Parse expression with support for multiple variables."""
        # Replace common notations
        expr_str = expr_str.replace('^', '**')
        expr_str = expr_str.replace('√', 'sqrt')
        
        # Try to parse
        try:
            return parse_expr(expr_str, local_dict={
                'x': self.x, 'y': self.y, 'z': self.z, 't': self.t, 'n': self.n
            })
        except:
            # Fallback: try with symbols auto-detection
            return parse_expr(expr_str)
    
    def _differentiate(self, expr):
        """Smart differentiation - detects variable automatically."""
        free_symbols = expr.free_symbols
        if self.x in free_symbols:
            return sympy.diff(expr, self.x)
        elif self.t in free_symbols:
            return sympy.diff(expr, self.t)
        elif len(free_symbols) == 1:
            return sympy.diff(expr, list(free_symbols)[0])
        else:
            return sympy.diff(expr, self.x)
    
    def _integrate(self, expr):
        """Smart integration - detects variable automatically."""
        free_symbols = expr.free_symbols
        if self.x in free_symbols:
            return sympy.integrate(expr, self.x)
        elif self.t in free_symbols:
            return sympy.integrate(expr, self.t)
        elif len(free_symbols) == 1:
            return sympy.integrate(expr, list(free_symbols)[0])
        else:
            return sympy.integrate(expr, self.x)
    
    def solve_equation(self, equation_str, variable='x'):
        """Solve algebraic equations."""
        try:
            # Parse equation (handle = sign)
            if '=' in equation_str:
                lhs, rhs = equation_str.split('=')
                lhs_expr = self._parse_expression(lhs.strip())
                rhs_expr = self._parse_expression(rhs.strip())
                equation = lhs_expr - rhs_expr
            else:
                equation = self._parse_expression(equation_str)
            
            var = symbols(variable)
            solutions = solve(equation, var)
            
            self.history.append(f"Solving: {equation} = 0")
            self.history.append(f"Solutions: {solutions}")
            
            return solutions, self.history
        except Exception as e:
            return f"Error: {e}", self.history
    
    def compute_limit(self, expr_str, point, variable='x'):
        """Compute limits."""
        try:
            expr = self._parse_expression(expr_str)
            var = symbols(variable)
            result = limit(expr, var, point)
            
            self.history.append(f"Computing limit of {expr} as {variable} → {point}")
            self.history.append(f"Result: {result}")
            
            return result, self.history
        except Exception as e:
            return f"Error: {e}", self.history
    
    def taylor_series(self, expr_str, point=0, order=5, variable='x'):
        """Compute Taylor series expansion."""
        try:
            expr = self._parse_expression(expr_str)
            var = symbols(variable)
            result = series(expr, var, point, order)
            
            self.history.append(f"Taylor series of {expr} at {variable}={point}")
            self.history.append(f"Result: {result}")
            
            return result, self.history
        except Exception as e:
            return f"Error: {e}", self.history
    
    def matrix_operations(self, operation, *matrices):
        """Perform matrix operations."""
        try:
            if operation == 'multiply':
                result = matrices[0] * matrices[1]
            elif operation == 'inverse':
                result = matrices[0].inv()
            elif operation == 'determinant':
                result = matrices[0].det()
            elif operation == 'eigenvalues':
                result = matrices[0].eigenvals()
            
            self.history.append(f"Matrix operation: {operation}")
            self.history.append(f"Result: {result}")
            
            return result, self.history
        except Exception as e:
            return f"Error: {e}", self.history
    
    def create_formula(self, name, expression_str, description=""):
        """Store a custom formula for future use."""
        try:
            expr = self._parse_expression(expression_str)
            self.custom_formulas[name] = {
                'expression': expr,
                'description': description,
                'variables': list(expr.free_symbols)
            }
            
            self.history.append(f"Created formula '{name}': {expr}")
            if description:
                self.history.append(f"Description: {description}")
            
            return f"Formula '{name}' created successfully", self.history
        except Exception as e:
            return f"Error: {e}", self.history
    
    def apply_formula(self, formula_name, **substitutions):
        """Apply a stored formula with given values."""
        if formula_name not in self.custom_formulas:
            return f"Formula '{formula_name}' not found", self.history
        
        try:
            formula = self.custom_formulas[formula_name]
            expr = formula['expression']
            
            # Substitute values
            result = expr.subs(substitutions)
            
            self.history.append(f"Applied formula '{formula_name}': {expr}")
            self.history.append(f"Substitutions: {substitutions}")
            self.history.append(f"Result: {result}")
            
            return result, self.history
        except Exception as e:
            return f"Error: {e}", self.history
    
    def list_formulas(self):
        """List all stored formulas."""
        if not self.custom_formulas:
            return "No custom formulas stored yet."
        
        output = "Stored Formulas:\n"
        for name, data in self.custom_formulas.items():
            output += f"\n{name}: {data['expression']}"
            if data['description']:
                output += f"\n  Description: {data['description']}"
            output += f"\n  Variables: {', '.join(str(v) for v in data['variables'])}\n"
        
        return output

if __name__ == "__main__":
    solver = AdvancedSymbolicSolver()
    
    print("=== Advanced Symbolic Solver Demo ===\n")
    
    # Test 1: Differentiation
    print("--- Test 1: Differentiation ---")
    expr = "x**3 + 2*x**2 + 5*x + 1"
    path = ["calculus", "differentiation"]
    res, hist = solver.execute_path(expr, path)
    print(f"Result: {res}")
    
    # Test 2: Integration
    print("\n--- Test 2: Integration ---")
    expr = "3*x**2 + 4*x + 5"
    path = ["calculus", "integration"]
    res, hist = solver.execute_path(expr, path)
    print(f"Result: {res}")
    
    # Test 3: Solve Equation
    print("\n--- Test 3: Solve Equation ---")
    solutions, hist = solver.solve_equation("x**2 - 5*x + 6 = 0")
    print(f"Solutions: {solutions}")
    
    # Test 4: Simplify
    print("\n--- Test 4: Simplify ---")
    expr = "(x + 1)**2"
    path = ["algebra", "expand"]
    res, hist = solver.execute_path(expr, path)
    print(f"Result: {res}")
    
    # Test 5: Create Custom Formula
    print("\n--- Test 5: Create Custom Formula ---")
    msg, hist = solver.create_formula(
        "quadratic", 
        "a*x**2 + b*x + c",
        "General quadratic formula"
    )
    print(msg)
    
    # Test 6: Apply Custom Formula
    print("\n--- Test 6: Apply Custom Formula ---")
    result, hist = solver.apply_formula("quadratic", a=2, b=3, c=1, x=5)
    print(f"Result: {result}")
    
    # Test 7: List Formulas
    print("\n--- Test 7: List Formulas ---")
    print(solver.list_formulas())
