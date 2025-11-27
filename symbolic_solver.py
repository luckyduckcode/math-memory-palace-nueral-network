import sympy
from sympy.parsing.sympy_parser import parse_expr

class SymbolicSolver:
    """
    Stage C: The Symbolic Solver Engine (CAS).
    Executes the path sequence derived by the MPNN.
    """
    def __init__(self):
        self.history = []

    def execute_path(self, initial_expression_str, path_sequence):
        """
        Executes the symbolic operations defined in the path_sequence.
        
        Args:
            initial_expression_str (str): The starting math expression (e.g., "x**2").
            path_sequence (list): List of concept keys (e.g., ["calculus", "differentiation"]).
            
        Returns:
            final_result (str): The result of the computation.
            history (list): Step-by-step execution log.
        """
        self.history = []
        
        try:
            # 1. Parse Initial State
            # We assume single variable 'x' for PoC simplicity
            x = sympy.symbols('x')
            current_expr = parse_expr(initial_expression_str)
            self.history.append(f"Initial State: {current_expr}")
            
            # 2. Execute Steps
            for step in path_sequence:
                if step == "differentiation" or step == "power_rule":
                    # Apply differentiation
                    # In a real system, 'power_rule' would be a specific sub-rule.
                    # Here we map it to the general diff operation.
                    new_expr = sympy.diff(current_expr, x)
                    self.history.append(f"Applied {step}: {new_expr}")
                    current_expr = new_expr
                    
                elif step == "integration":
                    new_expr = sympy.integrate(current_expr, x)
                    self.history.append(f"Applied {step}: {new_expr}")
                    current_expr = new_expr
                    
                elif step == "calculus" or step == "algebra":
                    # Context setting steps, no operation
                    self.history.append(f"Entered Domain: {step}")
                    
                else:
                    self.history.append(f"Traversed Loci: {step} (No Operation)")
            
            return str(current_expr), self.history
            
        except Exception as e:
            error_msg = f"Symbolic Execution Error: {e}"
            self.history.append(error_msg)
            return "ERROR", self.history

if __name__ == "__main__":
    solver = SymbolicSolver()
    
    # Test 1: Differentiation
    print("--- Test 1: Differentiation ---")
    expr = "x**2"
    path = ["calculus", "differentiation", "power_rule"]
    res, hist = solver.execute_path(expr, path)
    print(f"Result: {res}")
    print("History:")
    for h in hist:
        print(f"  {h}")
        
    # Test 2: Integration
    print("\n--- Test 2: Integration ---")
    expr = "2*x"
    path = ["calculus", "integration"]
    res, hist = solver.execute_path(expr, path)
    print(f"Result: {res}")
    print("History:")
    for h in hist:
        print(f"  {h}")
