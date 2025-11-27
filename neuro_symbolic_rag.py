import json
import torch
from sllm_wrapper import SLLMWrapper
from mnemonic_model import MPNN, DIM_Net
from symbolic_solver import SymbolicSolver

class NeuroSymbolicRAG:
    """
    The main pipeline orchestrating the 4-stage Neuro-Symbolic architecture.
    """
    def __init__(self, use_ollama=True):
        print("Initializing Neuro-Symbolic RAG System...")
        
        # Stage A & D
        self.use_ollama = use_ollama
        if use_ollama:
            self.sllm = SLLMWrapper()
        else:
            self.sllm = None
        
        # Stage B
        self.dim_net = DIM_Net()
        self.mpnn = MPNN(dim_net=self.dim_net)
        
        # Stage C
        self.solver = SymbolicSolver()
        
        print("System Initialized.")

    def solve(self, user_query, use_simple_parser=True):
        """
        Solves a math problem using the Memory Palace RAG.
        """
        print(f"\n--- Processing Query: '{user_query}' ---")
        
        # --- Stage A: Intent Parsing & Vectorization ---
        print("[Stage A] Parsing Intent...")
        
        if use_simple_parser or not self.use_ollama:
            # Simple rule-based parser for PoC
            intent_data = self._simple_parse(user_query)
        else:
            intent_data = self.sllm.parse_intent(user_query)
            
        if not intent_data:
            return {"error": "Could not parse intent."}
        
        print(f"  Intent: {intent_data.get('intent')}")
        print(f"  Domain: {intent_data.get('domain')}")
        print(f"  Expression: {intent_data.get('expression')}")
        
        expression = intent_data.get('expression')
        
        # --- Stage B: Memory Palace Search ---
        print("[Stage B] Searching Memory Palace...")
        
        # Map intent to goal concept
        intent = intent_data.get('intent')
        goal_concept = self._intent_to_concept(intent)
        coords = self.mpnn.concept_locations.get(goal_concept, (1,1,1))
        
        print(f"  Goal Loci: {goal_concept} at {coords}")
        
        # Path Derivation
        start_concept = intent_data.get('domain', 'calculus')
        if start_concept not in self.mpnn.knowledge_graph:
            start_concept = 'calculus'
            
        print(f"  Deriving Path from '{start_concept}' to '{goal_concept}'...")
        path_sequence = self.mpnn.derive_path(start_concept, goal_concept)
        print(f"  Path Sequence: {path_sequence}")
        
        # --- Stage C: Symbolic Execution ---
        print("[Stage C] Executing Symbolic Path...")
        final_result, history = self.solver.execute_path(expression, path_sequence)
        print(f"  Symbolic Result: {final_result}")
        
        # --- Stage D: Explanation Generation ---
        print("[Stage D] Generating Explanation...")
        explanation = self._generate_explanation(history, final_result, intent_data)
        
        # --- Consolidation: Store in Tier 2 Memory Palace ---
        print("[Consolidation] Storing result in Tier 2 Memory Palace...")
        storage_key, tier2_coords = self.mpnn.consolidate_memory(goal_concept, explanation, path_sequence)
        print(f"  Stored at Tier 2 Loci: {storage_key} (linked to {goal_concept})")
        
        return {
            "query": user_query,
            "intent": intent_data,
            "path": path_sequence,
            "result": final_result,
            "explanation": explanation,
            "tier2_location": storage_key
        }
    
    def _simple_parse(self, query):
        """Simple rule-based parser for common math queries."""
        query_lower = query.lower()
        
        # Detect intent
        if 'derivative' in query_lower or 'differentiate' in query_lower:
            intent = 'differentiation'
            domain = 'calculus'
        elif 'integrate' in query_lower or 'integral' in query_lower:
            intent = 'integration'
            domain = 'calculus'
        elif 'solve' in query_lower:
            intent = 'solve_equation'
            domain = 'algebra'
        else:
            intent = 'unknown'
            domain = 'math'
        
        # Extract expression (simple heuristic)
        expression = None
        if 'of ' in query_lower:
            expression = query.split('of ')[-1].strip()
            # Clean up common endings
            expression = expression.rstrip('.?!')
        elif 'solve ' in query_lower:
            expression = query.split('solve ')[-1].strip()
            expression = expression.rstrip('.?!')
        
        # Convert ^ to ** for Python/SymPy
        if expression:
            expression = expression.replace('^', '**')
        
        return {
            'intent': intent,
            'domain': domain,
            'expression': expression or 'x'
        }
    
    def _intent_to_concept(self, intent):
        """Map intent to goal concept."""
        mapping = {
            'differentiation': 'differentiation',
            'integration': 'integration',
            'solve_equation': 'linear_equations',
            'unknown': 'calculus'
        }
        return mapping.get(intent, 'calculus')
    
    def _generate_explanation(self, history, result, intent_data):
        """Generate explanation (simple version without LLM)."""
        if self.use_ollama and self.sllm:
            try:
                return self.sllm.explain_solution(history, result)
            except:
                pass
        
        # Fallback: Simple template-based explanation
        intent = intent_data.get('intent')
        expr = intent_data.get('expression')
        
        explanation = f"To solve this {intent} problem:\n"
        for step in history:
            explanation += f"  • {step}\n"
        explanation += f"\nFinal Result: {result}"
        
        return explanation

if __name__ == "__main__":
    # Initialize without requiring Ollama
    rag = NeuroSymbolicRAG(use_ollama=False)
    
    # Test queries
    test_queries = [
        "Find the derivative of x^2",
        "Find the derivative of x^3 + 2*x",
        "Integrate 2*x",
    ]
    
    for query in test_queries:
        print("\n" + "="*60)
        result = rag.solve(query, use_simple_parser=True)
        
        if "error" not in result:
            print("\n--- FINAL OUTPUT ---")
            print(f"Query: {result['query']}")
            print(f"Result: {result['result']}")
            print(f"Tier 2 Location: {result['tier2_location']}")
            print(f"\nExplanation:\n{result['explanation']}")
