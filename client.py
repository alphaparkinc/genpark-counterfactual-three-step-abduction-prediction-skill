"""
Counterfactual Three-Step Abduction Prediction Skill Client
Pure Python Standard Library implementation of Pearl's three-step counterfactual algorithm:
1. Abduction: Infer the specific exogenous noise terms U from observed factual evidence.
2. Action: Modify the structural equations according to the counterfactual intervention do(X = x').
3. Prediction: Compute the counterfactual outcome values using the abducted noise in the modified model.
"""

from typing import Dict, Any, List, Tuple


class CounterfactualReasoner:
    def __init__(self):
        self.equations: Dict[str, Dict[str, float]] = {}  # var -> {parent: weight}
        self.topological_order: List[str] = []

    def add_equation(self, var: str, parent_weights: Dict[str, float]):
        """Define linear structural equation: var = sum(weight * parent) + U_var."""
        self.equations[var] = dict(parent_weights)
        if var not in self.topological_order:
            self.topological_order.append(var)

    def abduce(self, evidence: Dict[str, float]) -> Dict[str, float]:
        """Step 1: Abduce exogenous noise variables U given observed factual evidence."""
        noise: Dict[str, float] = {}
        for var in self.topological_order:
            if var in evidence:
                val = evidence[var]
                pred = sum(coeff * evidence[parent] for parent, coeff in self.equations[var].items())
                noise[var] = val - pred
            else:
                noise[var] = 0.0
        return noise

    def counterfactual_predict(self, noise: Dict[str, float], interventions: Dict[str, float]) -> Dict[str, float]:
        """Steps 2 & 3: Apply action do(X = x') and predict counterfactual outcome."""
        cf_state: Dict[str, float] = {}
        for var in self.topological_order:
            if var in interventions:
                cf_state[var] = interventions[var]
            else:
                pred = sum(coeff * cf_state[parent] for parent, coeff in self.equations[var].items())
                cf_state[var] = pred + noise.get(var, 0.0)
        return cf_state
