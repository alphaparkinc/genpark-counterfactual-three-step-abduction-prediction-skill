"""
MCP Server for Counterfactual Three-Step Abduction Prediction Skill
"""

import json
import sys
from client import CounterfactualReasoner

def handle_call(name: str, args: dict) -> dict:
    if name == "compute_counterfactual":
        cf = CounterfactualReasoner()
        equations = args.get("equations", {
            "Education": {},
            "Experience": {"Education": 2.0},
            "Salary": {"Education": 10.0, "Experience": 5.0}
        })
        for var, parents in equations.items():
            cf.add_equation(var, parents)
        evidence = args.get("evidence", {"Education": 4.0, "Experience": 10.0, "Salary": 95.0})
        interventions = args.get("interventions", {"Education": 6.0})
        noise = cf.abduce(evidence)
        outcome = cf.counterfactual_predict(noise, interventions)
        return {"abduced_noise": noise, "counterfactual_outcome": outcome}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        res = handle_call(req.get("method"), req.get("params", {}))
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
