"""
Demonstration of Counterfactual Three-Step Abduction Prediction Skill
"""

from client import CounterfactualReasoner

def main():
    print("=== Pearl's Three-Step Counterfactual Algorithm Demonstration ===")
    cf = CounterfactualReasoner()

    # Structural model:
    # Education = U_E
    # Experience = 2.0 * Education + U_Exp
    # Salary = 10.0 * Education + 5.0 * Experience + U_Salary
    cf.add_equation("Education", {})
    cf.add_equation("Experience", {"Education": 2.0})
    cf.add_equation("Salary", {"Education": 10.0, "Experience": 5.0})

    # Factual observation:
    factual_evidence = {"Education": 4.0, "Experience": 10.0, "Salary": 95.0}
    print("Factual Observation:")
    for k, v in factual_evidence.items():
        print(f"  {k}: {v}")

    # Step 1: Abduction
    noise = cf.abduce(factual_evidence)
    print("\nStep 1: Abduced Latent Exogenous Factors (U):")
    for k, v in noise.items():
        print(f"  U_{k}: {v:.2f}")

    assert noise["Education"] == 4.0
    assert noise["Experience"] == 2.0
    assert noise["Salary"] == 5.0

    # Step 2 & 3: Action do(Education = 6.0) and Prediction
    print("\nSteps 2 & 3: Action do(Education = 6.0) & Counterfactual Prediction:")
    cf_outcome = cf.counterfactual_predict(noise, interventions={"Education": 6.0})
    for k, v in cf_outcome.items():
        print(f"  Counterfactual {k}: {v:.2f}")

    assert cf_outcome["Education"] == 6.0
    assert cf_outcome["Experience"] == 14.0
    assert cf_outcome["Salary"] == 135.0

    ite = cf_outcome["Salary"] - factual_evidence["Salary"]
    print(f"\nIndividual Treatment Effect (ITE): {ite:+.2f} Salary units")
    assert ite == 40.0

    print("\nCounterfactual Reasoning Verification PASS!")

if __name__ == "__main__":
    main()
