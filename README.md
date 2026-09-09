# GenPark Counterfactual Three-Step Abduction Prediction Skill

Pearl's 3-step counterfactual inference engine executing Abduction, Action, and Prediction.

Find more agent capabilities at [GenPark](https://genpark.ai) and the [GenPark MCP Catalog](https://genpark.ai/mcp).

```mermaid
graph TD
    A[Step 1: Factual Evidence E=e] --> B[Abduction: Infer Exogenous Noise U]
    B --> C[Step 2: Action do X=x']
    C --> D[Graph Surgery: Replace X Equation]
    D --> E[Step 3: Prediction: Forward Propagation]
    E --> F[Counterfactual Outcome Y* & ITE]
    style A fill:#e1f5fe
    style B fill:#fff9c4
    style C fill:#ffcdd2
    style E fill:#c8e6c9
    style F fill:#d1c4e9
```

## Features
- Pure Python implementation of Pearl's three-step counterfactual ladder.
- Individual Treatment Effect (ITE) calculation for individual units.
- Zero external dependencies.
