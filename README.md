# Probabl

```mermaid
flowchart TD
    subgraph "Type Structure"
        direction TB
        subgraph PDF ["ParametricDensityFn"]
            V[("Variable")]
            subgraph LDF ["LogDensityFn"]
                V2[("Variable")] --> LP{{"LogProb"}}
            end
        end
        V --> LDF
    end

    style V fill:#f9f9,stroke:#333
    style V2 fill:#f9f9,stroke:#333
    style LP fill:#ddd9,stroke:#333
    style PDF fill:#bbf9,stroke:#333,stroke-width:2px
    style LDF fill:#bfe9,stroke:#333,stroke-width:2px
```

Key Concepts:
- A Variable can be either observed (data) or unobserved (parameters)
- ParametricDensityFn is a higher-order function that:
  - Takes a Variable as input
  - Returns a LogDensityFn
- LogDensityFn is a function that:
  - Takes another Variable as input
  - Returns a LogProb value
- This structure allows composition of probabilistic models where:
  - Priors follow: observed → (unobserved → LogProb)
  - Likelihoods follow: unobserved → (observed → LogProb)
  - Posteriors combine these patterns
- The same structure applies recursively to build complex models

```mermaid
flowchart TD
    subgraph "Variable Types"
        direction TB
        V[("Variable")]
        OV[("Observed<br/>Data")]
        UV[("Unobserved<br/>Parameters")]
        V --> OV
        V --> UV
    end

    style V fill:#f9f9,stroke:#333
    style OV fill:#bfe9,stroke:#333
    style UV fill:#bbf9,stroke:#333
```

Prior Example: normal(0,1)
- A prior distribution shows how hyperparameters determine parameter uncertainty:
- Observed hyperparameters: loc=0, scale=1 (fixed)
- Unobserved parameter: θ (to be estimated)
- The structure demonstrates:
- Outer function takes observed hyperparameters
- Inner function takes unobserved parameter θ
- Returns LogProb of θ under N(0,1)
- This follows the pattern:
  - observed → (unobserved → LogProb)
- Matches the general prior structure
- Shows how fixed values influence parameter uncertainty

```mermaid
flowchart TD
    subgraph "Prior normal(0,1)"
        direction TB
        subgraph PDF ["ParametricDensityFn"]
            V[("Observed<br/>loc=0, scale=1")]
            subgraph LDF ["LogDensityFn"]
                V2[("Unobserved<br/>θ")] --> LP{{"LogProb"}}
            end
        end
        V --> LDF
    end

    style V fill:#bfe9,stroke:#333
    style V2 fill:#bbf9,stroke:#333
    style LP fill:#ddd9,stroke:#333
    style PDF fill:#bbf9,stroke:#333,stroke-width:2px
    style LDF fill:#bfe9,stroke:#333,stroke-width:2px
```

```mermaid
flowchart TD
    subgraph "Likelihood(θ,Data)"
        direction TB
        subgraph PDF ["ParametricDensityFn"]
            V[("Unobserved<br/>θ")]
            subgraph LDF ["LogDensityFn"]
                V2[("Observed<br/>Data")] --> LP{{"LogProb"}}
            end
        end
        V --> LDF
    end

    style V fill:#bbf9,stroke:#333
    style V2 fill:#bfe9,stroke:#333
    style LP fill:#ddd9,stroke:#333
    style PDF fill:#bbf9,stroke:#333,stroke-width:2px
    style LDF fill:#bfe9,stroke:#333,stroke-width:2px
```

```mermaid
flowchart TD
    subgraph "Posterior P(θ|Data)"
        direction TB
        subgraph PDF ["ParametricDensityFn"]
            V[("Parameters<br/>μ,σ")]
            subgraph LDF ["LogDensityFn"]
                V2[("Data")] --> LP{{"LogProb"}}
                
                subgraph P1 ["ParametricDensityFn"]
                    VP1[("Data<br/>loc=0,scale=1")]
                    subgraph LDP1 ["LogDensityFn"]
                        VP2[("Parameter<br/>μ")] --> LP1{{"LogProb"}}
                    end
                    VP1 --> LDP1
                end

                subgraph P2 ["ParametricDensityFn"]
                    VS1[("Data<br/>rate=1")]
                    subgraph LDP2 ["LogDensityFn"]
                        VS2[("Parameter<br/>σ")] --> LP2{{"LogProb"}}
                    end
                    VS1 --> LDP2
                end

                subgraph L ["ParametricDensityFn"]
                    VL1[("Parameters<br/>μ,σ")]
                    subgraph LDL ["LogDensityFn"]
                        VL2[("Data")] --> LP3{{"LogProb"}}
                    end
                    VL1 --> LDL
                end

                P1 --> LP
                P2 --> LP
                L --> LP
            end
        end
        V --> LDF
    end

    style V fill:#bbf9,stroke:#333
    style V2 fill:#bfe9,stroke:#333
    style VP1 fill:#bfe9,stroke:#333
    style VP2 fill:#bbf9,stroke:#333
    style VS1 fill:#bfe9,stroke:#333
    style VS2 fill:#bbf9,stroke:#333
    style VL1 fill:#bbf9,stroke:#333
    style VL2 fill:#bfe9,stroke:#333
    style LP fill:#ddd9,stroke:#333
    style LP1 fill:#ddd9,stroke:#333
    style LP2 fill:#ddd9,stroke:#333
    style LP3 fill:#ddd9,stroke:#333
    style PDF fill:#bbf9,stroke:#333,stroke-width:2px
    style LDF fill:#bfe9,stroke:#333,stroke-width:2px
    style P1 fill:#bbf9,stroke:#333,stroke-width:2px
    style P2 fill:#bbf9,stroke:#333,stroke-width:2px
    style L fill:#bbf9,stroke:#333,stroke-width:2px
    style LDP1 fill:#bfe9,stroke:#333,stroke-width:2px
    style LDP2 fill:#bfe9,stroke:#333,stroke-width:2px
    style LDL fill:#bfe9,stroke:#333,stroke-width:2px
```
