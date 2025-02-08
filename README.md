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
```mermaid

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
``

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
            V[("Observed<br/>Data")]
            subgraph LDF ["LogDensityFn"]
                V2[("Unobserved<br/>μ,σ")] --> LP{{"LogProb"}}
                
                subgraph P1 ["Prior μ"]
                    VP1[("Observed<br/>loc=0,scale=1")]
                    subgraph LDP1 ["LogDensityFn"]
                        VP2[("Unobserved<br/>μ")] --> LP1{{"LogProb"}}
                    end
                    VP1 --> LDP1
                end

                subgraph P2 ["Prior σ"]
                    VS1[("Observed<br/>rate=1")]
                    subgraph LDP2 ["LogDensityFn"]
                        VS2[("Unobserved<br/>σ")] --> LP2{{"LogProb"}}
                    end
                    VS1 --> LDP2
                end

                subgraph L ["Likelihood"]
                    VL1[("Unobserved<br/>μ,σ")]
                    subgraph LDL ["LogDensityFn"]
                        VL2[("Observed<br/>Data")] --> LP3{{"LogProb"}}
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

    style V fill:#bfe9,stroke:#333
    style V2 fill:#bbf9,stroke:#333
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
