# Architecture Diagram 

## PaliGemmaWithExpert

```mermaid
graph TD
    A[inputs_embeds List] -->|contains| B[PaliGemma Embeddings]
    A -->|contains| C[Expert Embeddings]
    
    subgraph "For each layer_idx in num_layers"
        B -->|Layer Norm| D[Normalized PaliGemma States]
        C -->|Layer Norm| E[Normalized Expert States]
        
        D -->|Linear Projections| F[Q1, K1, V1]
        E -->|Linear Projections| G[Q2, K2, V2]
        
        F -->|Concatenate| H[Combined Q states]
        G -->|Concatenate| H
        F -->|Concatenate| I[Combined K states]
        G -->|Concatenate| I
        F -->|Concatenate| J[Combined V states]
        G -->|Concatenate| J
        
        H --> K[Apply RoPE]
        I --> K
        
        K --> L[Attention Mechanism]
        J --> L
        
        L -->|Split & Process| M[Updated PaliGemma States]
        L -->|Split & Process| N[Updated Expert States]
    end
    
    M -->|Final Norm| O[Final PaliGemma Output]
    N -->|Final Norm| P[Final Expert Output]
```

## Pi0
---
    π0: A Vision-Language-Action Flow Model for General Robot Control

    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax code](https://github.com/Physical-Intelligence/openpi)

    Designed by Physical Intelligence. Ported from Jax by Hugging Face.
    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │PaliGemma│   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘

---


