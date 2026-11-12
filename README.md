# Drisyon Sadhana · AI System Design Starter Kit (`aisys40`)

Welcome to the **AI System Design & Production Architecture (40-Hour Track)** starter repository.

This lab kit contains the reference modules, test harnesses, and automated GitHub Actions CI gates required for your Sadhana Kosha portfolio.

## Structure
```
drisyon-sadhana-aisys40/
├── core/
│   ├── inference_calculator.py  # Model weight VRAM, KV cache, TTFT & TPS math
│   ├── semantic_cache.py        # Cosine similarity semantic caching
│   └── agent_supervisor.py      # ReAct loop step budgets & cycle breaks
├── tests/
│   └── test_gate1.py            # Automated Gate 1 test suite
├── .github/
│   └── workflows/
│       └── gate.yml             # GitHub Actions automated test gate
└── requirements.txt
```

## Local Setup & Gate Verification

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Gate 1 Verification
pytest tests/test_gate1.py -v
```

Once all tests turn green, push your code to your GitHub repository to trigger the automated CI Gate.
