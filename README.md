# Hiver SDE Intern Assignment — AI Customer Support Agent

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end production AI Support Agent, RAG Retriever, Escalation Engine, and LLM-as-a-Judge Evaluation Harness for **@AmazonHelp** customer support.

---

## ⚡ Quickstart & Interactive Demo (Under 2 Minutes)

When cloned, the repository runs completely **self-contained** without requiring external data downloads.

### 1. Clone & Install Dependencies
```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>
pip install -r requirements.txt
```

### 2. Configure API Key (Optional but Recommended for LLM Replies)
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Note: If no API key is provided, the system seamlessly falls back to the deterministic rule-based generator.)*

### 3. Run the Production AI Agent Pipeline
```bash
python main.py
```
- Type any customer question (e.g. *"My package is delayed"*, *"I got charged twice"*, *"Locked out of my account"*).
- Type **`demo`** to automatically test 3 built-in scenarios.
- Type **`quit`** to exit.

---

## 📊 Headline Benchmark Summary

| Metric / Dimension | Baseline 1 (Majority Class) | Baseline 2 (TF-IDF + LogReg) | Our AI Support Agent |
|---|:---:|:---:|:---:|
| **Intent Accuracy** | 22.50% | 86.00% | **96.00%** |
| **Macro F1 (8 Classes)** | 0.0459 | 0.8661 | **0.9472** |
| **Routing Decision Accuracy** | — | — | **99.50%** |
| **Escalation Recall (Safety Critical)** | — | — | **100.00%** (0% under-escalation) |
| **RAG Retrieval Recall@2** | — | — | **97.00%** |
| **LLM Judge Quality Score (1-5)** | — | — | **4.81 / 5.00** |
| **Human vs. LLM Judge Agreement** | — | — | **Pearson $r = 0.9049$ ($p < 10^{-14}$)** |

---

## 📁 Repository Structure
```
├── main.py                               # Complete production AI agent pipeline & interactive demo
├── stage1_data_exploration.ipynb         # EDA on ~3M tweets & @AmazonHelp selection rationale
├── FINAL_REPORT.md                       # Comprehensive 6-page submission report
├── DECISION_LOG.md                       # 12 non-obvious engineering decisions
├── requirements.txt                      # Python dependencies
├── .env.example                          # Environment variable template
└── data/
    └── processed/
        ├── intent_taxonomy.json          # 8 MECE Intent definitions & risk levels
        ├── golden_evaluation_set.csv     # 200 hand-curated benchmark examples
        ├── baseline_metrics_summary.csv  # Baselines 1 & 2 benchmark results
        ├── ai_agent_golden_predictions.csv # Predictions, confidence & escalation reasons
        ├── evaluation_summary_table.csv  # Multi-dimensional evaluation metrics
        ├── human_vs_llm_judge_40_examples.csv # 40 Human vs. LLM judge ratings
        ├── failure_analysis_table.csv    # Failure modes & proposed fixes
        └── evaluation_metrics_full.json  # Complete metric records
```

---

## 📑 Submission Documentation
- **Final Report**: [`FINAL_REPORT.md`](file:///FINAL_REPORT.md) (Problem framing, brand selection rationale, baselines, headline number analysis, failure modes, and next steps)
- **Decision Log**: [`DECISION_LOG.md`](file:///DECISION_LOG.md) (12 architectural and engineering decisions with context and rationale)
