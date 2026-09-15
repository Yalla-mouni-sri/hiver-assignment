# Hiver SDE Intern Assignment — Technical Report
**Project Title**: Production-Grade AI Support Agent & Evaluation Harness for AmazonHelp  
**Author**: SDE Intern Candidate  
**Repository**: [github.com/candidate/hiver-ai-support-agent](https://github.com/)  
**Target Brand**: AmazonHelp (Customer Support on Twitter)  

---

## 1. Executive Summary
This project builds, benchmarks, and critically evaluates an autonomous Customer Support AI Agent for **AmazonHelp** on Twitter. Addressing noisy, multi-turn, real-world customer support data, the system:
1. Classifies incoming customer messages into **8 business-grounded intents** with **96.00% accuracy** (surpassing a Classical ML baseline of 86.00% and a Majority-Class baseline of 22.50%).
2. Retrieves and grounds replies in historical Amazon resolution policies using **RAG vector search** (achieving **97.00% Recall@2** and a **4.86/5.0 Groundedness score**).
3. Executes a high-precision **Escalation Policy Engine** achieving **99.50% routing accuracy** with **0.00% under-escalation** on critical security and financial issues.
4. Validates an **LLM-as-a-Judge evaluation harness** against human expert ratings across 40 test interactions, achieving a statistically significant Pearson correlation of **$r = 0.9049$ ($p < 10^{-14}$)**.

---

## 2. Problem & Brand Selection
### Why AmazonHelp?
We analyzed 3+ million tweets across dozens of top brands (AppleSupport, AmazonHelp, Uber_Support, Delta, Tesco). **AmazonHelp was chosen over AppleSupport and Uber** based on four quantitative criteria:
1. **Diverse Business Domain**: Covers logistics (delivery/tracking), physical e-commerce (damaged goods/returns), digital services (Prime Video/Music), and financial transactions (billing/gift cards). AppleSupport data is overwhelmingly device troubleshooting; Uber is largely ride cancellations.
2. **High Interaction Volume**: Comprises **331,412 tweets** and **111,507 unique multi-turn conversations**, ensuring statistical significance without artificial data sparsity.
3. **Actionable Resolution Paths**: Amazon has clearly defined self-service URLs (`/returns`, `/your-orders`, `/mc`) and clear regulatory boundaries requiring human escalation (card fraud, account lockouts).

### What We Chose *Not* to Build
- **No Direct Financial Execution**: The agent intentionally does *not* execute refunds or card credits autonomously without human confirmation, preventing financial drain attacks.
- **No Multi-Brand Generalization**: The agent is deeply specialized for AmazonHelp; cross-brand generic models dilute intent precision.

---

## 3. Dataset & Preprocessing
The primary raw dataset (`twcs.csv`) is highly noisy. We engineered a 4-step pipeline:
1. **Multi-Turn Thread Reconstruction**: Reconstructed conversational DAGs by mapping `in_response_to_tweet_id` to assemble complete threads: `Customer -> Brand -> Customer -> Brand`.
2. **Root-Turn Extraction**: Discovered that multi-turn follow-up tweets (*"DM sent"*, *"Thanks"*) polluted intent modeling. We isolated the **first inbound customer message** per conversation to capture the true root intent.
3. **Selective Text Cleaning**: Stripped handle mentions (`@AmazonHelp`), URLs, and HTML entities while preserving currency symbols (`$`, `£`, `€`), punctuation questions (`?`), and physical damage tokens.
4. **Volume Filtering**: Discarded non-semantic fragments (<3 words) while preserving emoji sentiment signals in raw storage.

---

## 4. Intent Taxonomy Definition
Rather than relying on unconstrained unsupervised clustering (which fractured delivery issues into 5 redundant clusters), we established an 8-intent **MECE (Mutually Exclusive, Collectively Exhaustive)** taxonomy grounded in real Amazon business domains:

| Intent Code | Intent Name | Business Definition | Risk Level | Default Action |
|---|---|---|---|---|
| `DELIVERY_TRACKING` | Order Delivery & Tracking | Inquiries on delays, carrier tracking status, or missing packages. | LOW | `AUTO_HANDLE` |
| `RETURNS_REFUNDS` | Returns, Refunds & Replacements | Return requests, refund status, damaged/wrong/defective items. | MEDIUM | `AUTO_HANDLE` |
| `CANCELLATION_MODIFICATION` | Order Cancellation & Changes | Pre-dispatch cancellation or delivery address modifications. | MEDIUM | `AUTO_HANDLE` |
| `PAYMENT_BILLING` | Payment, Billing & Charges | Duplicate charges, unauthorized deductions, invoices, gift cards. | HIGH | `ESCALATE_HUMAN` |
| `SUBSCRIPTION_PRIME` | Prime & Digital Subscriptions | Prime membership, auto-renewals, Prime Video streaming bugs. | MEDIUM | `AUTO_HANDLE` |
| `ACCOUNT_SECURITY` | Account Access & Security | 2FA/OTP failures, locked/suspended accounts, suspected hacking. | CRITICAL | `ESCALATE_HUMAN` |
| `PRODUCT_INQUIRY` | Product & Stock Inquiry | Stock availability, compatibility, warranty coverage, pricing. | LOW | `AUTO_HANDLE` |
| `FEEDBACK_CHITCHAT` | General Feedback & Pleasantries | Compliments, casual greetings, or general brand frustration. | LOW | `AUTO_HANDLE` |

---

## 5. System Architecture
The production agent runs as a cohesive pipeline:

```
Incoming Message
       ↓
[Semantic Intent Classifier] (Multi-tier N-gram & Anchor Head)
       ↓
[Historical Case Retriever] (RAG Vector Index, Top-2 Resolutions)
       ↓
[Escalation Decision Engine] (Intent Risk + Confidence + Discrepancy Rules)
       ↓
   ┌───────────────────────┴───────────────────────┐
   ↓                                               ↓
AUTO_HANDLE (57.0%)                       ESCALATE_HUMAN (43.0%)
Grounded Self-Service Resolution          Secure DM Routing + Stated Reason
```

1. **Semantic Classifier**: Calibrated cosine similarity over domain anchors + linear embedding head.
2. **RAG Vector Retriever**: Indexes verified past customer resolutions to ground replies in official policies.
3. **Escalation Engine**: Evaluates financial liability, account takeover risks, driver misconduct, and SLA breaches.
4. **Grounded Generator**: Assembles empathetic, brand-compliant replies with verified navigation links.

---

## 6. Evaluation Methodology
### The Golden Evaluation Set ($N=200$)
We constructed a trusted **200-sample hand-labeled benchmark**:
- **Stratified Sampling**: 45 Delivery, 35 Returns/Refunds, 25 Cancellation, 25 Payment, 25 Prime, 20 Security, 15 Product, 10 Feedback.
- **Decision Ground Truth**: 115 `AUTO_HANDLE` (57.5%) vs. 85 `ESCALATE_HUMAN` (42.5%).
- **Stated Rationales**: Every example includes an explicit reason and grounded historical evidence.

### LLM-as-a-Judge Rubric & Human Calibration
We designed a 5-dimension rubric (1.0 to 5.0 scale) for **Correctness**, **Groundedness**, **Relevance**, **Helpfulness**, and **Tone**. We benchmarked 40 interactions with independent human vs. LLM judge ratings.

---

## 7. Results & Baseline Comparisons

| Model / System | Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1 | Decision Accuracy |
|---|---|---|---|---|---|---|
| **Baseline 1** | Majority Class (Trivial) | 22.50% | 0.0281 | 0.1250 | 0.0459 | — |
| **Baseline 2** | TF-IDF + Logistic Regression | 86.00% | 0.9075 | 0.8595 | 0.8661 | — |
| **Our AI Support Agent** | **Semantic Classifier + RAG + Escalation Engine** | **96.00%** | **0.9688** | **0.9400** | **0.9472** | **99.50%** |

### Escalation & Safety Metrics:
- **Routing Decision Accuracy**: **99.50%**
- **Escalation Precision**: **0.9884** | **Escalation Recall**: **1.0000**
- **Under-Escalation Rate**: **0.00%** (Zero safety-critical financial or security cases leaked to auto-handling).
- **Over-Escalation Rate**: **0.87%** (High operational efficiency).

### Human vs. LLM Judge Agreement ($N=40$):
- **Overall Pearson Correlation**: **$r = 0.9049$ ($p = 1.14 	imes 10^{-15}$)**
- **Correctness Correlation**: **$r = 0.8874$**
- **Mean Absolute Error (MAE)**: **$0.105$ points** on a 5.0 scale.

---

## 8. Failure Analysis (Top 5 Failure Modes)
1. **Multi-Intent Collisions**: Query bundles root cause with desired resolution (*"charged for Prime renewal, want refund"*). *Fix*: Hierarchical entity-action classification.
2. **Context-Free Follow-Ups**: Short queries (*"Still not working"*) lack domain nouns. *Fix*: Sliding-window conversational history concatenation.
3. **Sarcasm / Lexical Inversion**: Sarcastic praise (*"Great job Amazon, plates arrived in pieces"*) confuses sentiment. *Fix*: Prioritize physical condition damage tokens over adjectives.
4. **False Delivery Confirmations**: Carrier marked delivered, but package is missing. *Fix*: Explicit negation discrepancy detector between carrier status and user statement.
5. **Fulfillment State Blindness**: Customer asks to change address on an order that already shipped. *Fix*: Tool-augmented live API integration checking `OrderStatus == SHIPPED`.

---

## 9. What is Misleading About My Headline Number?
*(Mandatory Self-Critical Evaluation Section)*

While **96.00% intent accuracy** and **99.50% routing accuracy** appear outstanding, reporting them in isolation is misleading for four reasons:
1. **Stratified vs. In-the-Wild Imbalance**: Our Golden Set is stratified across all 8 intents (10–22% each). In raw Twitter data, Delivery queries represent >50% of volume. Real-world accuracy would skew heavily toward delivery performance.
2. **First-Turn Selection Bias**: Our evaluation is conducted on the *opening customer message*. Performance drops significantly on turn 3 or 4 where context is implicit.
3. **No Live Order DB Grounding**: The agent does not query a live ERP/database to verify if an order number actually exists or if tracking has genuinely stalled.
4. **Twitter-Specific Format**: The system assumes tweet-length queries (<280 characters). Long, rambling customer emails would degrade keyword density.

---

## 10. Limitations
- **English-Centric**: Multi-lingual tweets (Spanish/French/German) are filtered or mapped via fallback heuristics.
- **Stateless Execution**: The core classifier evaluates isolated turns without cross-turn coreference resolution.
- **Synthetic Augmentation in Sparse Classes**: Minor classes (Prime Student discrepancies) utilized template expansions during training.

---

## 11. What I'd Do with One More Week
1. **Cross-Encoder Re-Ranking**: Deploy a fine-tuned `ms-marco-MiniLM-L-6-v2` cross-encoder for RAG retrieval to reach 99%+ Recall@1.
2. **ReAct Tool Use**: Connect the agent to mock REST APIs (`getOrder(id)`, `cancelOrder(id)`, `checkCarrierStatus(id)`).
3. **Active Learning Queue**: Route low-confidence predictions (<0.70) to an annotator UI for continuous model retraining.
4. **Multi-Turn Thread Encoder**: Train a hierarchical transformer over entire conversation trees.
