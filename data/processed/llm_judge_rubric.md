# LLM-as-a-Judge Evaluation Rubric for Amazon Support AI

This rubric defines the standardized criteria used by both human annotators and the LLM Judge to evaluate customer support reply quality on a 1.0 to 5.0 discrete scale.

---

### 1. Correctness (1–5)
- **5 - Perfectly Correct**: Accurately addresses the core inquiry, correctly identifies whether human escalation is required, and specifies exact next actions.
- **4 - Mostly Correct**: Correct solution with minor omissions that do not hinder resolution.
- **3 - Partially Correct**: Identifies the general domain but suggests suboptimal workflow.
- **2 - Mostly Incorrect**: Misinterprets key constraints or suggests inapplicable policy.
- **1 - Completely Incorrect**: Dangerous misinformation, wrong order action, or inappropriate dismissal.

---

### 2. Groundedness (1–5)
- **5 - Fully Grounded**: 100% faithful to official Amazon support URLs (`/returns`, `/your-orders`, `/mc`, `/account-recovery`) and standard DM escalation paths. Zero hallucinations.
- **4 - High Groundedness**: Factual information with standard phrasing, no fictitious claims.
- **3 - Moderate Groundedness**: General advice without specific verified URL links.
- **2 - Low Groundedness**: Mentions non-standard policies or vague procedures.
- **1 - Hallucinated**: Invented policies, fake contact details, or false promises (e.g. "We will refund $500 right now").

---

### 3. Relevance (1–5)
- **5 - Highly Relevant**: Directly answers the user's specific scenario with zero irrelevant filler.
- **4 - Relevant**: Addresses the issue with slight boilerplate padding.
- **3 - Moderately Relevant**: Generic response that partially applies.
- **2 - Barely Relevant**: Misses the nuance of the user's inquiry.
- **1 - Irrelevant**: Completely off-topic canned response.

---

### 4. Helpfulness (1–5)
- **5 - Highly Actionable**: Provides unambiguous next steps, direct links, or immediate escalation route.
- **4 - Helpful**: Gives clear guidance requiring minimal customer effort.
- **3 - Somewhat Helpful**: Requires customer to figure out specific navigation paths.
- **2 - Unhelpful**: Vague instructions (e.g. "Try looking on our website").
- **1 - Counterproductive**: Leaves the customer frustrated without a clear path forward.

---

### 5. Tone & Empathy (1–5)
- **5 - Exemplary Tone**: Warm, empathetic, professional, and effectively de-escalating frustration.
- **4 - Good Professional Tone**: Polite and courteous.
- **3 - Neutral**: Robotic but not offensive.
- **2 - Cold / Brusque**: Curt or dismissive phrasing.
- **1 - Inappropriate / Rude**: Hostile, combative, or dismissive of customer distress.
