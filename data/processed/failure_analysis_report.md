# Stage 11 — Failure Mode Analysis & Engineering Hypotheses

An AI system in production must be rigorously audited for failure modes. Below is the deep dive into the **Top 5 failure modes** identified on Amazon customer support data, including real examples, root cause diagnoses, and engineering fixes.

---

### Failure Mode 1: Multi-Intent / Compound Problem Collisions
- **Customer Query**: *"@AmazonHelp I was charged £79 for annual Amazon Prime renewal today, but I didn't want to renew. Can I cancel and get a full refund?"*
- **Expected**: `SUBSCRIPTION_PRIME` | `AUTO_HANDLE`
- **Predicted**: `PAYMENT_BILLING` | `ESCALATE_HUMAN`
- **Root Cause**: Query activates 3 overlapping domains simultaneously: *"charged £79"* (Billing), *"refund"* (Returns), and *"Prime renewal"* (Subscription).
- **Hypothesis**: Single-label classifiers fail when a customer couples their root cause (Prime) with their desired resolution action (Refund/Cancel).
- **Potential Fix**: Deploy a **Hierarchical Two-Stage Classifier**: Stage 1 isolates the root business entity (Prime); Stage 2 routes the specific operational action (Cancel with automatic refund).

---

### Failure Mode 2: Context-Free Follow-Up Messages & Implicit References
- **Customer Query**: *"@AmazonHelp Still not working. I waited 3 hours as told and it still gives the exact same error code."*
- **Expected**: `ACCOUNT_SECURITY` / `SUBSCRIPTION_PRIME` | `ESCALATE_HUMAN`
- **Predicted**: `OTHER_UNCLASSIFIED` | `AUTO_HANDLE`
- **Root Cause**: Complete absence of domain nouns or order numbers; customer relies on implicit context from prior turns in the Twitter thread.
- **Hypothesis**: Single-tweet classification loses essential conversational state.
- **Potential Fix**: Implement **Sliding-Window Thread Concatenation**: Prepend the previous 2–3 thread messages before embedding and classification.

---

### Failure Mode 3: Sarcastic Lexical Polarity Clashing with Real Issue
- **Customer Query**: *"@AmazonHelp Great job Amazon! My new ceramic dinner set arrived in a million tiny pieces. Outstanding delivery service as always! 👏"*
- **Expected**: `RETURNS_REFUNDS` | `AUTO_HANDLE` (Replacement)
- **Predicted**: `FEEDBACK_CHITCHAT` | `AUTO_HANDLE` (Praise acknowledge)
- **Root Cause**: Superficial praise tokens (*"Great job"*, *"Outstanding delivery"*, clapping emoji) misguide positive sentiment weighting, overpowering the physical damage token (*"million tiny pieces"*).
- **Hypothesis**: Standard N-gram and embedding models without pragmatic context fail on irony and sarcasm.
- **Potential Fix**: Deploy a **Physical Condition Extraction Rule Layer**: Prioritize damage tokens (*"broken"*, *"shattered"*, *"pieces"*) over conversational adjectives.

---

### Failure Mode 4: False Delivery Confirmation ("Delivered" but Missing)
- **Customer Query**: *"@AmazonHelp Tracking says 'Handed to resident' at 2pm today, but I was at work and nobody was home. My porch and mailbox are completely empty!"*
- **Expected**: `DELIVERY_TRACKING` | `ESCALATE_HUMAN` (Carrier Trace / Theft)
- **Predicted**: `DELIVERY_TRACKING` | `AUTO_HANDLE` (Standard tracking link)
- **Root Cause**: Intent is correctly identified, but delivery keywords trigger the default auto-handle reply, missing the customer's negation that the package was falsely marked delivered.
- **Hypothesis**: Intent recognition alone is insufficient for high-stakes decision routing without explicit negation detection.
- **Potential Fix**: Implement a **Discrepancy Detector**: Trigger human escalation whenever carrier confirmation tokens co-occur with customer absence/negation tokens (*"nobody home"*, *"not received"*, *"empty"*).

---

### Failure Mode 5: Fulfillment State Blindness (Post-Dispatch Address Reroute)
- **Customer Query**: *"@AmazonHelp I moved last week and order #402-18291 just shipped to my old address 300 miles away. Please change delivery address right now!"*
- **Expected**: `CANCELLATION_MODIFICATION` | `ESCALATE_HUMAN` (Emergency Carrier Intercept)
- **Predicted**: `CANCELLATION_MODIFICATION` | `AUTO_HANDLE` (Self-service link)
- **Root Cause**: The model provides the self-service address change link, but once an order has dispatched, self-service editing is locked in Amazon's systems.
- **Hypothesis**: Language models lacking real-time order lifecycle state cannot distinguish between pre-dispatch self-service and post-dispatch emergency interventions.
- **Potential Fix**: Build a **Tool-Augmented (ReAct) Decision Agent**: Query the internal Order API (`OrderStatus`) to branch routing dynamically based on fulfillment state.
