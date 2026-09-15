# Golden Evaluation Set — Annotation & Sampling Methodology

## Overview
- **Dataset Size**: 200 hand-curated customer queries
- **Target Brand**: AmazonHelp (Twitter Customer Support)
- **Primary Purpose**: Serving as the trusted ground truth benchmark to evaluate Intent Classification, RAG Reply Groundedness, and Auto-vs-Escalate Decision Routing.

## Sampling Methodology
1. **Stratified Domain Sampling**: Rather than simple random sampling (which is 80%+ dominated by delivery queries), we stratified samples across all 8 business intents to guarantee representation of high-risk / low-frequency classes (Account Security, Payment Disputes, Prime Subscriptions).
2. **First-Turn Filtering**: We extracted the initial inbound customer query from multi-turn threads to capture true customer root cause before conversational noise.
3. **Boundary & Edge-Case Inclusion**: Included subtle cases (e.g. false delivery marked as delivered, driver misconduct, composite Prime auto-renewal billing disputes).

## Intent Distribution
```
intent
DELIVERY_TRACKING            45
RETURNS_REFUNDS              35
CANCELLATION_MODIFICATION    25
PAYMENT_BILLING              25
SUBSCRIPTION_PRIME           25
ACCOUNT_SECURITY             20
PRODUCT_INQUIRY              15
FEEDBACK_CHITCHAT            10
```

## Routing Decision Breakdown
```
expected_decision
AUTO_HANDLE       115
ESCALATE_HUMAN     85
```

## Annotation Guidelines
- **AUTO_HANDLE**: Informational inquiries, standard delivery tracking lookups, self-service returns/cancellations, and general pleasantries where public knowledge or standard workflows suffice without access to sensitive financial credentials.
- **ESCALATE_HUMAN**: High-risk financial issues (double charges, unauthorized charges), security lockouts (2FA failure, hacked accounts), driver misconduct, missed guaranteed delivery SLAs requiring monetary compensation, or third-party seller disputes.
