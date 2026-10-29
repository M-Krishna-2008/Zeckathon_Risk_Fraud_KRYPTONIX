# Credit Risk Policy & Early Warning Alert Framework

## 1. Objective & Decision-Theoretic Foundation

The Early Warning System for Indian Corporate Distress (EWS-India) provides quarterly, point-in-time risk rankings for an 800-borrower corporate credit portfolio at least 4 quarters in advance of credit distress or default.

### Cost Model & Theoretical Alert Threshold
In wholesale commercial banking, the cost of a missed default ($C_{FN}$) vastly outweighs the administrative review cost of a false alarm ($C_{FP}$). Under institutional workout studies:
- $C_{FN} = 20$ (Loss Given Default, provisions, capital charge, asset-sale haircuts)
- $C_{FP} = 1$ (Analyst review time, borrower inquiries, enhanced covenant audit)

By Bayes decision theory, the cost-minimizing threshold on calibrated probability $p(y=1|x)$ is:
$$\tau^* = \frac{C_{FP}}{C_{FN} + C_{FP}} = \frac{1}{20 + 1} = \frac{1}{21} \approx 0.047619$$

Any corporate borrower with a calibrated default probability $P \ge 4.76\%$ mathematically warrants early surveillance.

---

## 2. Operational Capacity Constraint: The Top 5% Cap

While $\tau^* = 1/21$ defines the statistical alert threshold, practical risk teams have finite operational bandwidth.
- **Monitored Portfolio**: 800 listed corporate borrowers.
- **Quarterly Review Capacity**: Exactly **5%** of the universe = **Top 40 Borrowers**.

### Protocol When Cost-Optimal Candidates Exceed 40
In cyclical downturns (e.g., sector-wide stress), the number of borrowers exceeding $\tau^* = 0.0476$ may rise to 70 or 100. When $N_{\tau^*} > 40$, the following two-tier capacity protocol is triggered:

1. **Tier 1 (Priority Action Queue — Top 40 by Composite Risk Score)**:
   - Ranked by combined probability, soft-signal severity, and accounting flags.
   - Assigned immediately to credit analysts for full deep-dive forensic review.
   - Requires formal credit committee resolution within 14 business days.

2. **Tier 2 (Surveillance Watchlist — Candidates Ranked 41 to $N_{\tau^*}$)**:
   - Retained on automated quarterly watchlist.
   - Trigger automated automated covenant verification:
     - Request latest quarterly stock statements and debtor aging.
     - Verify monthly promoter pledge disclosures with BSE/NSE.
     - Review ALM gap ratios and working capital facility drawdown velocity.
   - No immediate committee triage unless an acute soft signal occurs (e.g. auditor resignation or default rating).

---

## 3. Three-Tier Review Workflow

```text
[ Quarterly Pipeline Run ] 
          │
          ▼
   Top 40 Flagged
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 1: Triage (Days 1–2)                                   │
│ Responsibility: Junior Credit Risk Analyst                  │
│ Action: Verify source filings, check for data omissions,    │
│         confirm signal dates, validate sector adjustments.  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Validated Alerts
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 2: Analyst Deep-Dive (Days 3–7)                        │
│ Responsibility: Senior Sector Specialist                    │
│ Action: Investigate top 3 SHAP drivers, Beneish M-Score     │
│         accrual distortions, promoter pledge trends,        │
│         and related-party fund transfers.                   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Detailed Credit Memo
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 3: Credit Risk Committee (Days 8–14)                   │
│ Responsibility: Head of Credit, CRO, Sector Business Head  │
│ Action: Executive decision on exposure limits, collateral,  │
│         restructuring, or exit plan.                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Immediate Escalation Path (Red-Flag Fast-Track)

If any borrower exhibits an acute governance or solvency trigger, standard 14-day workflow is bypassed and an **Immediate Escalation Notice** is delivered to the Chief Risk Officer (CRO) within **24 hours**:

1. **Statutory Auditor Resignation**:
   - Auditor departs prior to completing annual accounts citing scope limitation, fraud (Sec 143(12)), or unverified transactions (e.g., PwC at Reliance Capital, Deloitte at DHFL).
2. **Promoter Pledge Surge**:
   - Promoter pledge exceeding 50% of total promoter holding, or a >20% single-quarter surge.
3. **Credit Rating Downgrade to 'D' or Sub-Investment Grade**:
   - Immediate freeze on unutilized credit limits.
4. **Undisclosed Related-Party Fund Diversion**:
   - Forensic or statutory auditor notes identifying lending to promoter affiliates.

---

## 5. False Positive vs. Missed Default Trade-off Analysis

| Metric | Cost Weight | Impact of False Alarm (FP) | Impact of Missed Default (FN) |
|---|---|---|---|
| **Direct Financial Cost** | 1x vs 20x | ₹50,000 – ₹100,000 analyst labor hours | ₹50 Cr – ₹500 Cr credit write-off |
| **Operational Impact** | Low vs Catastrophic | Routine borrower inquiry | Capital adequacy erosion, regulatory scrutiny |
| **Mitigation** | System Tuning | Explainable SHAP rationale prevents wasted investigation | $1/21$ threshold ensures high recall |
