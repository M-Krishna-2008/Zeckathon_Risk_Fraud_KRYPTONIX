# Missing Data Audit Log

In adherence to **Standing Rule 2: Zero Financial Data Fabrication**, this document logs every financial item that was omitted from `data/india/financials.csv` because it was not directly or separately reported in public annual reports, regulatory filings, or stock exchange disclosures. Values are **never** estimated, interpolated, or guessed.

---

## Sector-Specific Structural Omissions

### 1. NBFCs and Housing Finance Companies (e.g., DHFL, IL&FS, Reliance Capital)
- **`cogs` (Cost of Goods Sold)**:
  - *Reason*: Financial intermediaries do not manufacture goods or carry merchandise inventory. Their core funding cost is reported under `interest_expense` (Finance Costs), not COGS.
  - *Status*: Blank / Omitted.
- **`receivables` (Trade Receivables)**:
  - *Reason*: Loans and mortgages extended to borrowers are classified under Advances / Loan Assets / AUM, rather than commercial trade receivables.
  - *Status*: Blank / Omitted.
- **`current_assets` / `current_liabilities`**:
  - *Reason*: NBFCs historically reported balance sheets under Indian GAAP / RBI non-banking balance sheet formats where current/non-current bifurcation was replaced by structural liquidity ALM maturity buckets.
  - *Status*: Blank / Omitted where not separately disclosed.

### 2. Aviation & Services (e.g., Jet Airways, InterGlobe Aviation)
- **`sga` (Selling, General and Administrative)**:
  - *Reason*: Airline filings under Ind AS report Aircraft Operating Expenses, Aircraft Maintenance, Lease Rentals, and Employee Benefits as distinct line items; SG&A is not reported as a separate single aggregate in many disclosures.
  - *Status*: Omitted to avoid arbitrary aggregation without explicit statutory disclosure.
- **`retained_earnings`**:
  - *Reason*: Indian corporate balance sheets report "Reserves and Surplus" as a composite line item encompassing securities premium, general reserves, foreign currency translation reserves, and retained earnings. Where standalone retained earnings sub-schedules were omitted from summary disclosures, the item is left blank.
  - *Status*: Blank / Omitted.

---

## Item-Level Omission Audit Table

| Company | Fiscal Year | Omitted Items | Reporting Source Inspected | Reason / Note |
|---|---|---|---|---|
| DHFL | FY17, FY18, FY19 | `cogs`, `receivables`, `sga` | BSE Annual Disclosures & Results | NBFC accounting format; interest expenses reported instead of COGS; loan assets instead of trade debtors. |
| Jet Airways | FY16, FY17, FY18 | `sga`, `retained_earnings` | BSE Audited Results & Annual Reports | Airline operating expenses broken out by fuel/lease/maintenance; composite reserves & surplus reported. |
| Reliance Communications | FY15, FY16, FY17 | `cogs`, `sga`, `retained_earnings` | BSE Audited Financial Results | Telecom services report network operation costs and spectrum amortization rather than COGS/SG&A. |
| Future Retail | FY19, FY20, FY21 | `retained_earnings` | BSE Corporate Filings & Annual Reports | Reported composite Reserves & Surplus under Ind AS Balance Sheet format. |
| Coffee Day Enterprises | FY17, FY18, FY19 | `sga`, `retained_earnings` | BSE Annual Reports & Filings | Operating costs reported under store expenses/raw materials consumed; consolidated reserves reported composite. |
| InterGlobe Aviation | FY16, FY17, FY18 | `sga`, `retained_earnings` | BSE Filings & Investor Disclosures | Operating expenditure broken into aviation turbine fuel (ATF), lease rental, and crew costs. |
| Avenue Supermarts | FY19, FY20, FY21 | `retained_earnings` | BSE Annual Reports | Composite Reserves & Surplus reported on Face of Balance Sheet. |

---

## Standing Audit Assurance
- **Zero Fabrication**: All rows populated in `data/india/financials.csv` correspond directly to original reported figures.
- **Audit Traceability**: Every populated row is linked to an official `source_url` and point-in-time `filing_date`.
