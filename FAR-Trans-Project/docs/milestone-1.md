# Milestone 1 — Define / Team

**Team name:** FAR-Trans Lab

**Project title:** Profile-Aware LLM Re-Ranking for Financial Asset Recommendation

**Members:** Носов Владимир Андреевич (team leader); Ворожцов Пётр Алексеевич; Шигапова Диана Рашидовна.

**Leader's Telegram ID:** given separately in instructor-facing PDF to avoid exposing it publicly.

## One research question

For warm-start investors in FAR-Trans, does an LLM re-ranker using historical purchases, as-of-cutoff customer risk profiles, and asset metadata improve nDCG@10 compared with the original user-based kNN order, when both methods use the same fixed set of up to 50 candidates?

## Data example

Public [FAR-Trans data](https://researchdata.gla.ac.uk/1658/) (2018–2022; CC BY 4.0). Fields confirmed by [official schema](https://researchdata.gla.ac.uk/1658/2/README.txt):

| File | Relevant columns | Purpose |
|---|---|---|
| `transactions.csv` | customerID, ISIN, transactionType, timestamp | Historical purchases and held-out future buys |
| `customer_information.csv` | customerID, riskLevel, investmentCapacity, timestamp | As-of-cutoff user profile |
| `asset_information.csv` | ISIN, assetName, assetCategory, sector, industry, timestamp | Candidate metadata |

Synthetic demonstration (not actual data): customer demo_customer_001 has a Balanced risk level and CAP_30K_80K investment capacity. The system retrieves candidate assets DEMO_ISIN_B (stock), DEMO_ISIN_C (bond) and DEMO_ISIN_D (mutual fund). The LLM may only rearrange the input list; no new IDs may be introduced. Full production candidate sets will contain up to 50 IDs.

## Comparison

| Dimension | Baseline | Treatment |
|---|---|---|
| Retrieval | User-based kNN up to 50 | Same candidate list |
| Ranking | Original kNN order | LLM re-ranking of exact candidate list |
| Inputs | Historical transactions | Historical transactions, investor profile, asset metadata; no future data |
| Output | Top-10 unique assets | Top-10 unique assets among candidates |
| Evaluation | nDCG@10 on future purchases | The same nDCG@10 on the same split and cohort |

The result is **unknown**. Ranking agreement with purchases does not prove investment returns or suitability. Evaluation will use temporal splits and track invalid IDs, latency and token use as secondary evidence.

## First check

Download official CSV files, inspect joins and timestamps, and estimate cohort size and missingness. No baseline or LLM evaluation has been run in Session 1.

## References

- Sanz-Cruzado, J., Droukas, N., & McCreadie, R. (2024). *FAR-Trans: An Investment Dataset for Financial Asset Recommendation*. https://arxiv.org/abs/2407.08692
- Official dataset: https://researchdata.gla.ac.uk/1658/
- Original benchmark implementation: https://github.com/JavierSanzCruza/far-trans
