# China Anomaly Factors

A Python library for computing **Chinese A-share market anomaly factors**, covering momentum, value, investment, profitability, intangibles, and trading frictions. Based on academic literature on cross-sectional return anomalies in the Chinese equity market.

---

## Project Structure

```
China-anormaly-factors/
├── china_anomalies_factors.py     # Core factor library (81 factor classes)
├── Factor_code_revise_Qiu.py      # Revised/fixed factor implementations (21 classes)
├── Factor_code_transfer_Qiu.py    # Extended & new factor implementations (40 classes)
├── Factor_factory_Qiu.py          # Factory registry + batch runner
├── data/                          # Input data (RESSET / Wind format)
├── output/                        # Raw factor outputs
└── clean_output/                  # Final factor outputs (Parquet files)
```

---

## Factor Categories

| Category | Code | # Factors | Examples |
|----------|------|-----------|---------|
| Momentum | A1 | 12 | SUE, ABR, RE, R6, R11, RS, TES, NEI, W52, RM6, RM11 |
| Value | A2 | 18+ | BM, DM, AM, EP, CP, SP, BMJ, BMQ, SR, SG, OCP |
| Investment | A3 | 25+ | AGR, IA, NSI, CEI, CDI, OA, TA, DWC, POA, NXF |
| Profitability | A4 | 18 | ROE, ROA, GPA, OPA, CTO, OLA, OLE, OPE, COP |
| Intangibles | A5 | 9 | AGE, DSI, ANA, RDM, RDS, ADM, TAN |
| Trading Frictions | A6 | 12+ | ME, TURN, TUR, DTV, PPS, AMI, SREV, TS, SHL, MDR |

**Total: 100+ factors** registered across all modules.

---

## Design

### Priority System (3-layer override)
```
Transfer (highest) > Revised Fixed* > Original (lowest)
```

- **`china_anomalies_factors.py`** — original implementations
- **`Factor_code_revise_Qiu.py`** — bug-fixed `Fixed*` variants that override originals
- **`Factor_code_transfer_Qiu.py`** — rewritten versions + brand-new factors

### Factor Interface
Every factor inherits from `BaseFactor` and implements:
```python
factor.calculate()        # → raw DataFrame
factor.as_factor_frame()  # → standardized panel DataFrame
factor.save()             # → Parquet file in clean_output/
```

---

## Usage

### Run all factors (batch mode)
```bash
python Factor_factory_Qiu.py
```
Results are saved to `clean_output/` as Parquet files named:
```
{factor_id}_{abbr}_{start_date}-{end_date}.parquet
```

### Use a single factor
```python
from Factor_factory_Qiu import create

factor = create("bm")          # Book-to-Market
df_raw = factor.calculate()
df_ff  = factor.as_factor_frame(df_raw)
factor.save(df_ff)
```

### List all available factors
```python
from Factor_factory_Qiu import REGISTRY
print(sorted(REGISTRY.keys()))  # 100+ factor abbreviations
```

---

## Data Requirements

Input data should be placed in the `data/` directory. Stock codes are expected in either:
- **RESSET format** — numeric (e.g., `1`, `600000`)
- **Wind format** — with exchange suffix (e.g., `000001.SZ`, `600000.SH`)

The library handles automatic conversion between both formats.

---

## Dependencies

```
numpy
pandas
statsmodels
matplotlib
tqdm
pyarrow   # for Parquet I/O
```

Install with:
```bash
pip install numpy pandas statsmodels matplotlib tqdm pyarrow
```

---

## Author

**Jaden Qiu** — factor revisions and extensions (`Factor_code_revise_Qiu.py`, `Factor_code_transfer_Qiu.py`, `Factor_factory_Qiu.py`)

Original factor library refactored from `china-anomalies-compiled.ipynb` (2025-10-16).
