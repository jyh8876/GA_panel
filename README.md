# Panel Study: Mis-selling, Lapse and Planner Settlement (Korean Insurers, 2021–2025)

Company-year panel study of how mis-selling, claim denial and planner (agent) settlement
relate to policy lapse (1 − persistency) among Korean life and non-life insurers.

## Data sources

| Source | What | Files in `data/` |
|---|---|---|
| FSS insurance company disclosure ("contract management") | Planner settlement rate, 13th/25th-month persistency (half-year and annual periods) | `fss_YYYYMM_YYYYMM.html` |
| KNIA consumer portal, item 04 | Mis-selling rate (non-life only; `F` = first half, `L` = second half) | `knia04_YYYYF/L.html` |
| KNIA consumer portal, item 07 | Long-term claim denial rate and post-claim cancellation rate (non-life only) | `knia07_YYYYF/L.html` |

KNIA covers non-life insurers only, so mis-selling is missing for life insurers. The life sample
is used for settlement vs. lapse only.

## Project structure

```
panel_study/
├── data/                  Raw saved HTML disclosure pages (FSS, KNIA 04, KNIA 07)
├── build_panel.py         Step 1: parse HTML -> company-year panel
├── analysis.py            Step 2: regressions, tables, figures, results.json
├── panel_analysis.qmd     Standalone Quarto/R document: TWFE models + residual diagnostics
│                          (raw data hard-coded; needs R packages plm, lmtest, sandwich)
├── panel.csv              OUTPUT of build_panel.py: one row per company x year
├── fss_raw_long.csv       OUTPUT: every FSS row as published, all periods
├── fss_sector_avg.csv     OUTPUT: sector-level FSS averages by year
├── results.json           OUTPUT of analysis.py: headline numbers
├── run_log.txt            Console log of the analysis run (correlations, model coefficients)
├── tables/                OUTPUT: LaTeX table fragments (git-ignored)
├── fig/                   OUTPUT: figures as PDF (git-ignored)
└── panel_report.*         LaTeX report and build files (git-ignored)
```

## Pipeline

```
data/*.html ──build_panel.py──> panel.csv, fss_raw_long.csv
panel.csv   ──analysis.py────> tables/*.tex, fig/*.pdf, results.json
tables/, fig/ ──LaTeX────────> panel_report.pdf
```

### Run

```bash
pip install numpy pandas scipy linearmodels matplotlib
python build_panel.py
python analysis.py
latexmk -pdf panel_report.tex          # optional; needs a LaTeX install
quarto render panel_analysis.qmd       # optional; needs R + plm, lmtest, sandwich
```

## Panel variables (`panel.csv`)

| Column | Meaning | Unit |
|---|---|---|
| `sector`, `company`, `year` | Panel keys (`life` / `nonlife`) | |
| `settle` | Planner settlement rate (0 treated as not applicable) | % |
| `p13`, `p25` | 13th / 25th-month persistency | % |
| `lapse13`, `lapse25` | 100 − `p13`, 100 − `p25` | % |
| `ms`, `ms_halves` | Mis-selling rate (half-year rates averaged) and number of halves used | % |
| `claims` | Number of long-term claims (KNIA 07) | count |
| `deny_rate`, `post_claim_cxl_rate` | Claim denial and post-claim cancellation rates | % |

`analysis.py` also derives `ms_bp` (mis-selling in basis points) and one/two-year lags
(`ms_bp_l1`, `ms_bp_l2`, `settle_l1`) on a balanced company x year grid.

## Methods

- Pooled OLS, random effects, firm fixed effects, and two-way fixed effects (TWFE) via `linearmodels`.
- Between/within correlation decomposition.
- Samples: non-life (FSS x KNIA), excluding digital/thin-data insurers (신한EZ, 카카오페이, 카디프손보);
  life (FSS only).
- `panel_analysis.qmd` repeats the preferred TWFE specifications in R and adds residual diagnostics.

## Git policy

`.gitignore` excludes secrets/credentials, documents (`.tex`, `.pdf`, `.doc(x)`, spreadsheets, slides),
LaTeX build files, archives, Python/R caches, and editor/OS files. Generated tables, figures and the
report are therefore not versioned; regenerate them with the pipeline above.
