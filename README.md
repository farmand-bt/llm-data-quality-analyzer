# Data Quality Analyzer

A Streamlit app for automated data profiling, interactive cleaning, and optional LLM-powered quality insights — upload a CSV or Excel file and instantly get a comprehensive report with actionable recommendations.

## Features

- ✅ **Data Profiling** — per-column statistics, type inference, missing value analysis with MCAR/MAR/MNAR classification, outlier detection (IQR & Z-score), normality tests, correlation heatmap
- ✅ **Quality Score** — composite 0–100 score and letter grade (A–F) computed from missing rate, duplicates, type mismatches, and outlier prevalence
- ✅ **Interactive Cleaning** — remove duplicates, impute missing values (mean/median/mode/constant/ffill/bfill), fix type mismatches, handle outliers; every step logged, nothing mutated
- ✅ **LLM Insights (optional)** — plain-English explanations and a free-form Q&A interface, powered by any OpenAI-compatible API; summary stats only — raw data never leaves your machine
- ✅ **Export** — download the cleaned dataset as CSV or Excel
- ✅ **Built-in Samples** — two ready-to-explore datasets (Housing, Titanic) with realistic data quality issues; no upload required

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/llm-data-quality-analyzer.git
cd llm-data-quality-analyzer
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

### 2. Run

```bash
uv run streamlit run app/app.py
# or
make run
```

Open [http://localhost:8501](http://localhost:8501). Upload a CSV / Excel file — or click one of the built-in sample buttons to explore immediately.

### 3. (Optional) Enable LLM Insights

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

```env
GWDG_API_KEY=your_key_here
GWDG_API_BASE=https://your-api-endpoint
GWDG_MODEL_NAME=llama-3.3-70b-instruct
```

Any OpenAI-compatible provider works (GWDG, Together AI, Groq, local Ollama with OpenAI-compat layer, etc.). Restart the app, then toggle **🤖 Enable AI Insights** in the sidebar.

## How It Works

### 1. File Loading
CSV files use `chardet` for encoding detection and auto-detect the separator (`,` vs `;`). Excel files support multi-sheet selection. Files up to 200 MB are accepted.

### 2. Type Inference
Each column is inspected beyond its pandas dtype: numeric strings, boolean-like text (`yes`/`no`/`true`/`false`), mixed types, and datetime patterns are detected and a `type_mismatch` flag is set when pandas and inferred types disagree.

### 3. Statistics & Distributions
Numeric columns receive mean, median, standard deviation, min, and max. Categorical columns receive unique count and top value. Distribution analysis runs normality tests (Shapiro–Wilk for small columns, Kolmogorov–Smirnov for larger ones), plus skewness and kurtosis labelling.

### 4. Missing Values
Missing count, percentage, and a heuristic pattern classification:
- **MCAR** — missingness appears random
- **MAR** — strongly overlaps with another missing column
- **MNAR** — structural (>60% absent)

### 5. Outlier Detection
Both IQR (1.5× multiplier) and Z-score (±3σ) methods return counts, percentages, and the exact row indices — so the cleaning pipeline can target them precisely without index drift.

### 6. Quality Score
Points are deducted for: missing cell rate, duplicate row rate, per-column type mismatches, and outlier prevalence. Clean columns add small bonuses. Final score maps to A–F grades using configurable thresholds in `config/settings.py`.

### 7. Cleaning Pipeline
Non-destructive three-tier pipeline:
1. **Tier 0** — type fixes and fill operations (no row count change)
2. **Tier 1** — outlier removal (uses profiler-computed indices; must run before row drops)
3. **Tier 2** — row-dropping operations (drop_rows, drop_column, remove_duplicates)

Every step is logged (action, column, rows before/after). The original DataFrame is never mutated.

### 8. LLM Insights
Only summary statistics from the report dict are sent to the API — raw data rows are never transmitted. Insights are cached in session state after first generation. All LLM calls are user-triggered.

## Architecture

```mermaid
flowchart TD
    A("📁 Upload / Sample Dataset") --> B["profiler/loader.py\nCSV + Excel · chardet encoding"]
    B --> C[("profiler/report.py\nUnified Report Dict")]
    C --> D["Streamlit UI\n8 Components"]
    C --> E["cleaner/pipeline.py\nNon-destructive 3-tier pipeline"]
    C --> F["narrator/narrator.py\nPrompt construction"]
    E --> G("✅ Cleaned CSV / Excel")
    F --> H["narrator/client.py\nOpenAI-compatible SDK"]
    H --> I("🤖 Any OpenAI-compatible LLM")
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI framework | Streamlit 1.30+ |
| Data processing | pandas 2.0+, NumPy 1.24+ |
| Statistical analysis | SciPy 1.10+ |
| Visualizations | Plotly 5.0+ |
| Excel support | openpyxl |
| Encoding detection | chardet |
| LLM integration (optional) | OpenAI SDK — any compatible provider |
| Environment variables | python-dotenv |
| Linting | Ruff |
| Testing | pytest 8.0+ (200+ tests) |
| Package management | uv |

## Project Structure

```
llm-data-quality-analyzer/
├── app/
│   ├── app.py                  # Streamlit entry point
│   └── components/             # UI: overview, column_report, outliers,
│                               #   duplicates, correlations, recommendations,
│                               #   cleaning_panel, llm_insights
├── profiler/                   # Analysis engine
│   ├── loader.py               # CSV / Excel parsing
│   ├── type_detector.py        # Inferred-type logic
│   ├── stats.py                # Descriptive statistics
│   ├── missing.py              # Missing-value analysis + MCAR/MAR/MNAR
│   ├── outliers.py             # IQR and Z-score detection
│   ├── duplicates.py           # Exact and near-duplicate detection
│   ├── distributions.py        # Normality tests, skew, kurtosis
│   ├── correlations.py         # Pearson + Cramér's V
│   ├── recommendations.py      # Rule-based severity recommendations
│   └── report.py               # Aggregates all sub-modules into one dict
├── cleaner/                    # Cleaning engine
│   ├── missing_handler.py
│   ├── duplicate_handler.py
│   ├── type_fixer.py
│   ├── outlier_handler.py
│   └── pipeline.py             # Orchestrates steps in priority order
├── narrator/                   # LLM integration
│   ├── client.py               # API calls with retry/backoff
│   ├── prompts.py              # Prompt builders
│   └── narrator.py             # High-level narration methods
├── config/
│   └── settings.py             # All thresholds and constants
├── tests/                      # pytest suite (200+ tests)
├── scripts/
│   └── generate_messy_data.py  # Synthetic dataset generator
├── sample_data/
│   ├── housing_messy.csv       # 520 rows — price, sqft, outliers, mixed types
│   └── titanic_messy.csv       # 911 rows — missing age/cabin, inconsistent labels
├── .streamlit/
│   └── config.toml             # UI theme
└── requirements.txt            # Streamlit Cloud deployment
```

## Sample Datasets

Two built-in datasets are included for instant demo — no upload needed. Click the sample buttons in the sidebar.

### 🏠 Housing (520 rows)
Synthetic housing sale data with:
- Missing values in `price`, `sqft`, `school_rating` (~8% random + structural)
- Inconsistent date formats (`YYYY-MM-DD`, `MM/DD/YYYY`, `DD-Mon-YYYY`) in `sale_date`
- Inconsistent boolean representations in `garage` (`Yes`/`No`/`Y`/`N`/`TRUE`/`FALSE`)
- Mixed types in `sqft` (some cells are `"N/A"`, `"unknown"`, `"TBD"`)
- Extreme price outliers (values like `–1000`, `9999999`)
- 20 exact duplicate rows

### 🚢 Titanic (911 rows)
Synthetic Titanic-style passenger data with:
- ~20% missing `Age`, ~77% missing `Cabin`, a few missing `Embarked`
- Inconsistent `Sex` labels (`male`/`Male`/`MALE`/`M`/`female`/`Female`/`F`)
- `Fare` column with mixed types (most are numeric, ~25 rows stored as `"$X.XX"` strings)
- Outlier fares (0.0, 512.33, 263.0)
- 20 exact duplicate rows

Regenerate with:
```bash
uv run python scripts/generate_messy_data.py
```

## Configuration

All thresholds live in `config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_UPLOAD_SIZE_MB` | 200 | File size limit |
| `MISSING_THRESHOLD` | 0.30 | Bar chart red-flag threshold |
| `COLUMN_WARNING_MISSING_THRESHOLD` | 0.50 | Column-level warning threshold |
| `OUTLIER_IQR_MULTIPLIER` | 1.5 | IQR fence multiplier |
| `OUTLIER_ZSCORE_THRESHOLD` | 3.0 | Z-score cutoff |
| `HIGH_CORRELATION_THRESHOLD` | 0.9 | Pair correlation flag |
| `HIGH_CARDINALITY_THRESHOLD` | 0.5 | Unique ratio for cardinality warning |
| `MAX_PREVIEW_ROWS` | 10 | Rows shown in data preview |
| `QUALITY_GRADE_THRESHOLDS` | A:90, B:80, C:70, D:60 | Score-to-grade mapping |

## Development

```bash
make install   # uv sync
make run       # uv run streamlit run app/app.py
make lint      # ruff check --fix + ruff format
make test      # pytest tests/
make clean     # remove __pycache__ and .pytest_cache
```

## Deployment (Streamlit Community Cloud)

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set **Main file path** to `app/app.py`
4. (Optional) Add `GWDG_API_KEY`, `GWDG_API_BASE`, and `GWDG_MODEL_NAME` under **Secrets**
5. Deploy — Streamlit Cloud uses `requirements.txt` automatically

## Future Improvements

Listed by implementation priority (easiest and highest-value first):

1. **More file formats (JSON, Parquet)** — Only `profiler/loader.py` needs extending with `pd.read_json()` / `pd.read_parquet()`; Parquet requires adding `pyarrow` to dependencies. SQL connections are more involved but follow the same pattern. High value for data engineering use cases.

2. **PDF report export** — Render the profiler report dict to a downloadable PDF using a library like `weasyprint` or `reportlab`. The main effort is layout: recreating the charts and tables in a static format. High value for sharing results with non-technical stakeholders.

3. **Pipeline export as Python script** — Serialize `pipeline.get_log()` into a runnable `.py` file with the correct imports and `pd` operations. Useful for teams who want to apply the same cleaning steps in a production pipeline without the UI.

4. **Custom quality rules** — A rule-builder UI (column + operator + threshold, e.g. "price must be > 0") that feeds into the recommendations engine. Makes the tool useful for data contracts and domain-specific validation.

5. **Comparison mode** — Upload two datasets and diff their quality profiles side-by-side. Useful for comparing a dataset before and after an ETL step, or two data exports from different sources. Touches almost every component, so it is the largest single feature.

6. **Scheduled profiling** — Connect to a live data source, run profiling on a schedule, and alert when quality degrades. This is more infrastructure than analytics (needs job scheduling, persistent storage, alerting) and is effectively a different product category.

## License

[MIT](LICENSE)
