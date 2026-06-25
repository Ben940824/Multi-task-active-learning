# data/ — script outputs

| Path | Description |
|------|-------------|
| `imdb/features.parquet` / `features.csv` | 45 feature columns (paper Table 1) |
| `imdb/targets.parquet` / `targets.csv` | 3 target columns |
| `imdb/combined.parquet` / `combined.csv` | features + targets (48 columns) |
| `imdb/preprocessing_config.yaml` | preprocessing decisions and column lists |
| `imdb/preprocessing_report.json` | row counts, IR vs paper |
| `imdb/split.json` | train / test / pool row indices |
| `imdb/split_report.json` | subset sizes and target class counts |

Each table is saved in both Parquet (for training scripts) and CSV (for manual inspection).

Raw inputs are under `raw_data/`.
