Implementation caveats / decisions to confirm:

1. Random Forest is CPU-only. No GPU is required for either single-output or multi-output versions.

2. Start with local execution on Apple M3 Pro. Expected runtime for IMDB single-output baseline is small: usually seconds to a few minutes for one seed, depending on n_estimators and implementation.

3. Use n_jobs=-1 for RandomForestClassifier to utilize available CPU cores.

4. Need to confirm the IMDB dataset version:
   - whether row count is close to 3,737
   - whether column count is close to 45
   - whether the three target columns match the report

5. Need to confirm target preprocessing:
   - IMDb score: convert into Bad / Average / Good
   - Gross: binary threshold at 15 million
   - Content Rating: need to inspect missing values and rare classes; relabeling rule may differ from the report

6. Need to decide duplicate handling during active learning query:
   - Each target selects 15 low-confidence samples.
   - The same sample may be selected by multiple targets.
   - Preferred first version: take the union, then fill additional lowest-confidence samples until reaching 45 unique samples per round.

7. First implementation target:
   - IMDB only
   - single-output only
   - three independent Random Forest classifiers
   - no resampling first
   - 100 initial training samples
   - 1000 fixed test samples
   - remaining data as pool
   - 20 query steps
   - 45 queried samples per step
   - save accuracy and macro-F1 curves