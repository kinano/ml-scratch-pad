# StandardScaler

Most ML algorithms are sensitive to the scale of input features. If one feature ranges from 0–1 and another from 0–1,000,000, the algorithm will treat the large-valued feature as more important — not because it is, but because its numbers are bigger. `StandardScaler` fixes this by bringing all features to the same numeric scale so the algorithm evaluates them on equal footing.

## What It Does

For each feature column, it computes the mean and standard deviation from the training data, then transforms every value:

```
scaled_value = (value - mean) / std_dev
```

The result: every feature has mean=0 and std=1. This is called **standardization** (z-score normalization).

## Why It Matters

Two families of algorithms break down without scaling:

- **Distance-based models (KNN, SVM)** — these ask "how close is this point to others?" A $10,000 salary difference will numerically dwarf a 10-year age difference, even if age is the more predictive signal. Scaling puts both on equal footing.
- **Gradient-based models (neural nets, logistic regression)** — these learn by iteratively adjusting weights. Uneven feature scales cause uneven gradient steps, slowing down or destabilizing training.

Tree-based models (decision trees, random forests) split on thresholds and are scale-invariant — they don't need scaling, but it doesn't hurt them either.

## Base Class

`StandardScaler` extends sklearn's `TransformerMixin` and `BaseEstimator`. `TransformerMixin` gives it `fit()`, `transform()`, and `fit_transform()` — the same interface contract all sklearn preprocessors follow. This is what allows it to slot seamlessly into a `Pipeline`.

## Why Test Data Uses Training Stats

Once the scaler learns mean and std from training data, it applies those same numbers to the test data — it does NOT re-learn from the test set.

This is critical for consistency: a salary of $80,000 must represent the same thing in training and test. If the scaler computed different stats for each, you'd be feeding the model data that looks different than what it was trained on.

## How It Is Used in This Repo

```python
clf = make_pipeline(StandardScaler(), clf)
```

The repo tests many classifier types in a single loop. Wrapping each in `StandardScaler` is a safe blanket — required for KNN, SVM, and neural nets; harmless for decision trees. The pipeline ensures the scaler fits only on training data and correctly transforms test data at evaluation time.
