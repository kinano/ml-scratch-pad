# make_pipeline

`make_pipeline` from scikit-learn chains preprocessing steps and a model into a single object. Instead of manually managing each step, you hand the pipeline your data and it handles the rest — in the right order, on the right data, every time.

## What It Takes

Any number of sklearn-compatible objects in sequence: transformers (preprocessing steps like `StandardScaler`) followed by a final estimator (the classifier or model). Each transformer must implement `fit` and `transform`. The final estimator just needs `fit` and `predict`/`score`.

```python
clf = make_pipeline(StandardScaler(), SVC())
```

## What It Returns

A `Pipeline` object that behaves exactly like a single sklearn model — it exposes `.fit()`, `.predict()`, and `.score()` just like any classifier would.

## What Happens Under the Hood

When you call `clf.fit(X_train, y_train)`, the pipeline runs in sequence:

1. `StandardScaler.fit_transform(X_train)` — scaler learns the mean and std from training data, then scales it
2. `classifier.fit(X_train_scaled, y_train)` — classifier trains on the scaled data

When you call `clf.score(X_test, y_test)`:

1. `StandardScaler.transform(X_test)` — scales test data using the stats learned from training (does NOT re-learn)
2. `classifier.score(X_test_scaled, y_test)` — evaluates on scaled test data

## Why Use It

Without a pipeline, you'd manage each step manually:

```python
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
clf.fit(X_train_scaled, y_train)

X_test_scaled = scaler.transform(X_test)  # easy to forget
score = clf.score(X_test_scaled, y_test)
```

Two things can go wrong here:

1. **You forget to scale the test set** — the model trained on scaled data but evaluates on raw data. Silent bug, garbage results.
2. **You accidentally call `fit` on the test set** — the scaler learns from test data, leaking its distribution into training.

The pipeline makes both mistakes impossible. `.fit()` and `.score()` always run the full sequence correctly.

## How It Is Used in This Repo

```python
for name, clf in zip(names, classifiers):
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
```

The repo tests many classifier types in a single loop. Wrapping each one in a pipeline with `StandardScaler` is a safe blanket — it's required for distance- and gradient-based models (KNN, SVM, neural nets) and doesn't hurt tree-based ones.
