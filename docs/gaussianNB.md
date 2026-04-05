# GaussianNB

`GaussianNB` from `sklearn.naive_bayes` is a probabilistic classifier. Instead of learning a decision boundary like SVM or a tree like RandomForest, it models the probability that a data point belongs to each class, then picks the most probable one.

## Bayes' Theorem

The foundation is Bayes' theorem:

```
P(class | features) ∝ P(features | class) × P(class)
```

In plain English: "Given these features, what's the probability of each class?" You compute it by asking "how likely are these features if we're in class C?" and weighting by how common class C is overall.

## The "Naive" Assumption

Computing `P(features | class)` is hard when features are correlated — you'd need a full joint distribution. The "Naive" shortcut: assume all features are **independent of each other** within a class.

```
P(features | class) = P(f1 | class) × P(f2 | class) × ... × P(fn | class)
```

This is almost always wrong in practice — feature correlation is everywhere. But the classifier is surprisingly robust to this violation, especially when you just want the most probable class rather than a calibrated probability.

## The "Gaussian" Assumption

To estimate `P(fi | class)`, the model assumes each feature follows a **normal (bell curve) distribution** within each class. During training, it computes the mean (μ) and variance (σ²) of each feature, separately for each class.

The probability of a single feature value given a class:

```
P(x | class) = (1 / √(2π σ²)) × exp(-(x - μ)² / (2σ²))
```

This is just plugging the value into the Gaussian formula using the class-specific mean and variance. No hyperparameters — the data fully determines the model.

## Prediction

For a new point, compute the above product across all features for each class, multiply by the class prior `P(class)`, and pick the class with the highest result.

## Advantages

- **Fast**: single pass through training data to compute means and variances
- **No hyperparameters**: nothing to tune — the data does all the work
- **Small data friendly**: even with few samples it can fit a Gaussian
- **High-dimension friendly**: each feature is modeled independently, so adding more features doesn't cause the curse of dimensionality the way joint distributions do

## Weaknesses

- **Correlated features**: the independence assumption breaks down. If two features are correlated, their combined evidence gets double-counted, skewing probabilities.
- **Non-Gaussian data**: if your feature distribution is skewed, bimodal, or bounded, forcing a bell curve on it produces garbage probability estimates.

## Scaling

`GaussianNB` does **not** need `StandardScaler`. Each feature is modeled independently using its own mean and variance — scale differences between features don't contaminate each other. The pipeline wrapper in this repo doesn't hurt it, but it's not necessary.

## How It Is Used in This Repo

```python
GaussianNB()
```

No arguments. The model is fully determined by the training data. Wrapped in a `make_pipeline(StandardScaler(), clf)` loop with all other classifiers — harmless, not needed.
