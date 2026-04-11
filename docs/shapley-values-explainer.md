# Shapley Values — A Plain-English Walkthrough

For a full-stack engineer learning ML.

---

## The Problem

You've got a machine learning model. It takes in data about something — say, a flower — and spits out a prediction: *"this is a setosa."*

Great. But WHY? Which of the four measurements actually drove that call? Was it the petal length? The sepal width? ALL of them? NONE of them?

That's the problem. The model is a black box. It gives you an answer but no receipt.

Shapley values are the receipt.

---

## The Core Question

For ONE specific prediction, for ONE specific input, Shapley values answer:

> **How much did each feature contribute to pushing this prediction above (or below) the average prediction?**

Notice what that question is NOT asking. It's not "which feature is important in general." It's not "what would happen if I changed the feature." It's: *for this flower, right now, what did each measurement DO?*

---

## The Baseline

Before you can say a feature pushed something *up*, you need a "up from where."

That reference point is the **average prediction** across your whole dataset. If you trained on 1000 flowers and asked the model to predict each one, the average probability it assigned to "setosa" across all of them — that's your baseline.

Think of it as: *what would the model guess if it knew absolutely nothing specific about this flower?*

Every Shapley value is measured as a deviation from that baseline.

---

## The Analogy

Four engineers ship a feature. Revenue goes up $10k. How do you credit each engineer?

You can't say "split it equally" — that ignores who actually did the work. You can't say "whoever committed last gets credit" — that's just arbitrary.

The Shapley solution: **imagine every possible order in which the team could have assembled, and for each order, see how much revenue appeared when each person joined.**

- Alice joins first → revenue jumps $6k. Alice gets +$6k in this ordering.
- Bob joins second → revenue jumps $1k more. Bob gets +$1k in this ordering.
- Try every other ordering. Average each person's contribution.

That average is their Shapley value. It's the only mathematically fair way to split the credit.

**In ML**: engineers = features. Revenue = model prediction.

---

## How It's Computed

You can't just "remove" a feature to see what happens — ML models don't have an "ignore this column" mode. So instead, Shapley uses **hybrid instances** — Frankenstein inputs.

To measure how much petal length contributed:

1. Take your flower.
2. Take a random *other* flower from the dataset (the background).
3. Build two hybrids:
   - **With** petal length: some features from your flower, some from the background — but petal length comes from **your flower**.
   - **Without** petal length: same mix — but petal length comes from the **background flower**.
4. The difference in predictions between those two hybrids = petal length's contribution in this one trial.
5. Repeat with different background flowers and different orderings. Average everything.

That average, across all possible orderings and backgrounds, is the Shapley value.

The reason you try all orderings is so the result doesn't depend on some arbitrary order you picked. It's the fairest possible average.

---

## Handling Too Many Features

Trying every possible subset is exponential — 2^n subsets. With 4 features that's 16. With 20 it's over a million. With 50 it's impossible.

Three ways to deal with this:

**1. Monte Carlo sampling**

Instead of trying all subsets, randomly sample a fixed number of orderings (e.g. 500). You get an approximation, not the exact answer. More samples = more accurate, but slower. This is `KernelExplainer` in the shap library.

**2. Exploit the model structure**

For tree-based models, the tree structure lets you compute exact Shapley values in polynomial time by walking the branches directly — no sampling needed. This is `TreeExplainer`. Same idea exists for linear models (`LinearExplainer`). Both are fast *and* exact.

**3. Kernel trick**

`KernelExplainer` uses a weighted linear regression over sampled subsets to approximate Shapley values. It's model-agnostic (works on anything — neural nets, SVMs, whatever) but slow. Use it when no model-specific explainer exists.

**Practical rule:**

| Model type | Use |
|---|---|
| Random Forest, XGBoost, LightGBM | `TreeExplainer` — fast, exact |
| Linear / logistic regression | `LinearExplainer` — fast, exact |
| Anything else (neural net, SVM…) | `KernelExplainer` — slow, approximate |

---

## What the Numbers Mean

Say petal length gets a Shapley value of **+0.25** for predicting "setosa."

That means: across all the hybrid combinations tried, petal length's presence pushed the setosa probability up by 0.25 on average, compared to the baseline.

And they all add up. Sum every feature's Shapley value and you get exactly `prediction − baseline`. The whole gap between "average flower" and "this specific flower" is fully accounted for. Nothing invented, nothing lost.

---

## The Plots

**Waterfall plot**: one prediction. Starts at the baseline, each bar is one feature's Shapley value stacking up or down, lands at the final prediction. Like a bank statement — starting balance, transactions, ending balance.

**Summary plot**: all predictions at once. Y-axis = features ranked by importance. X-axis = Shapley value. Each dot is one sample. Color = the actual feature value (red = high, blue = low). Red dots on the right = high feature values pushed the prediction up.

**Bar plot**: the average size of each feature's impact across all samples. "Which features moved predictions the most, on average?"

---

## The Three Traps

**Trap 1 — It's not "what if I removed the feature."**

Shapley value of +0.25 for petal length does NOT mean "delete petal length and the prediction drops by 0.25." The hybrid approach is doing something subtler — it's averaging over partial information, not removal.

**Trap 2 — It's not causal.**

High Shapley value for income doesn't mean income *caused* the outcome. It means the model *used* income heavily. Shapley explains the model's behavior, not the world.

**Trap 3 — Correlated features pollute each other.**

The hybrid approach mixes features freely. But real features are often correlated — long petals tend to come with wide petals. Naive mixing creates impossible combinations the model has never seen, making predictions on them unreliable. TreeExplainer's `tree_path_dependent` mode handles this better for tree models.

---

## The One-Sentence Version

> Shapley values fairly split the credit for a prediction among the features, by averaging each feature's marginal contribution across every possible order in which the features could have been considered.

---

## Further Reading

- [Interpretable ML Book — Shapley Values](https://christophm.github.io/interpretable-ml-book/shapley.html)
- [SHAP library docs](https://shap.readthedocs.io/)
- [Original SHAP paper (Lundberg & Lee, 2017)](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)
- See also: `src/random_and_isolation_forests_revised.ipynb` in this repo for runnable code examples
