---
layout: page
title: "Implementing a forecasting paper, and what the evaluation revealed"
permalink: /projects/tnn-stoxx600/
sitemap: false
---

# What I learned implementing a forecasting paper properly

I implemented the Time-series Neural Network (TNN) from a 2023 paper in PyTorch and benchmarked it against LSTM and ARIMA on the STOXX Europe 600, building the whole thing as a production repository rather than a notebook.

The goal was to learn what separates ML research code from ML engineering. I expected that to be mostly about tooling. In practice, the tooling took a week and the interesting part was discovering that my results did not mean what they appeared to mean.

## The result that changed the project

The first working version reproduced the paper's central claim. TNN came in at RMSE 0.025 against LSTM's 0.029 on a one-day horizon, with TNN ahead across all three horizons I tested. That is what the paper reports, so I took it as a successful replication and moved on.

Then I tuned the hyperparameters properly on a held-out validation set, and LSTM overtook TNN. Then I noticed the two searches had used different learning rate grids, which meant the comparison was unfair to TNN. I reran the search with matched grids and got a *worse* best-of-grid score than the narrower search had found, even though the narrower grid was a strict subset of the wider one. That is arithmetically impossible unless the differences between configurations are smaller than the noise between runs of the same configuration.

So I trained each model five times with different random seeds:

| Horizon | LSTM (mean ± std) | TNN (mean ± std) |
|---------|-------------------|------------------|
| 1 day   | 0.0242 ± 0.0017   | 0.0249 ± 0.0007  |
| 7 days  | 0.0584 ± 0.0029   | 0.0571 ± 0.0022  |
| 30 days | 0.1193 ± 0.0119   | 0.1181 ± 0.0066  |

At every horizon, the gap between the two models is smaller than the variation between runs of the same model. They are indistinguishable. The apparent TNN advantage I had recorded earlier was one lucky initialization against one unlucky one.

The paper reports single-run figures with no measure of variance, which means its comparison is subject to the same problem. This is a non-replication under different conditions rather than a refutation: different index, different period, different preprocessing. But the claimed advantage is not visible here at any horizon, including the thirty-day case where the paper reports its largest margin.

One thing did survive. TNN's standard deviation is consistently lower than LSTM's, at every horizon. It is not more accurate, but it is more stable across initializations, which is a real if unglamorous property.

## The result that changed it more

Having fixed the comparison between models, I added a control I should have had from the start: a persistence baseline that predicts each value as the last observed value. Five lines, no model, no training.

| Horizon | Persistence | ARIMA  | LSTM   | TNN    |
|---------|-------------|--------|--------|--------|
| 1 day   | **0.0205**  | 0.0321 | 0.0242 | 0.0249 |
| 7 days  | **0.0529**  | 0.0576 | 0.0584 | 0.0571 |
| 30 days | **0.1032**  | 0.1049 | 0.1193 | 0.1181 |

It beats everything, at every horizon, by a margin well outside the seed variance.

This is not a bug in the pipeline. Daily index price levels behave close to a random walk, and under a random walk the best available forecast of tomorrow is today. The models were not failing to learn. There was very little in the target for them to learn.

What this reframes is the metric the whole field reports. My models reached R² between 0.92 and 0.95, which reads as excellent, and the paper reports 0.95 as evidence its architecture works. Persistence reaches 0.958 on this data while fitting nothing at all. On price levels, a high R² mostly reflects the autocorrelation of the series rather than any predictive skill. Without a naive baseline sitting in the same table, it is close to uninformative, and it will flatter whatever model you put next to it.

The underlying problem is the choice of target. Forecasting levels is a task where doing nothing is nearly optimal. Forecasting returns, where persistence predicts zero by construction and any signal has to be earned, is the version of this problem worth solving. That is the next iteration.

## Things I got wrong along the way

**I set up the package layout wrong and then tried to patch it.** Using `src/` as the package directory broke the editable install. I went through three fixes that did not hold up: a legacy setuptools backend, renaming the package to match the project name, and adding directories to the build configuration one at a time. I discarded all three because none of them explained why the problem existed, and a fix I cannot explain is a fix I will not be able to maintain. The correct answer was three lines of hatchling configuration declaring where the package lived. I lost an hour and came out understanding how Python resolves imports, which I had previously been taking on faith.

**I implemented the attention mechanism incorrectly.** The code multiplied the attention weights by the attention scores rather than by the input features. It ran without error and produced entirely sensible-looking output. I found it by going back to the equations in the paper and asking what each line of code was supposed to correspond to. That check is the only reason the model is a TNN rather than something adjacent to one.

**A missing `return` statement cost me a debugging session.** One preprocessing method did its work and returned `None`, which surfaced as a confusing type error two functions downstream. Ruff did not catch it: the function was annotated as returning a DataFrame, but the annotation is not enforced at runtime and the path fell through. Static analysis is not a substitute for reading what you wrote.

**My `save` method was an empty stub.** I did not find out until the inference script failed to load a preprocessor that had never been written to disk. Nothing in the training path exercised saving, so nothing surfaced it until the moment it mattered.

**A test I wrote found a real crash.** Asking the sequence builder for a window longer than the available data produced a negative array dimension. It would never have come up on the full dataset, and came up immediately on a short slice.

**I once concluded a fix had failed when I had not applied it.** The ARIMA numbers after a change were identical to the numbers before it, to every decimal place. That cannot happen if a different code path ran. The tell was in the numbers, not in the code.

**My first ARIMA evaluation was scoring a single prediction.** The neural models were evaluated on roughly 500 rolling windows. ARIMA was fitted once at the train/test boundary and scored on however many points the horizon called for, which at a one-day horizon is exactly one, hence an undefined R² rather than a bad one. The table looked like a comparison. It was two different tasks in adjacent columns. Replacing it with a rolling-origin forecast changed ARIMA's thirty-day RMSE from 0.46 to 0.10, moving it from worst to best of the three models.

**My inference script was predicting the past.** It built input windows using the same function as training, which necessarily drops the final rows because every window needs a known target. The most recent window therefore ended well before the last observation, and its "prediction" referred to a value I already had on hand. Nothing about the output looked wrong. It took writing a test asserting that the inference window differs from the last training window to make the problem visible.

**The persistence baseline was off by one and scored a perfect prediction.** RMSE of exactly 0.0 and R² of exactly 1.0 at a one-day horizon, because it was predicting each point using itself. An impossible result is much easier to debug than a plausible one, and the class of error was the same one I had already hit twice: converting between a length and an index without accounting for zero-based indexing.

## What I would do differently

Start with the baseline. Not as a formality at the end, but as the first thing built, before any model exists. Every number I produced for weeks was uninterpretable until I had something trivial to compare it against, and I would not have found that out at all if I had stopped when the results agreed with the paper.

Measure variance before comparing anything. Five seeds cost about an hour of compute and invalidated two conclusions I had already written down as findings.

Be more suspicious when results confirm expectations. Every error above that survived longest was one where the output looked like what I wanted to see. The ones I caught within minutes were the ones that produced something obviously absurd. Agreement with a published result is not validation. It is the condition under which I check least carefully.

## Limitations I have not resolved

The ARIMA order is hardcoded rather than selected by AIC or BIC, so its position in the ranking is partly arbitrary. There is a single chronological train/test split with no walk-forward validation. Hyperparameters were tuned at a one-day horizon and reused at seven and thirty. The model comparison rests on overlapping intervals rather than a formal significance test, over five seeds. Min-max normalization does not extrapolate, so live inference on current prices, which sit above the training maximum, produces values outside [0, 1] that should not be read as forecasts.

These are all in the repository README too. I would rather they be visible than discovered.

## What the repository contains

uv for dependency management, Ruff and mypy enforced through pre-commit hooks and GitHub Actions, pytest, Docker, MLflow with a SQLite backend, Pydantic-validated YAML configuration, and separate entry points for training and inference.

That was the original point of the exercise and it is the part I am least inclined to write about. It is largely a matter of following conventions carefully. Working out that my results did not mean what I thought they meant took longer and taught me considerably more.

Repository: [github.com/paulricc/tnn-stoxx600-forecasting](https://github.com/paulricc/tnn-stoxx600-forecasting)
