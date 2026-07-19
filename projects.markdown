---
layout: page
title: Projects
permalink: /projects/
---

### Stock Price Forecasting with Time Series Neural Networks
*Python · PyTorch · LSTM · ARIMA · Time Series Analysis · MLOps*

Can deep learning outperform classical approaches at forecasting stock prices? I
implemented the Time-series Neural Network (TNN) from a 2023 paper in PyTorch and
benchmarked it against LSTM and ARIMA on the STOXX Europe 600, across horizons of
1, 7, and 30 days.

The interesting part turned out to be the evaluation. Averaged over five random
seeds, TNN and LSTM perform within the margin of run-to-run variance, and a simple
persistence baseline holds up well against all three models. Forecasting price
levels rewards doing very little, which makes naive baselines essential for reading
the results properly.

Built as a production repository: uv, Ruff, mypy, pre-commit, pytest, Docker,
GitHub Actions, and MLflow experiment tracking.

[Read the full write-up](/projects/tnn-stoxx600/) ·
[View on GitHub](https://github.com/paulricc/tnn-stoxx600-forecasting){:target="_blank"}


### Constrained MDP with Lagrangian Policy Gradient
*Python · PyTorch · Reinforcement Learning · Constrained Optimisation*

How do you train an agent to perform well while respecting hard behavioural constraints?
This project implements a primal-dual policy gradient method on a Constrained Markov
Decision Process (CMDP), where the agent must maximise cumulative reward subject to a
budget constraint on a secondary cost signal.

The approach is grounded in Lagrangian relaxation: a dual variable adapts dynamically
during training, penalising constraint violations and driving the policy toward feasible
behaviour. The constrained agent learns to keep costs below the specified budget while
the unconstrained baseline, trained on the same environment, consistently violates it.

The mathematical backbone of the method, primal-dual optimisation and Lagrangian duality,
will be familiar to anyone with a background in constrained optimisation. What this project
explores is how those classical ideas extend naturally to the sequential decision making
setting.

[View on GitHub](https://github.com/paulricc/cmdp-lagrangian){:target="_blank"}