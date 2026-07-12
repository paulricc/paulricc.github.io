---
layout: page
title: Projects
permalink: /projects/
---

### Stock Price Forecasting with Time Series Neural Networks
*Python · PyTorch · LSTM · ARIMA · Time Series Analysis · MLOps*

Can deep learning models outperform classical statistical approaches in forecasting stock 
prices? That is the question this project sets out to answer. Using the STOXX Europe 600 
index as the target, I compared the predictive performance of a Temporal Neural Network 
(TNN), an LSTM, and two ARIMA configurations across three forecasting horizons: 1, 7, 
and 30 days.

The TNN came out on top at shorter horizons, achieving an RMSE of 0.0268 and an R² of 
0.95 at the 1-day horizon. As the forecasting horizon grows, predicting stock prices 
becomes — unsurprisingly — harder for everyone.

Originally developed as my master's thesis at Bocconi University, I am now rewriting the 
codebase from scratch into a production-ready repository, applying the software engineering 
and MLOps practices I have built since then.

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