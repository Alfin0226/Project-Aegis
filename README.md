# Project Aegis

Closed-source algorithmic trading system for probability-based signal generation, autonomous paper execution, and live performance reporting.

## Live Dashboard

**[View Dashboard](https://alfin0226.github.io/Project-Aegis/)**

The dashboard is updated automatically and displays current paper-trading performance, open positions, equity history, and trade statistics.

---

## Overview

Project Aegis is an end-to-end algorithmic trading platform built to:

- generate directional trade signals from market data using deep learning
- apply portfolio-level and position-level risk controls
- execute paper trades automatically through the Alpaca API
- publish a live dashboard for monitoring system behavior and results

The production system is hosted privately in Microsoft Azure, while the dashboard is published separately for public viewing.

---

## Key Highlights

### Strategy Design
- Deep learning–based signal engine trained on engineered market features
- Multi-asset portfolio spanning equities, ETFs, and defensive assets
- Probability-driven entry and exit framework
- Tiered position sizing with rule-based capital allocation

### Risk Management
- Trailing-stop protection
- Per-position exposure caps
- Volatility-aware circuit-breaker logic
- Portfolio monitoring with drawdown tracking

### Automation
- Fully automated paper-trading workflow
- Cloud-hosted execution in Azure
- Scheduled reporting and dashboard refresh
- CI/CD integration for publishing updates

---

## Performance Snapshot

### Backtest Period
**Jan 2025 – Mar 2026**

- **+64.99% cumulative return** vs **+14.61% SPY buy-and-hold**
- **Sharpe ratio: 1.67** vs **0.71 for SPY**
- **Max drawdown: -16.56%** vs **-18.76% for SPY**
- **Recovery period: 36 days** vs **126 days for SPY**
- **62% win rate across 217 closed trades**
- Monte Carlo validation showed **95.5% of 1,000 resampled paths remained profitable**

### Live Paper Trading
See the live dashboard for current paper-trading results and active positions.

> Note: Live results are still early-stage and should be interpreted with appropriate caution until a larger sample of trades is accumulated.

---

## System Architecture

The system has four main components:

1. **Signal Engine**  
   Processes market data and produces probability-based trade signals.

2. **Execution Layer**  
   Translates signals into paper orders through Alpaca.

3. **Risk Engine**  
   Applies stop logic, exposure limits, and portfolio controls.

4. **Reporting Layer**  
   Generates dashboard-ready outputs for visualization and monitoring.

---

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **pandas / numpy**
- **scikit-learn**
- **alpaca-py**
- **GitHub Actions**
- **GitHub Pages**
- **Microsoft Azure**

---

## Public Repo Scope

This repository is intentionally limited to public-facing documentation and shareable reporting utilities.

Included here:
- project overview
- dashboard link
- sanitized reporting code/examples

Not included:
- production trading logic
- proprietary model code
- training pipeline
- model artifacts
- deployment credentials
- operational playbooks

---

## Notes

This project is operated as a research and paper-trading system.  
Nothing in this repository constitutes financial advice or an offer to trade securities.

---

## Contact

If you'd like to discuss the system design, infrastructure, modeling choices, or risk framework, feel free to connect.
