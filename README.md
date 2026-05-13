# Scorecast

A machine learning web app that predicts international soccer match outcomes — built around the 2026 FIFA World Cup.

![Scorecast screenshot](screenshot.png)

## How it works

Logistic Regression trained on international match results from 1990 to present. For each matchup, the model uses:

- Recent win rate (last 10 matches per team)
- FIFA ranking
- Home vs. away vs. neutral venue
- Whether the match is competitive (World Cup, qualifiers) or a friendly

## Features

- **2026 World Cup tab** — all 48 qualified nations, neutral venue (matches are played in the US, Canada, and Mexico)
- **General International tab** — any two teams from the full historical dataset, with home/neutral/away venue selector
- Probability bar chart with predicted outcome highlighted
- Team stats card showing FIFA ranking, win rate, and last 5 results

## Tech stack

Python · scikit-learn · pandas · Gradio

**Datasets**
- [International Football Results (Kaggle)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- [FIFA World Rankings (Kaggle)](https://www.kaggle.com/datasets/caspersolheim/fifa-world-rankings)

## Run locally

```bash
git clone https://github.com/pitzilscript/scorecast.git
cd scorecast
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Then open `http://localhost:7860` in your browser.
