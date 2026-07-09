"""
Scorecast training script (v2).

Improvements over v1:
  * Rolling Elo ratings computed chronologically (no ranking anachronism —
    v1 applied 2024 FIFA ranks to 1990s matches).
  * Single O(n) pass over history instead of O(n^2) form lookups.
  * Poisson goals model (expected goals per team -> full scoreline grid),
    which fixes systematic draw under-prediction and enables scoreline
    output + tournament simulation.
  * Recency-weighted form, goals for/against, and head-to-head features.
  * Proper time-based holdout with log loss / Brier / accuracy, compared
    against a multinomial logistic-regression baseline.
  * Saves prediction-time team state so app.py can build features without
    re-scanning history.

Outputs:
  model_bundle.pkl   {'poisson_home', 'poisson_away', 'scaler', 'features'}
  team_state.json    {team: {elo, form, gf, ga}}
  h2h.json           pairwise head-to-head records
"""

import json
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

from scorecast_model import (
    FEATURES, TeamState, build_features, elo_update,
    h2h_record, outcome_probs,
)

ELO_WARMUP_START = '1970-01-01'   # Elo starts accumulating here...
TRAIN_START = '1990-01-01'        # ...but only matches from here go in the training set
TEST_START = '2023-01-01'         # holdout: everything from this date forward
GOAL_CAP = 8                      # cap freak scorelines (31-0 etc.) for stability

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

results = pd.read_csv('results.csv', parse_dates=['date'])
results = results[results['home_score'].notna()].copy()
results = results[results['date'] >= ELO_WARMUP_START].sort_values('date').reset_index(drop=True)
results['home_score'] = results['home_score'].astype(int).clip(upper=GOAL_CAP)
results['away_score'] = results['away_score'].astype(int).clip(upper=GOAL_CAP)

# 0 = away win, 1 = draw, 2 = home win (same encoding app.py already expects)
results['result'] = 1
results.loc[results['home_score'] > results['away_score'], 'result'] = 2
results.loc[results['home_score'] < results['away_score'], 'result'] = 0

# ---------------------------------------------------------------------------
# Single chronological pass: build features BEFORE updating state
# ---------------------------------------------------------------------------

states = defaultdict(TeamState)
h2h = {}
rows = []

for row in results.itertuples(index=False):
    home, away = row.home_team, row.away_team
    neutral = str(row.neutral).upper() == 'TRUE' or row.neutral is True
    competitive = 'Friendly' not in str(row.tournament)
    hs, aw = states[home], states[away]

    if row.date >= pd.Timestamp(TRAIN_START):
        feats = build_features(hs, aw, h2h, home, away, neutral, competitive)
        feats['home_goals'] = row.home_score
        feats['away_goals'] = row.away_score
        feats['result'] = row.result
        feats['date'] = row.date
        rows.append(feats)

    # --- update state AFTER extracting features (no leakage) ---
    hs.elo, aw.elo = elo_update(hs.elo, aw.elo, row.home_score, row.away_score,
                                row.tournament, neutral)
    hs.record(row.home_score, row.away_score)
    aw.record(row.away_score, row.home_score)
    h2h_record(h2h, home, away, row.home_score, row.away_score)

df = pd.DataFrame(rows)
print(f"Training set: {len(df)} matches "
      f"({df['date'].min().date()} -> {df['date'].max().date()})")

# ---------------------------------------------------------------------------
# Time-based split (never random-split time series!)
# ---------------------------------------------------------------------------

train = df[df['date'] < TEST_START]
test = df[df['date'] >= TEST_START]
print(f"Train: {len(train)} matches | Test (holdout, >= {TEST_START}): {len(test)}")

scaler = StandardScaler()
X_train = scaler.fit_transform(train[FEATURES])
X_test = scaler.transform(test[FEATURES])

# ---------------------------------------------------------------------------
# Poisson goals model: one regressor per side
# ---------------------------------------------------------------------------

poisson_home = PoissonRegressor(alpha=1e-4, max_iter=1000)
poisson_away = PoissonRegressor(alpha=1e-4, max_iter=1000)
poisson_home.fit(X_train, train['home_goals'])
poisson_away.fit(X_train, train['away_goals'])

mu_h = poisson_home.predict(X_test)
mu_a = poisson_away.predict(X_test)
probs_poisson = np.array([outcome_probs(m1, m2)[1] for m1, m2 in zip(mu_h, mu_a)])

# ---------------------------------------------------------------------------
# Baseline: multinomial logistic regression (the v1 approach, on new features)
# ---------------------------------------------------------------------------

baseline = LogisticRegression(max_iter=1000, random_state=42)
baseline.fit(X_train, train['result'])
probs_baseline = baseline.predict_proba(X_test)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def brier_multiclass(y_true, probs, n_classes=3):
    onehot = np.eye(n_classes)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

def report(name, probs, y_true):
    pred = probs.argmax(axis=1)
    print(f"\n--- {name} ---")
    print(f"log loss : {log_loss(y_true, probs, labels=[0, 1, 2]):.4f}  (lower is better)")
    print(f"Brier    : {brier_multiclass(y_true, probs):.4f}  (lower is better)")
    print(f"accuracy : {accuracy_score(y_true, pred):.4f}")
    actual_draws = float((np.asarray(y_true) == 1).mean())
    pred_draw_mass = float(probs[:, 1].mean())
    pred_draw_calls = float((pred == 1).mean())
    print(f"draws    : actual {actual_draws:.1%} | avg predicted prob {pred_draw_mass:.1%}"
          f" | predicted as most likely {pred_draw_calls:.1%}")

y_test = test['result'].values
report("Poisson goals model", probs_poisson, y_test)
report("Logistic regression baseline", probs_baseline, y_test)

# ---------------------------------------------------------------------------
# Refit on ALL data (train + holdout) for the deployed model, then save
# ---------------------------------------------------------------------------

scaler_full = StandardScaler()
X_full = scaler_full.fit_transform(df[FEATURES])
poisson_home_full = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(X_full, df['home_goals'])
poisson_away_full = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(X_full, df['away_goals'])

joblib.dump({
    'poisson_home': poisson_home_full,
    'poisson_away': poisson_away_full,
    'scaler': scaler_full,
    'features': FEATURES,
}, 'model_bundle.pkl')

# Prediction-time state: final Elo/form/goal averages for every team
team_state = {
    team: {'elo': round(s.elo, 1), 'form': round(s.form, 4),
           'gf': round(s.gf, 3), 'ga': round(s.ga, 3)}
    for team, s in states.items()
}
with open('team_state.json', 'w') as f:
    json.dump(team_state, f, indent=1)

with open('h2h.json', 'w') as f:
    json.dump(h2h, f)

print("\nSaved model_bundle.pkl, team_state.json, h2h.json")
top = sorted(team_state.items(), key=lambda kv: -kv[1]['elo'])[:10]
print("\nTop 10 by Elo:")
for team, s in top:
    print(f"  {s['elo']:7.1f}  {team}")