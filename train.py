import pandas as pd
import numpy as np
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = ['home_form', 'away_form', 'form_diff', 'neutral', 'is_competitive',
            'home_rank', 'away_rank', 'rank_diff']

results = pd.read_csv('results.csv', parse_dates=['date'])
results = results[results['home_score'].notna()].copy()
results = results[results['date'] >= '1990-01-01'].sort_values('date').reset_index(drop=True)
results['result'] = 1
results.loc[results['home_score'] > results['away_score'], 'result'] = 2
results.loc[results['home_score'] < results['away_score'], 'result'] = 0

latest_rankings = pd.read_csv('latest_rankings.csv', index_col=0).squeeze()

with open('name_map.json') as f:
    name_map = json.load(f)

def get_form(team, before_idx, n=10):
    past = results.iloc[:before_idx]
    matches = past[(past['home_team'] == team) | (past['away_team'] == team)].tail(n)
    if len(matches) == 0:
        return 0.5
    wins = (
        ((matches['home_team'] == team) & (matches['result'] == 2)) |
        ((matches['away_team'] == team) & (matches['result'] == 0))
    ).sum()
    return wins / len(matches)

rows = []
for i, row in results.iterrows():
    h, a = row['home_team'], row['away_team']
    h_form = get_form(h, i)
    a_form = get_form(a, i)
    h_r = name_map.get(h, h)
    a_r = name_map.get(a, a)
    h_rank = float(latest_rankings.get(h_r, 100))
    a_rank = float(latest_rankings.get(a_r, 100))
    neutral = 1 if str(row['neutral']).upper() == 'TRUE' else 0
    is_competitive = 0 if 'Friendly' in str(row['tournament']) else 1
    rows.append({
        'home_form': h_form,
        'away_form': a_form,
        'form_diff': h_form - a_form,
        'neutral': neutral,
        'is_competitive': is_competitive,
        'home_rank': h_rank,
        'away_rank': a_rank,
        'rank_diff': h_rank - a_rank,
        'result': row['result'],
    })

df = pd.DataFrame(rows)
X = df[FEATURES]
y = df['result']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Done. model.pkl and scaler.pkl saved.")
