import gradio as gr
import pandas as pd
import numpy as np
import joblib
import json

# Load model and supporting data
model           = joblib.load('model.pkl')
scaler          = joblib.load('scaler.pkl')
latest_rankings = pd.read_csv('latest_rankings.csv', index_col=0).squeeze()

with open('name_map.json') as f:
    name_map = json.load(f)
with open('wc_teams.json') as f:
    wc_teams = json.load(f)
with open('all_teams.json') as f:
    all_teams = json.load(f)

results = pd.read_csv('results.csv', parse_dates=['date'])
results = results[results['home_score'].notna()].copy()
results = results[results['date'] >= '1990-01-01'].sort_values('date').reset_index(drop=True)
results['result'] = 1
results.loc[results['home_score'] > results['away_score'], 'result'] = 2
results.loc[results['home_score'] < results['away_score'], 'result'] = 0

FEATURES = ['home_form', 'away_form', 'form_diff', 'neutral', 'is_competitive',
            'home_rank', 'away_rank', 'rank_diff']

BALL_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"
  style="display:inline-block;width:0.82em;height:0.82em;vertical-align:middle;margin:0 2px -6px 2px;">
  <defs><clipPath id="bc"><circle cx="50" cy="50" r="44"/></clipPath></defs>
  <circle cx="50" cy="50" r="44" fill="white"/>
  <polygon points="50,7 61,16 57,29 43,29 39,16"       fill="#1a1a1a" clip-path="url(#bc)"/>
  <polygon points="81,23 85,35 74,43 62,37 64,24"       fill="#1a1a1a" clip-path="url(#bc)"/>
  <polygon points="86,65 78,76 66,74 62,61 72,53"       fill="#1a1a1a" clip-path="url(#bc)"/>
  <polygon points="50,93 39,84 43,71 57,71 61,84"       fill="#1a1a1a" clip-path="url(#bc)"/>
  <polygon points="14,65 22,76 34,74 38,61 28,53"       fill="#1a1a1a" clip-path="url(#bc)"/>
  <polygon points="19,23 15,35 26,43 38,37 36,24"       fill="#1a1a1a" clip-path="url(#bc)"/>
  <circle cx="50" cy="50" r="44" fill="none" stroke="#1a1a1a" stroke-width="5"/>
</svg>"""


def get_team_form(team, n=10):
    matches = results[(results['home_team'] == team) | (results['away_team'] == team)].tail(n)
    if len(matches) == 0:
        return 0.5
    wins = (
        ((matches['home_team'] == team) & (matches['result'] == 2)) |
        ((matches['away_team'] == team) & (matches['result'] == 0))
    ).sum()
    return wins / len(matches)


def get_last_5(team):
    matches = results[(results['home_team'] == team) | (results['away_team'] == team)].tail(5)
    icons = []
    for _, m in matches.iterrows():
        if m['home_team'] == team:
            icons.append('W' if m['result'] == 2 else ('D' if m['result'] == 1 else 'L'))
        else:
            icons.append('W' if m['result'] == 0 else ('D' if m['result'] == 1 else 'L'))
    return icons


def badge(r):
    colors = {'W': '#27ae60', 'D': '#555555', 'L': '#c0392b'}
    return (f'<span style="background:{colors[r]};color:white;padding:5px 12px;'
            f'border-radius:5px;font-weight:bold;font-size:1em;margin:2px;display:inline-block;">{r}</span>')


def prediction_html(team1, win1, draw, win2, team2):
    rows = [
        (f"{team1.upper()} WIN", win1),
        ("DRAW", draw),
        (f"{team2.upper()} WIN", win2),
    ]
    top_prob = max(win1, draw, win2)

    html = '<div style="font-family:\'Inter\',Arial,sans-serif;padding:4px 0;">'
    for label, prob in rows:
        is_top = abs(prob - top_prob) < 0.001
        bar_pct = f"{prob*100:.1f}%"
        bar_fill = "#0d0d1a" if is_top else "#b0b0b8"
        row_bg   = "#f8f8f8" if is_top else "transparent"
        label_color = "#0d0d1a"
        label_weight = "800" if is_top else "600"
        pct_color = "#0d0d1a" if is_top else "#555"
        marker = ('<span style="background:#0d0d1a;color:white;font-size:0.72em;'
                  'padding:2px 8px;border-radius:3px;margin-left:10px;'
                  'letter-spacing:1px;font-weight:700;">PREDICTED</span>') if is_top else ""

        html += f"""
        <div style="display:flex;align-items:center;gap:14px;padding:11px 14px;
                    border-radius:8px;background:{row_bg};margin-bottom:5px;">
          <div style="width:200px;font-size:1.05em;font-weight:{label_weight};
                      color:{label_color};letter-spacing:1px;white-space:nowrap;
                      display:flex;align-items:center;">
            {label}{marker}
          </div>
          <div style="flex:1;background:#e0e0e0;border-radius:4px;height:13px;">
            <div style="width:{bar_pct};background:{bar_fill};height:13px;border-radius:4px;"></div>
          </div>
          <div style="width:52px;text-align:right;font-size:1.15em;font-weight:800;color:{pct_color};">
            {prob:.1%}
          </div>
        </div>"""
    html += "</div>"
    return html


def predict_and_explain(team1, team2, venue):
    if not team1 or not team2:
        return "", ""
    if team1 == team2:
        return "<p style='color:white;'>Select two different teams.</p>", ""

    # Determine home/away
    if "Neutral" in str(venue):
        home, away, neutral, swapped = team1, team2, 1, False
        venue_text = "Neutral venue — no home advantage"
    elif team2 in str(venue):
        home, away, neutral, swapped = team2, team1, 0, True
        venue_text = f"{team2} is playing at home"
    else:
        home, away, neutral, swapped = team1, team2, 0, False
        venue_text = f"{team1} is playing at home"

    h_r = name_map.get(home, home)
    a_r = name_map.get(away, away)
    h_form = get_team_form(home)
    a_form = get_team_form(away)
    h_rank = float(latest_rankings.get(h_r, 100))
    a_rank = float(latest_rankings.get(a_r, 100))

    t1_form  = get_team_form(team1)
    t2_form  = get_team_form(team2)
    t1_rank  = float(latest_rankings.get(name_map.get(team1, team1), 100))
    t2_rank  = float(latest_rankings.get(name_map.get(team2, team2), 100))
    t1_last  = get_last_5(team1)
    t2_last  = get_last_5(team2)

    X_pred = pd.DataFrame([[
        h_form, a_form, h_form - a_form,
        neutral, 1,
        h_rank, a_rank, h_rank - a_rank
    ]], columns=FEATURES)

    X_scaled = scaler.transform(X_pred)
    probs = model.predict_proba(X_scaled)[0]
    t1_win, t2_win = (probs[0], probs[2]) if swapped else (probs[2], probs[0])
    draw = probs[1]

    pred = prediction_html(team1, float(t1_win), float(draw), float(t2_win), team2)

    rank_diff = abs(t1_rank - t2_rank)
    rank_edge_text = (
        f"{team1} has a {rank_diff:.0f} point ranking advantage"
        if t1_rank < t2_rank else
        f"{team2} has a {rank_diff:.0f} point ranking advantage"
        if t2_rank < t1_rank else
        "Rankings are closely matched"
    )
    form_diff = abs(t1_form - t2_form)
    form_edge_text = (
        f"{team1} has better recent form ({t1_form:.0%} vs {t2_form:.0%})"
        if t1_form > t2_form + 0.05 else
        f"{team2} has better recent form ({t2_form:.0%} vs {t1_form:.0%})"
        if t2_form > t1_form + 0.05 else
        f"Both teams have similar recent form ({t1_form:.0%} vs {t2_form:.0%})"
    )

    card = """
    <link href="https://fonts.googleapis.com/css2?family=Anton&family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@400;600&display=swap" rel="stylesheet">
    <style>
      .why-wrap { font-family: 'Inter', Arial, sans-serif; padding: 4px; }
      .teams-grid { display: grid; grid-template-columns: 1fr 56px 1fr; gap: 14px; align-items: start; margin-bottom: 16px; }
      .team-card { background: white; border: 1px solid #ddd;
                   border-radius: 12px; padding: 20px; color: #0d0d1a; text-align: center; }
      .team-name { font-family: 'Barlow Condensed', sans-serif; font-style: italic;
                   font-weight: 800; font-size: 1.5em; color: #0d0d1a;
                   letter-spacing: 2px; margin-bottom: 16px; }
      .stat-label { color: #555; font-size: 0.78em; text-transform: uppercase;
                    letter-spacing: 1px; margin-bottom: 2px; font-weight: 500; }
      .stat-value { font-family: 'Barlow Condensed', sans-serif; font-style: italic;
                    font-weight: 800; font-size: 2.2em;
                    letter-spacing: 1px; margin-bottom: 12px; color: #0d0d1a; }
      .vs-col { text-align: center; padding-top: 55px; font-family: 'Inter', sans-serif;
                font-size: 1em; font-weight: 500; color: #999; letter-spacing: 2px; }
      .factors { background: white; border: 1px solid #ddd;
                 border-radius: 10px; padding: 16px; }
      .factors-title { font-family: 'Barlow Condensed', sans-serif; font-style: italic;
                       font-weight: 800; color: #0d0d1a; font-size: 1.05em;
                       letter-spacing: 3px; margin-bottom: 10px; }
      .factor-row { margin-bottom: 7px; font-size: 0.95em; color: #333; font-weight: 400; }
      .factor-key { color: #0d0d1a; font-weight: 600; }
    </style>
    """

    card += f"""
    <div class="why-wrap">
      <div class="teams-grid">
        <div class="team-card">
          <div class="team-name">{team1.upper()}</div>
          <div class="stat-label">FIFA Ranking</div>
          <div class="stat-value">#{int(t1_rank)}</div>
          <div class="stat-label">Win Rate (last 10)</div>
          <div class="stat-value">{t1_form:.0%}</div>
          <div class="stat-label" style="margin-bottom:6px;">Last 5 Results</div>
          <div>{''.join(badge(r) for r in t1_last)}</div>
        </div>

        <div class="vs-col">VS</div>

        <div class="team-card">
          <div class="team-name">{team2.upper()}</div>
          <div class="stat-label">FIFA Ranking</div>
          <div class="stat-value">#{int(t2_rank)}</div>
          <div class="stat-label">Win Rate (last 10)</div>
          <div class="stat-value">{t2_form:.0%}</div>
          <div class="stat-label" style="margin-bottom:6px;">Last 5 Results</div>
          <div>{''.join(badge(r) for r in t2_last)}</div>
        </div>
      </div>

      <div class="factors">
        <div class="factors-title">KEY FACTORS</div>
        <div class="factor-row"><span class="factor-key">Ranking edge:</span> {rank_edge_text}</div>
        <div class="factor-row"><span class="factor-key">Form edge:</span> {form_edge_text}</div>
        <div class="factor-row"><span class="factor-key">Venue:</span> {venue_text}</div>
      </div>
    </div>
    """

    return pred, card


def update_venue_choices(t1, t2):
    t1_label = f"{t1} is Home" if t1 else "Team 1 is Home"
    t2_label = f"{t2} is Home" if t2 else "Team 2 is Home"
    return gr.Radio(choices=[t1_label, "Neutral Venue", t2_label], value="Neutral Venue")


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:ital,wght@1,800&family=Inter:wght@400;500&display=swap');

body, .gradio-container {
    background: #ebebeb !important;
    min-height: 100vh;
    font-family: 'Inter', Arial, sans-serif !important;
}
button[role="tab"] {
    color: #0d0d1a !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9em !important;
    font-weight: 500 !important;
    letter-spacing: 1px !important;
    background: transparent !important;
    border: 2px solid #0d0d1a !important;
    border-radius: 999px !important;
}
button[role="tab"][aria-selected="true"] {
    background: #0d0d1a !important;
    color: white !important;
}
label span, .block label span {
    color: #0d0d1a !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-size: 0.85em !important;
    font-family: 'Inter', sans-serif !important;
}
.wrap label span { color: #0d0d1a !important; font-size: 0.95em !important; font-weight: 500 !important; }
input[type="radio"] { accent-color: #27ae60 !important; }
select:focus, .wrap select:focus { border-color: #27ae60 !important; outline-color: #27ae60 !important; }
.svelte-1gfkn6j:focus, input:focus, textarea:focus { border-color: #27ae60 !important; outline-color: #27ae60 !important; }
:root {
    --color-accent: #27ae60 !important;
    --color-accent-soft: #d4edda !important;
    --primary-500: #27ae60 !important;
    --primary-600: #219a52 !important;
}
button.primary {
    background: #0d0d1a !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1em !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
}
button.primary:hover { background: #2a2a3a !important; }
.section-label {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    color: #0d0d1a !important;
    font-size: 1em !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid #0d0d1a;
    padding-bottom: 6px;
    margin-bottom: 14px;
}
"""

HEADER = """
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:ital,wght@1,800&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<div style="text-align:center;padding:24px 0 28px 0;">
  <div style="font-family:'Barlow Condensed',sans-serif;font-style:italic;font-weight:800;
              color:#0d0d1a;font-size:3.8em;letter-spacing:6px;line-height:1;">
    SCORECAST
  </div>
  <div style="font-family:'Inter',sans-serif;color:#666;font-size:0.9em;
              letter-spacing:2px;margin-top:10px;font-weight:500;">
    2026 FIFA WORLD CUP &nbsp;&middot;&nbsp; MATCH OUTCOME PREDICTOR
  </div>
</div>
"""

FOOTER = """
<div style="text-align:center;padding:28px 0 16px 0;font-family:'Barlow Condensed',sans-serif;">
  <div style="color:#999;font-size:0.85em;letter-spacing:1px;">
    Dataset credits:
    <a href="https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017"
       style="color:#555;text-decoration:underline;" target="_blank">FIFA Rankings</a>
    &nbsp;&middot;&nbsp;
    <a href="https://www.kaggle.com/datasets/caspersolheim/fifa-world-rankings"
       style="color:#555;text-decoration:underline;" target="_blank">Historical Match Data</a>
  </div>
</div>
"""


with gr.Blocks(css=CSS, title="Scorecast") as demo:
    gr.HTML(HEADER)

    with gr.Tabs():

        with gr.Tab("2026 World Cup"):
            gr.HTML('<p style="color:#555;font-size:1.05em;margin-bottom:16px;">Select two of the 48 qualified nations. All World Cup matches are at neutral venues.</p>')
            with gr.Row():
                t1_wc = gr.Dropdown(choices=wc_teams, label="Team 1")
                t2_wc = gr.Dropdown(choices=wc_teams, label="Team 2")
            venue_wc = gr.State("Neutral Venue")
            btn_wc   = gr.Button("Predict", variant="primary")
            gr.HTML('<div class="section-label">THE PREDICTION</div>')
            out_pred_wc = gr.HTML()
            gr.HTML('<div class="section-label">WHY</div>')
            out_why_wc  = gr.HTML()
            btn_wc.click(fn=predict_and_explain,
                         inputs=[t1_wc, t2_wc, venue_wc],
                         outputs=[out_pred_wc, out_why_wc])

        with gr.Tab("General International"):
            gr.HTML('<p style="color:#555;font-size:1.05em;margin-bottom:16px;">Any two international teams from the full historical dataset.</p>')
            with gr.Row():
                t1_gen = gr.Dropdown(choices=all_teams, label="Team 1")
                t2_gen = gr.Dropdown(choices=all_teams, label="Team 2")
            venue_gen = gr.Radio(
                choices=["Team 1 is Home", "Neutral Venue", "Team 2 is Home"],
                value="Neutral Venue",
                label="Venue"
            )
            t1_gen.change(fn=update_venue_choices, inputs=[t1_gen, t2_gen], outputs=venue_gen)
            t2_gen.change(fn=update_venue_choices, inputs=[t1_gen, t2_gen], outputs=venue_gen)
            btn_gen = gr.Button("Predict", variant="primary")
            gr.HTML('<div class="section-label">THE PREDICTION</div>')
            out_pred_gen = gr.HTML()
            gr.HTML('<div class="section-label">WHY</div>')
            out_why_gen  = gr.HTML()
            btn_gen.click(fn=predict_and_explain,
                          inputs=[t1_gen, t2_gen, venue_gen],
                          outputs=[out_pred_gen, out_why_gen])

    gr.HTML(FOOTER)

if __name__ == '__main__':
    demo.launch()
