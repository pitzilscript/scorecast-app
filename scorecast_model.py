"""
Shared model utilities for Scorecast.

Used by both train.py (to build the training set chronologically) and
app.py (to build features for a new matchup and turn goal expectations
into outcome probabilities).
"""

import numpy as np
from collections import deque
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEATURES = [
    'elo_home', 'elo_away', 'elo_diff',
    'form_home', 'form_away',
    'gf_home', 'ga_home', 'gf_away', 'ga_away',
    'h2h_home',
    'neutral', 'is_competitive',
]

ELO_START = 1500.0
ELO_HOME_ADV = 60.0          # Elo points added to home side when not neutral
FORM_WINDOW = 10             # matches considered for form / goals averages
FORM_DECAY = 0.85            # exponential decay per match (most recent = weight 1)
MAX_GOALS = 10               # grid size when converting Poisson means to W/D/L


def elo_k(tournament: str) -> float:
    """K-factor by match importance (standard football-Elo convention)."""
    t = str(tournament)
    if 'FIFA World Cup' in t and 'qualification' not in t.lower():
        return 60.0
    if 'Friendly' in t:
        return 20.0
    return 40.0                # qualifiers, continental cups, Nations League...


def elo_expected(elo_h: float, elo_a: float, neutral: bool) -> float:
    """Expected score (0..1) for the home side."""
    adv = 0.0 if neutral else ELO_HOME_ADV
    return 1.0 / (1.0 + 10 ** (-((elo_h + adv) - elo_a) / 400.0))


def elo_update(elo_h, elo_a, home_goals, away_goals, tournament, neutral):
    """Return updated (elo_home, elo_away) after a match."""
    if home_goals > away_goals:
        actual = 1.0
    elif home_goals < away_goals:
        actual = 0.0
    else:
        actual = 0.5
    k = elo_k(tournament)
    # goal-difference multiplier (as used by eloratings.net)
    gd = abs(home_goals - away_goals)
    if gd == 2:
        k *= 1.5
    elif gd >= 3:
        k *= (1.75 + (gd - 3) / 8.0)
    exp = elo_expected(elo_h, elo_a, neutral)
    delta = k * (actual - exp)
    return elo_h + delta, elo_a - delta


# ---------------------------------------------------------------------------
# Per-team rolling state
# ---------------------------------------------------------------------------

class TeamState:
    """Rolling Elo + recent-match state for one team."""

    __slots__ = ('elo', 'recent')

    def __init__(self):
        self.elo = ELO_START
        # each entry: (points, goals_for, goals_against); newest last
        self.recent = deque(maxlen=FORM_WINDOW)

    def record(self, gf: int, ga: int):
        pts = 3 if gf > ga else (1 if gf == ga else 0)
        self.recent.append((pts, gf, ga))

    def _weighted(self, idx: int, default: float) -> float:
        if not self.recent:
            return default
        vals = [m[idx] for m in self.recent]
        w = np.array([FORM_DECAY ** (len(vals) - 1 - i) for i in range(len(vals))])
        return float(np.dot(vals, w) / w.sum())

    @property
    def form(self) -> float:
        """Recency-weighted points per game, scaled to 0..1."""
        return self._weighted(0, default=1.5) / 3.0

    @property
    def gf(self) -> float:
        return self._weighted(1, default=1.2)

    @property
    def ga(self) -> float:
        return self._weighted(2, default=1.2)


def h2h_key(team_a: str, team_b: str) -> str:
    return '|'.join(sorted([team_a, team_b]))


def h2h_home_share(h2h_store: dict, home: str, away: str) -> float:
    """Home team's historical points share vs this opponent (0..1, default 0.5)."""
    rec = h2h_store.get(h2h_key(home, away))
    if not rec or rec['n'] == 0:
        return 0.5
    pts = rec['pts'].get(home, 0.0)
    return pts / (3.0 * rec['n'])


def h2h_record(h2h_store: dict, home: str, away: str, hg: int, ag: int):
    key = h2h_key(home, away)
    rec = h2h_store.setdefault(key, {'n': 0, 'pts': {}})
    rec['n'] += 1
    if hg > ag:
        rec['pts'][home] = rec['pts'].get(home, 0.0) + 3
    elif ag > hg:
        rec['pts'][away] = rec['pts'].get(away, 0.0) + 3
    else:
        rec['pts'][home] = rec['pts'].get(home, 0.0) + 1
        rec['pts'][away] = rec['pts'].get(away, 0.0) + 1


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_features(home_state: TeamState, away_state: TeamState,
                   h2h_store: dict, home: str, away: str,
                   neutral: bool, is_competitive: bool) -> dict:
    """Feature dict for one matchup, using only information known beforehand."""
    return {
        'elo_home': home_state.elo,
        'elo_away': away_state.elo,
        'elo_diff': home_state.elo - away_state.elo,
        'form_home': home_state.form,
        'form_away': away_state.form,
        'gf_home': home_state.gf,
        'ga_home': home_state.ga,
        'gf_away': away_state.gf,
        'ga_away': away_state.ga,
        'h2h_home': h2h_home_share(h2h_store, home, away),
        'neutral': int(neutral),
        'is_competitive': int(is_competitive),
    }


# ---------------------------------------------------------------------------
# Poisson goals -> outcome probabilities / scorelines
# ---------------------------------------------------------------------------

def outcome_probs(mu_home: float, mu_away: float):
    """
    Convert expected goals (Poisson means) into a score-probability matrix
    and (P_away_win, P_draw, P_home_win) — ordered to match the original
    label encoding: 0 = away win, 1 = draw, 2 = home win.
    """
    mu_home = float(np.clip(mu_home, 0.05, 6.0))
    mu_away = float(np.clip(mu_away, 0.05, 6.0))
    goals = np.arange(MAX_GOALS + 1)
    p_h = poisson.pmf(goals, mu_home)
    p_a = poisson.pmf(goals, mu_away)
    grid = np.outer(p_h, p_a)          # grid[i, j] = P(home scores i, away scores j)
    grid /= grid.sum()                 # renormalise tail mass
    p_home = float(np.tril(grid, -1).sum())
    p_draw = float(np.trace(grid))
    p_away = float(np.triu(grid, 1).sum())
    return grid, (p_away, p_draw, p_home)


def top_scorelines(grid: np.ndarray, k: int = 3):
    """Most likely scorelines: list of ((home_goals, away_goals), prob)."""
    flat = [((i, j), grid[i, j]) for i in range(grid.shape[0])
            for j in range(grid.shape[1])]
    flat.sort(key=lambda x: -x[1])
    return flat[:k]