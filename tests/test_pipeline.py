from __future__ import annotations
from datetime import timedelta
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline import build_snapshot, normalize_name, within_trial_typical_error
from analysis import analyze_drops


def example_sources():
    start = pd.Timestamp('2026-04-01')
    games = []
    jumps = []
    for who, batter in [('Ada A', 'Ada A'), ('Beth B', 'B, Beth')]:
        for n in range(50):
            date = start + pd.Timedelta(days=n)
            games.append(dict(batter=batter, game_date=date.strftime('%Y-%m-%d'),
                              mlbam_game_pk=f'{who}-{n}', max_effort_runs=1,
                              max_effort_distance_covered_yards=10,
                              distance_covered_yards=100))
        for n in [5, 28, 29, 30, 32, 35, 40, 49, 50]:
            date = start + pd.Timedelta(days=n)
            impulse = 90.0 if n == 35 else 100.0
            for offset in [0, 2]:
                jumps.append(dict(Athlete=who, Date=date.strftime('%Y-%m-%d'),
                                  **{'Concentric Impulse [N s]': impulse + offset,
                                     'Concentric Impulse 100 ms [N s]': 50,
                                     'mRSI': 0.5}))
    return pd.DataFrame(games), pd.DataFrame(jumps)


def test_lookbacks_and_test_day_exclusion():
    pp, jumps = example_sources()
    # Make test-day load huge; none of 1/3/7/14/21/28 windows should include it.
    pp.loc[pp['game_date'].eq('2026-05-01'), 'max_effort_distance_covered_yards'] = 9000
    snapshot, audit, replicates = build_snapshot(pp, jumps)
    row = snapshot[(snapshot['Athlete'] == 'Ada A') & (snapshot['jump_date'] == pd.Timestamp('2026-05-01'))].iloc[0]
    assert row['max_effort_distance_covered_yards_1d'] == 10
    assert row['max_effort_distance_covered_yards_3d'] == 30
    assert row['max_effort_distance_covered_yards_7d'] == 70
    assert row['max_effort_distance_covered_yards_14d'] == 140
    assert row['max_effort_distance_covered_yards_21d'] == 210
    assert row['max_effort_distance_covered_yards_28d'] == 280
    assert audit['eligible_athletes'] == 2
    assert len(replicates) > 0


def test_doubleheaders_dedup_and_names():
    pp, jumps = example_sources()
    row = pp.iloc[1].copy()
    row['mlbam_game_pk'] = 'extra game'
    pp = pd.concat([pp, pd.DataFrame([row, row])], ignore_index=True)
    snapshot, audit, _ = build_snapshot(pp, jumps)
    assert audit['exact_duplicate_games_removed'] == 1
    result = snapshot[(snapshot['Athlete'] == 'Ada A') & (snapshot['jump_date'] == pd.Timestamp('2026-04-29'))].iloc[0]
    assert result['games_played_28d'] == 29  # day 1 doubleheader
    assert normalize_name('B, Beth') == normalize_name('Beth B')


def test_acute_drop_and_trial_sd():
    pp, jumps = example_sources()
    snapshot, _, repeats = build_snapshot(pp, jumps)
    summary, events = analyze_drops(snapshot, 'distance_covered_yards', 'ci_ns', 'prior1')
    assert len(summary) == 6
    assert events['drop_flag'].sum() == 2  # one per athlete on day 35
    te = within_trial_typical_error(repeats, 'ci_ns')
    assert te['groups'] > 0
    assert te['te'] > 0


def test_coverage_excludes_early_and_far_late_jumps():
    pp, jumps = example_sources()
    jumps = pd.concat([jumps, pd.DataFrame([{'Athlete':'Ada A','Date':'2026-06-10','Concentric Impulse [N s]':110}])], ignore_index=True)
    snapshot, audit, _ = build_snapshot(pp, jumps)
    assert (snapshot['jump_date'] >= pd.Timestamp('2026-04-29')).all()
    assert not (snapshot['jump_date'] >= pd.Timestamp('2026-06-10')).any()
    assert audit['excluded_player_days'] > 0


def test_missing_load_not_automatically_zero():
    pp, jumps = example_sources()
    pp.loc[0,'max_effort_runs'] = None
    try:
        build_snapshot(pp, jumps)
    except ValueError as exc:
        assert 'missing/non-numeric' in str(exc)
    else:
        raise AssertionError('missing load was silently imputed as zero')
    _, audit, _ = build_snapshot(pp, jumps, blanks_are_zero=True)
    assert audit['blank_game_metrics_treated_as_zero']['max_effort_runs'] == 1
