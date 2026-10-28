import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import fit_window


def test_random_slope_fit_and_actual_pair_count():
    rng = np.random.default_rng(31)
    rows = []
    for i in range(9):
        for t in range(24):
            x = 300 + 30*i + rng.normal(0, 70)
            date = pd.Timestamp('2026-04-01') + pd.Timedelta(days=t*3)
            y = 245 + 11*i - 0.02*(x-300-30*i) + rng.normal(0, 6)
            rows.append({'player_key': f'player_{i}', 'jump_date': date,
                         'days_since_prior_jump': 3, 'ci_ns': y,
                         'max_effort_distance_covered_yards_7d': x})
    result = fit_window(pd.DataFrame(rows), 'max_effort_distance_covered_yards',
                        'ci_ns', 7, compare_intercept=True)
    assert result['Converged']
    assert result['Athletes'] == 9
    assert result['Jump observations'] == 216
    assert result['Residual pairs'] == 216 - 9
    assert np.isfinite(result['Within-X SD'])
    assert np.isfinite(result['Standardized effect'])
    assert np.isfinite(result['Random-slope AIC'])
