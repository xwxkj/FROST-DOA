from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from frost_doa import FROSTDOAEstimator, cardinality_aware_set_error
from frost_doa.fixture import iter_fixture

ROOT = Path(__file__).resolve().parents[1]


def test_first_sample_of_each_scenario_matches_frozen_reference():
    reference = pd.read_csv(ROOT / "reference" / "frost_doa_frozen_sample_scores.csv")
    expected = reference[reference["sample_id"].str.endswith("#0000")].set_index("sample_id")
    estimator = FROSTDOAEstimator()
    seen = set()
    for sample in iter_fixture(ROOT / "data" / "smoke_fixture.h5"):
        if sample.scenario in seen:
            continue
        seen.add(sample.scenario)
        result = estimator.estimate(sample.snapshots)
        rmse, mae = cardinality_aware_set_error(result.doas_deg, sample.true_doas_deg)
        row = expected.loc[sample.sample_id]
        assert np.isclose(rmse, row["set_rmse"], atol=1e-10)
        assert np.isclose(mae, row["set_mae"], atol=1e-10)
        assert result.num_sources == int(row["k_hat"])
    assert len(seen) == 12
