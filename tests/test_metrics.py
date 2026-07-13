from frost_doa.metrics import cardinality_aware_set_error


def test_equal_sets_are_zero():
    assert cardinality_aware_set_error([-10, 20], [20, -10]) == (0.0, 0.0)


def test_cardinality_penalty():
    rmse, mae = cardinality_aware_set_error([0], [0, 10], penalty_deg=25)
    assert abs(rmse - (625 / 2) ** 0.5) < 1e-12
    assert abs(mae - 12.5) < 1e-12
