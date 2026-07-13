from frost_doa import FROSTDOAEstimator


def test_public_class_name_has_no_development_version():
    assert "v16" not in FROSTDOAEstimator.__name__.lower()
