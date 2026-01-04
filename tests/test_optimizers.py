import numpy as np

from src.strategies.allocations import equal_weight, mean_variance_weights, min_variance_weights, risk_parity_weights


def test_weights_sum_to_one_and_nonnegative():
    cov = np.array(
        [
            [0.01, 0.002, 0.001],
            [0.002, 0.015, 0.003],
            [0.001, 0.003, 0.02],
        ]
    )
    mu = np.array([0.08, 0.06, 0.07])

    w_min = min_variance_weights(cov)
    w_mv = mean_variance_weights(mu, cov, gamma=5.0)
    w_rp = risk_parity_weights(cov)
    w_eq = equal_weight(3)

    for w in [w_min, w_mv, w_rp, w_eq]:
        assert np.isclose(w.sum(), 1.0)
        assert np.all(w >= -1e-8)

    # stability: risk parity should tilt toward lower variance asset
    assert w_rp[0] >= w_rp[2]
