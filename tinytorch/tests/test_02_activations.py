import numpy as np

from tinytorch import Tensor, ReLU, Sigmoid, Tanh, GELU, Softmax


def test_relu_zeroes_negatives_keeps_positives():
    y = ReLU()(Tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(y.data, [0, 0, 0, 1, 2])
    assert y.shape == (5,)
    assert y.dtype == np.float32


def test_activations_keep_shape_and_have_no_parameters():
    x = Tensor.randn(4, 3, 5, seed=0)
    for act in (ReLU(), Sigmoid(), Tanh(), GELU(), Softmax()):
        assert act(x).shape == (4, 3, 5)
        assert act.parameters() == []


def test_sigmoid_values_and_symmetry():
    s = Sigmoid()
    np.testing.assert_allclose(s(Tensor([0.0])).data, [0.5])
    x = Tensor([-3.0, -1.0, 0.5, 2.0])
    # σ(-x) = 1 - σ(x)
    np.testing.assert_allclose(s(-x).data, 1 - s(x).data, atol=1e-6)


def test_sigmoid_is_stable_at_extremes():
    with np.errstate(over="raise", invalid="raise"):   # tràn số hay NaN thành lỗi
        y = Sigmoid()(Tensor([-1000.0, -100.0, 100.0, 1000.0]))
    np.testing.assert_allclose(y.data, [0, 0, 1, 1], atol=1e-30)
    assert not np.isnan(y.data).any()


def test_tanh_range_and_identity():
    x = Tensor(np.linspace(-5, 5, 11))
    y = Tanh()(x)
    assert np.all(np.abs(y.data) <= 1)
    np.testing.assert_allclose(y.data, -Tanh()(-x).data, atol=1e-6)
    # tanh(x) = 2σ(2x) - 1
    np.testing.assert_allclose(y.data, 2 * Sigmoid()(x * 2).data - 1, atol=1e-6)


def test_gelu_matches_exact_formula():
    from math import erf, sqrt
    xs = np.linspace(-4, 4, 81)
    exact = np.array([v * 0.5 * (1 + erf(v / sqrt(2))) for v in xs])
    approx = GELU()(Tensor(xs)).data
    assert np.max(np.abs(approx - exact)) < 0.03
    np.testing.assert_allclose(GELU()(Tensor([0.0])).data, [0.0])
    assert approx.min() > -0.2                  # cực tiểu ≈ -0.17 quanh x ≈ -0.75


def test_softmax_sums_to_one_along_dim():
    x = Tensor([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0]])
    p = Softmax()(x)
    np.testing.assert_allclose(p.data.sum(axis=-1), [1, 1], rtol=1e-6)
    np.testing.assert_allclose(p.data[0], [0.659, 0.2424, 0.0986], atol=1e-3)
    np.testing.assert_allclose(p.data[1], [1 / 3] * 3, rtol=1e-6)
    cols = Softmax(dim=0)(x)
    np.testing.assert_allclose(cols.data.sum(axis=0), [1, 1, 1], rtol=1e-6)


def test_softmax_shift_invariance_and_large_inputs():
    x = Tensor([1.0, 2.0, 3.0])
    with np.errstate(over="raise", invalid="raise"):
        big = Softmax()(x + 1000)
    np.testing.assert_allclose(big.data, Softmax()(x).data, rtol=1e-5)


def test_softmax_masks_with_negative_infinity():
    x = Tensor([[1.0, -np.inf, 1.0], [-np.inf, -np.inf, -np.inf]])
    with np.errstate(over="raise", invalid="raise"):
        p = Softmax()(x)
    np.testing.assert_allclose(p.data, [[0.5, 0, 0.5], [0, 0, 0]])


def test_two_class_softmax_equals_sigmoid_of_difference():
    z = Tensor([[3.0, 1.0]])
    p = Softmax()(z).data[0, 0]
    np.testing.assert_allclose(p, Sigmoid()(Tensor([2.0])).data[0], rtol=1e-6)
