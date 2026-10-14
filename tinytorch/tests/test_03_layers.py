import numpy as np
import pytest

from tinytorch import Tensor, ReLU, Layer, Linear, Dropout, Sequential, manual_seed


def test_layer_base_contract():
    class Double(Layer):
        def forward(self, x):
            return x * 2

    np.testing.assert_array_equal(Double()(Tensor([1.0, 2.0])).data, [2, 4])
    assert Double().parameters() == []
    with pytest.raises(NotImplementedError):
        Layer()(Tensor([1.0]))


def test_linear_shapes_and_parameters():
    layer = Linear(784, 256)
    assert layer.weight.shape == (784, 256)
    assert layer.bias.shape == (256,)
    assert layer.parameters() == [layer.weight, layer.bias]
    assert all(p.requires_grad for p in layer.parameters())
    assert Linear(784, 256, bias=False).parameters()[0].shape == (784, 256)
    assert len(Linear(784, 256, bias=False).parameters()) == 1
    assert Linear(4, 3)(Tensor.randn(32, 4, seed=0)).shape == (32, 3)


def test_linear_forward_matches_numpy():
    layer = Linear(3, 2)
    layer.weight = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    layer.bias = Tensor([0.5, -0.5])
    y = layer(Tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(y.data, [[4.5, 4.5], [0.5, -0.5]])


def test_linear_init_scale_and_zero_bias():
    manual_seed(0)
    layer = Linear(1000, 500)
    assert abs(layer.weight.data.std() - np.sqrt(1 / 1000)) < 0.002
    assert abs(layer.weight.data.mean()) < 0.002
    np.testing.assert_array_equal(layer.bias.data, 0)


def test_linear_handles_1d_and_3d_inputs():
    layer = Linear(8, 4)
    assert layer(Tensor.randn(8, seed=1)).shape == (4,)
    assert layer(Tensor.randn(2, 5, 8, seed=1)).shape == (2, 5, 4)   # (batch, seq, features)


def test_linear_shape_mismatch_names_the_layer():
    with pytest.raises(ValueError, match="128"):
        Linear(256, 10)(Tensor.randn(32, 128, seed=0))


def test_linear_parameter_memory():
    layer = Linear(784, 256)
    n = sum(p.size for p in layer.parameters())
    assert n == 784 * 256 + 256 == 200_960
    assert sum(p.memory_footprint() for p in layer.parameters()) == 803_840   # float32


def test_dropout_eval_and_edge_probabilities():
    x = Tensor.randn(100, seed=0)
    np.testing.assert_array_equal(Dropout(0.5)(x, training=False).data, x.data)
    np.testing.assert_array_equal(Dropout(0.0)(x, training=True).data, x.data)
    with np.errstate(divide="raise", invalid="raise"):
        y = Dropout(1.0)(x, training=True)
    np.testing.assert_array_equal(y.data, 0)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            Dropout(bad)


def test_dropout_training_masks_and_rescales():
    manual_seed(0)
    x = Tensor(np.ones(100_000))
    y = Dropout(0.3)(x, training=True)
    dropped = (y.data == 0).mean()
    assert abs(dropped - 0.3) < 0.01
    np.testing.assert_allclose(y.data[y.data != 0], 1 / 0.7, rtol=1e-6)   # phần sống được nhân 1/(1-p)
    assert abs(y.data.mean() - 1.0) < 0.01                                 # kỳ vọng giữ nguyên


def test_dropout_follows_training_flag():
    d = Dropout(0.5)
    x = Tensor(np.ones(1000))
    assert (d(x).data == 0).any()               # mặc định: training
    d.eval()
    np.testing.assert_array_equal(d(x).data, 1)
    assert (d(x, training=True).data == 0).any()  # tham số truyền vào thắng cờ


def test_sequential_forward_matches_manual_composition():
    manual_seed(1)
    l1, l2 = Linear(4, 8), Linear(8, 3)
    model = Sequential(l1, ReLU(), l2)
    x = Tensor.randn(5, 4, seed=2)
    expected = l2(ReLU()(l1(x)))
    np.testing.assert_allclose(model(x).data, expected.data, rtol=1e-6)
    assert model.parameters() == [l1.weight, l1.bias, l2.weight, l2.bias]
    assert Sequential([l1, ReLU(), l2])(x).shape == (5, 3)
    assert len(model) == 3 and model[0] is l1


def test_sequential_eval_reaches_nested_dropout():
    inner = Sequential(Linear(4, 4), Dropout(0.5))
    model = Sequential(inner, ReLU(), Linear(4, 2))
    x = Tensor.randn(3, 4, seed=0)
    model.eval()
    assert inner[1].training is False
    np.testing.assert_array_equal(model(x).data, model(x).data)   # không còn ngẫu nhiên
    model.train()
    assert inner[1].training is True


def test_sequential_counts_shared_parameters_once():
    shared = Linear(4, 4)
    model = Sequential(shared, ReLU(), shared)
    assert len(model.parameters()) == 2


def test_manual_seed_reproducible():
    manual_seed(42)
    a = Linear(5, 5).weight.data.copy()
    manual_seed(42)
    np.testing.assert_array_equal(Linear(5, 5).weight.data, a)
