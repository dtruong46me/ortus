import numpy as np
import pytest

from tinytorch import Tensor


def test_creation_casts_to_float32():
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3)
    assert t.size == 6
    assert t.ndim == 2
    assert t.dtype == np.float32
    assert Tensor(3.5).shape == ()
    assert Tensor(t).shape == (2, 3)


def test_memory_footprint():
    assert Tensor.zeros(1000, 1000).memory_footprint() == 4_000_000


def test_arithmetic_and_broadcasting():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = Tensor([10, 20, 30])
    np.testing.assert_allclose((a + b).data, [[11, 22, 33], [14, 25, 36]])
    np.testing.assert_allclose((a * 2).data, [[2, 4, 6], [8, 10, 12]])
    np.testing.assert_allclose((2 * a).data, (a * 2).data)
    np.testing.assert_allclose((10 - a).data, [[9, 8, 7], [6, 5, 4]])
    np.testing.assert_allclose((1 / Tensor([2, 4])).data, [0.5, 0.25])
    with pytest.raises(ValueError):
        a + Tensor([1, 2])          # (2, 3) + (2,) -> 3 != 2


def test_matmul():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    np.testing.assert_allclose((a @ b).data, [[19, 22], [43, 50]])
    m = Tensor([[1, 2, 3], [4, 5, 6]])
    assert (m @ Tensor([1, 1, 1])).shape == (2,)
    with pytest.raises(ValueError, match="Inner dimensions"):
        m @ Tensor([[1, 2], [3, 4]])
    with pytest.raises(TypeError):
        m.matmul([[1], [2], [3]])


def test_linear_layer_forward():
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    W = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    b = Tensor([0.1, 0.2])
    np.testing.assert_allclose((x @ W + b).data, [[2.3, 3.0], [5.0, 6.6]], rtol=1e-6)


def test_reshape():
    t = Tensor([1, 2, 3, 4, 5, 6])
    assert t.reshape(2, 3).shape == (2, 3)
    assert t.reshape((3, 2)).shape == (3, 2)
    assert t.reshape(2, -1).shape == (2, 3)
    assert t.reshape(-1).shape == (6,)
    for bad in [(4, -1), (2, 4), (-1, -1), (0, 6)]:
        with pytest.raises(ValueError):
            t.reshape(*bad)


def test_transpose():
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(t.transpose().data, [[1, 4], [2, 5], [3, 6]])
    x = Tensor.zeros(2, 3, 4)
    assert x.transpose().shape == (2, 4, 3)
    assert x.transpose(0, 2).shape == (4, 3, 2)
    assert t.T.shape == (3, 2)


def test_flatten_squeeze_unsqueeze():
    x = Tensor.zeros(8, 3, 32, 32)
    assert x.flatten(1).shape == (8, 3072)
    assert Tensor([1, 2, 3]).unsqueeze(0).shape == (1, 3)
    assert Tensor.zeros(1, 3, 1).squeeze().shape == (3,)


def test_reductions():
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    assert t.sum().data == 21
    np.testing.assert_allclose(t.sum(axis=0).data, [5, 7, 9])
    np.testing.assert_allclose(t.sum(axis=1).data, [6, 15])
    assert t.mean(axis=1, keepdims=True).shape == (2, 1)
    np.testing.assert_allclose((t - t.mean(axis=1, keepdims=True)).data, [[-1, 0, 1], [-1, 0, 1]])
    np.testing.assert_allclose(t.max(axis=0).data, [4, 5, 6])


def test_operations_do_not_modify_inputs():
    a = Tensor([1, 2, 3])
    _ = a + 1
    _ = a.reshape(3, 1)
    np.testing.assert_allclose(a.data, [1, 2, 3])
