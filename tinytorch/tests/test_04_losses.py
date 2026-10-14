import numpy as np
import pytest

from tinytorch import (
    Tensor, Sigmoid, Softmax, Linear, ReLU, Sequential, manual_seed,
    log_softmax, Loss, MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss, BCEWithLogitsLoss,
)


def test_log_softmax_matches_definition():
    x = Tensor.randn(4, 5, seed=0)
    expected = np.log(np.exp(x.data) / np.exp(x.data).sum(axis=-1, keepdims=True))
    out = log_softmax(x)
    np.testing.assert_allclose(out.data, expected, atol=1e-6)
    np.testing.assert_allclose(np.exp(out.data).sum(axis=-1), 1.0, atol=1e-6)
    assert np.all(out.data <= 0)


def test_log_softmax_survives_huge_logits():
    x = Tensor([[100.0, 200.0, 300.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(x.data) / np.exp(x.data).sum()       # float32 tràn từ e^89
    assert np.isnan(naive).all()
    np.testing.assert_allclose(log_softmax(x).data, [[-200.0, -100.0, 0.0]])


def test_log_softmax_dim():
    x = Tensor.randn(3, 4, seed=1)
    np.testing.assert_allclose(log_softmax(x, dim=0).data, log_softmax(x.T).data.T, atol=1e-6)


def test_losses_return_scalar_tensor():
    assert isinstance(MSELoss()(Tensor([1.0]), Tensor([2.0])), Tensor)
    assert CrossEntropyLoss()(Tensor([[1.0, 2.0]]), Tensor([0])).shape == ()
    assert isinstance(MSELoss(), Loss)
    assert repr(CrossEntropyLoss()) == "CrossEntropyLoss()"


def test_mse_values():
    mse = MSELoss()
    assert mse(Tensor([1.0, 2.0, 3.0]), Tensor([1.5, 2.5, 2.8])).data == pytest.approx(0.18)
    assert mse(Tensor([3.0, 4.0]), Tensor([3.0, 4.0])).data == 0
    assert mse(Tensor([-10.0]), Tensor([0.0])).data == mse(Tensor([10.0]), Tensor([0.0])).data
    assert mse(Tensor([10.0]), Tensor([0.0])).data == 4 * mse(Tensor([5.0]), Tensor([0.0])).data


def test_mse_rejects_broadcasting_shapes():
    # (4, 1) - (4,) sẽ broadcast thành (4, 4): loss sai mà không ai biết.
    with pytest.raises(ValueError, match="cùng shape"):
        MSELoss()(Tensor(np.zeros((4, 1))), Tensor(np.zeros(4)))


def test_cross_entropy_matches_manual():
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 1.5, 0.8]])
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    expected = -(np.log(probs[0, 0]) + np.log(probs[1, 1])) / 2
    loss = CrossEntropyLoss()(Tensor(logits), Tensor([0, 1]))
    assert loss.data == pytest.approx(expected, rel=1e-5)


def test_cross_entropy_uniform_is_log_classes():
    for c in (2, 10, 1000):
        loss = CrossEntropyLoss()(Tensor(np.zeros((8, c))), Tensor(np.arange(8) % c))
        assert loss.data == pytest.approx(np.log(c), rel=1e-5)


def test_cross_entropy_stable_and_unbounded():
    ce = CrossEntropyLoss()
    logits = Tensor([[100.0, 200.0, 300.0]])
    assert ce(logits, Tensor([2])).data == pytest.approx(0.0, abs=1e-6)
    assert ce(logits, Tensor([0])).data == pytest.approx(200.0)    # tự tin mà sai: phạt tuyến tính, không có trần


def test_cross_entropy_rejects_bad_targets():
    ce, logits = CrossEntropyLoss(), Tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        ce(logits, Tensor([5]))
    with pytest.raises(ValueError, match="ngoài khoảng"):
        ce(logits, Tensor([-1]))            # NumPy sẽ hiểu -1 là lớp cuối nếu không kiểm tra
    with pytest.raises(ValueError, match="nguyên"):
        ce(logits, Tensor([1.5]))
    with pytest.raises(ValueError):
        ce(logits, Tensor([0, 1]))          # 2 nhãn cho 1 mẫu


def test_cross_entropy_sequence_shape():
    logits = Tensor.randn(2, 5, 7, seed=2)                  # (batch, độ dài chuỗi, số lớp)
    targets = Tensor(np.random.default_rng(3).integers(0, 7, size=(2, 5)))
    ce = CrossEntropyLoss()
    flat = ce(logits.reshape(10, 7), targets.reshape(10))
    assert ce(logits, targets).data == pytest.approx(flat.data)


def test_double_softmax_has_a_floor():
    # Softmax hai lần: đầu vào của loss bị nén vào [0, 1], loss không thể xuống dưới log(1 + (C-1)/e).
    c = 10
    logits = np.full((1, c), -50.0)
    logits[0, 0] = 50.0
    good = CrossEntropyLoss()(Tensor(logits), Tensor([0])).data
    double = CrossEntropyLoss()(Softmax()(Tensor(logits)), Tensor([0])).data
    assert good == pytest.approx(0.0, abs=1e-6)
    assert double == pytest.approx(np.log(1 + (c - 1) / np.e), rel=1e-5)


def test_bce_values():
    bce = BinaryCrossEntropyLoss()
    loss = bce(Tensor([0.9, 0.1, 0.7, 0.3]), Tensor([1.0, 0.0, 1.0, 0.0]))
    assert loss.data == pytest.approx(-(2 * np.log(0.9) + 2 * np.log(0.7)) / 4, rel=1e-5)
    assert bce(Tensor([0.5]), Tensor([1.0])).data == pytest.approx(np.log(2), rel=1e-5)


def test_bce_clips_instead_of_nan():
    bce = BinaryCrossEntropyLoss()
    worst = bce(Tensor([0.0, 1.0]), Tensor([1.0, 0.0])).data
    assert np.isfinite(worst) and 15 < worst < 17                  # ≈ -log(1e-7) ≈ 16.1


def test_bce_rejects_logits_and_bad_shapes():
    bce = BinaryCrossEntropyLoss()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bce(Tensor([1.2]), Tensor([1.0]))                          # quên sigmoid: đưa logit vào
    with pytest.raises(ValueError):
        bce(Tensor([0.5, 0.5]), Tensor([1.0]))


def test_bce_with_logits_matches_and_does_not_saturate():
    z = Tensor([-3.0, 0.0, 2.0, 5.0])
    y = Tensor([1.0, 0.0, 1.0, 0.0])
    assert BCEWithLogitsLoss()(z, y).data == pytest.approx(
        BinaryCrossEntropyLoss()(Sigmoid()(z), y).data, rel=1e-5)
    # z = -100, y = 1: sigmoid rồi kẹp chỉ cho ≈ 16.1; công thức gộp cho đúng 100.
    assert BCEWithLogitsLoss()(Tensor([-100.0]), Tensor([1.0])).data == pytest.approx(100.0)
    assert BinaryCrossEntropyLoss()(Sigmoid()(Tensor([-100.0])), Tensor([1.0])).data < 17


def test_loss_on_model_output():
    manual_seed(0)
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 3))
    x = Tensor.randn(16, 4, seed=4)
    y = Tensor(np.random.default_rng(5).integers(0, 3, size=16))
    loss = CrossEntropyLoss()(model(x), y)
    assert np.isfinite(loss.data) and 0.5 < loss.data < 2.5      # mạng chưa học: quanh log(3) ≈ 1.1
