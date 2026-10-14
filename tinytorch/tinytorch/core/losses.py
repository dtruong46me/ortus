"""Module 04 · Losses: biến "dự đoán sai bao nhiêu" thành một con số.

Loss nhận dự đoán của mô hình và đáp án đúng, trả về một số vô hướng: càng nhỏ càng tốt.
Module 06 (autograd) sẽ lấy đạo hàm của con số này theo từng tham số để biết phải sửa theo hướng nào.

Bốn loss: MSELoss (hồi quy), CrossEntropyLoss (phân loại nhiều lớp, nhận logits),
BinaryCrossEntropyLoss (có/không, nhận xác suất) và BCEWithLogitsLoss (có/không, nhận logit).
Tất cả dựa trên log_softmax ổn định số học (mẹo log-sum-exp).
"""
import numpy as np

from tinytorch.core.tensor import Tensor

# Kẹp xác suất vào [EPSILON, 1 - EPSILON] để log() không bao giờ gặp 0.
EPSILON = 1e-7


def _data(x):
    """Lấy mảng NumPy từ Tensor; nhận cả list hay mảng NumPy cho tiện."""
    return x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)


def _log_softmax(z, axis=-1):
    """log_softmax trên mảng NumPy: z - max - log(sum(exp(z - max)))."""
    z_max = np.max(z, axis=axis, keepdims=True)          # 1. tìm max để dời
    shifted = z - z_max                                  # 2. mọi giá trị <= 0, nên exp(.) <= 1
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))   # 3. tổng nằm trong [1, C]
    return shifted - log_sum_exp                         # 4. = z - max - log_sum_exp


def log_softmax(x, dim=-1):
    """log(softmax(x)) tính theo chiều ``dim`` mà không bao giờ tràn số.

    Toán học: log_softmax(x)_i = x_i - log Σ_j e^{x_j}. Trừ max trước khi lấy mũ nên số mũ
    lớn nhất là e^0 = 1, và tổng nằm trong [1, C]: log của nó luôn hữu hạn.
    """
    return Tensor(_log_softmax(x.data, axis=dim))


class Loss:
    """Lớp cha: loss gọi được như hàm, ``loss_fn(predictions, targets)`` trả về Tensor vô hướng."""

    def forward(self, predictions, targets):
        raise NotImplementedError(f"{type(self).__name__} chưa cài đặt forward()")

    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)

    def __repr__(self):
        return f"{type(self).__name__}()"


def _check_same_shape(name, a, b):
    # Không cho broadcasting: predictions (N, 1) với targets (N,) sẽ âm thầm thành ma trận (N, N)
    # và cho ra một con số sai mà không báo lỗi gì.
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"{name} cần predictions và targets cùng shape, khác rỗng; nhận {a.shape} và {b.shape}")


def _check_unit_interval(name, what, a):
    if not np.all(np.isfinite(a)) or np.any((a < 0) | (a > 1)):
        raise ValueError(f"{name}: {what} phải là số hữu hạn trong [0, 1]")


class MSELoss(Loss):
    """Trung bình bình phương sai số: (1/N) Σ (ŷ - y)². Dùng cho hồi quy (dự đoán một con số)."""

    def forward(self, predictions, targets):
        pred, target = _data(predictions), _data(targets)
        _check_same_shape("MSELoss", pred, target)
        diff = pred - target                              # 1. sai số từng phần tử
        return Tensor(np.mean(diff * diff))               # 2-3. bình phương rồi lấy trung bình


class CrossEntropyLoss(Loss):
    """Cross-entropy cho phân loại nhiều lớp: -mean(log softmax(logits)[đáp án đúng]).

    logits: shape (..., C), số thực bất kỳ (CHƯA qua softmax).
    targets: shape (...), chỉ số lớp nguyên trong [0, C-1].
    """

    def forward(self, logits, targets):
        z, target = _data(logits), _data(targets)
        if z.ndim < 2 or z.size == 0 or target.shape != z.shape[:-1]:
            raise ValueError(f"CrossEntropyLoss cần logits (..., C) và targets (...); nhận {z.shape} và {target.shape}")
        if not np.all(np.isfinite(target)) or np.any(target != np.floor(target)):
            raise ValueError("CrossEntropyLoss: targets phải là chỉ số lớp nguyên")
        num_classes = z.shape[-1]
        # Kiểm tra trước khi đánh chỉ số: NumPy hiểu -1 là "lớp cuối cùng" và sẽ lặng lẽ chọn sai lớp.
        bad = (target < 0) | (target >= num_classes)
        if np.any(bad):
            raise ValueError(f"CrossEntropyLoss: nhãn {np.unique(target[bad]).astype(int).tolist()} "
                             f"nằm ngoài khoảng [0, {num_classes - 1}] của {num_classes} lớp")
        log_probs = _log_softmax(z.reshape(-1, num_classes))         # 1. log-xác suất ổn định
        idx = target.reshape(-1).astype(np.int64)
        picked = log_probs[np.arange(idx.size), idx]                 # 2. mỗi hàng lấy cột đáp án
        return Tensor(-np.mean(picked))                              # 3. negative log-likelihood


class BinaryCrossEntropyLoss(Loss):
    """BCE cho bài toán có/không: -mean(y log p + (1 - y) log(1 - p)), p là XÁC SUẤT trong [0, 1].

    p được kẹp vào [EPSILON, 1 - EPSILON], nên loss lớn nhất khoảng 16 thay vì vô cùng.
    Cái giá: ngoài khoảng kẹp, loss là hằng số và Module 06 sẽ cho gradient bằng 0.
    """

    def forward(self, predictions, targets):
        p, y = _data(predictions), _data(targets)
        _check_same_shape("BinaryCrossEntropyLoss", p, y)
        _check_unit_interval("BinaryCrossEntropyLoss", "predictions", p)
        _check_unit_interval("BinaryCrossEntropyLoss", "targets", y)
        p = np.clip(p, EPSILON, 1 - EPSILON)              # thanh chắn: không bao giờ log(0)
        return Tensor(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class BCEWithLogitsLoss(Loss):
    """BCE nhận LOGIT z (trước sigmoid): gộp sigmoid và log thành một công thức ổn định.

    loss = max(z, 0) - z·y + log(1 + e^{-|z|}), đúng bằng BCE(sigmoid(z), y) nhưng không cần kẹp:
    với z rất âm mà y = 1, loss ≈ -z (tăng tuyến tính) chứ không bị chặn ở 16.
    """

    def forward(self, logits, targets):
        z, y = _data(logits), _data(targets)
        _check_same_shape("BCEWithLogitsLoss", z, y)
        _check_unit_interval("BCEWithLogitsLoss", "targets", y)
        loss = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
        return Tensor(np.mean(loss))
