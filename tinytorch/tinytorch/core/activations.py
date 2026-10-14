"""Module 02 · Activations: các hàm phi tuyến của TinyTorch.

Mỗi activation là một lớp không có tham số học được, gọi như hàm: ``ReLU()(x)``.
Tất cả nhận và trả về Tensor; phần tính toán giao cho NumPy.
"""
import numpy as np

from tinytorch.core.tensor import Tensor


def _stable_sigmoid(z):
    """σ(z) trên mảng NumPy, không bao giờ tính e mũ một số dương.

    Với z ≥ 0 dùng 1 / (1 + e^{-z}); với z < 0 dùng e^{z} / (1 + e^{z}).
    Cả hai nhánh chỉ cần e^{-|z|} ∈ [0, 1], nên không thể tràn số (chỉ có thể underflow về 0, vô hại).
    """
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


class Activation:
    """Lớp cha: activation không có trọng số và gọi được như một hàm."""

    def parameters(self):
        # Module 03 (layers) và 07 (optimizers) sẽ gom tham số; activation không có gì để học.
        return []

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.forward(x, *args, **kwargs)

    def __repr__(self):
        return f"{type(self).__name__}()"


class ReLU(Activation):
    """f(x) = max(0, x). Chi phí 1x (mốc so sánh)."""

    def forward(self, x: Tensor) -> Tensor:
        # np.maximum: so sánh TỪNG PHẦN TỬ với 0 (khác np.max là phép rút gọn).
        return Tensor(np.maximum(0, x.data))


class Sigmoid(Activation):
    """σ(x) = 1 / (1 + e^{-x}), ra khoảng (0, 1)."""

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(_stable_sigmoid(x.data))


class Tanh(Activation):
    """tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x}), ra khoảng (-1, 1), đối xứng quanh 0."""

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.tanh(x.data))


class GELU(Activation):
    """GELU(x) = x · Φ(x) ≈ x · σ(1.702 x).

    1.702 là hằng số được khớp (fit) để σ(1.702 x) bám sát hàm phân phối tích luỹ Φ(x)
    của phân phối chuẩn; sai số tuyệt đối của Φ dưới 0.01.
    """

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(x.data * _stable_sigmoid(1.702 * x.data))


class Softmax(Activation):
    """softmax(x)_i = e^{x_i} / Σ_j e^{x_j}, chuẩn hoá theo chiều ``dim``.

    Trừ max trước khi lấy mũ (mẹo max-subtraction) để không tràn số. Phần tử bằng -inf
    được hiểu là "bị che" (mask) và nhận xác suất 0; một hàng bị che toàn bộ trả về toàn 0.
    """

    def __init__(self, dim: int = -1):
        self.dim = dim

    def forward(self, x: Tensor, dim: int = None) -> Tensor:
        dim = self.dim if dim is None else dim
        z = x.data
        z_max = np.max(z, axis=dim, keepdims=True)
        # Hàng toàn -inf: max = -inf, và -inf - (-inf) = nan. Thay max bằng 0 để e^{-inf} = 0.
        z_max = np.where(np.isneginf(z_max), 0.0, z_max)
        e = np.exp(z - z_max)                       # số mũ lớn nhất là e^0 = 1
        total = np.sum(e, axis=dim, keepdims=True)
        return Tensor(e / np.where(total == 0, 1.0, total))

    def __repr__(self):
        return f"Softmax(dim={self.dim})"
