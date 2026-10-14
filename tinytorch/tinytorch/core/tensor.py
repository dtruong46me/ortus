"""Module 01 · Tensor: lớp dữ liệu nền móng của TinyTorch.

Tensor bọc một mảng NumPy float32 và thêm các quy tắc của deep learning:
kiểm tra shape, broadcasting, matmul, reshape/transpose và các phép rút gọn.
"""
import numpy as np


def _unwrap(x):
    """Lấy mảng NumPy bên trong nếu x là Tensor, giữ nguyên nếu là số/mảng."""
    return x.data if isinstance(x, Tensor) else x


class Tensor:
    """Mảng N chiều kiểu float32, giao diện giống torch.Tensor."""

    def __init__(self, data, requires_grad=False):
        if isinstance(data, Tensor):
            data = data.data
        # Ép mọi thứ về float32: một nửa bộ nhớ so với float64, đúng chuẩn GPU.
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
        # Module 06 (autograd) sẽ dùng ba thuộc tính này.
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None

    # ---------- tiện ích ----------
    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        return self.size

    def numpy(self):
        return self.data

    def memory_footprint(self):
        """Số byte dữ liệu chiếm (1000x1000 float32 = 4 MB)."""
        return self.data.nbytes

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape})"

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        return Tensor(self.data[key])

    # ---------- tạo tensor nhanh ----------
    @staticmethod
    def zeros(*shape):
        return Tensor(np.zeros(shape))

    @staticmethod
    def ones(*shape):
        return Tensor(np.ones(shape))

    @staticmethod
    def randn(*shape, seed=None):
        return Tensor(np.random.default_rng(seed).standard_normal(shape))

    # ---------- số học từng phần tử (có broadcasting) ----------
    def __add__(self, other):
        return Tensor(self.data + _unwrap(other))

    def __radd__(self, other):          # 2 + t
        return Tensor(_unwrap(other) + self.data)

    def __sub__(self, other):
        return Tensor(self.data - _unwrap(other))

    def __rsub__(self, other):          # 2 - t
        return Tensor(_unwrap(other) - self.data)

    def __mul__(self, other):           # nhân TỪNG PHẦN TỬ, không phải matmul
        return Tensor(self.data * _unwrap(other))

    def __rmul__(self, other):
        return Tensor(_unwrap(other) * self.data)

    def __truediv__(self, other):
        return Tensor(self.data / _unwrap(other))

    def __rtruediv__(self, other):
        return Tensor(_unwrap(other) / self.data)

    def __neg__(self):
        return Tensor(-self.data)

    def __pow__(self, p):
        return Tensor(self.data ** p)

    # ---------- nhân ma trận ----------
    def _validate_matmul_shapes(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f"Expected Tensor, got {type(other).__name__}")
        if self.ndim == 0 or other.ndim == 0:
            raise ValueError(f"matmul cần tensor ít nhất 1D, nhận {self.shape} @ {other.shape}; số vô hướng hãy dùng *")
        inner_self = self.shape[-1]
        inner_other = other.shape[-2] if other.ndim >= 2 else other.shape[0]
        if inner_self != inner_other:
            raise ValueError(
                f"Inner dimensions must match: {self.shape} @ {other.shape} "
                f"({inner_self} != {inner_other})"
            )

    def matmul(self, other):
        self._validate_matmul_shapes(other)
        return Tensor(np.matmul(self.data, other.data))

    def __matmul__(self, other):        # cho phép viết x @ W
        return self.matmul(other)

    # ---------- đổi hình dạng ----------
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        new_shape = list(shape)
        if any(d == 0 or d < -1 for d in new_shape):
            raise ValueError(f"Kích thước không hợp lệ trong {tuple(shape)}: chỉ dùng số dương hoặc -1")
        if new_shape.count(-1) > 1:
            raise ValueError("Chỉ được một chiều là -1")
        if -1 in new_shape:
            known_size = 1
            for d in new_shape:
                if d != -1:
                    known_size *= d
            if self.size % known_size != 0:
                raise ValueError(f"Không suy ra được -1: {self.size} không chia hết cho {known_size}")
            new_shape[new_shape.index(-1)] = self.size // known_size
        if int(np.prod(new_shape)) != self.size:
            raise ValueError(f"Cannot reshape {self.shape} ({self.size} phần tử) thành {tuple(new_shape)}")
        return Tensor(np.reshape(self.data, tuple(new_shape)))

    def transpose(self, dim0=None, dim1=None):
        if self.ndim < 2:
            return Tensor(self.data)
        if (dim0 is None) != (dim1 is None):
            raise ValueError("Truyền cả dim0 và dim1, hoặc bỏ trống cả hai")
        axes = list(range(self.ndim))
        if dim0 is None:                # mặc định: đổi hai chiều cuối (ma trận chuyển vị)
            dim0, dim1 = -2, -1
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return Tensor(np.transpose(self.data, axes))

    @property
    def T(self):
        return self.transpose()

    def flatten(self, start_dim=0):
        """(B, C, H, W).flatten(1) -> (B, C*H*W)."""
        start_dim = start_dim % self.ndim if self.ndim else 0
        return self.reshape(*self.shape[:start_dim], -1)

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self.data, dim))

    def squeeze(self, dim=None):
        return Tensor(np.squeeze(self.data, axis=dim))

    # ---------- rút gọn ----------
    def sum(self, axis=None, keepdims=False):
        return Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))

    def mean(self, axis=None, keepdims=False):
        return Tensor(np.mean(self.data, axis=axis, keepdims=keepdims))

    def max(self, axis=None, keepdims=False):
        return Tensor(np.max(self.data, axis=axis, keepdims=keepdims))
