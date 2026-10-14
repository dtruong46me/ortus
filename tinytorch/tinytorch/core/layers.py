"""Module 03 · Layers: các khối xây dựng của mạng nơ-ron.

Layer gói hai thứ vào một chỗ: phép tính (forward) và trạng thái (tham số học được).
Bốn mảnh: lớp cha Layer, Linear (y = xW + b), Dropout và Sequential để xếp chồng.
"""
import numpy as np

from tinytorch.core.tensor import Tensor

# Một bộ sinh số ngẫu nhiên dùng chung cho khởi tạo trọng số và mask của Dropout.
_rng = np.random.default_rng()


def manual_seed(seed):
    """Cố định ngẫu nhiên để chạy lại ra đúng kết quả (giống torch.manual_seed)."""
    global _rng
    _rng = np.random.default_rng(seed)


class Layer:
    """Hợp đồng chung: forward() tính đầu ra, parameters() khai báo trọng số, gọi được như hàm."""

    # Thuộc tính của lớp: mọi layer mặc định ở chế độ huấn luyện, kể cả khi
    # lớp con quên gọi super().__init__().
    training = True

    def forward(self, x):
        raise NotImplementedError(f"{type(self).__name__} chưa cài đặt forward()")

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    def parameters(self):
        """Danh sách Tensor cần học. Layer không có trọng số trả về []."""
        return []

    def train(self, mode=True):
        """Bật chế độ huấn luyện (mode=False là chế độ đánh giá). Trả về chính nó."""
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def __repr__(self):
        return f"{type(self).__name__}()"


class Linear(Layer):
    """Lớp kết nối đầy đủ: y = xW + b, với W có shape (in_features, out_features)."""

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        # Khởi tạo theo fan_in: Var(W) = 1/in_features để phương sai tín hiệu
        # không phình ra hay teo lại qua từng lớp (slide gọi là Xavier; đúng tên là LeCun).
        scale = np.sqrt(1.0 / in_features)
        weight_data = _rng.standard_normal((in_features, out_features)) * scale
        self.weight = Tensor(weight_data, requires_grad=True)
        if bias:
            self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"{self!r} cần chiều cuối của đầu vào bằng {self.in_features}, "
                f"nhận shape {x.shape}"
            )
        output = x.matmul(self.weight)          # (..., in) @ (in, out) -> (..., out)
        if self.bias is not None:
            output = output + self.bias         # broadcasting: (out,) cộng vào mọi hàng
        return output

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __repr__(self):
        return (f"Linear(in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None})")


class Dropout(Layer):
    """Khi huấn luyện: tắt ngẫu nhiên mỗi phần tử với xác suất p, nhân phần còn lại với 1/(1-p).

    Khi đánh giá: trả nguyên đầu vào. Nhờ phép nhân 1/(1-p) (inverted dropout), kỳ vọng của
    đầu ra bằng đúng đầu vào ở cả hai chế độ.
    """

    def __init__(self, p=0.5):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p là xác suất TẮT một phần tử, phải nằm trong [0, 1], nhận {p}")
        self.p = p

    def forward(self, x, training=None):
        if training is None:                    # không truyền thì theo cờ của layer
            training = self.training
        if not training or self.p == 0.0:
            return x
        if self.p == 1.0:                       # tắt hết; tránh chia 1/0 ra inf rồi 0*inf = nan
            return x * 0.0
        keep_prob = 1.0 - self.p
        mask = _rng.random(x.shape) < keep_prob
        return x * Tensor(mask / keep_prob)     # gộp mask và hệ số vào một tensor

    def __repr__(self):
        return f"Dropout(p={self.p})"


class Sequential(Layer):
    """Xếp các layer nối tiếp: đầu ra lớp trước là đầu vào lớp sau."""

    def __init__(self, *layers):
        # Nhận cả Sequential(a, b, c) lẫn Sequential([a, b, c]).
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = layers[0]
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # Gom tham số của mọi lớp con; một tham số dùng chung chỉ được tính một lần.
        params, seen = [], set()
        for layer in self.layers:
            for p in layer.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        return params

    def train(self, mode=True):
        self.training = mode
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train(mode)
            else:
                # Activation của Module 02 không kế thừa Layer: chỉ cần gắn cờ.
                layer.training = mode
        return self

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

    def __repr__(self):
        inner = "".join(f"\n  ({i}): {layer!r}" for i, layer in enumerate(self.layers))
        return f"Sequential({inner}\n)"
