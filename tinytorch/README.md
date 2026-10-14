# TinyTorch (tự xây)

Bản tự viết lại khóa [TinyTorch](https://mlsysbook.ai/tinytorch/): một framework deep learning nhỏ trên NumPy.

```
tinytorch/
├── pyproject.toml          # khai báo package, cài bằng: pip install -e .
├── tinytorch/
│   ├── __init__.py         # from tinytorch import Tensor
│   └── core/
│       ├── tensor.py       # Module 01: Tensor
│       ├── activations.py  # Module 02: ReLU, Sigmoid, Tanh, GELU, Softmax
│       ├── layers.py       # Module 03: Layer, Linear, Dropout, Sequential
│       └── losses.py       # Module 04: log_softmax, MSE, CrossEntropy, BCE
└── tests/
    ├── test_01_tensor.py
    ├── test_02_activations.py
    ├── test_03_layers.py
    └── test_04_losses.py
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
python -m pytest -q                         # toàn bộ test
python -m pytest tests/test_04_losses.py -q # chỉ một module
```

Quy trình mỗi module: viết `tinytorch/core/<tên>.py` → xuất tên mới trong `tinytorch/__init__.py` → viết `tests/test_0N_<tên>.py` → chạy `python -m pytest -q`.

Dùng `python -m pytest` thay vì `pytest` trần để chắc chắn chạy đúng Python đã cài NumPy và TinyTorch. Từ thư mục cha cũng chạy được: `python -m pytest tinytorch/tests`.
