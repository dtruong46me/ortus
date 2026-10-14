# TinyTorch (tự xây)

Bản tự viết lại khóa [TinyTorch](https://mlsysbook.ai/tinytorch/): một framework deep learning nhỏ trên NumPy.

```
tinytorch/
├── pyproject.toml          # khai báo package, cài bằng: pip install -e .
├── tinytorch/
│   ├── __init__.py         # from tinytorch import Tensor
│   └── core/
│       ├── tensor.py       # Module 01: Tensor
│       └── activations.py  # Module 02: ReLU, Sigmoid, Tanh, GELU, Softmax
└── tests/
    ├── test_01_tensor.py
    └── test_02_activations.py
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
pytest -q                              # toàn bộ test
pytest tests/test_02_activations.py -q # chỉ một module
```

Quy trình mỗi module: viết `tinytorch/core/<tên>.py` → xuất tên mới trong `tinytorch/__init__.py` → viết `tests/test_0N_<tên>.py` → chạy `pytest -q`.
