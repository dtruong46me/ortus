# TinyTorch (tự xây)

Bản tự viết lại khóa [TinyTorch](https://mlsysbook.ai/tinytorch/): một framework deep learning nhỏ trên NumPy.

```
tinytorch/
├── pyproject.toml          # khai báo package, cài bằng: pip install -e .
├── tinytorch/
│   ├── __init__.py         # from tinytorch import Tensor
│   └── core/
│       └── tensor.py       # Module 01: Tensor
└── tests/
    └── test_01_tensor.py   # chạy: pytest -q
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
pytest -q
```
