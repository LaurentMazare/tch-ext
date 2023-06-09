## Python extensions using tch to interact with PyTorch

This sample crate shows how to use
[tch](https://github.com/LaurentMazare/tch-rs) to write a Python extension
that manipulates PyTorch tensors via [PyO3](https://github.com/PyO3/pyo3).

In order to build the extension and test the plugin, run the following command
from the root of the github repo. This requires a Python environment that has
the appropriate torch version installed.

```bash
LIBTORCH_USE_PYTORCH=1 cargo build && cp -f target/debug/libtch_ext.so tch_ext.so
python test.py
```

Setting `LIBTORCH_USE_PYTORCH` results in using the libtorch C++ library from the
Python install in `tch` and ensures that this is at the proper version (having `tch`
using a different libtorch version from the one used by the Python runtime may result
in segfaults).

## Colab Notebook

`tch` based plugins can easily be used from colab (though it might be a bit slow
to download all the crates and compile), see this [example
notebook](https://colab.research.google.com/drive/1bXVQ2TaKABI4bBG9IL0QFkmvhhf8Tsyl?usp=sharing).
