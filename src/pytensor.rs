use pyo3::{exceptions, prelude::*, AsPyPointer};

pub struct PyTensor(pub tch::Tensor);

pub fn wrap_tch_err(err: tch::TchError) -> PyErr {
    PyErr::new::<exceptions::PyValueError, _>(format!("{err:?}"))
}

impl<'source> FromPyObject<'source> for PyTensor {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let ptr = ob.as_ptr() as *mut tch::python::CPyObject;
        match unsafe { tch::Tensor::pyobject_unpack(ptr) } {
            Err(err) => Err(wrap_tch_err(err))?,
            Ok(None) => {
                let msg = format!("expected a torch.Tensor, got {}", ob.get_type());
                Err(PyErr::new::<exceptions::PyTypeError, _>(msg))?
            }
            Ok(Some(tensor)) => Ok(PyTensor(tensor)),
        }
    }
}

impl IntoPy<PyObject> for PyTensor {
    fn into_py(self, py: Python<'_>) -> PyObject {
        // There is no fallible alternative to ToPyObject/IntoPy at the moment so we return
        // None on errors. https://github.com/PyO3/pyo3/issues/1813
        self.0.pyobject_wrap().map_or_else(
            |_| py.None(),
            |ptr| unsafe { PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject) },
        )
    }
}
