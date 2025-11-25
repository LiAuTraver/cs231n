from distutils.core import setup  # type: ignore
from distutils.extension import Extension  # type: ignore
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "im2col_cython", ["im2col_cython.pyx"], include_dirs=[numpy.get_include()]
    ),
]

setup(ext_modules=cythonize(extensions),)
