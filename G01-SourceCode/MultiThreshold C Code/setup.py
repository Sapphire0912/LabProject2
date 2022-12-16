from distutils.core import setup, Extension
import numpy as np

module = Extension(
    '_Cconvolution2',
    sources=['convolution2_wrap.c', 'convolution2.c']
)

setup(
    name='Cconvolution2',
    version='0.1',
    author='SWIG Docs',
    description="convolution v2",
    ext_modules=[module],
    py_modules=['Cconvolution2'],
    include_dirs=[np.get_include()]
)
