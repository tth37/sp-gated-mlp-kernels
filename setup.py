# setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

if CUDA_HOME is None:
    raise EnvironmentError("CUDA_HOME not found. Make sure CUDA is installed and CUDA_HOME is set.")

setup(
    name='convert_bcsr_ext',
    ext_modules=[
        CUDAExtension(
            name='convert_bcsr_ext',
            sources=['csrc/convert_bcsr.cpp', 'csrc/convert_bcsr_kernel.cu'],
            libraries=['cublas'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)