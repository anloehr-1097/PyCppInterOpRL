from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='np_interop',
    ext_modules=[
        CppExtension(
            name='np_interop',
            sources=['np_interop.cpp'],
            include_dirs=['/opt/Eigen/include', "/opt/venv/lib/python3.12/site-packages/pybind11/include",
                          '/opt/Eigen/include/eigen3', '/opt/venv/include/python3.12'],
            extra_compile_args=['-std=c++17'])
    ],

    cmdclass={
        'build_ext': BuildExtension
    }
)
