from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

openmp_include = "/opt/homebrew/opt/libomp/include"
openmp_lib = "/opt/homebrew/opt/libomp/lib"
openmp_compile_args = ['-Xpreprocessor', '-fopenmp']
openmp_link_args = ['-lomp']

ext_modules = [
    Pybind11Extension(
        "QuantCpp.Time2Image._core",   
        ["QuantCpp/Time2Image/binding.cpp",
         "QuantCpp/Time2Image/GramianAngularField.cpp",
         "QuantCpp/Time2Image/MarkovTransitionField.cpp"],
        include_dirs=["QuantCpp", openmp_include],
        library_dirs=[openmp_lib],
        extra_compile_args=openmp_compile_args,
        extra_link_args=openmp_link_args,
    ),

    Pybind11Extension(
        "QuantCpp.Test._core",   
        ["QuantCpp/Test/binding.cpp",
         "QuantCpp/Test/test_func.cpp"],
        include_dirs=["QuantCpp", openmp_include],
        library_dirs=[openmp_lib],
        extra_compile_args=openmp_compile_args,
        extra_link_args=openmp_link_args,
    ),
]

setup(
    name="QuantCpp",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)