from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys
import platform

# Detect platform
is_macos = platform.system() == "Darwin"
is_linux = platform.system() == "Linux"

# Platform-specific OpenMP configuration
if is_macos:
    openmp_include = "/opt/homebrew/opt/libomp/include"
    openmp_lib = "/opt/homebrew/opt/libomp/lib"
    openmp_compile_args = ['-Xpreprocessor', '-fopenmp']
    openmp_link_args = ['-lomp']
    include_dirs = ["QuantCpp", openmp_include]
    library_dirs = [openmp_lib]
elif is_linux:
    # On Linux, OpenMP is built into GCC
    openmp_compile_args = ['-fopenmp']
    openmp_link_args = ['-fopenmp']
    include_dirs = ["QuantCpp"]
    library_dirs = []
else:
    raise RuntimeError(f"Unsupported platform: {platform.system()}")

# Add optimization flags for both platforms
openmp_compile_args += ['-O3', '-march=native', '-std=c++17']

base_compile_args = ['-O3', '-march=native', '-std=c++17']

ext_modules = [
    Pybind11Extension(
        "QuantCpp.Time2Image._core",   
        ["QuantCpp/Time2Image/binding.cpp",
         "QuantCpp/Time2Image/GramianAngularField.cpp",
         "QuantCpp/Time2Image/MarkovTransitionField.cpp",
         "QuantCpp/Time2Image/MatrixInfo.cpp",],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=openmp_compile_args,
        extra_link_args=openmp_link_args,
    ),

    Pybind11Extension(
        "QuantCpp.Regime._core",   
        ["QuantCpp/Regime/binding.cpp",
         "QuantCpp/Regime/MarkovTransitionMatrix.cpp",],
        include_dirs=["QuantCpp"],
        extra_compile_args=base_compile_args,
    ),

    Pybind11Extension(
        "QuantCpp.Test._core",   
        ["QuantCpp/Test/binding.cpp",
         "QuantCpp/Test/test_func.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
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