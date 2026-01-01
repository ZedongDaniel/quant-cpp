from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "QuantCpp.Test.test_func",   
        ["QuantCpp/Test/binding.cpp",
        "QuantCpp/Test/test_func.cpp",],
        include_dirs=["QuantCpp"], 
    ),

]

setup(
    name="QuantCpp",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)