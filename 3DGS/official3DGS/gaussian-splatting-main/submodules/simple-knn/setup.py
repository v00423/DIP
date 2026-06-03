from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch.utils.cpp_extension

# Disable CUDA version mismatch check
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None

ROOT = Path(__file__).resolve().parent

setup(
    name="simple_knn",
    packages=["simple_knn"],
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
                "spatial.cu",
                "simple_knn.cu",
                "ext.cpp",
            ],
            include_dirs=[
                str(ROOT),
            ],
            extra_compile_args={
                "cxx": ["/O2", "/Zc:preprocessor"],
                "nvcc": [
                    "-O2",
                    "--use_fast_math",
                    "-Xcompiler",
                    "/Zc:preprocessor",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)