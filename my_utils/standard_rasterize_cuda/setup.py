from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# python setup.py build_exit -i

setup(
    name='standard_rasterize_cuda',
    ext_modules=[
	CUDAExtension('standard_rasterize_cuda', [
        'standard_rasterize_cuda.cpp',
        'standard_rasterize_cuda_kernel.cu',
        ])
	],
    cmdclass = {'build_ext': BuildExtension}
)
