# cd into dir
# pip install .

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name='OBDS_zoo',
    version='1.0',
    author='piplupx',
    author_email='piplupx96@outlook.com',
    description='object-based diamond search dependencies',
    ext_modules=[
        CUDAExtension(
            name='OBDS_zoo',
            sources=['OBDS_zoo.cpp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)