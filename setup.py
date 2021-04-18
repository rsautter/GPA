from distutils.core import setup
from setuptools import find_packages
from Cython.Build import cythonize

import numpy

setup(name="GPA",
      version="3.1",
      ext_modules=cythonize("*.pyx"),
      author='Rubens Andreas Sautter',
      author_email='rubens.sautter@gmail.com',
      url='https://github.com/rsautter/GPA',
      include_dirs=[numpy.get_include()],
      packages=find_packages())

