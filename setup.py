from distutils.core import setup
from setuptools import find_packages

import numpy

setup(name="GPA",
      version="4.0",
      ext_modules=cythonize("*.py"),
      author='Rubens Andreas Sautter',
      author_email='rubens.sautter@gmail.com',
      url='https://github.com/rsautter/GPA',
      include_dirs=[numpy.get_include()],
      packages=find_packages())

