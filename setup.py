from setuptools import setup

setup(name="GPA",
      version="4.0",
      ext_modules=cythonize("*.py"),
      author='Rubens Andreas Sautter',
      author_email='rubens.sautter@gmail.com',
      url='https://github.com/rsautter/GPA')

