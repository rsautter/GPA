from setuptools import find_packages, Extension, setup

setup(name="GPA",
      version="1.5",
      ext_modules=[Extension("GPA",["GPA.c"]),Extension("GPA3D",["GPA3D.c"])],
      author='Rubens Andreas Sautter',
      author_email='rubens.sautter@gmail.com',
      url='https://github.com/rsautter/GPA',
      packages=find_packages())

