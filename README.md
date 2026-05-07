# GPA

Gradient Pattern Analysis (GPA) developed in Python.

### Requirements
- Python 2.7 or greater
- Numpy
- Scipy
- Optional - CuPy

### Install
```
pip install git+https://github.com/rsautter/GPA
```

Colab:
```
!pip install git+https://github.com/rsautter/GPA
```

## Log
May 07, 2026 -  Changed the language to Python and included GPU acceleration via CuPy.

Oct. 7, 2022 - Updated Gradient Moments - new $G_1$ equation. 

Apr. 18, 2021 - Added symmetrical and full analysis.          
  - now uses __call__ function to measure the gradient moments.          
  - vectors are now classified into symmetrical, asymmetrical and unknown.
  -
Jun. 26, 2020 - Changed deploy system to setuptools

Jul. 01, 2019 - Changed GPA constructor,it receives modular and angular tolerance as input.   
  - Now the evaluation function receives the matrix as an argument.
  - ScaleGPA has a better performance.
  - Now GPA works with double precision. 

Nov. 27, 2018 - Added the FFT-Hilbert approach (file fragGPA.py) 

Nov. 26, 2018 - Changed the user interface (it automatically identifies whether the file is a list or a data) 
  - Results are saved at reult.csv
  - Tests updated

Mar. 27, 2018 - Added the Third and Fourth Gradient Moments ( $G_3$ and $G_4$ )

Aug. 30, 2017 - Added the First Gradient Moment ( $G_1$ )

