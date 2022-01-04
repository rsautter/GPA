# Gradient Pattern Analysis
Gradient Pattern Analysis (GPA) prototype developed in Cython.

### Requirements
 - Python 2.7 or greater
 - Numpy
 - Numba 
 - Scipy
 
### Install
    pip install git+https://github.com/rsautter/GPA
    
To notebook envoirment (i.e. Colab):

    !pip install git+https://github.com/rsautter/GPA
    
## Log
Jan. 03, 2022 - Changed the code to python (with numba) - version 4.0
&emsp;&emsp; &emsp; &emsp; &emsp; - removed fragGPA and ScaleGPA (it is necessary a more detailed study) .		

Apr. 18, 2021 - Added symmetrical and full analysis.\
&emsp;&emsp; &emsp; &emsp; &emsp; - renamed evaluate to __call__ 
&emsp;&emsp; &emsp; &emsp; &emsp; - vectors are now classified into symmetrical, asymmetrical and unknown.
Jun. 26, 2020 - Changed deploy system to setuptools

Jul. 01, 2019 - Changed GPA constructor, it receives modular and angular tolerance as input.\
&emsp;&emsp; &emsp; &emsp; &emsp; - Now the evaluation function receives the matrix as an argument.\
&emsp;&emsp; &emsp; &emsp; &emsp; - ScaleGPA has now a better performance.\
&emsp;&emsp; &emsp; &emsp; &emsp; - Now GPA works with double precision.\
Nov. 27, 2018 - Added the FFT-Hilbert approach (file fragGPA.py)\
Nov. 26, 2018 - Changed the user interface (it automatically identifies whether the file is a list or a data)\
&emsp;&emsp; &emsp; &emsp; &emsp; - Results are saved at reult.csv\
&emsp;&emsp; &emsp; &emsp; &emsp; - Tests updated
              
Mar. 27, 2018 - Added the Third and Fourth Gradient Moments (G3 and G4)

Aug. 30, 2017 - Added the First Gradient Moment (G1)




## References
https://en.wikipedia.org/wiki/Gradient_pattern_analysis

https://numba.pydata.org/

https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html