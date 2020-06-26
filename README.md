# Gradient Pattern Analysis
This is a Gradient Pattern Analysis (GPA) prototype developed in Cython.

### Requirements
 - Python 2.7 or greater
 - Numpy
 - Cython (for an optimized compilation)
 
### Install
    pip install git+https://github.com/rsautter/GPA
    
To collab:

    !pip install git+https://github.com/rsautter/GPA
    
### Compilation

This code uses the Cython library, to improve its performance. 
To get a better performance, edit compile.py and run:

    python compile.py build_ext --inplace

Obs: Requires Cython

## Log
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

http://cython.org/
