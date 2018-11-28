# Concentric Gradient Pattern Analysis
This is a Concentric Gradient Pattern Analysis (CGPA) prototype developed in Cython.

### Requirements
 - Python 2.7 or greater
 - Matplotlib
 - Cython
 - Numpy
 - Pandas

### Compilation

This code uses the Cython library, to improve its performance. 
Go to the gpa Folder and type:

    python compile.py build_ext --inplace

### Execution

If you want to analyse a single image, and/or if you want to display it:

    python runGPA.py Gn filename tol posTol

If you want to compute GPA for multiple images:

    python runGPA.py Gn  filelist tol posTol

The parameters tol and rad_tol are the vectorial modulus and phase tolerance (float). Gn is the gradient moment (G1, G2, G3, or G4). 

### Execution Examples
#### Single file

    python runGPA.py G1 test/m4.txt 0.05 0.05

Must show the image:

![mapExampleIt19](/gpa/Figures/exampleOutput_m4.png)

#### Multiple files

    python main.py G1 configexample.txt 0.01 0.01

Must write in result.csv:

File | G1	| Nc |	Nv | t1 | t2 | t3
------- | ------- | ------- | ------- | ------- | ------- | -------
test/m2.txt | 1.82899999619 | 314 | 111| 0.0005| 0.001| 0.001
test/m3.txt | 1.81700003147 | 338 | 120| 0.0006| 0.001| 0.001
test/m4.txt | 1.96000003815 | 950 | 321| 0.0005| 0.001| 0.001

#### Fragmentation example

    python fragGPA.py G2 test/t1350.csv 0.01 0.01

![mapExampleIt19](/gpa/Figures/frag1350.png)

## Log

Nov. 27, 2018 - Added the FFT-Hilbert approach (file fragGPA.py)\
Nov. 26, 2018 - Changed the user interface (it automatically identifies whether the file is a list or a data)\
&emsp;&emsp; &emsp; &emsp; &emsp; - Results are saved at reult.csv\
&emsp;&emsp; &emsp; &emsp; &emsp; - Tests updated
              
Mar. 27, 2018 - Added the Third and Fourth Gradient Moments (G3 and G4)

Aug. 30, 2017 - Added the First Gradient Moment (G1)




## References
https://en.wikipedia.org/wiki/Gradient_pattern_analysis

http://cython.org/
