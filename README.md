# Concentric Gradient Pattern Analysis
This is a Concentric Gradient Pattern Analysis (CGPA) prototype developed in Cython + Python

### Requirements
 - Python 2.7 or greater
 - Matplotlib
 - Cython
 - Numpy

### Compilation

This code uses the Cython library, to improve the perfomence. 
Go to the gpa Folder and type:

    python compile.py build_ext --inplace

### Execution

If you want to analyse a single image, and/or if you want to display it:

    python main.py G2 filename tol posTol

If you want to compute multiple images:

    python main.py G1 -l filelist tol posTol output

The parameter tol is the vectorial modulus and phase tolerance (float), and posTol is the position tolerance (integer)

### Execution Examples
#### Single file

    python main.py test/m4.txt 0.02 1

Must output the image:

![mapExampleIt19](/gpa/Figures/exampleOutput_m4.png)

#### Multiple files

    python main.py G1 -l configexample.txt 0.03 1 gas.csv

Must outputs in file gas.csv:

\# | Ga	| Nc |	Nv
------- | ------- | ------- | -------

test/m2.txt | 1.70833337307 | 130 | 48
test/m3.txt | 1.60526311398 | 99 | 38
test/m4.txt | 1.93684208393 | 558 | 190


## References
https://en.wikipedia.org/wiki/Gradient_pattern_analysis

http://cython.org/
