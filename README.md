# Concentric GPA
This is a concentric GPA prototype developed in Cython + Python

### Requirements
 - Python 2.7 or greater
 - Matplotlib
 - Cython
 - Numpy

### Compilation

This code uses the Cython library, to improve the perfomence. To compile the cython part:

    python compile.py build_ext --inplace

### Execution

If you want to analyse a single image, and/or if you want to display it:

    python main.py filename tol rad_tol

If you want to compute multiple images, and/or want to display it:

    python main.py -l filelist tol rad_tol output

### Execution Examples
#### Single file

    python main.py test/m4.txt 0.02 1

Must output the image:

![mapExampleIt19](/gpa/Figures/exampleOutput_m4.png)

#### Multiple files

    python main.py -l configexample.txt 0.03 1 gas.csv

Must outputs in file gas.csv:

\# | Ga	| Nc |	Nv
------- | ------- | ------- | -------
test/m2.txt |	1.67999994755	| 67	| 25
test/m3.txt	| 1.46153843403	| 32	| 13
test/m4.txt	| 1.89075624943 |	344	| 119


## References
https://en.wikipedia.org/wiki/Gradient_pattern_analysis

http://cython.org/
