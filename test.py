import GPA 
import numpy as np

ga = GPA.GPA()
#m = np.arange(25).reshape(5,5).astype(np.float64)
m = np.arange(128*128).reshape(128,128).astype(np.float64)
print(ga(m,moment=['G1','G2','G3']))
