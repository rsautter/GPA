import GPA
import numpy as np

mat = np.array([[0.0,0.0,0.0],
				[1.0,0.0,0.0],
				[0.0,0.0,0.0],
				[0.0,1.0,0.0]])

a = GPA.GPA(0.01)
print("Resultado:", a.evaluate(mat,moment=['G1','G2','G3']))