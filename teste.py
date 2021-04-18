import GPA
import numpy as np

mat = np.array([[0.0,0.0,0.0],
				[1.0,0.0,0.0],
				[0.0,0.0,0.0],
				[0.0,1.0,0.0]])

a = GPA.GPA(0.01)
print("GPA version: ", a.version())
print("Resultado (simetrico):", a(mat,moment=['G1','G2','G3'], symmetrycalGrad='S'))
print("Resultado (assimetrico):", a(mat,moment=['G1','G2','G3'],symmetrycalGrad='A'))
print("Resultado (completo exceto desconhecido):", a(mat,moment=['G1','G2','G3'],symmetrycalGrad='L'))
print("Resultado (completo):", a(mat,moment=['G1','G2','G3'],symmetrycalGrad='F'))