import GPA
import pandas as pd
import numpy as np

class GPA1D(GPA.GPA):
	def __init__(self,tol=0.03, spaceFilling='lines', splitWidth=3):
		'''
			tol - muduli tolerance
			spaceFilling - spatial curve to transform the time series into matrices ('lines','hilbert') 
			scale - lattice size generated for spacefilling curve
		'''
		GPA.GPA.__init__(tol)
		self.splitWidth = splitWidth
		self.spaceFilling = spaceFilling
	
	def verifyPower2(self,value):
		i = int(np.log2(value))//2
		while (i>=1):
			if value % (2**i) > 0:
				return False
			i= i-1
		return True

	def _transformData(self,vet):
		if self.spaceFilling == 'hilbert':
			mat = gilbert.vec2mat(vet, self.splitWidth)
		elif self.spaceFilling == 'lines':
			mat = vet.reshape(self.splitWidth,self.splitWidth)
			for i in range(1,len(mat),2):
				mat[i] = np.flip(mat[i])
		else:
			mat = vet.reshape(self.splitWidth,self.splitWidth)
		return mat

	def __call__(self,timeSeries,moment=["G2"],symmetrycalGrad='A'):
		if len(timeSeries) % (self.splitWidth**2) != 0:
			raise Exception(f"The time series must be multiple of {self.splitWidth**2}. You can interpolate or remove some elements.")
		splitted = np.array_split(timeSeries, len(timeSeries) // (self.splitWidth**2))
		res = []
		for s in splitted:
			mat = self._transformData(s)
			res.append(super.__call__(mat,moment,symmetrycalGrad))
		return pd.DataFrame(res)
