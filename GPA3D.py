import numpy as np
import math
from scipy.spatial import Delaunay as Delanuay
import itertools
from sympy.algebras.quaternion import Quaternion
from time import time as time
from numba import jit, prange



class GPA3D:

	def __init__(self,tol):
		'''
		Constructor - tol is the summation tolerance for symmetry detection
		'''
		self.tol = tol
		self.triangulation_points = []
	
	def setPosition(self,cx, cy,cz):
		self.cx = cx
		self.cy = cy
		self.cz = cz
	
	def getMod(self,x,y,z):
		'''
		Returns the module normalized by maxGrad
		'''
		return np.sqrt(np.power(x,2.0)+np.power(y,2.0)+np.power(z,2.0))/self.maxGrad

	def _setGradients(self):
		self.gradient_dx, self.gradient_dy, self.gradient_dz = np.gradient(self.mat)
		self._setMaxGrad()
		self._setModulusPhase()

	def _setMaxGrad(self):
		'''
		Determines the largest vector in the gradient field, ts a tolerance of 10^-6 (which case it is considered maxGrad = 1)
		'''
		gx = self.gradient_dx
		gy = self.gradient_dy
		gz = self.gradient_dz
		mods = np.sqrt((gx**2+gy**2+gz**2))
		self.maxGrad = np.max(mods,axis=None)	
		if self.maxGrad < 1e-6:
			self.maxGrad = 1.0
	
	def _setModulusPhase(self):
		'''
		Measures the spherical coordinate system
		'''
		gx = self.gradient_dx
		gy = self.gradient_dy
		gz = self.gradient_dz
		
		self.phasesTheta = np.arctan2(gy,gx)
		self.phasesPhi = np.arctan2(np.sqrt(gy**2+gx**2),gz)
		self.mods = self.getMod(gx, gy, gz) 
	
	def version():
		return "GPA - 4.0"
	
	@staticmethod
	@jit(nopython=True,parallel=True) 
	def staticUpdateSymmetry(mods,gx,gy,gz,dists,maxGrad,tol):
		unknownP = np.zeros(mods.shape,dtype=np.int32)
		symmetricalP = np.zeros(mods.shape,dtype=np.int32)
		d1,d2, d3 = mods.shape
		for x in prange(d1):
			for y in range(d2):
				for z in range(d3):
					position = (x,y,z)
					if mods[position]<tol:
						unknownP[position] = 1
						continue
					dx,dy,dz = gx[position],gy[position],gz[position]
					d = dists[position]
					sumup = np.sqrt((gx+dx)**2+(gy+dy)**2+(gz+dz)**2)/maxGrad 
					if np.logical_and( dists==d , sumup<=tol).any():
						symmetricalP[position] = 1
		return unknownP,symmetricalP

	def __selectSymmAnalysis(self,symm):
		'''
			Returns the boolean matrix of analysis 
		'''
		if symm == 'S':# Symmetrical matrix
			targetMat = self.symmetricalP
			opositeMat = self.asymmetricalP
		elif symm == 'A':# Asymmetrical matrix
			targetMat = self.asymmetricalP
			opositeMat = self.symmetricalP
		elif symm == 'F':# Full Matrix, including unknown vectors
			targetMat = np.ones_like(self.symmetricalP)
			opositeMat = np.zeros_like(self.symmetricalP)
		elif symm == 'K': # Full Matrix, excluding unknown vectors
			targetMat = np.logical_or(self.symmetricalP,self.asymmetricalP)
			opositeMat = np.logical_not(targetMat)
		else:
			raise Exception("Unknown analysis type (should be S,A,F or K), got: "+symm)
		return targetMat, opositeMat

	def _G1(self, symm):
		self.triangulation_points = []
		targetMat, opositeMat = self.__selectSymmAnalysis(symm)

		iterator = np.where(targetMat>0)
		
		for i,j,k in zip(*iterator):
			self.triangulation_points.append([j+0.5*self.gradient_dx[i, j, k], i+0.5*self.gradient_dy[i, j, k], k+0.5*self.gradient_dz[i, j, k] ])
					
		self.triangulation_points = np.array(self.triangulation_points)
		self.n_points = len(self.triangulation_points)
		if self.n_points < 5:
			self.n_edges = 0
			self.G1 = 0.0
		else:
			self.triangles = Delanuay(self.triangulation_points)
			neigh = self.triangles.vertex_neighbor_vertices
			self.n_edges = len(neigh[1])/2
			self.G1 = float(self.n_edges-self.n_points)/float(self.n_points)
		if self.G1<0.0:
			self.G1 = 0.0
		
	def _G1N(self, symm):
		self.triangulation_points = []
		
		targetMat, opositeMat = self.__selectSymmAnalysis(symm)

		iterator = np.where(targetMat>0)
		
		for i,j,k in zip(*iterator):
			self.triangulation_points.append([j+0.5*self.gradient_dx[i, j, k], i+0.5*self.gradient_dy[i, j, k], k+0.5*self.gradient_dz[i, j, k] ])
					
		self.triangulation_points = np.array(self.triangulation_points)
		self.n_points = len(self.triangulation_points)
		if self.n_points < 5:
			self.n_edges = 0
			self.G1N = 0.0
		else:
			self.triangles = Delanuay(self.triangulation_points)
			neigh = self.triangles.vertex_neighbor_vertices
			self.n_edges = len(neigh[1])/2
			self.G1N = np.exp(- float(self.n_points)/float(self.n_edges) )
		if self.G1N<0.0:
			self.G1N = 0.0
	
	def _G2(self,symm):
		somax = 0.0
		somay = 0.0
		somaz = 0.0
		smod = 0.0
		targetMat, opositeMat = self.__selectSymmAnalysis(symm)
		
		if np.sum(targetMat)<1:
			self.G2 = 0.0
			return
		
		alinhamento = 0.0
		
		iterator = np.where(targetMat>0)
		
		
		if symm != 'S':
			somax = np.sum(self.gradient_dx[np.where(targetMat>0)])/self.maxGrad
			somay = np.sum(self.gradient_dy[np.where(targetMat>0)])/self.maxGrad
			somaz = np.sum(self.gradient_dz[np.where(targetMat>0)])/self.maxGrad
			smod = np.sum(self.mods[np.where(targetMat>0)])
			if smod <= 0.0:
				alinhamento = 0.0
			else:
				alinhamento = np.sqrt(np.power(somax,2.0)+np.power(somay,2.0)+np.power(somaz,2.0))/(2*smod)
			if np.sum(opositeMat)+np.sum(targetMat)> 0:
				self.G2 = (float(np.sum(targetMat))/float(np.sum(opositeMat)+np.sum(targetMat)) )*(1.0-alinhamento)
			else: 
				self.G2 = 0.0
		else:
			probabilityMat = self.mods*np.array(targetMat,dtype=float)
			probabilityMat = probabilityMat/np.sum(probabilityMat)
			maxEntropy = np.log(float(np.sum(targetMat)))
			alinhamento = - np.sum(probabilityMat[np.where(targetMat>0)]*np.log(probabilityMat[np.where(targetMat>0)]))/maxEntropy
			self.G2 = alinhamento

	
	def distAngle(self,a1,a11,a2,a21):
		'''
		Internal product between two phases
		'''
		return (np.cos(a1)*np.cos(a2)*np.sin(a11)*np.sin(a21)+
				np.sin(a1)*np.sin(a2)*np.sin(a11)*np.sin(a21)+
				np.cos(a11)*np.cos(a21))
	
	def _G3(self,symm):
		
		targetMat, opositeMat = self.__selectSymmAnalysis(symm)
		
		targetList = np.zeros((np.sum(targetMat),3),dtype=np.int32)
		
		i = 0
		for ty in range(self.rows):
			for tx in range(self.cols):
				for tz in range(self.depth):
					if targetMat[ty,tx,tz]>0:
						targetList[i,0] = ty
						targetList[i,1] = tx
						targetList[i,2] = tz
						i = i+1
		
		newReferencePhase = []
		for i in range(len(targetList)):
			y1, x1, z1  = targetList[i,0],targetList[i,1],targetList[i,2]
			y2, x2, z2  = y1-int(self.cy), x1-int(self.cx),z1-int(self.cz)
			angle1 = np.arctan2(y2,x2) 
			angle2 = np.arctan2(np.sqrt(y2**2+x2**2),z2)
			newReferencePhase.append(self.distAngle(self.phasesTheta[x1,y1,z1],self.phasesPhi[x1,y1,z1],angle1,angle2))
		if np.sum(targetMat)> 1:
			self.G3 = np.std(newReferencePhase)
		else: 
			self.G3 = 0.0
	
	def quaternion2Numpy(quat):
		out = np.zeros(4)
		out[0] = quat.a
		out[1] = quat.b
		out[2] = quat.c
		out[3] = quat.d
		return out
			
	def _G4(self,symm):
		
		targetMat, opositeMat = self.__selectSymmAnalysis(symm)
			
		self.G4 =  Quaternion(0,0,0,0)
		for ty in range(self.rows):
			for tx in range(self.cols):
				for tz in range(self.depth):
					if targetMat[ty,tx,tz]>0:
						
						z = Quaternion(self.gradient_dx[ty,tx,tz]/self.maxGrad,self.gradient_dy[ty,tx,tz]/self.maxGrad,self.gradient_dz[ty,tx,tz]/self.maxGrad,0)
						z2 = z*(z._ln())
						'''
						Considering a quaternion of the form:

							z = a + b*i + c*j + + d*k = a + v

						The log function is:
							ln(z) = ln |z| + (v/|v|)*arccos(a/|q|)

						When v is null, which is the multicomplex part, the log function becames Nan.
						This is a numerical method error, which should be accomplished by the library, but it is not.
						Therefore we treat as null the complex part as null
						'''
						if math.isnan(z2.b) and math.isnan(z2.c) and math.isnan(z2.d):
							z2 = z*Quaternion(np.log(np.abs(float(z.a))),0.0,0.0,0.0)
						self.G4 = self.G4 - z2
						
	
	def __call__(self,mat=None,gx=None,gy=None,gz=None,moment=["G2"],symmetrycalGrad='A',showTimer=False):
		'''
		There is two input options:
			- mat -> amplitude field
			- gx,gy,gz -> gradient field	
		'''
		if (mat is None) and (gx is None) and (gy is None) and (gx is None):
			raise Exception("Matrix or gradient must be stated!")
		if (mat is None) and ((gy is None) or (gx is None) or (gz is None)):
			raise Exception("Gradient must have 3 components (gx, gy and gz)")
		if not(mat is None) and not(gx is None):
			raise Exception("Matrix or gradient must be stated, not both")
		
		if not(mat is None):
			return self._eval(mat,moment,symmetrycalGrad,showTimer)
		else:
			return self._evalGradient(gx,gy,gz,moment,symmetrycalGrad)
	
	def _eval(self,mat,moment=["G2"],symmetrycalGrad='A',showTimer=False):
		self.mat = mat
		self.depth = len(self.mat[0,0])
		self.cols = len(self.mat[0])
		self.rows = len(self.mat)
		
		
		if showTimer:
			timer = time()
		self.setPosition(float(self.rows/2),float(self.cols/2),float(self.depth/2))
		self._setGradients()
		if showTimer:
			print("Gradient:",time()-timer, "seconds")
		
		x,y,z = np.meshgrid(np.linspace(-0.5,0.5,self.rows),np.linspace(-0.5,0.5,self.cols),np.linspace(-0.5,0.5,self.depth))
		
		if showTimer:
			timer = time()
		dists = np.sqrt(x**2+y**2+z**2)
		dists = np.digitize(dists, np.linspace(0.0,0.5,np.max([self.depth,self.rows,self.cols])))
		if showTimer:
			print("Distance:",time()-timer, "seconds")
		
		if showTimer:
			timer = time()
		self.unknownP,self.symmetricalP =  self.staticUpdateSymmetry(self.mods,self.gradient_dx,self.gradient_dy,self.gradient_dz,dists,self.maxGrad,self.tol)
		self.asymmetricalP = np.zeros(self.mods.shape,dtype=np.int32)
		self.asymmetricalP[np.logical_and(self.symmetricalP== 0, self.unknownP ==0)] = 1
		if showTimer:
			print("Symmetry:",time()-timer, "seconds")
		
		#gradient moments:
		retorno = {}
		if showTimer:
			timer = time()
		for gmoment in moment:
			if("G4" == gmoment):
				self._G4(symmetrycalGrad)
				retorno["G4"] = self.G4
			if("G3" == gmoment):
				self._G3(symmetrycalGrad)
				retorno["G3"] = self.G3
			if("G2" == gmoment):
				self._G2(symmetrycalGrad)
				retorno["G2"] = self.G2
			if("G1" == gmoment):
				self._G1(symmetrycalGrad)
				retorno["G1"] = self.G1
			if("G1N" == gmoment):
				self._G1N(symmetrycalGrad)
				retorno["G1N"] = self.G1N
		if showTimer:
			print("Moment:",time()-timer, "seconds")
		
		return retorno
	
	def _evalGradient(self,gx, gy, gz,moment=["G2"],symmetrycalGrad='A',showTimer=False):
		self.gradient_dx = gx
		self.gradient_dy = gy
		self.gradient_dz = gz
		self.depth = len(gx[0,0])
		self.cols = len(gx[0])
		self.rows = len(gx)
		
		self.setPosition(float(self.rows/2),float(self.cols/2),float(self.depth/2))
		self._setMaxGrad()
		self._setModulusPhase()
		
		x,y,z = np.meshgrid(np.linspace(-0.5,0.5,self.rows),np.linspace(-0.5,0.5,self.cols),np.linspace(-0.5,0.5,self.depth))
		
		dists = np.sqrt(x**2+y**2+z**2)
		dists = np.digitize(dists, np.linspace(0.0,0.5,np.max([self.depth,self.rows,self.cols])))

		
		# removes the symmetry in gradient_asymmetric_dx and gradient_asymmetric_dy:
		if showTimer:
			timer = time()
		self.unknownP,self.symmetricalP =  self.staticUpdateSymmetry(self.mods,self.gradient_dx,self.gradient_dy,self.gradient_dz,dists,self.maxGrad,self.tol)
		self.asymmetricalP = np.zeros(self.mods.shape,dtype=np.int32)
		self.asymmetricalP[np.logical_and(self.symmetricalP== 0, self.unknownP ==0)] = 1
		if showTimer:
			print("Symmetry:",time()-timer, "seconds")
		
		#gradient moments:
		retorno = {}
		for gmoment in moment:
			if("G4" == gmoment):
				self._G4(symmetrycalGrad)
				retorno["G4"] = self.G4
			if("G3" == gmoment):
				self._G3(symmetrycalGrad)
				retorno["G3"] = self.G3
			if("G2" == gmoment):
				self._G2(symmetrycalGrad)
				retorno["G2"] = self.G2
			if("G1" == gmoment):
				self._G1(symmetrycalGrad)
				retorno["G1"] = self.G1
			if("G1N" == gmoment):
				self._G1N(symmetrycalGrad)
				retorno["G1N"] = self.G1N
		return retorno
