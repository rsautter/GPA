import numpy as np
from scipy.spatial import Delaunay as Delanuay
import itertools
from time import time as time
from numba import jit, prange


class GPA2D:

	def __init__(self,tol):
		self.tol = tol
		self.triangulation_points = []
	
	def setPosition(self,cx, cy):
		self.cx = cx
		self.cy = cy
	
	def getMod(self,x,y):
		return np.sqrt(np.power(x,2.0)+np.power(y,2.0))/self.maxGrad

	def _setGradients(self):
		self.gradient_dx, self.gradient_dy = np.gradient(self.mat)
		self._setMaxGrad()
		self._setModulusPhase()

	def _setMaxGrad(self):
		gx = self.gradient_dx
		gy = self.gradient_dy
		mods = np.sqrt((gx**2+gy**2))
		self.maxGrad = np.max(mods,axis=None)	
		if self.maxGrad < 1e-6:
			self.maxGrad = 1.0
	
	def _setModulusPhase(self):
		gx = self.gradient_dx
		gy = self.gradient_dy
		self.phases = np.arctan2(gy,gx)
		self.mods = self.getMod(gx, gy) 
	
	def version():
		return "GPA - 4.0"
	
	@staticmethod
	@jit(nopython=True,parallel=True) 
	def staticUpdateSymmetry(mods,gx,gy,dists,maxGrad,tol):
		unknownP = np.zeros(mods.shape,dtype=np.int32)
		symmetricalP = np.zeros(mods.shape,dtype=np.int32)
		d1,d2 = mods.shape
		
		for x in prange(d1):
			for y in range(d2):
				position = (x,y)
				if mods[position]<tol:
					unknownP[position] = 1
					continue
				dx,dy = gx[position],gy[position]
				d = dists[position]
				sumup = np.sqrt((gx+dx)**2+(gy+dy)**2)/maxGrad 
				if np.logical_and( dists==d , sumup<=tol).any():
					symmetricalP[position] = 1
		return unknownP,symmetricalP
	
	def _G1(self, symm):
		self.triangulation_points = []
		
		if symm == 'S':# Symmetrical matrix 
			targetMat = self.symmetricalP
		elif symm == 'A':# Asymmetrical matrix 
			targetMat = self.asymmetricalP
		elif symm == 'F': # Full Matrix, including unknown vectors
			targetMat = np.ones((self.symmetricalP.shape[0],self.symmetricalP.shape[1]),dtype=np.int32)
		elif symm == 'K': # Full Matrix, excluding unknown vectors
			targetMat = np.logical_or(self.symmetricalP,self.asymmetricalP).astype(np.int32)
		else:
			raise Exception("Unknown analysis type (should be S,A,F or K), got: "+symm)
		
		iterator = np.where(targetMat>0)
		
		for i,j in zip(*iterator):
			self.triangulation_points.append([j+0.5*self.gradient_dx[i, j], i+0.5*self.gradient_dy[i, j] ])
					
		self.triangulation_points = np.array(self.triangulation_points)
		self.n_points = len(self.triangulation_points)
		if self.n_points < 3:
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
		
		if symm == 'S':# Symmetrical matrix 
			targetMat = self.symmetricalP
		elif symm == 'A':# Asymmetrical matrix 
			targetMat = self.asymmetricalP
		elif symm == 'F': # Full Matrix, including unknown vectors
			targetMat = np.ones((self.symmetricalP.shape[0],self.symmetricalP.shape[1]),dtype=np.int32)
		elif symm == 'K': # Full Matrix, excluding unknown vectors
			targetMat = np.logical_or(self.symmetricalP,self.asymmetricalP).astype(np.int32)
		else:
			raise Exception("Unknown analysis type (should be S,A,F or K), got: "+symm)
		
		iterator = np.where(targetMat>0)
		
		for i,j in zip(*iterator):
			self.triangulation_points.append([j+0.5*self.gradient_dx[i, j], i+0.5*self.gradient_dy[i, j] ])
					
		self.triangulation_points = np.array(self.triangulation_points)
		self.n_points = len(self.triangulation_points)
		if self.n_points < 3:
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
		
		if symm == 'S':# Symmetrical matrix
			targetMat = self.symmetricalP
			opositeMat = self.asymmetricalP
		elif symm == 'A':# Asymmetrical matrix
			targetMat = self.asymmetricalP
			opositeMat = self.symmetricalP
		elif symm == 'F':# Full Matrix, including unknown vectors
			targetMat = np.ones((self.symmetricalP.shape[0],self.symmetricalP.shape[1]),dtype=np.int32)
			opositeMat = np.zeros((self.symmetricalP.shape[0],self.symmetricalP.shape[1]),dtype=np.int32)
		elif symm == 'K': # Full Matrix, excluding unknown vectors
			targetMat = np.logical_or(self.symmetricalP,self.asymmetricalP).astype(dtype=np.int32)
			opositeMat = np.zeros((self.symmetricalP.shape[0],self.symmetricalP.shape[1]),dtype=np.int32)
		else:
			raise Exception("Unknown analysis type (should be S,A,F or K), got: "+symm)
		
		if np.sum(targetMat)<1:
			self.G2 = 0.0
			return
		
		alinhamento = 0.0
		
		iterator = np.where(targetMat>0)
		
		
		if symm != 'S':
			somax = np.sum(self.gradient_dx[np.where(targetMat>0)])/self.maxGrad
			somay = np.sum(self.gradient_dy[np.where(targetMat>0)])/self.maxGrad
			smod = np.sum(self.mods[np.where(targetMat>0)])
			if smod <= 0.0:
				alinhamento = 0.0
			else:
				alinhamento = np.sqrt(np.power(somax,2.0)+np.power(somay,2.0))/smod
			if np.sum(opositeMat)+np.sum(targetMat)> 0:
				self.G2 = (float(np.sum(targetMat))/float(np.sum(opositeMat)+np.sum(targetMat)) )*(2.0-alinhamento)
			else: 
				self.G2 = 0.0
		else:
			probabilityMat = self.mods*np.array(targetMat,dtype=float)
			probabilityMat = probabilityMat/np.sum(probabilityMat)
			maxEntropy = np.log(float(np.sum(targetMat)))
			alinhamento = - np.sum(probabilityMat[np.where(targetMat>0)]*np.log(probabilityMat[np.where(targetMat>0)]))/maxEntropy
			self.G2 = alinhamento

	
	def distAngle(self,a1,a2):
		'''
		Produto interno no sistema de coordenada polares, somado 1 e dividido por 2 para normalizar a distancia entre 0 e 1
		'''
		return (np.cos(a1)*np.cos(a2)+np.sin(a1)*np.sin(a2))
	
	def _G3(self,symm):
		
		if symm == 'S':# Symmetrical matrix 
			targetMat = self.symmetricalP
			opositeMat = self.asymmetricalP
		elif symm == 'A':# Asymmetrical matrix 
			targetMat = self.asymmetricalP
			opositeMat = self.symmetricalP
		elif symm == 'F': # Full Matrix, including unknown vectors
			targetMat = np.ones((self.symmetricalP.shape[0],self.symmetricalP.shape[1]),dtype=np.int32)
			opositeMat = np.zeros((self.symmetricalP.shape[0],self.symmetricalP.shape[1]),dtype=np.int32)
		elif symm == 'K': # Full Matrix, excluding unknown vectors
			targetMat = np.logical_or(self.symmetricalP,self.asymmetricalP).astype(dtype=np.int32)
			opositeMat = np.zeros((self.symmetricalP.shape[0],self.symmetricalP.shape[1]),dtype=np.int32)
		else:
			raise Exception("Unknown analysis type (should be S,A,F or K), got: "+symm)
		
	
		targetList = np.zeros((np.sum(targetMat),3),dtype=np.int32)
		
		i = 0
		for ty in range(self.rows):
			for tx in range(self.cols):
				if targetMat[ty,tx]>0:
					targetList[i,0] = ty
					targetList[i,1] = tx
					i = i+1

		newReferencePhase = []
		for i in range(len(targetList)):
			y1, x1  = targetList[i,0],targetList[i,1]
			y2, x2  = y1-int(self.cy), x1-int(self.cx)
			angle = np.arctan2(y2,x2) 
			newReferencePhase.append(self.distAngle(self.phases[x1,y1],angle))
		if np.sum(targetMat)> 1:
			self.G3 = np.std(newReferencePhase)
		else: 
			self.G3 = 0.0
	
	def __call__(self,mat=None,gx=None,gy=None,moment=["G2"],symmetrycalGrad='A',showTimer=False):
		if (mat is None) and (gx is None) and (gy is None) :
			raise Exception("Matrix or gradient must be stated!")
		if (mat is None) and ((gy is None) or (gx is None) ):
			raise Exception("Gradient must have 3 components (gx, gy and gz)")
		if not(mat is None) and not(gx is None):
			raise Exception("Matrix or gradient must be stated, not both")
		
		if not(mat is None):
			return self._eval(mat,moment,symmetrycalGrad,showTimer)
		else:
			return self._evalGradient(gx,gy,moment,symmetrycalGrad)
	
	def _eval(self,mat,moment=["G2"],symmetrycalGrad='A',showTimer=False):
		self.mat = mat
		self.cols = len(self.mat[0])
		self.rows = len(self.mat)
		
		
		if showTimer:
			timer = time()
		self.setPosition(float(self.rows/2),float(self.cols/2))
		self._setGradients()
		if showTimer:
			print("Gradient:",time()-timer, "seconds")
		
		x,y = np.meshgrid(np.linspace(-0.5,0.5,self.rows),np.linspace(-0.5,0.5,self.cols))
		
		if showTimer:
			timer = time()
		dists = np.sqrt(x**2+y**2)
		dists = np.digitize(dists, np.linspace(0.0,0.5,np.max([self.rows,self.cols])))
		if showTimer:
			print("Distance:",time()-timer, "seconds")
		
		if showTimer:
			timer = time()
		self.unknownP,self.symmetricalP =  self.staticUpdateSymmetry(self.mods,self.gradient_dx,self.gradient_dy,dists,self.maxGrad,self.tol)
		self.asymmetricalP = np.zeros(self.mods.shape,dtype=np.int32)
		self.asymmetricalP[np.logical_and(self.symmetricalP== 0, self.unknownP ==0)] = 1
		if showTimer:
			print("Symmetry:",time()-timer, "seconds")
		
		#gradient moments:
		retorno = {}
		if showTimer:
			timer = time()
		for gmoment in moment:
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
	
	def _evalGradient(self,gx, gy, moment=["G2"],symmetrycalGrad='A',showTimer=False):
		self.gradient_dx = gx
		self.gradient_dy = gy
		self.cols = len(gx[0])
		self.rows = len(gx)
		
		self.setPosition(float(self.rows/2),float(self.cols/2))
		self._setMaxGrad()
		self._setModulusPhase()
		
		x,y = np.meshgrid(np.linspace(-0.5,0.5,self.rows),np.linspace(-0.5,0.5,self.cols))
		
		dists = np.sqrt(x**2+y**2)
		dists = np.digitize(dists, np.linspace(0.0,0.5,np.max([self.rows,self.cols])))

		
		# removes the symmetry in gradient_asymmetric_dx and gradient_asymmetric_dy:
		if showTimer:
			timer = time()
		self.unknownP,self.symmetricalP =  self.staticUpdateSymmetry(self.mods,self.gradient_dx,self.gradient_dy,dists,self.maxGrad,self.tol)
		self.asymmetricalP = np.zeros(self.mods.shape,dtype=np.int32)
		self.asymmetricalP[np.logical_and(self.symmetricalP== 0, self.unknownP ==0)] = 1
		if showTimer:
			print("Symmetry:",time()-timer, "seconds")
		
		#gradient moments:
		retorno = {}
		for gmoment in moment:
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
