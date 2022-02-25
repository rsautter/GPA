import numpy
from libc.math cimport pow, fabs, sqrt, M_PI, sin, cos,tan,floor
from math import radians, atan2,factorial
from scipy.spatial import Delaunay as Delanuay
import matplotlib.pyplot as plt

from cpython cimport bool
cimport numpy
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef class GPA3D:
	cdef public double[:,:,:] mat,gradient_dx,gradient_dy,gradient_dz
	cdef public double cx, cy, cz
	cdef public int rows, cols, depth
	
	cdef public double[:,:,:] phasesTheta, phasesPhi, mods
	cdef public int[:,:,:] symmetricalP, asymmetricalP, unknownP
	cdef public object triangulation_points,triangles
	cdef public double  maxGrad,tol
	cdef public object cvet

	cdef public int n_edges, n_points
	cdef public double G1, G2, G3
	cdef public object G4

	#@profile
	def __cinit__(self, double tol):
		self.tol = tol
		self.triangulation_points = []

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.nonecheck(False)
	@cython.cdivision(True)
	cpdef void setPosition(self, double cx, double cy, double cz):
		self.cx = cx
		self.cy = cy
		self.cz = cz

	@cython.cdivision(True)
	cdef double getMod(self,x,y,z):
		return sqrt(pow(x,2.0)+pow(y,2.0)+pow(z,2.0))/self.maxGrad

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.nonecheck(False)
	@cython.cdivision(True)
	cdef void _setGradients(self):
		cdef int w, h,i,j,k
		cdef double[:,:,:] gx, gy,gz
		
		gx, gy, gz = self.gradient(self.mat)
		self.gradient_dx = gx
		self.gradient_dy = gy
		self.gradient_dz = gz
		
		self._setMaxGrad()
		self._setModulusPhase()

		
	
	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.nonecheck(False)
	@cython.cdivision(True)
	cdef void _setMaxGrad(self):
		cdef int i,j,k,w,h,d
		cdef double[:,:,:] gx, gy, gz
		
		gx = self.gradient_dx
		gy = self.gradient_dy
		gz = self.gradient_dz
		
		
		self.maxGrad = -1.0
		w, h, d = self.cols, self.rows, self.depth 
		for i in range(w):
			for j in range(h):
				for k in range(d):
					if self.maxGrad<0.0 or sqrt(pow(gx[i,j,k],2.0)+pow(gy[i,j,k],2.0)+pow(gz[i,j,k],2.0))>self.maxGrad:
						self.maxGrad = abs(self.getMod(gx[i,j,k],gy[i,j,k],gz[i,j,k]))
		if self.maxGrad < 1e-5:
			self.maxGrad = 1.0

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.nonecheck(False)
	@cython.cdivision(True)
	cdef void _setModulusPhase(self):
		cdef int w, h, i, j
		cdef double[:,:,:] gx, gy, gz
		
		gx = self.gradient_dx
		gy = self.gradient_dy
		gz = self.gradient_dz
		w, h = self.cols, self.rows 
		
		self.phasesTheta = numpy.array([[[atan2(gy[j, i,k],gx[j, i,k]) for i in range(self.rows) ] for j in range(self.cols)] for k in range(self.depth) ],dtype=float)
		self.phasesPhi = numpy.array([[[atan2(sqrt(gy[j, i,k]**2+gx[j, i,k]**2),gz[j, i,k]) for i in range(self.rows) ] for j in range(self.cols)] for k in range(self.depth) ],dtype=float)
		self.mods = numpy.array([[[self.getMod(gx[j, i, k], gy[j, i, k],gz[j, i, k]) for i in range(self.rows) ] for j in range(self.cols)] for k in range(self.depth)],dtype=float)
		


	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.nonecheck(False)
	@cython.cdivision(True)
	cpdef char* version(self):
		return "GPA - 3.2"

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.nonecheck(False)
	@cython.cdivision(True)
	cdef void _update_asymmetric_mat(self,double[:] index_dist,double[:,:,:] dists,double tol, double ptol):
		cdef int ind, lx, px, py,pz, px2, py2, pz2, i, j
		cdef int[:] x, y
		
		self.symmetricalP = numpy.zeros((self.rows,self.cols,self.depth),dtype=numpy.int32)
		self.asymmetricalP = numpy.zeros((self.rows,self.cols,self.depth),dtype=numpy.int32)
		self.unknownP = numpy.zeros((self.rows,self.cols,self.depth),dtype=numpy.int32)
		
		# distances loop
		for ind in range(0, len(index_dist)):
			x2, y2, z2 =[], [], []
			for py in range(self.rows):
				for px in range(self.cols):
					for pz in range(self.depth):
						if (fabs(dists[py, px, pz]-index_dist[ind]) <= fabs(ptol)):
							x2.append(px)
							y2.append(py)
							z2.append(pz)
			x, y, z = numpy.array(x2,dtype=numpy.int32), numpy.array(y2,dtype=numpy.int32), numpy.array(z2,dtype=numpy.int32)
			lx = len(x)
			# compare each point in the same distance
			for i in range(lx):
				px, py, pz = x[i], y[i], z[i]
				if self.mods[py,px, pz]<= tol:
					self.unknownP[py, px, pz] = 1
					continue
				for j in range(lx):
					px2, py2, pz2 = x[j], y[j], z[j]
					if self.mods[py2,px2, pz2]<= tol:
						continue	
					dx = self.gradient_dx[py, px, pz]+self.gradient_dx[py2, px2, pz2]
					dy = self.gradient_dy[py, px, pz]+self.gradient_dy[py2, px2, pz2]
					dz = self.gradient_dy[py, px, pz]+self.gradient_dy[py2, px2, pz2]
					# Vetores sao simetricos
					if self.getMod(dx,dy,dz)<= tol:
						self.symmetricalP[py,px,pz] = 1
						# se outro for simetrico ele vai se marcar
						break
					
		# Caso nao seja desconhecido ou simetrico, ele eh asimmetrico
		for py in range(self.rows):
			for px in range(self.cols):
				for pz in range(self.depth):
					if self.symmetricalP[py,px,pz] == 0 and self.unknownP[py,px,pz] ==0:
						self.asymmetricalP[py,px,pz] = 1

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.nonecheck(False)
	@cython.cdivision(True)
	def _G1(self,str symm):
		cdef int w, h, i, j, k
		cdef int[:,:,:] targetMat
		self.triangulation_points = []


		if symm == 'S':# Symmetrical matrix 
			targetMat = self.symmetricalP
		elif symm == 'A':# Asymmetrical matrix 
			targetMat = self.asymmetricalP
		elif symm == 'F': # Full Matrix, including unknown vectors
			targetMat = numpy.ones((self.symmetricalP.shape[0],self.symmetricalP.shape[1]),dtype=numpy.int32)
		elif symm == 'K': # Full Matrix, excluding unknown vectors
			targetMat = numpy.logical_or(self.symmetricalP,self.asymmetricalP).astype(numpy.int32)
		else:
			raise Exception("Unknown analysis type (should be S,A,F or K), got: "+symm)
		
		for i in range(self.rows):
			for j in range(self.cols):
				for k in range(self.depth):
					if targetMat[i,j,k] > 0:
						self.triangulation_points.append([j+0.5*self.gradient_dx[i, j, k], i+0.5*self.gradient_dy[i, j, k], k+0.5*self.gradient_dz[i, j, k] ])
					
		self.triangulation_points = numpy.array(self.triangulation_points)
		self.n_points = len(self.triangulation_points)
		if self.n_points < 3:
			self.n_edges = 0
			self.G1 = 0.0
		else:
			self.triangles = Delanuay(self.triangulation_points)
			neigh = self.triangles.vertex_neighbor_vertices
			self.n_edges = len(neigh[1])/2
			self.G1 = round(float(self.n_edges-self.n_points)/float(self.n_points),3)
		if self.G1<0.0:
			self.G1 = 0.0

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.nonecheck(False)
	@cython.cdivision(True)
	cdef void _G2(self,str symm):
		cdef int i,j,k
		cdef double somax, somay,somaz, phase, alinhamento, mod, smod, maxEntropy
		cdef int[:,:,:] targetMat,opositeMat
		cdef double[:,:,:] probabilityMat
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
			targetMat = numpy.ones((self.symmetricalP.shape[0],self.symmetricalP.shape[1]),dtype=numpy.int32)
			opositeMat = numpy.zeros((self.symmetricalP.shape[0],self.symmetricalP.shape[1]),dtype=numpy.int32)
		elif symm == 'K': # Full Matrix, excluding unknown vectors
			targetMat = numpy.logical_or(self.symmetricalP,self.asymmetricalP).astype(dtype=numpy.int32)
			opositeMat = numpy.zeros((self.symmetricalP.shape[0],self.symmetricalP.shape[1]),dtype=numpy.int32)
		else:
			raise Exception("Unknown analysis type (should be S,A,F or K), got: "+symm)
		
		if numpy.sum(targetMat)<1:
			self.G2 = 0.0
			return
		
		alinhamento = 0.0
		
		if symm != 'S':
			for i in range(self.rows):
				for j in range(self.cols):
					for k in range(self.depth):
						if targetMat[i,j,k] == 1:
							somax += self.gradient_dx[i,j,k]
							somay += self.gradient_dy[i,j,k]
							somaz += self.gradient_dz[i,j,k]
							smod += self.mods[i,j,k]
			if smod <= 0.0:
				alinhamento = 0.0
			else:
				alinhamento = sqrt(pow(somax,2.0)+pow(somay,2.0))/smod
			if numpy.sum(opositeMat)+numpy.sum(targetMat)> 0:
				self.G2 = (float(numpy.sum(targetMat))/float(numpy.sum(opositeMat)+numpy.sum(targetMat)) )*(2.0-alinhamento)
			else: 
				self.G2 = 0.0
		else:
			probabilityMat = self.mods*numpy.array(targetMat,dtype=float)
			probabilityMat = probabilityMat/numpy.sum(probabilityMat)
			maxEntropy = numpy.log(float(numpy.sum(targetMat)))
			for i in range(self.rows):
				for j in range(self.cols):
					for k in range(self.depth):
						if targetMat[i,j,k] == 1:
							alinhamento = alinhamento - probabilityMat[i,j,k]*numpy.log(probabilityMat[i,j,k])/maxEntropy
			self.G2 = 2*alinhamento

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.nonecheck(False)
	@cython.cdivision(True)
	cdef double distAngle(self,double a1,double a11,double a2,double a21):
		'''
		Produto interno no sistema de coordenada esferico
		'''
		return (cos(a1)*cos(a2)*sin(a11)*sin(a21)+
				sin(a1)*sin(a2)*sin(a11)*sin(a21)+
				cos(a11)*cos(a21)
				+1)/2
				
	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.nonecheck(False)
	@cython.cdivision(True)
	cdef void _G3(self,str symm):
		cdef int x1, y1, z1, x2, y2,z2, i, j, div
		cdef double sumPhases, alinhamento,nterms,angle
		cdef int[:,:,:] targetMat,opositeMat
		cdef int[:,:] targetList

		if symm == 'S':# Symmetrical matrix 
			targetMat = self.symmetricalP
			opositeMat = self.asymmetricalP
		elif symm == 'A':# Asymmetrical matrix 
			targetMat = self.asymmetricalP
			opositeMat = self.symmetricalP
		elif symm == 'F': # Full Matrix, including unknown vectors
			targetMat = numpy.ones((self.symmetricalP.shape[0],self.symmetricalP.shape[1]),dtype=numpy.int32)
			opositeMat = numpy.zeros((self.symmetricalP.shape[0],self.symmetricalP.shape[1]),dtype=numpy.int32)
		elif symm == 'K': # Full Matrix, excluding unknown vectors
			targetMat = numpy.logical_or(self.symmetricalP,self.asymmetricalP).astype(dtype=numpy.int32)
			opositeMat = numpy.zeros((self.symmetricalP.shape[0],self.symmetricalP.shape[1]),dtype=numpy.int32)
		else:
			raise Exception("Unknown analysis type (should be S,A,F or K), got: "+symm)
		
	
		targetList = numpy.zeros((numpy.sum(targetMat),3),dtype=numpy.int32)
		
		i = 0
		for ty in range(self.rows):
			for tx in range(self.cols):
				for tz in range(self.depth):
					if targetMat[ty,tx,tz]>0:
						targetList[i,0] = ty
						targetList[i,1] = tx
						targetList[i,2] = tz
						i = i+1
		
		sumPhases = 0.0
		nterms = 0.0
		alinhamento = 0.0
		for i in range(len(targetList)):
			y1, x1, z1  = targetList[i,0],targetList[i,1],targetList[i,2]
			y2, x2, z2  = y1-int(self.cy), x1-int(self.cx),z1-int(self.cz)
			angle1 = atan2(y2,x2) if atan2(y2,x2)>0 else atan2(y2,x2)+2.0*M_PI
			angle2 = atan2(sqrt(y2**2+x2**2),z2) if atan2(sqrt(y2**2+x2**2),z2)>0 else atan2(sqrt(y2**2+x2**2),z2)+2.0*M_PI
			sumPhases += self.distAngle(self.phasesTheta[x1,y1,z1],self.phasesPhi[x1,y1,z1],angle1,angle2)
			nterms = nterms + 1.0
		if nterms>0.0:
			alinhamento = sumPhases / nterms
		else:
			alinhamento = 0.0
		if numpy.sum(opositeMat)+numpy.sum(targetMat)> 0:
			self.G3 = (float(numpy.sum(targetMat))/float(numpy.sum(opositeMat)+numpy.sum(targetMat)) ) + alinhamento
		else: 
			self.G3 = 0.0
			
	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.nonecheck(False)
	@cython.cdivision(True)
	def __call__(self,double[:,:,:] mat=None, double[:,:,:] gx=None,double[:,:,:] gy=None,double[:,:,:] gz=None,list moment=["G2"],str symmetrycalGrad='A'):
		if (mat is None) and (gx is None) and (gy is None) and (gx is None):
			raise Exception("Matrix or gradient must be stated!")
		if (mat is None) and ((gy is None) or (gx is None) or (gz is None)):
			raise Exception("Gradient must have 3 components (gx, gy and gz)")
		if not(mat is None) and not(gx is None):
			raise Exception("Matrix or gradient must be stated, not both")
		
		if not(mat is None):
			return self._eval(mat,moment,symmetrycalGrad)
		else:
			return self._evalGradient(gx,gy,gz,moment,symmetrycalGrad)
			
	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.nonecheck(False)
	@cython.cdivision(True)
	def _eval(self,double[:,:,:] mat,list moment=["G2"],str symmetrycalGrad='A'):
		cdef int[:] i
		cdef int x, y
		cdef double minimo, maximo
		cdef dict retorno
		
		self.mat = mat
		self.depth = len(self.mat[0,0])
		self.cols = len(self.mat[0])
		self.rows = len(self.mat)
		
		self.setPosition(float(self.rows/2),float(self.cols/2),float(self.depth/2))
		self._setGradients()
		

		cdef numpy.ndarray dists = numpy.array([[[sqrt(pow(float(x)-self.cx, 2.0)+pow(float(y)-self.cy, 2.0)+pow(float(z)-self.cz, 2.0)) \
												  for x in range(self.cols)] for y in range(self.rows)] for z in range(self.depth)])
		
		minimo, maximo = numpy.min(dists),numpy.max(dists)
		sequence = numpy.arange(minimo,maximo,0.705).astype(dtype=float)
		cdef numpy.ndarray uniq = numpy.array([minimo for minimo in  sequence])
		
		# removes the symmetry in gradient_asymmetric_dx and gradient_asymmetric_dy:
		self._update_asymmetric_mat(uniq.astype(dtype=float), dists.astype(dtype=float), self.tol, float(1.41))
		
		#gradient moments:
		retorno = {}
		for gmoment in moment:
			
			#if("G4" == gmoment):
			#	self._G4(symmetrycalGrad)
			#	retorno["G4"] = self.G4
			if("G3" == gmoment):
				self._G3(symmetrycalGrad)
				retorno["G3"] = self.G3
			if("G2" == gmoment):
				self._G2(symmetrycalGrad)
				retorno["G2"] = self.G2
			if("G1" == gmoment):
				self._G1(symmetrycalGrad)
				retorno["G1"] = self.G1
		return retorno
		
	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.nonecheck(False)
	@cython.cdivision(True)
	def _evalGradient(self,double[:,:,:] gx, double[:,:,:] gy, double[:,:,:] gz,list moment=["G2"],str symmetrycalGrad='A'):
		cdef int[:] i
		cdef int x, y
		cdef double minimo, maximo
		cdef dict retorno
		
		self.gradient_dx = gx
		self.gradient_dy = gy
		self.gradient_dz = gz
		self.depth = len(self.gx[0,0])
		self.cols = len(self.gx[0])
		self.rows = len(self.gx)
		
		self.setPosition(float(self.rows/2),float(self.cols/2),float(self.depth/2))
		self._setMaxGrad()
		self._setModulusPhase()
		

		cdef numpy.ndarray dists = numpy.array([[[sqrt(pow(float(x)-self.cx, 2.0)+pow(float(y)-self.cy, 2.0)+pow(float(z)-self.cz, 2.0)) \
												  for x in range(self.cols)] for y in range(self.rows)] for z in range(self.depth)])
		
		minimo, maximo = numpy.min(dists),numpy.max(dists)
		sequence = numpy.arange(minimo,maximo,0.705).astype(dtype=float)
		cdef numpy.ndarray uniq = numpy.array([minimo for minimo in  sequence])
		
		# removes the symmetry in gradient_asymmetric_dx and gradient_asymmetric_dy:
		self._update_asymmetric_mat(uniq.astype(dtype=float), dists.astype(dtype=float), self.tol, float(1.41))
		
		#gradient moments:
		retorno = {}
		for gmoment in moment:
			
			#if("G4" == gmoment):
			#	self._G4(symmetrycalGrad)
			#	retorno["G4"] = self.G4
			if("G3" == gmoment):
				self._G3(symmetrycalGrad)
				retorno["G3"] = self.G3
			if("G2" == gmoment):
				self._G2(symmetrycalGrad)
				retorno["G2"] = self.G2
			if("G1" == gmoment):
				self._G1(symmetrycalGrad)
				retorno["G1"] = self.G1
		return retorno


	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.nonecheck(False)
	@cython.cdivision(True)
	cdef tuple gradient(self,double[:,:,:] mat):
		cdef double[:,:,:] dx, dy, dz
		cdef double divx, divy,divz
		cdef int i, j, k, w, h,p, i1,j1,k1,i2,j2,k2
		w, h, p = len(mat), len(mat[0]),len(mat[0,0])
		dx = numpy.array([[[0.0 for i in range(w) ] for j in range(h)] for k in range(p)],dtype=float)
		dy = numpy.array([[[0.0 for i in range(w) ] for j in range(h)] for k in range(p)],dtype=float)
		dz = numpy.array([[[0.0 for i in range(w) ] for j in range(h)] for k in range(p)],dtype=float)
		for i in range(w):
			for j in range(h):
				for k in range(p):
					divz =  2.0 if (k<p-1 and k>0) else 1.0
					divy =  2.0 if (j<h-1 and j>0) else 1.0
					divx =  2.0 if (i<w-1 and i>0) else 1.0

					i1 = (i+1) if i<w-1 else i
					j1 = (j+1) if j<h-1 else j
					k1 = (k+1) if k<p-1 else k

					i2 = (i-1) if i>0 else i
					j2 = (j-1) if j>0 else j
					k2 = (k-1) if k>0 else k

					dz[i,j,k] = (mat[i1,j,k]-mat[i2,j,k])/divx
					dy[i,j,k] = (mat[i,j1,k]-mat[i,j2,k])/divy
					dx[i,j,k] = (mat[i,j,k1]-mat[i,j,k2])/divz
		return dx, dy, dz

 
			   


