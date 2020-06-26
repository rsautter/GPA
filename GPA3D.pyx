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
cdef class GPA:
    cdef public double[:,:,:] mat,gradient_dx,gradient_dy,gradient_dz,gradient_asymmetric_dy,gradient_asymmetric_dx,gradient_asymmetric_dz
    cdef public double cx, cy, cz
    cdef public int rows, cols, depth
    
    cdef int[:,:] removedP, nremovedP
    cdef public object triangulation_points,triangles
    cdef public int totalAssimetric, totalVet
    cdef public double phaseDiversity,modDiversity, maxGrad,tol
    cdef public object cvet

    cdef public int n_edges, n_points
    cdef public double G1, G2, G3
    cdef public object G4

    #@profile
    def __cinit__(self, double tol):
        self.tol = tol
   
        # percentual Ga proprieties
        self.removedP = numpy.array([[]],dtype=numpy.int32)
        self.nremovedP = numpy.array([[]],dtype=numpy.int32)
        self.triangulation_points = []
        self.phaseDiversity = 0.0

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
        return sqrt(pow(x,2.0)+pow(y,2.0)+pow(z,2.0))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void _setGradients(self):
        cdef int w, h,i,j,k
        cdef double[:,:,:] gx, gy,gz
        
        gx, gy, gz = self.gradient(self.mat)
        
        self.maxGrad = -1.0
        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(self.depth):
                    if self.maxGrad<0.0 or abs(self.getMod(gx[i,j,k],gy[i,j,k],gz[i,j,k]))>self.maxGrad:
                        self.maxGrad = abs(self.getMod(gx[i,j,k],gy[i,j,k],gz[i,j,k]))
        #initialization
        self.gradient_dx = numpy.array([[[gx[i,j,k]/self.maxGrad for i in range(self.rows) ] for j in range(self.cols)] for k in range(self.depth)],dtype=numpy.float)
        self.gradient_dy = numpy.array([[[gy[i,j,k]/self.maxGrad for i in range(self.rows) ] for j in range(self.cols)] for k in range(self.depth)],dtype=numpy.float)
        self.gradient_dz = numpy.array([[[gz[i,j,k]/self.maxGrad for i in range(self.rows) ] for j in range(self.cols)] for k in range(self.depth)],dtype=numpy.float)

        # copying gradient field to asymmetric gradient field
        self.gradient_asymmetric_dx = numpy.array([[[gx[i,j,k]/self.maxGrad for i in range(self.rows) ] for j in range(self.cols)] for k in range(self.depth)],dtype=numpy.float)
        self.gradient_asymmetric_dy = numpy.array([[[gy[i,j,k]/self.maxGrad for i in range(self.rows) ] for j in range(self.cols)] for k in range(self.depth)],dtype=numpy.float)
        self.gradient_asymmetric_dz = numpy.array([[[gz[i,j,k]/self.maxGrad for i in range(self.rows) ] for j in range(self.cols)] for k in range(self.depth)],dtype=numpy.float)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void _update_asymmetric_mat(self,int[:,:,:] dists,double tol):
        cdef int ind, lx, px, py, pz, px2, py2, pz2, i, j
        cdef int[:] x, y, z
        cdef double dx,dy,dz, mod
        
        # distances loop
        for ind in range(0, numpy.max(dists)):
            x2, y2, z2 =[], [], []
            for px in range(self.rows):
                for py in range(self.cols):
                    for pz in range(self.depth):
                        if (dists[px,py,pz]==ind):
                            x2.append(px)
                            y2.append(py)
                            z2.append(pz)
            x, y, z = numpy.array(x2,dtype=numpy.int32), numpy.array(y2,dtype=numpy.int32), numpy.array(z2,dtype=numpy.int32)
            lx = len(x)
            # compare each point in the same distance
            for i in range(lx):
                px, py, pz = x[i], y[i], z[i]
                # is it negligible gradient?                
                if (self.getMod(self.gradient_dx[px, py, pz],self.gradient_dy[px, py, pz],self.gradient_dz[px, py, pz]) <= tol):
                    self.gradient_asymmetric_dx[px, py, pz] = 0.0
                    self.gradient_asymmetric_dy[px, py, pz] = 0.0
                    self.gradient_asymmetric_dy[px, py, pz] = 0.0
                    continue
                for j in range(lx):
                    px2, py2, pz2 = x[j], y[j], z[j]
                    dx = self.gradient_dx[px, py, pz]+self.gradient_dx[px2, py2, pz2]
                    dy = self.gradient_dy[px, py, pz]+self.gradient_dy[px2, py2, pz2]
                    dz = self.gradient_dz[px, py, pz]+self.gradient_dz[px2, py2, pz2]
                    if self.getMod(dx,dy,dz)<= tol:
                        self.gradient_asymmetric_dx[px, py, pz] = 0.0
                        self.gradient_asymmetric_dy[px, py, pz] = 0.0
                        self.gradient_asymmetric_dz[px, py, pz] = 0.0
                        self.gradient_asymmetric_dx[px2, py2, pz2] = 0.0
                        self.gradient_asymmetric_dy[px2, py2, pz2] = 0.0
                        self.gradient_asymmetric_dz[px2, py2, pz2] = 0.0

        self.totalVet = 0
        self.totalAssimetric = 0
        nremovedP = []
        removedP = []

        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(self.depth):
                    mod = self.getMod(self.gradient_asymmetric_dx[i,j,k],self.gradient_asymmetric_dy[i,j,k],self.gradient_asymmetric_dz[i,j,k])
                    if (mod <= tol):
                        removedP.append([i, j, k])
                        self.totalVet = self.totalVet+1
                    else:
                        nremovedP.append([i, j, k])
                        self.totalVet = self.totalVet+1
                        self.totalAssimetric = self.totalAssimetric+1
        if(len(nremovedP)>0):
            self.nremovedP = numpy.array(nremovedP,dtype=numpy.int32)
        if(len(removedP)>0):
            self.removedP = numpy.array(removedP,dtype=numpy.int32)	

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _modVariety(self):
        cdef int i
        cdef double somax,somay,somaz, alinhamento, smod
        somax = 0.0
        somay = 0.0
        somaz = 0.0
        smod = 0.0
        if(self.totalAssimetric<1):
            return 0.0
        for i in range(self.totalAssimetric):
            somax += self.gradient_dx[self.nremovedP[i,0],self.nremovedP[i,1],self.nremovedP[i,2]]
            somay += self.gradient_dy[self.nremovedP[i,0],self.nremovedP[i,1],self.nremovedP[i,2]]
            somaz += self.gradient_dz[self.nremovedP[i,0],self.nremovedP[i,1],self.nremovedP[i,2]]
            smod += self.getMod(self.gradient_dx[self.nremovedP[i,0],self.nremovedP[i,1],self.nremovedP[i,2]],
                                self.gradient_dy[self.nremovedP[i,0],self.nremovedP[i,1],self.nremovedP[i,2]],
                                self.gradient_dz[self.nremovedP[i,0],self.nremovedP[i,1],self.nremovedP[i,2]])
        if smod <= 0.0:
            return 0.0
        alinhamento = self.getMod(somax,somay,somaz)/smod
        return alinhamento    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void _G2(self):
        self.modDiversity = self._modVariety()
        self.G2 = ((self.totalAssimetric)/float(self.totalVet))*(2.0-self.modDiversity)
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    # This function estimates both asymmetric gradient coeficient (geometric and algebric), with the given tolerances
    cpdef list evaluate(self,double[:,:,:] mat,list moment=["G2"]):
        cdef int x, y
        cdef double minimo, maximo
        
        self.mat = mat
        self.depth = len(self.mat[0,0])
        self.cols = len(self.mat[0])
        self.rows = len(self.mat)
        self.totalVet =self.rows*self.cols*self.depth
        self.setPosition(float(self.rows/2),float(self.cols/2),float(self.depth/2))
        self._setGradients()
        

        cdef numpy.ndarray dists = numpy.array([[[self.getMod(float(x)-self.cx,float(y)-self.cy,float(z)-self.cz) \
                                                  for x in range(self.rows)] for y in range(self.cols)] for z in range(self.depth)])
        
        # removes the symmetry in gradient_asymmetric_dx and gradient_asymmetric_dy:
        self._update_asymmetric_mat(dists.astype(dtype=numpy.int32), self.tol)
        
        #gradient moments:
        retorno = []
        '''        
        if("G4" in moment):
            self._G4()
            retorno.append(self.G4)
        if("G3" in moment):
            self._G3()
            retorno.append(self.G3)
        '''
        if("G2" in moment):
            self._G2()
            retorno.append(self.G2)
        '''        
        if("G1" in moment):
            self._G1(self.mtol)
            retorno.append(self.G1)
        '''
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
        dx = numpy.array([[[0.0 for i in range(w) ] for j in range(h)] for k in range(p)],dtype=numpy.float)
        dy = numpy.array([[[0.0 for i in range(w) ] for j in range(h)] for k in range(p)],dtype=numpy.float)
        dz = numpy.array([[[0.0 for i in range(w) ] for j in range(h)] for k in range(p)],dtype=numpy.float)
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

 
               


