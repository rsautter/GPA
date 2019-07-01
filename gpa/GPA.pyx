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
    cdef public double[:,:] mat,gradient_dx,gradient_dy,gradient_asymmetric_dy,gradient_asymmetric_dx
    cdef public double cx, cy
    cdef public int rows, cols
    
    cdef double[:,:] phases, mods
    cdef int[:,:] removedP, nremovedP
    cdef public object triangulation_points,triangles
    cdef public int totalAssimetric, totalVet
    cdef public double phaseDiversity,modDiversity, maxGrad,mtol,ftol
    cdef public object boundaryType,ignoreBoundary,cvet

    cdef public int n_edges, n_points
    cdef public double G1, G2, G3
    cdef public object G4

    #@profile
    def __cinit__(self, double mtol,double ftol):
        # setting matrix,and calculating the gradient field
        self.boundaryType = "reflexive"
        self.ignoreBoundary = False
        self.mtol = mtol
        self.ftol = ftol

   
        # percentual Ga proprieties
        self.removedP = numpy.array([[]],dtype=numpy.int32)
        self.nremovedP = numpy.array([[]],dtype=numpy.int32)
        self.triangulation_points = []
        self.phaseDiversity = 0.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef void setPosition(self, double cx, double cy):
        self.cx = cx
        self.cy = cy

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void _setGradients(self):
        cdef int w, h,i,j
        cdef double[:,:] gx, gy
        
        gy, gx = self.gradient(self.mat)
        w, h = len(gx[0]),len(gx)
        
       
        self.maxGrad = -1.0
        for i in range(w):
            for j in range(h):
                if(self.maxGrad<0.0) or (sqrt(pow(gy[j, i],2.0)+pow(gx[j, i],2.0))>self.maxGrad):
                    self.maxGrad = sqrt(pow(gy[j, i],2.0)+pow(gx[j, i],2.0))
        
        #initialization
        self.gradient_dx=numpy.array([[gx[j, i] for i in range(w) ] for j in range(h)],dtype=numpy.float)
        self.gradient_dy=numpy.array([[gy[j, i] for i in range(w) ] for j in range(h)],dtype=numpy.float)

        # copying gradient field to asymmetric gradient field
        self.gradient_asymmetric_dx = numpy.array([[gx[j, i] for i in range(w) ] for j in range(h)],dtype=numpy.float)
        self.gradient_asymmetric_dy = numpy.array([[gy[j, i] for i in range(w) ] for j in range(h)],dtype=numpy.float)
        
        
        # calculating the phase and mod of each vector
        self.phases = numpy.array([[atan2(gy[j, i],gx[j, i]) if atan2(gy[j, i],gx[j, i])>0 else atan2(gy[j, i],gx[j, i])+2.0*M_PI
                                     for i in range(w) ] for j in range(h)],dtype=numpy.float)
        self.mods = numpy.array([[sqrt(pow(gy[j, i],2.0)+pow(gx[j, i],2.0)) for i in range(w) ] for j in range(h)],dtype=numpy.float)
   
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _min(self,double a, double b):
        if a < b:
            return a
        else:
            return b

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _angleDifference(self, double a1,double a2):
        return self._min(fabs(a1-a2), fabs(fabs(a1-a2)-2.0*M_PI))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void _update_asymmetric_mat(self,double[:] index_dist,double[:,:] dists,double mtol,double ftol,double ptol):
        cdef int ind, lx, px, py, px2, py2, i, j
        cdef int[:] x, y

        # distances loop
        for ind in range(0, len(index_dist)):
            x2, y2 =[], []
            for py in range(self.rows):
                for px in range(self.cols):
                    if (fabs(dists[py, px]-index_dist[ind]) <= fabs(ptol)):
                        x2.append(px)
                        y2.append(py)
            x, y =numpy.array(x2,dtype=numpy.int32), numpy.array(y2,dtype=numpy.int32)
            lx = len(x)

            # compare each point in the same distance
            for i in range(lx):
                px, py = x[i], y[i]
                if (self.mods[py, px]/self.maxGrad <= mtol):
                    self.gradient_asymmetric_dx[py, px] = 0.0
                    self.gradient_asymmetric_dy[py, px] = 0.0
                if(self.ignoreBoundary):
                    if(px<2 or px>self.rows-2 or py<2 or py>self.cols-2):
                        self.gradient_asymmetric_dx[py, px] = 0.0
                        self.gradient_asymmetric_dy[py, px] = 0.0
                        continue
                for j in range(lx):
                    px2, py2 = x[j], y[j]
                    if (fabs((self.mods[py, px]- self.mods[py2, px2])/self.maxGrad )<= mtol):
                        if (fabs(self._angleDifference(self.phases[py, px], self.phases[py2, px2])-M_PI)  <= ftol) :
                            if not((self.ignoreBoundary) and (px2<2 or px2>self.rows-2 or py2<2 or py2>self.cols-2)):
                                self.gradient_asymmetric_dx[py, px] = 0.0
                                self.gradient_asymmetric_dy[py, px] = 0.0
                                self.gradient_asymmetric_dx[py2, px2] = 0.0
                                self.gradient_asymmetric_dy[py2, px2] = 0.0
                                break

        self.totalVet = 0
        self.totalAssimetric = 0
        nremovedP = []
        removedP = []
        for j in range(self.rows):
            for i in range(self.cols):
                if (sqrt(pow(self.gradient_asymmetric_dy[j,i],2.0)+pow(self.gradient_asymmetric_dx[j,i],2.0)) <= mtol):
                    removedP.append([j,i])
                    self.totalVet = self.totalVet+1
                else:
                    nremovedP.append([j,i])
                    self.totalVet = self.totalVet+1
                    self.totalAssimetric = self.totalAssimetric+1
        if(self.ignoreBoundary):
            self.totalVet = self.rows*self.cols-2*self.rows-2*self.cols+4
        else:
            self.totalVet = self.rows*self.cols
        if(len(nremovedP)>0):
            self.nremovedP = numpy.array(nremovedP,dtype=numpy.int32)
        if(len(removedP)>0):
            self.removedP = numpy.array(removedP,dtype=numpy.int32)	

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double distAngle(self,double a1,double a2):
        return (cos(a1)*cos(a2)+sin(a1)*sin(a2)+1.0)/2.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double sumAngle(self,double a1,double a2):
        turns = floor((a1+a2)/(2.0*M_PI)) 
        return a1+a2-turns*2.0*M_PI

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _modVariety(self):
        cdef int i
        cdef double somax,somay, phase, alinhamento, mod, smod
        somax = 0.0
        somay = 0.0
        smod = 0.0
        if(self.totalAssimetric<1):
            return 0.0
        for i in range(self.totalAssimetric):
            phase = self.phases[self.nremovedP[i,0],self.nremovedP[i,1]]
            mod = self.mods[self.nremovedP[i,0],self.nremovedP[i,1]]
            somax += self.gradient_dx[self.nremovedP[i,0],self.nremovedP[i,1]]
            somay += self.gradient_dy[self.nremovedP[i,0],self.nremovedP[i,1]]
            smod += mod
        if smod <= 0.0:
            return 0.0
        alinhamento = sqrt(pow(somax,2.0)+pow(somay,2.0))/smod
        return alinhamento

    # Versao proposta
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double _phaseVariety(self):
        cdef int x1, y1, x2, y2, i, j, div
        cdef double sumPhases, variety
        sumPhases = 0.0
        for i in range(self.totalAssimetric-1):
            x1,y1 = self.nremovedP[i,0],self.nremovedP[i,1]
            for j in range(i+1,self.totalAssimetric):
                x2,y2 = self.nremovedP[j,0],self.nremovedP[j,1]
                sumPhases += self.distAngle(self.phases[x1,y1],self.phases[x2,y2])
        div = ((self.totalAssimetric)*(self.totalAssimetric-1))/2
        variety = sumPhases / float(div)
        return variety
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void _G4(self):
        cdef int w, h 
        cdef double sumMod
        cdef double[:,:] nmods
        w, h = self.cols,self.rows
        if(len(self.nremovedP[:,0])>3):
            self.totalAssimetric = len(self.nremovedP[:,0])
        else:
            self.totalAssimetric = 0
        nmods = self.mods
        sumMod = 0.0 
        for i in self.nremovedP:
            sumMod += self.mods[i[0],i[1]]
        for i in range(len(nmods)):
            for j in range(len(nmods[i])):
                nmods[i,j] = nmods[i,j]/sumMod
        self.cvet = numpy.array([-nmods[i[0],i[1]]*numpy.log(nmods[i[0],i[1]])+1.0j*nmods[i[0],i[1]]*self.phases[i[0],i[1]]
                                 if nmods[i[0],i[1]] > 0.0 else
                                 1.0j*nmods[i[0],i[1]]*self.phases[i[0],i[1]]
                    for i in self.nremovedP],dtype=numpy.complex64)
        self.G4 = numpy.sum(self.cvet)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void _G3(self):
        if(len(self.nremovedP[:,0])>3):
            self.totalAssimetric = len(self.nremovedP[:,0])
        else:
            self.totalAssimetric = 0
        self.phaseDiversity = self._phaseVariety()
        self.G3 = ((self.totalAssimetric)/float(self.totalVet))*(2.0-self.phaseDiversity)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void _G2(self):
        if(len(self.nremovedP[:,0])>3):
            self.totalAssimetric = len(self.nremovedP[:,0])
        else:
            self.totalAssimetric = 0
        self.modDiversity = self._modVariety()
        self.G2 = ((self.totalAssimetric)/float(self.totalVet))*(2.0-self.modDiversity)
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    # This function estimates both asymmetric gradient coeficient (geometric and algebric), with the given tolerances
    cpdef list evaluate(self,double[:,:] mat,list moment=["G2"]):
        cdef int[:] i
        cdef int x, y
        cdef double minimo, maximo
        
        self.mat = mat
        self.cols = len(self.mat[0])
        self.rows = len(self.mat)
        self.totalVet =self.rows*self.cols
        self.setPosition(float(self.rows/2),float(self.cols/2))
        self._setGradients()
        

        cdef numpy.ndarray dists = numpy.array([[sqrt(pow(float(x)-self.cx, 2.0)+pow(float(y)-self.cy, 2.0)) \
                                                  for x in range(self.cols)] for y in range(self.rows)])
        
        minimo, maximo = numpy.min(dists),numpy.max(dists)
        sequence = numpy.arange(minimo,maximo,0.705).astype(dtype=numpy.float)
        cdef numpy.ndarray uniq = numpy.array([minimo for minimo in  sequence])
        
        # removes the symmetry in gradient_asymmetric_dx and gradient_asymmetric_dy:
        self._update_asymmetric_mat(uniq.astype(dtype=numpy.float), dists.astype(dtype=numpy.float), self.mtol, self.ftol, numpy.float(1.41))
        
        #gradient moments:
        retorno = []
        if("G4" in moment):
            self._G4()
            retorno.append(self.G4)
        if("G3" in moment):
            self._G3()
            retorno.append(self.G3)
        if("G2" in moment):
            self._G2()
            retorno.append(self.G2)
        if("G1" in moment):
            self._G1(self.mtol)
            retorno.append(self.G1)
        return retorno

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef bool _wasRemoved(self, int j,int i):
        cdef int rp
        for rp in range(self.totalVet - self.totalAssimetric):
            if(self.removedP[rp,0] == j) and(self.removedP[rp,1] == i):
                return True
        return False 
   
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef tuple gradient(self,double[:,:] mat):
        cdef double[:,:] dx, dy
        cdef double divx, divy
        cdef int i, j,w,h,i1,j1,i2,j2
        w, h = len(mat[0]),len(mat)
        dx = numpy.array([[0.0 for i in range(w) ] for j in range(h)],dtype=numpy.float)
        dy = numpy.array([[0.0 for i in range(w) ] for j in range(h)],dtype=numpy.float)
        for i in range(w):
           for j in range(h):
              if(self.boundaryType == "periodic"):
                 divx,divy = 2.0, 2.0
                 i1 = (i+1)%len(mat[j])
                 j1 = (j+1)%len(mat)
                 j2 = (j-1) if j>0 else len(mat)-1
                 i2 = (i-1) if i>0 else len(mat[j])-1
              elif(self.boundaryType == "reflexive"):
                 divy =  2.0 if (j<len(mat)-1 and j>0) else 1.0
                 divx =  2.0 if (i<len(mat[j])-1 and i>0) else 1.0
                 i1 = (i+1) if i<len(mat[j])-1 else i
                 j1 = (j+1) if j<len(mat)-1 else j
                 i2 = (i-1) if i>0 else i
                 j2 = (j-1) if j>0 else j
              else:
                 divx,divy = 2.0, 2.0
                 i1 = (i+1)%len(mat[j])
                 j1 = (j+1)%len(mat)
                 j2 = (j-1) if j>0 else len(mat)-1
                 i2 = (i-1) if i>0 else len(mat[j])-1
              dy[j, i] = (mat[j1, i] - mat[j2, i])/divy
              dx[j, i] = (mat[j, i1] - mat[j, i2])/divx
        return dy,dx

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    def _G1(self,double tol):
        cdef int w, h, i, j
        cdef double mod

        for i in range(self.rows):
            for j in range(self.cols):
                mod = (self.gradient_asymmetric_dx[i, j]**2+self.gradient_asymmetric_dy[i, j]**2)**0.5
                if mod > tol:
                    self.triangulation_points.append([j+0.5*self.gradient_asymmetric_dx[i, j], i+0.5*self.gradient_asymmetric_dy[i, j]])
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
        return self.G1

                 
                 
 
               


