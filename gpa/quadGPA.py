import matplotlib.pyplot as plt
import sys
from GPA import GPA as ga
import numpy as np
import pandas as pd
import math  



def quadSplit(mat):
   '''
   Input:
      mat - the matrix to be splitted
   Output:
      ul - Upper-Left split
      ul - Upper-Right split
      ll - Lower-Left SPlit
      lr - Lower-Right split
   '''
   x, y = mat.shape
   hx, hy = x/2,y/2
   ul = np.array([[mat[i,j] for i in range(hx)] for j in range(hy)]).T.astype(np.float32)
   ll = np.array([[mat[i,j] for i in range(hx,x)] for j in range(hy)]).T.astype(np.float32)
   ur = np.array([[mat[i,j] for i in range(hx)] for j in range(hy,y)]).T.astype(np.float32)
   lr = np.array([[mat[i,j] for i in range(hx,x)] for j in range(hy,y)]).T.astype(np.float32)
   return ul,ur,ll,lr

def recursiveGN(inputMatrix, gn, tol, rad_tol, maxDepth, glist=[],level=0):
   '''
   Input:
      inputMatrix - the matrix
      gn - Gradient moment (i.e. G1)
      tol - GPA modulus tolerance(0.0-1.0)
      rad_tol - GPA radius tolerance (0-2*pi)
      maxDepth - the maximum tree depth
      glist - List of GN in each level - defaul = empty
   Output:
      glist - List of GN in each level 
   Description:
      Recursevly evaluates GN, spliting in 4 segments 
   until the maximal depth is achieved 
   '''

   #Local analysis
   
   gaObject = ga(inputMatrix)
   gaObject.cx, gaObject.cy = len(inputMatrix[0])/2., len(inputMatrix)/2.
   gaObject.evaluate(tol,rad_tol,1.0,[gn])
   mlen = math.sqrt(pow(gaObject.cx,2.0)+pow(gaObject.cy,2.0))
   if(maxDepth<=level):
      return glist
   
   if(gn == "G1"):
      glist.append([level,gaObject.G1])
   elif(gn == "G2"):
      glist.append([level,gaObject.G2])
   elif(gn == "G3"):
      glist.append([level,gaObject.G3])
   elif(gn == "G4"):
      glist.append([level,gaObject.G4])

   #recursion part
   if(maxDepth>level and  inputMatrix.shape[0]>2 and inputMatrix.shape[1]>2):
      ul,ur,ll,lr = quadSplit(inputMatrix)
      glist = recursiveGN(ul,gn,tol,rad_tol,maxDepth,glist=glist,level=level+1)
      glist = recursiveGN(ur,gn,tol,rad_tol,maxDepth,glist=glist,level=level+1)
      glist = recursiveGN(ll,gn,tol,rad_tol,maxDepth,glist=glist,level=level+1)
      glist = recursiveGN(lr,gn,tol,rad_tol,maxDepth,glist=glist,level=level+1)

   return glist

def quadGPA(inputMatrix, gn, tol, rad_tol, levels=3):
   points = np.array(recursiveGN(inputMatrix,gn, tol, rad_tol,levels)).T
   levels = np.array(points[0])
   glist = np.array(points[1])
   ulevels = np.unique(levels)
   avgList = []
   for l in ulevels:
      pts = np.where(levels == l)[0]
      avg = np.average(glist[pts])
      avgList.append(avg)
   alpha,beta,gamma = np.polyfit(ulevels,avgList,2)
   return alpha, beta, gamma, [levels,glist], [ulevels,avgList]

def singleFile(fileName, gn, tol, rad_tol):
   print("Reading "+fileName)
   inputMatrix = pd.read_csv(fileName, sep = "\s+|,| ",engine="python").as_matrix()
   inputMatrix=inputMatrix.astype(np.float32)
   alpha, beta, gamma, raw, avg = quadGPA(inputMatrix,gn, tol, rad_tol,5)
   
   plt.figure(figsize=(12.8,4.8))
   plt.subplot(1,2,1)
   plt.plot(raw[0],raw[1],'.b',label="Local sample")
   plt.plot(avg[0],avg[1],'--r',label="Average per level") 
   xs = np.arange(min(avg[0]),max(avg[0]),(max(avg[0])-min(avg[0]))/1000.0)
   ys = np.polyval([alpha,beta,gamma],xs)
   plt.plot(xs,ys,'-k',label="Fitted")

   plt.xlabel("Tree Depth")
   plt.ylabel(gn)
   plt.title(r"$\alpha _{"+gn+"}$ = "+str(np.round(alpha,3))+"\n"+r"$\beta _{"+gn+"}$ = "+str(np.round(beta,3)))

   plt.legend()
   plt.subplot(1,2,2)
   plt.imshow(inputMatrix)
   plt.tight_layout()
   plt.show()

   

def multipleFiles(fileName, gn, tol, rad_tol):
   files = [line.rstrip() for line in open(fileName)]
   save = []

   for f in files:
      print(f)
      inputMatrix = pd.read_csv(f, sep = "\s+|,| ",engine="python").as_matrix()
      inputMatrix=inputMatrix.astype(np.float32)
      alpha, beta, gamma, _, _ = quadGPA(inputMatrix,gn, tol, rad_tol)
      save.append([f,gn,alpha,beta,gamma])
   np.savetxt("result.csv",save,header="file,Gn,alpha,beta,gamma",fmt="%s",comments="")
   print("Result save at result.csv")
   

def printError():
   print('================================')
   print('Syntax:')
   print('python fragGPA.py Gn file tol rad_tol')
   print('')
   print('Gn - Gradient Moment (i.e. G1)')
   print('file - Raw csv matrix data (no header), or list of csv data')
   print('tol - modulus tolerance (0.0 - 1.0)')
   print('rad_tol - angular tolerance (rad)')
   print('================================')
   print('')
   exit() 

def isSingleFile(fileName):
   ofile = pd.read_csv(fileName, sep = "\s+|,| ",engine="python")
   if(ofile.select_dtypes([np.number]).empty):
      return False
   return True

if __name__ == "__main__":
   if('-h' in sys.argv) or ('--help' in sys.argv):
      printError()
   if(len(sys.argv) != 5):
      printError()     
   gn, fileName, tol, rad_tol = sys.argv[1], sys.argv[2],float(sys.argv[3]),float(sys.argv[4])
   if isSingleFile(fileName):  
      singleFile(fileName, gn, tol, rad_tol)
   else:
      multipleFiles(fileName,gn, tol, rad_tol)     

    
    
