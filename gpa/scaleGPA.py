import matplotlib.pyplot as plt
import sys
from GPA import GPA as ga
import numpy as np
import pandas as pd
import math  
import scipy.interpolate as interpolate

def iterativeScaling(inputMatrix,gn, tol, rad_tol,trim=0,step=1,glist=[]):
   '''
   Input:
      inputMatrix - the matrix to be analysed
      trim - the spacing (must be positive or zero - check to be done)
      step - trim step (must be positive - negative check to be done)
      glist - list containing the level(the matrix diagonal) and GPA result
   Output:
      glist - list containing the level(the matrix diagonal) and GPA result
   Descrition:
      Evaluates GPA iteratively, reducing the matrix size according to the step
      Required at least 4x4 submatrices (it will return the input list otherwise).
   '''
   y, x = inputMatrix.shape
   dminY, dmaxY =  trim, y - trim
   dminX, dmaxX =  trim, x - trim
   level = max(dmaxX-dminX,dmaxY-dminY)

   #  at least 4x4 matrices condition
   if(dmaxX<dminX+3 or dmaxY<dminY+3):
      return glist

   gaObject = ga(inputMatrix[dminY:dmaxY,dminX:dmaxX])
   gaObject.cx, gaObject.cy = len(inputMatrix[0])/2., len(inputMatrix)/2.
   gaObject.evaluate(tol,rad_tol,1.0,[gn])
   if(gn == "G1"):
      glist.append([level,gaObject.G1])
   elif(gn == "G2"):
      glist.append([level,gaObject.G2])
   elif(gn == "G3"):
      glist.append([level,gaObject.G3])
   elif(gn == "G4"):
      glist.append([level,gaObject.G4])
   
   glist = iterativeScaling(inputMatrix,gn, tol, rad_tol,trim+step,int(round(1.3*step)),glist)
   
   return glist

def scaleGPA(inputMatrix,gn, tol, rad_tol):
   level, gn = np.array(iterativeScaling(inputMatrix,gn, tol, rad_tol)).T
   argsort = np.argsort(level)
   level, gn = level[argsort], gn[argsort]
   #fit = interpolate.PchipInterpolator(level,gn)
   fit = np.polyfit(np.log(level),gn,1)
   return [level,gn],fit,fit[0]
   
def singleFile(fileName,gn, tol, rad_tol):
   print("Reading "+fileName)
   inputMatrix = pd.read_csv(fileName, sep = "\s+|,| ",engine="python").as_matrix()
   inputMatrix=inputMatrix.astype(np.float32)
   raw,fit,res = scaleGPA(inputMatrix,gn, tol, rad_tol)   

   plt.figure(figsize=(12.8,4.8))
   plt.subplot(1,2,1)
   plt.loglog(raw[0],raw[1],'.b',label="Local sample")
   xs = np.arange(min(raw[0]),max(raw[0]),(max(raw[0])-min(raw[0]))/1000.0)
   ys = np.polyval(fit,np.log(xs))
   print(fit[0])
   plt.plot(xs,ys,'-k',label="Fitted")
   #plt.axvline(xs[np.argmax(ys)])
   
   plt.xlabel("Matrix diagonal (l)")
   plt.ylabel(gn)
   
   plt.title(r"$\int G_"+gn[1]+"\; /\; L \enspace dl$ = "+str(res))

   plt.legend()
   plt.subplot(1,2,2)
   plt.imshow(inputMatrix)
   plt.tight_layout()
   plt.show()

   

def multipleFiles(fileName,gn, tol, rad_tol):
   files = [line.rstrip() for line in open(fileName)]
   save = []
   for f in files:
      inputMatrix = pd.read_csv(f, sep = "\s+|,| ",engine="python").as_matrix()
      inputMatrix = inputMatrix.astype(np.float32)
      _, _, alpha = scaleGPA(inputMatrix,gn, tol, rad_tol)
      print(f,alpha)
      save.append([f,gn,alpha])

   np.savetxt("scaleResult.csv",save,header="file,Gn,alpha",fmt="%s",comments="")
   print("Result save at scaleResult.cs")
   


def isSingleFile(fileName):
   ofile = pd.read_csv(fileName, sep = "\s+|,| ",engine="python")
   if(ofile.select_dtypes([np.number]).empty):
      return False
   return True

def printError():
   print('================================')
   print('Syntax:')
   print('python runGPA.py Gn file tol rad_tol')
   print('')
   print('Gn - Gradient Moment (i.e. G1)')
   print('file - Raw csv matrix data (no header), or list of csv data')
   print('tol - modulus tolerance (0.0 - 1.0)')
   print('rad_tol - angular tolerance (rad)')
   print('================================')
   print('')
   exit() 



if __name__ == "__main__":
   if('-h' in sys.argv) or ('--help' in sys.argv):
        printError()
   if(len(sys.argv) != 5):
        printError()   
   gn, fileName, tol, rad_tol = sys.argv[1], sys.argv[2],float(sys.argv[3]),float(sys.argv[4])
   if isSingleFile(fileName):  
        singleFile(fileName,gn, tol, rad_tol)
   else:
        multipleFiles(fileName,gn, tol, rad_tol)    

     

    
