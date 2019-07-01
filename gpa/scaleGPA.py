import matplotlib.pyplot as plt
import sys
from GPA import GPA as ga
import numpy as np
import pandas as pd
import math  
import scipy.interpolate as interpolate
import copy 
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter, LogLocator,FormatStrFormatter


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

   gaObject = ga(tol,rad_tol)
   gaObject.cx, gaObject.cy = len(inputMatrix[0])/2., len(inputMatrix)/2.
   gaObject.evaluate(inputMatrix[dminY:dmaxY,dminX:dmaxX],[gn])
   golist = copy.deepcopy(glist)
   if(gn == "G1"):
      golist.append([level,gaObject.G1])
   elif(gn == "G2"):
      golist.append([level,gaObject.G2])
   elif(gn == "G3"):
      golist.append([level,gaObject.G3])
   elif(gn == "G4"):
      golist.append([level,gaObject.G4])
   
   golist = iterativeScaling(inputMatrix,gn, tol, rad_tol,trim+step,int(round(1.3*step)),golist)
   
   return golist

def scaleGPA(inputMatrix,gn, tol, rad_tol):
   level, gn = np.array(iterativeScaling(inputMatrix,gn, tol, rad_tol)).T
   argsort = np.argsort(level)
   level, gn = level[argsort], gn[argsort]
   #fit = interpolate.PchipInterpolator(level,gn)
   fit = np.polyfit(np.log(level),gn,1)
   
   size = len(level)
   chunkSize = size
   fits = []
   objFunc = []
   selected = []
   chSize = []
   while chunkSize> size/5:
    for chunk in range(0,size,1):
        maxChunk = min(chunk+chunkSize,size-1)
        if(maxChunk-chunk<3):
            continue
        if(chunk+chunkSize>size-1):
            continue
        lfit = np.polyfit(np.log(level[chunk:maxChunk]),gn[chunk:maxChunk],1)
        lobjFunc = lfit[0]*chunkSize*np.log(np.sum((np.polyval(lfit,np.log(level[chunk:maxChunk]))-gn[chunk:maxChunk])**2.0))
        selected.append([level[chunk:maxChunk], gn[chunk:maxChunk]])
        fits.append(lfit)  
        objFunc.append(lobjFunc) 
        chSize.append(chunkSize) 
    chunkSize = chunkSize-1
   best = np.argmax(objFunc)
   return [level,gn], selected[best],fits[best],-fits[best][0]
   
def singleFile(fileName,gn, tol, rad_tol):
   print("Reading "+fileName)
   inputMatrix = pd.read_csv(fileName, sep = "\s+|,| ",engine="python").values
   inputMatrix=inputMatrix.astype(np.float)
   raw, sel,fit,res = scaleGPA(inputMatrix,gn, tol, rad_tol)   

   fig = plt.figure(figsize=(7,3))
   
   gs = gridspec.GridSpec(1, 2,width_ratios=[2,1])
   axes0 = plt.subplot(gs[0])
   axes1 = plt.subplot(gs[1])
   axes0.set_title(r"$\alpha(G_"+gn[1]+") =$ = "+str("%.2f"%res))
   axes0.loglog(raw[0],raw[1],'.b')
   axes0.loglog(sel[0],sel[1],'.r',label=r"Fitting sample")
   xs = np.arange(min(sel[0]),max(sel[0]),(max(sel[0])-min(sel[0]))/1000.0)
   ys = np.polyval(fit,np.log(xs))
   print("a", fit[0])

   axes0.plot(xs,ys,'-k')
   axes0.legend()
   loc = LogLocator(subs=np.linspace(min(sel[0]),max(sel[0]),3,endpoint=True))
   majorFormatter = FormatStrFormatter('%1.0f')
   axes0.xaxis.set_major_locator(loc)
   axes0.xaxis.set_major_formatter(majorFormatter)
   #axes[0].xlabel("Matrix length")
   
   loc = LogLocator(subs=np.linspace(min(raw[1]),max(raw[1]),7,endpoint=False))
   majorFormatter = FormatStrFormatter('%.2f')
   axes0.yaxis.set_major_locator(loc)
   axes0.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
   #axes[0].ylabel(r"$G_2$")
  
   #plt.axvline(xs[np.argmax(ys)])
   
   axes0.set_xlabel("Matrix length (L)")
   axes0.set_ylabel(r"$G_"+str(gn[1])+"$")

   axes1.imshow(inputMatrix,cmap = 'hot')
   plt.tight_layout()
   plt.show()

   

def multipleFiles(fileName,gn, tol, rad_tol):
   files = [line.rstrip() for line in open(fileName)]
   save = []
   alphaList = []
   n = 0
   for f in files:
      inputMatrix = pd.read_csv(f, sep = "\s+|,| ",engine="python").values
      inputMatrix = inputMatrix.astype(np.float)
      _, _, _, alpha = scaleGPA(inputMatrix,gn, tol, rad_tol)
      print(f,alpha,n)
      save.append([f,gn,alpha,n])
      alphaList.append(alpha)
      n = n+1

   np.savetxt("scaleResult.csv",save,header="file,Gn,alpha,n",fmt="%s",delimiter=",",comments="")
   print("Result save at scaleResult.csv")

   plt.plot(alphaList,'k-')
   plt.show()

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

     

    
