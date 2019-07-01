import matplotlib.pyplot as plt
import sys
from GPA import GPA as ga
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
import timeit
from matplotlib.colors import LightSource
from matplotlib import cm
import pandas as pd

def plot_matrix2(g):
    plt.figure(figsize=(9,3))
    azimuth = 137
    altitude = 40
    mat = g.mat
    plt.axis('off')
    ax = plt.subplot(131, projection='3d')  
    light = LightSource(90, 45)
    green = np.array([0,1.0,0])
    X, Y = np.meshgrid([m for m in range(len(mat[0]))], [n for n in range(len(mat))])
    z = np.array(mat)
    ax.view_init(altitude, azimuth)
    illuminated_surface = light.shade(z, cmap=cm.coolwarm)
    rgb = np.ones((z.shape[0], z.shape[1], 3))
    green_surface = light.shade_rgb(rgb * green, z)
    ax.plot_surface(X, Y, g.mat,rstride=1, cstride=1, linewidth=0, antialiased=True, facecolors=illuminated_surface)
    ax.set_zticks([])
    ax.grid(False)
    sbplt = plt.subplot(132)
    plt.contour(g.mat, cmap=plt.get_cmap('gray'), origin='lower')
    # plotting the asymmetric gradient field
    plt.subplot(133)
    plt.quiver(g.gradient_asymmetric_dx,g.gradient_asymmetric_dy,scale=g.rows*g.cols/2)
    plt.tight_layout()
    plt.show()

def singleFile(fileName, tol, rad_tol):
   print("Reading "+fileName)
   inputMatrix = pd.read_csv(fileName, sep = "\s+|,| ",engine="python").values
   inputMatrix=inputMatrix.astype(np.float)
   gaObject = ga(tol,rad_tol)
   gaObject.cx, gaObject.cy = len(inputMatrix[0])/2., len(inputMatrix)/2.
   gaObject.evaluate(inputMatrix,[sys.argv[1]])
   if(sys.argv[1] == "G1"):
      print("G1: "+str(gaObject.G1))
      print("Nc: "+str(gaObject.n_edges))
      print("Nv: "+str(gaObject.n_points))
   if(sys.argv[1] == "G2"):
      print("G2 "+str(gaObject.G2)) 
      print("Number of Vectors: "+str(gaObject.totalVet)) 
      print("Asymmetric: "+str(gaObject.totalAssimetric))
      print("Diversity: "+str(gaObject.modDiversity)) 
   if(sys.argv[1] == "G3"):
      print("G3 ",gaObject.G3) 
      print("Number of Vectors: "+str(gaObject.totalVet))  
      print("Asymmetric: "+str(gaObject.totalAssimetric))
      print("Diversity: "+str(gaObject.phaseDiversity)) 
   if(sys.argv[1] == "G4"):
      print("G4 "+str(gaObject.G4)) 
      print("Number of Vectors:  "+str(gaObject.totalVet)) 
      print("Asymmetric: "+str(gaObject.totalAssimetric))
   plot_matrix2(gaObject)
   plt.show()

def multipleFiles(filename,tol, rad_tol):
   files = [line.rstrip() for line in open(filename)]
   save = []
   header = ""

   for f in files:
      inputMatrix = pd.read_csv(f, sep = "\s+|,| ",engine="python").as_matrix()
      inputMatrix=inputMatrix.astype(np.float)
      gaObject = ga(tol,rad_tol)
      gaObject.cx, gaObject.cy = len(inputMatrix[0])/2., len(inputMatrix)/2.
      gaObject.evaluate(inputMatrix,[sys.argv[1]])
      if(sys.argv[1] == "G1"):
         print(f+" - G1 -",gaObject.G1)
         newline = [f,gaObject.G1,gaObject.n_edges,gaObject.n_points,gaObject.t1,gaObject.t2,gaObject.t3]
         save.append(newline)
         header="G1,Nc,Nv,t1,t2,t3"
      elif(sys.argv[1] == "G2"):
         print(f+" - G2 -",gaObject.G2)
         newline = [f,gaObject.G2,float(gaObject.totalAssimetric)/          float(gaObject.totalVet),gaObject.modDiversity,gaObject.t1,gaObject.t2,gaObject.t3]
         save.append(newline)
         header = "File,G2,Va,Md,t1,t2,t3"
      elif(sys.argv[1] == "G3"):
         print(f+" - G3 -",gaObject.G3)
         newline = [f,gaObject.G3,float(gaObject.totalAssimetric)/float(gaObject.totalVet),gaObject.phaseDiversity,gaObject.t1,gaObject.t2,gaObject.t3]
         save.append(newline)
         header = "File,G3,Va,Fd,t1,t2,t3"
      elif(sys.argv[1] == "G4"):
         print(f+" - G4 -",gaObject.G4)
         header = "File,G4,Va,t1,t2,t3"
         newline = [f,gaObject.G4,float(gaObject.totalAssimetric)/float(gaObject.totalVet),gaObject.t1,gaObject.t2,gaObject.t3]
         save.append(newline)
         
   np.savetxt("result.csv", np.array(save), fmt="%s", header=header, delimiter=',',comments='')
   print("Saved at result.csv")

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
    fileName, tol, rad_tol = sys.argv[2],float(sys.argv[3]),float(sys.argv[4])
    if isSingleFile(fileName):  
        singleFile(fileName, tol, rad_tol)
    else:
        multipleFiles(fileName, tol, rad_tol)    

    
