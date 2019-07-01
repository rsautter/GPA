import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
from GPA import GPA as ga
import numpy as np
import pandas as pd
import math  
import scipy.optimize
import scipy.interpolate as interpolate
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import signal
import time
import hilbert

def butterworthBandPass(b,order,band):
   '''
   Input:
      b - frequency distance (must be normalized: 0.0-1.0)
      order - butterworth filter order
      band - minimum and maximum band (must be normalized: 0.0-1.0)
   Output:
      Frequency weight
   '''
   window = float(max(band)-min(band))
   bmin = float(min(band)) + window/2.0
   distance = lambda b1, b2: pow(b1,2.0)-pow(b2,2.0)
   return (1.0-1.0/(1.0+ pow(b*window/distance(b,bmin), 2.0*order)) if distance(b,bmin) else 1.0)

def butterworthLowPass(d,d0,n):
    return 1.0/(1.0+pow(d/d0,2.0*n))

def butterBP2D(w,h,order,band):
   '''
   Description:
      Returns the bandpass weight matrix, with size w,h
   Input:
      w - matrix width
      h - matrix height
      order - butterworth filter order
      band - minimum and maximum band (must be normalized: 0.0-1.0)
   Output:
      Butterworth filter image
   '''
   maxdist = math.sqrt(pow(w,2.0)+pow(h,2.0))
   dist = lambda x, y: math.sqrt(pow(float(x)-float(w)/2.0,2.0)+pow(float(y)-float(h)/2.0,2.0))/maxdist
   nBand= np.array(band).astype(np.float32)
   output = [[butterworthBandPass(dist(i,j),order,nBand) for i in range(w)] for j in range(h)]
   return np.array(output)

def filterFreq(freq,order,band):
   '''
   Description:
      Given a complex frequency matrix, it build and applies the weights 
   Input:
      freq - 2D frequencies (already shifted)
      order - butterworth filter order
      band - minimum and maximum band (normalized: 0.0-1.0)
   output:
      filtered image 
   '''
   h, w = freq.shape
   filt = butterBP2D(w,h,order,band)
   output = [[freq[j,i]*filt[j,i] for i in range(w)] for j in range(h)]
   return np.array(output)

def crop_center(img,cropx,cropy):
    '''
    Description:
      Removes the image boundaries of size cropx and cropy
    Input:
      img - input matrix
      cropx, cropy - x and y proportion to cut (0.0-1.0) 
    Output:
      Cropped image 
    '''
    y,x = img.shape
    cx,cy = int(x*cropx), int(y*cropy)
    return img[cy:y-cy,cx:x-cx] 

def fragGPA(inputMatrix,gn,tol,rad_tol,bandwidth=0.4,nbands=20,hbands=5):
   '''
   Description:
      Measures the GPA, for a set of filtered images (made with butterworth).
      As increases the band, crops the boundaries (avoiding patterns affected by boundaries)
   Input:
      inputMatrix - the matrix to be analyzed
      gn - Gradient Moment (i.e. G1)
      tol - GPA modulus tolerance (0.0-1.0)
      rad_tol - GPA phase tolerance (0-2*pi)
      bandwidth - 0.0-1.0
      nbands - number of bands
      hbands - hilbert curve interpolator size (total of bands = 2^(2*hbands))
   Output:
      glist - list of GPA results
      avgFreq - list of the band average
      imgList - list of filtered images
      interp - Scipy interpolator, for the GPA sequence 
      layer - GPA frequency series transformed in matrix, via PCHIP interpolation and Hilbert curves 
      gssa - GPA of layer (it uses flexible tolerance based on hbands)
   '''
   glist = []
   avgFreq = []
   imgList = []
   freq = np.fft.fft2(inputMatrix)
   sfreq = np.fft.fftshift(freq)
   seq = [[0.5*float(i)/float(nbands)-bandwidth/2.0,0.5*float(i)/float(nbands)+bandwidth/2.0] for i in range(1,nbands+1)]
   
   # Frequencies loop (to build the series)
   for fb in seq:
      avg =np.average(fb)
      fsfreq = filterFreq(sfreq,3,fb)
      filtered = np.real(np.fft.ifft2(np.fft.ifftshift(fsfreq))).astype(np.float)
      filtered = crop_center(filtered,avg/4.0,avg/4.0)
      imgList.append(filtered)
      avgFreq.append(np.average(fb))
      gaObject = ga(tol,rad_tol)
      gaObject.evaluate(filtered,[gn])
      if(gn == "G1"):
         glist.append(gaObject.G1)
      elif(gn == "G2"):
         glist.append(gaObject.G2)
      elif(gn == "G3"):
         glist.append(gaObject.G3)
      elif(gn == "G4"):
         glist.append(gaObject.G4)

   # Starts transforming the series in a matrix
   dim = pow(2,hbands)
   interp = interpolate.PchipInterpolator(avgFreq,glist)
   xs = np.linspace(min(avgFreq),max(avgFreq),dim*dim)
   ys = interp(xs)
   hcurve = hilbert.HilbertCurve(hbands,2)
   layer = np.array([[ys[hcurve.distance_from_coordinates([i,j])] for i in range(dim)] for j in range(dim)]).astype(np.float)

   # Now measures the gradient moment   
   gaObject = ga(1.0/float(dim*dim),1.0/float(dim*dim))
   gaObject.evaluate(layer,[gn])
   if(gn == "G1"):
      gssa = gaObject.G1
   elif(gn == "G2"):
      gssa = gaObject.G2
   elif(gn == "G3"):
      gssa = gaObject.G3
   elif(gn == "G4"):
      gssa = gaObject.G4
   return np.array(glist),np.array(avgFreq),np.array(imgList),interp,layer,gssa
      


def singleFile(fileName, gn, tol, rad_tol):
   print("Reading "+fileName)
   inputMatrix = pd.read_csv(fileName, sep = "\s+|,| ",engine="python").values
   inputMatrix = inputMatrix.astype(np.float)
   glist,avgFreq,imgList,interp,affMat,gssa = fragGPA(inputMatrix,gn,tol,rad_tol)
   print("Gradient symmetry self-affinity: "+str(round(gssa,3)))
   
   fig = plt.figure(figsize=(10,5))
   gs = gridspec.GridSpec(2,4)
   
   mainAxes = plt.subplot(gs[:,0:2])   
   xs = np.linspace(min(avgFreq),max(avgFreq),1000)
   ys = interp(xs)
   plt.title("Gradient symmetry self-affinity: "+str(round(gssa,3)))
   plt.plot(np.array(avgFreq)*100.0,glist,'.')
   plt.plot(np.array(xs)*100.0,ys,'-k')
   plt.xlabel("Average frequency band (%)")
   plt.ylabel(gn)
   
   plt.subplot(gs[0,2])
   plt.axis('off')
   plt.imshow(imgList[0],cmap='jet')
   plt.title("Band %.2f" % (100.0*avgFreq[0]))
   plt.subplot(gs[0,3])
   plt.axis('off')
   plt.imshow(imgList[int(len(imgList)/4)],cmap='jet')
   plt.title("Band %.2f "% (100.0*avgFreq[int(len(imgList)/4)]) )
   plt.subplot(gs[1,2])
   plt.axis('off')
   plt.imshow(imgList[int(3*len(imgList)/4)],cmap='jet')
   plt.title("Band %.2f" % (100.0*avgFreq[int(3*len(imgList)/4)]) )
   plt.subplot(gs[1,3])
   plt.axis('off')
   plt.imshow(imgList[int(len(imgList)-1)],cmap='jet')
   plt.title("Band %.2f" % (100.0*avgFreq[int(len(imgList)-1)]))

   inset_axes(mainAxes,width="20%",height="20%",loc=2)
   plt.imshow(affMat)
   plt.axis('off')

   plt.tight_layout()
   plt.subplots_adjust(wspace=0.125,hspace=0.15)
   
   plt.figure()
   plt.imshow(affMat)
   plt.show()



def multipleFiles(fileName,gn, tol, rad_tol):
   files = [line.rstrip() for line in open(fileName)]
   save = []
   header = "File,Gssa,time"

   for f in files:
      inputMatrix = pd.read_csv(f, sep = "\s+|,| ",engine="python").values
      inputMatrix=inputMatrix.astype(np.float)
      t = time.clock()
      glist,avgFreq,imgList,interp,affMat,gssa = fragGPA(inputMatrix,gn,tol,rad_tol)
      t = time.clock()-t
      print("File: "+f+", Gssa:"+str(gssa)+", time:"+str(t))
      save.append([f,gssa,t])
         
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
   print('python fragGPA.py Gn file tol rad_tol')
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
      singleFile(fileName, gn, tol, rad_tol)
   else:
      multipleFiles(fileName,gn, tol, rad_tol)     

    
