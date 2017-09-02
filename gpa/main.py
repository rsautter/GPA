import matplotlib.pyplot as plt
import sys
from GPA import GPA as ga
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
import timeit

def plot3D(g):
    fig = plt.figure()
    ax = plt.subplot(1,2,1, projection='3d')
    X = np.arange(0, len(g.mat), 1)
    Y = np.arange(0, len(g.mat[0]), 1)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, g.mat,rstride=1, cstride=1,cmap=plt.get_cmap('jet'),linewidth=0, antialiased=False, shade=False)

    #plt.imshow(g.mat, cmap=plt.get_cmap('gray'), origin='lower')

    if g.n_edges > 0:
        fig.add_subplot(1,2,2)
        plt.xlim(0,len(g.mat[0]))
        plt.ylim(0,len(g.mat))
        plt.triplot(g.triangulation_points[:,0], g.triangulation_points[:,1], g.triangles.simplices.copy())
        plt.title("Triangulation")
    
def invertY(m):
    mat = np.array([mat2[:] for mat2 in m])
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            mat[len(mat)-1-j,i] = m[j,i]
    return mat
    #
def plot_matrix2(g):
    plt.suptitle("Ga: "+str(max(g.G2,g.G1)),fontsize=18)
    sbplt = plt.subplot(131)
    plt.title("Original Image")
    plt.contour(g.mat, cmap=plt.get_cmap('gray'), origin='lower')

    # plotting the asymmetric gradient field
    plt.subplot(132)
    plt.quiver(g.gradient_asymmetric_dx,g.gradient_asymmetric_dy, scale =1.0/0.01)
    plt.title("Asymmetric Gradient Field")
    #plt.title("C",fontsize=20)

    # plotting the triangulation
    if g.n_edges > 0:
        plt.subplot(133)
        #plt.xlim(0,len(g.mat[0]))
        #plt.ylim(0,len(g.mat))
        plt.triplot(g.triangulation_points[:,0], g.triangulation_points[:,1], g.triangles.simplices.copy())
        plt.title("Triangulation")
        #plt.title("D",fontsize=20)

    plt.show()

def plot_matrix(g):
    plt.suptitle("Ga: "+str(g.Ga),fontsize=18)
    plt.subplot(221)
    #plt.title("A",fontsize=20)
    plt.title("Original Image")
    plt.contour(invertY(g.mat), cmap=plt.get_cmap('gray'), origin='lower')

    #plt.gca().invert_yaxis()

    # plotting the asymmetric gradient field
    plt.subplot(223)
    plt.quiver(g.gradient_asymmetric_dy,g.gradient_asymmetric_dx, scale =1.0/0.06)
    print(np.array(g.gradient_dy))
    plt.xlim(-1,len(g.mat[0])+1)
    plt.ylim(-1,len(g.mat)+1)
    plt.title("Asymmetric Gradient Field")
    #plt.title("C",fontsize=20)

    # plotting the gradient field
    plt.subplot(222)
    plt.quiver(g.gradient_dy,g.gradient_dx, scale =1.0/0.06)
    plt.title("Gradient Field")
    plt.xlim(-1.,len(g.gradient_asymmetric_dy)+1)
    plt.ylim(-1.,len(g.gradient_asymmetric_dy)+1)
    #plt.title("B",fontsize=20)

    # plotting the triangulation
    if g.n_edges > 0:
        plt.subplot(224)
        #plt.xlim(-1,len(g.mat[0])+1)
        #plt.ylim(-1,len(g.mat)+1)
        plt.triplot(g.triangulation_points[:,0], g.triangulation_points[:,1], g.triangles.simplices.copy())
        plt.title("Triangulation")
        #plt.title("D",fontsize=20)

    plt.show()

def printError():
        print('================================')
        print('Syntax:')
        print('python main.py Gn filename tol rad_tol')
        print('python main.py Gn -l filelist tol rad_tol output')
        print('================================')
        print('')
        exit() 

if __name__ == "__main__":
    if('-h' in sys.argv) or ('--help' in sys.argv):
        printError()
    if(sys.argv[2] == "-l") and (len(sys.argv) != 7):
        printError()
    if(sys.argv[2] != "-l") and (len(sys.argv) != 5):
        printError()        
    if not("-l" in sys.argv):  
        fileName = sys.argv[2]
        tol = float(sys.argv[3])
        rad_tol = float(sys.argv[4])

        print("Reading "+fileName)
        inputMatrix = np.loadtxt(fileName)
        inputMatrix=inputMatrix.astype(np.float32)
        gaObject = ga(inputMatrix)
        gaObject.cx, gaObject.cy = len(inputMatrix[0])/2., len(inputMatrix)/2.
        gaObject.evaluate(tol,rad_tol,0.5,[sys.argv[1]])
        if(sys.argv[1] == "G1"):
            print("Nc,Nv,G1 ",gaObject.n_edges,gaObject.n_points,gaObject.G1)
        if(sys.argv[1] == "G2"):
            print("G2 ",gaObject.G2)       
        plot_matrix2(gaObject)
    else:
        files = [line.rstrip() for line in open(sys.argv[3])]
        tol = float(sys.argv[4])
        rad_tol = float(sys.argv[5])
        save = []

        for f in files:
            inputMatrix = np.loadtxt(f)
            inputMatrix=inputMatrix.astype(np.float32)
            gaObject = ga(inputMatrix)
            gaObject.cx, gaObject.cy = len(inputMatrix[0])/2., len(inputMatrix)/2.
            gaObject.evaluate(tol,rad_tol,float(0.5),[sys.argv[1]])
            if(sys.argv[1] == "G1"):
                print(f+" - G1 -",gaObject.G1)
                newline = [f,gaObject.G1,gaObject.n_edges,gaObject.n_points]
                save.append(newline)
                np.savetxt(sys.argv[6], np.array(save), fmt="%s", header="Ga,Nc,Nv", delimiter=',')
            else:
                print(f+" - G2 -",gaObject.G2)
                newline = [f,gaObject.G2,float(gaObject.totalAssimetric)/float(gaObject.totalVet),gaObject.phaseDiversity]
                save.append(newline)
                np.savetxt(sys.argv[6], np.array(save), fmt="%s", header="G2,Na,Diversity", delimiter=',')
        
       
    plt.show()
