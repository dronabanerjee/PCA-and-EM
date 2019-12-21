# -*- coding: utf-8 -*-
"""
File:   hw03.py
"""

"""
====================================================
================ Import Packages ===================
====================================================
"""
import sys

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import skimage.filters as filt
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler

"""
====================================================
================ Define Functions ==================
====================================================
"""

def EM(Y, Ps, p, q):
    
    MaximumNumberOfIterations = 20
    N = Y.shape[0] #number of data points
    mu0 = (Ps*(1-p))/((Ps*(1-p)) + (1-Ps)*(1-q))
    mu1 = (Ps*p)/((Ps*p) + (1-Ps)*q)
    sum = 0
    mean_mu = 0
    for i in range(N):
        if Y[i] == 0:
            sum = sum + mu0
        else:
            sum = sum + mu1

    mean_mu = sum/N

    print("\nTheta(0) values are:")
    print("Pie(0)=", Ps)
    print("p(0)=", p)
    print("q(0)=", q)
    print("\nFor all values of observable data = 0:")
    print("mu(1)=", mu0)
    print("For all values of observable data = 1:")
    print("mu(1)=", mu1, "\n")
    print("Mean mu(1)=", mean_mu, "\n")

    """
    #Initialize Parameters
    Means = X[rp[0:NumberOfComponents],:]
    Sigs = np.zeros((d,d,NumberOfComponents))
    Ps = np.zeros((NumberOfComponents,))
    pZ_X = np.zeros((N,NumberOfComponents))
    """

    NumberIterations = 1
    while NumberIterations <= MaximumNumberOfIterations:
        
        Ps = 0
        p = 0
        q = 0
        print("Iteration number: ", NumberIterations)

        #Ps
        for j in range(N):

            if Y[j] == 0:
                Ps = Ps + mu0
            else:
                Ps = Ps + mu1

        Ps = Ps/10
        print("Pie(",NumberIterations,")=",Ps)

        #p
        p_num = 0
        p_den = 0
        for j in range(N):

            if Y[j] == 1:
                p_num = p_num + mu1
                p_den = p_den + mu1
            else:
                p_den = p_den + mu0

        p = p_num/p_den
        print("p(",NumberIterations,")=",p)

        #q
        q_num = 0
        q_den = 0
        for j in range(N):

            if Y[j] == 1:
                q_num = q_num + (1-mu1)
                q_den = q_den + (1-mu1)
            else:
                q_den = q_den + (1-mu0)

        q = q_num/q_den
        print("q(",NumberIterations,")=",q)

        mu0 = (Ps*(1-p))/((Ps*(1-p)) + (1-Ps)*(1-q))
        mu1 = (Ps*p)/((Ps*p) + (1-Ps)*q)
        sum = 0
        mean_mu = 0
        for i in range(N):
            if Y[i] == 0:
                sum = sum + mu0
            else:
                sum = sum + mu1

        mean_mu = sum/N

        print("\nFor all values of observable data = 0:")
        print("mu(",NumberIterations+1,")=",mu0)
        print("For all values of observable data = 1:")
        print("mu(",NumberIterations+1,")=",mu1,"\n")
        print("Mean mu(",NumberIterations+1,")=",mean_mu,"\n")

        NumberIterations = NumberIterations + 1
    





def process_image(in_fname,debug=False):

    # load image
    x_in = np.array(Image.open(in_fname))

    # convert to grayscale
    x_gray = 1.0-rgb2gray(x_in)

    if debug:
        plt.figure(1)
        plt.imshow(x_gray)
        plt.title('original grayscale image')
        plt.show()

    # threshold to convert to binary
    thresh = filt.threshold_minimum(x_gray)
    fg = x_gray > thresh

    if debug:
        plt.figure(2)
        plt.imshow(fg)
        plt.title('binarized image')
        plt.show()

    # find bounds
    nz_r,nz_c = fg.nonzero()
    n_r,n_c = fg.shape
    l,r = max(0,min(nz_c)-1),min(n_c-1,max(nz_c)+1)+1
    t,b = max(0,min(nz_r)-1),min(n_r-1,max(nz_r)+1)+1

    # extract window
    win = fg[t:b,l:r]

    if debug:
        plt.figure(3)
        plt.imshow(win)
        plt.title('windowed image')
        plt.show()

    # resize so largest dim is 48 pixels 
    max_dim = max(win.shape)
    new_r = int(round(win.shape[0]/max_dim*48))
    new_c = int(round(win.shape[1]/max_dim*48))

    win_img = Image.fromarray(win.astype(np.uint8)*255)
    resize_img = win_img.resize((new_c,new_r))
    resize_win = np.array(resize_img).astype(bool)

    # embed into output array with 1 pixel border
    out_win = np.zeros((resize_win.shape[0]+2,resize_win.shape[1]+2),dtype=bool)
    out_win[1:-1,1:-1] = resize_win

    if debug:
        plt.figure(4)
        plt.imshow(out_win,cmap='Greys')
        plt.title('resized windowed image')
        plt.show()

    #save out result as numpy array
    #np.save(out_fname,out_win)
    return out_win


"""
====================================================
========= Generate Features and Labels =============
====================================================
"""

if __name__ == '__main__':

    # To not call from command line, comment the following code block and use example below 
    # to use command line, call: python hw03.py K.jpg output

    """
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print('usage: {} <in_filename> <out_filename> (--debug)'.format(sys.argv[0]))
        sys.exit(0)
    
    in_fname = sys.argv[1]
    out_fname = sys.argv[2]

    if len(sys.argv) == 4:
        debug = sys.argv[3] == '--debug'
    else:
        debug = False
    """

#    #e.g. use

    """
    ***************Code to generate data for assignment 3C*********************

    img_list = []
    for i in range(1, 81):
        bin_img=process_image('C:\\Users\\user\\Desktop\\MS\\Fall_2019\\FML\\ass3\\3C\\{}.jpeg'.format(i),debug=False)
        img_list.append(bin_img)

    img_list = np.asarray(img_list)
    np.save('data.npy',img_list)
    data = np.load('data.npy', allow_pickle =True) 
    print("Shape of data.npy is:\n", data.shape)

    #generating labels
    labels = np.arange(80)
    for i in range(80):
        if i>=0 and i<=9:
            labels[i] = 1
        if i>=10 and i<=19:
            labels[i] = 2
        if i>=20 and i<=29:
            labels[i] = 3
        if i>=30 and i<=39:
            labels[i] = 4
        if i>=40 and i<=49:
            labels[i] = 5
        if i>=50 and i<=59:
            labels[i] = 6
        if i>=60 and i<=69:
            labels[i] = 7
        if i>=70 and i<=79:
            labels[i] = 8

    np.save('labels.npy',labels)
    labels = np.load('labels.npy', allow_pickle =True)
    #process_image(in_fname,out_fname)
    print("\nShape of labels.npy is:\n", labels.shape)

    """

    X = np.array([(2,3,3,4,5,7), (2,4,5,5,6,8)])

    #normalizing the data for question 4 PCA
    sc = StandardScaler()
    X_std = sc.fit_transform(X.T)
    X_std = X_std.T

    #mean_X = X.mean(1)
    #norm_X = (X.T - mean_X).T
    #cov_X = np.cov(norm_X)
    cov_X = np.cov(X_std)

    #calculating eigenvalues for PCA question 1
    w, v = LA.eig(X.T@X)

    #calculating eigenvalues and eigenvectors for PCA question 2
    w1, v1 = LA.eig(X@X.T)

    w2, v2 = LA.eig(cov_X)

    #perform dimensionality reduction
    eigen_pairs = [(np.abs(w2[i]), v2[:,i]) for i in range(len(w2))]
    eigen_pairs.sort(reverse=True)

    PM = eigen_pairs[0][1][:, np.newaxis]
    #print('Matrix PM:\n', PM)

    #X_pca = norm_X.T.dot(PM)
    X_pca = X_std.T.dot(PM)
    X_pca = X_pca.T

    
    """
    print ("v:\n",X_std)
    print ("\n\n")
    print ("sd of X: \n",np.std(X, axis=1))
    """
    
    print ("*****PCA*******\n")
    print ("Question 1")
    print ("\nEigen values of X.T@X are:")
    print (w)
    print ("\n")
    print ("Question 2\n")
    print ("Eigenvalues of X@X.T are:")
    print (w1)
    print ("\n")
    print ("Eigenvectors of X@X.T are:")
    print (v1)
    print ("\n")
    #print (w2)
    print ("Question 4\n")
    print('Shape of transformed data:\n', X_pca.shape)

    print('\nTransformed data after dimension reduction:\n', X_pca)

    print ("\n*****EM*******")
    Y = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    EM(Y, 0.4, 0.6, 0.7)
    print("\n\n")
    EM(Y, 0.5, 0.5, 0.5)
