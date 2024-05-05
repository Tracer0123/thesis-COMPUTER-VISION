"""
------ Batchaya Noumeme Yacynte Divan ------
------ BEng Thesis THWS in Schweinfurt -----
------ WS 2023/24 -----

"""


from operator import le
import cv2
import numpy as np
import math
from rejection import inlier



def optimize_x_y(q,sh,l,wz):
    # create window size
    w = 2*wz+1

    # generate 2d array of points in the window size of x,y for all points
    
    range_x = [np.arange(val-w, val+w+1) for val in q[:,0]]
    range_y = [np.arange(val-w, val+w+1) for val in q[:,1]]
    x = np.minimum(np.array(range_x),sh[l,1]-1)

    # limit the maximum value for y to the size of the current array -1
    y = np.minimum(np.array(range_y),sh[l,0]-1)

    # limit the minimum points to atleast 0
    x[x<0] = 0
    y[y<0] = 0
    
    # create different combinations of the window sized element to index, e.g for x [1 1 2 2 3 3] for y [1 2 3 1 2 3]
    x = np.repeat(x,w,axis=1) # x,y is 2D array, x dimension is diffrent x,y y dimension are different points
    y = np.tile(y,w)
    x_ = x-1
    x_[x_< 1] = 0
    y_ = y - 1
    y_[y_<1] = 0
    return np.intp(x),np.intp(y),np.intp(x_),np.intp(y_)

def optimize_Ix_and_Iy(I_L,sh,l, x,y,x_,y_):
    #I_x = np.zeros((S[0],w**2), dtype=np.float32)
    #I_y = np.zeros((S[0],w**2), dtype=np.float32)
    #j = 0
    #print(np.shape(y),np.shape(np.minimum(x + 1, sh[l, 1] - 1)))
    I_x = (I_L[y, np.minimum(x + 1, sh[l, 1] - 1)] - I_L[y, x_ ]) / 2
    I_y = (I_L[np.minimum(y + 1, sh[l, 0] - 1), x] - I_L[y_, x,]) / 2
    
    return I_x, I_y

def optimized_dIk(I_L,J_L,x,y,v_k,g_L,l,sh):
    vy = (v_k[:,1]).reshape(len(v_k),1)
    vx = (v_k[:,0]).reshape(len(v_k),1)
    gy = (g_L[:,1]).reshape(len(g_L),1)
    gx = (g_L[:,0]).reshape(len(g_L),1)
    #print(np.shape(vy),np.shape(gy))
    k = np.intp(np.round(y+vy+gy))
    m = np.intp(np.round(x+vx+gx))
    k[k<1] = 0
    m[m<1] = 0
    k[k>sh[l,0]-1] = sh[l,0]-1
    m[m>sh[l,1]-1] = sh[l,1]-1
    
    dI_k = I_L[y,x] - J_L[k,m]
    #print("dI_k",dI_k)
    return dI_k




def lucas_pyramidal(img1,img2
                    ,box
                    ,level # level of the LK pyramid
                    ,wz  # windows size
                    ,k  # Number of Iterations
                    ):
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    I1 = np.array(img1)
    I2 = np.array(img2)
    S = np.shape(I1)
    I_L = np.empty((S[0],S[1],level),dtype=np.float32)
    J_L = np.empty((S[0],S[1],level),dtype=np.float32)
    I_L[:,:,0] = I1
    J_L[:,:,0] = I2
    sh = np.empty((level,2),dtype=int)
    sh[0,:] = S
    
    # create image levels
    for i in range(1,level):
        sh[i,:] = np.shape(cv2.resize(I_L[0:sh[i-1,0],0:sh[i-1,1],i-1], None, fx = 0.5, fy = 0.5))
        I_L[0:sh[i,0],0:sh[i,1],i] = cv2.resize(I_L[0:sh[i-1,0],0:sh[i-1,1],i-1], None, fx = 0.5, fy = 0.5)
        J_L[0:sh[i,0],0:sh[i,1],i] = cv2.resize(J_L[0:sh[i-1,0],0:sh[i-1,1],i-1], None, fx = 0.5, fy = 0.5)
    
    feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)
    # get good features to track and convert them to an nx2 array
    box_lens = []
    box_used = []
    q2 = []
    for i in range(0, len(box)):
        x01, y01, x02, y02 = box[i, 0, 0]
        I_min = I1[int(x01):(int(x02)+1), int(y01):(int(y02)+1)]
        features = cv2.goodFeaturesToTrack(I_min, mask=None, **feature_params)
        # print(features)
        if features is None:
            continue
        feature = np.int0(features)
        # print(feature)
        box_lens.append(len(feature))
        box_used.append(i)
        for i in range(len(feature)):
            q2.append([feature[i,0,0]+x01, feature[i,0,1]+y01])
    # print(np.shape(q2))
    # print(box_lens)
    
    # Initial guess
    g_Lm = np.zeros((len(q2),2),dtype=np.float32)

    # calculate the optical flow for all points at a time for each level    
    for l in range(level-1,-1,-1):
        # print(q2)
        q2 = np.array(q2)
        # print(np.shape(q2))
        q = np.intp(q2/2**l)

        # convert the points to a 2d array of the points and their window size combinations for indexing
        x,y,x_,y_ = optimize_x_y(q,sh,l,wz)
        # print(x,y,x_, y_)
        # Calculate the derivatives of the intensity I_L
        Ix,Iy = optimize_Ix_and_Iy(I_L[:,:,l],sh,l, x,y,x_,y_)
        Ixy = Ix*Iy

        #form the spatial gradient matix of the points around x,y as an nx2x2
        I2x = Ix*Ix
        I2y = Iy*Iy
        Ix_ = np.sum(I2x, axis=1)
        Iy_ = np.sum(I2y, axis=1)
        Ixy_ = np.sum(Ixy, axis=1)
        a = np.dstack((Ix_,Ixy_,Ixy_,Iy_))
        G = a.reshape(len(Ix),2,2)
        # g_not_zero = np.where(np.linalg.det(G) != 0)
        G = G + 0.0000000001
        G_ = np.linalg.inv(G)

        # initialize the oprical flow for level k=0
        v_k = np.zeros((len(x),2),dtype=np.float32)

        for j in range(1,k):

            # get the image differnce
            dIk = optimized_dIk(I_L[:,:,l],J_L[:,:,l],x,y,v_k,g_Lm,l,sh)
            # create the image mismatch vector b
            dIkx = dIk*Ix
            dIky = dIk*Iy
            dIkx = np.sum(dIkx, axis=1)
            dIky = np.sum(dIky, axis=1)
            b = np.dstack((dIkx,dIky)).reshape(len(Ix),2,1)

            # optical flow LK
            n_k = np.matmul(G_,b)
            n_k = n_k.reshape(np.shape(v_k))

            # guess for the next iteration
            v_k += n_k
        g_Lm = 2*(v_k+g_Lm)

    d = g_Lm/2
    """
    # dm = np.median(d, axis=0)
    #print(np.shape(d))
    d1 = d.copy()
    q5 = q2.copy()
    q2, d, foes = brut_force(q2,d,S,1)
    if len(d) < 6:
        d = d1
        q2 = q5
    """
    q2, d, foes, _ = inlier(q2,d,S,1)
    q3 = q2 + d
    #print(np.shape(I1))
    
    return q2, q3, box_lens, box_used #, foes
