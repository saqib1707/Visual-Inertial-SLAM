import os
import argparse
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import scipy.linalg

from pr3_utils import *


MISSOBS = np.array([-1,-1,-1,-1])
P_const = np.zeros((3,4))
P_const[:3,:3] = np.eye(3)


def estPoseEKFPred(twistvel, stamps, poseCov, W):
    N = stamps.shape[0]
    timediff = np.squeeze(stamps[1:] - stamps[0:-1], axis=1)   # (N-1,)

    twistvelhat = axangle2twist(twistvel)   # (N,4,4)
    twistvelcurlyhat = axangle2adtwist(twistvel)  # (N,6,6)

    twistdisthat = twistvelhat[:-1] * timediff[:,None,None]           # (N-1,4,4)
    twistdistcurlyhat = twistvelcurlyhat[:-1] * timediff[:,None,None]   # (N-1,6,6)

    # initialize the state at t=0 as identity pose matrix in SE(3)
    meanPose = np.zeros((N,4,4))
    meanPose[0] = np.eye(4)
    covPose = np.eye(6) * poseCov     # (6,6)

    for t in tqdm(range(0, N-1)):
        expm_twistdisthat = scipy.linalg.expm(twistdisthat[t])               # (4,4)
        expm_twistdistcurlyhat = scipy.linalg.expm(-twistdistcurlyhat[t])    # (6,6)

        meanPose[t+1] = np.matmul(meanPose[t], expm_twistdisthat)
        covPose = np.matmul(np.matmul(expm_twistdistcurlyhat, covPose), expm_twistdistcurlyhat.T) + W

    return meanPose


def getValidLandmarks(features, robotPose, IF_T_CF, Ks, distThresh=200):
    N,M,_ = features.shape

    # LMweights = np.zeros((N,M))
    nearLM2Robot = np.ones((M))
    deadReckLM = np.array([0,0,0])
    meanLMInit = np.zeros((M,3))

    for t in tqdm(range(0, N-1)):
        tmp1 = features[t] - MISSOBS          # (M,4)
        tmp2 = np.sum(np.abs(tmp1), axis=1)   # (M,)
        invalidObsIdx = np.where(tmp2 == 0)[0]   
        validObsIdx = np.where(tmp2 > 0)[0]     # (Nt,)
        Nt = validObsIdx.shape[0]

        validFeat = features[t,validObsIdx,:]              # (Nt,4)
        z = -Ks[2,3] / (validFeat[:,0] - validFeat[:,2])   # (Nt,)
        x = (1/Ks[0,0]) * (validFeat[:,0] - Ks[0,2]) * z   # (Nt,)
        y = (1/Ks[1,1]) * (validFeat[:,1] - Ks[1,2]) * z   # (Nt,)

        OFcoord = np.stack((x,y,z), axis=1)   # (Nt,3)
        OFcoordHom = np.ones((Nt,4))
        OFcoordHom[:,:3] = OFcoord
#         OFcoordHom = np.matmul(RF_T_OF, OFcoordHom.T).T     # (Nt,4)
        IFcoordHom = np.matmul(IF_T_CF, OFcoordHom.T).T     # (Nt,4)
        WFcoordHom = np.matmul(robotPose[t], IFcoordHom.T).T  # (Nt,4)
        WFcoord = WFcoordHom[:,:3]    # (Nt,3)

        meanLMInit[validObsIdx] = WFcoord    # (M,3)
        distLMRobot = np.linalg.norm(WFcoord[:,:3] - robotPose[t,:3,3], axis=1)  # (Nt,)
        farLM2RobotIdx = validObsIdx[np.where(distLMRobot > distThresh)[0]]
        nearLM2Robot[farLM2RobotIdx] = 0

#         nearLM2RobotIdx = validObsIdx[np.where(distLMRobot <= distThresh)[0]]
#         LMweights[t,nearLM2RobotIdx] = 1
        deadReckLM = np.vstack((deadReckLM, WFcoord[np.where(distLMRobot <= distThresh)[0]]))

    deadReckLM = deadReckLM[1:]
    nearLM2RobotIdx = np.where(nearLM2Robot > 0)[0]      # (<M,)

    return nearLM2RobotIdx, meanLMInit, deadReckLM


def mappingEKFUpdate(features, meanPose, meanLMInit, CF_T_IF, Ks, initLMCov, V):
    N, M, _ = features.shape
    meanPoseInv = inversePose(meanPose)   # (N,4,4)

    # initialize the gaussian mean and covariance matrix for M landmarks
    meanLM = np.matrix.flatten(meanLMInit)
    covLM = np.diag(np.full(shape=(3*M), fill_value=initLMCov))   # (3M,3M)
    IkroV = getIkroV(V, M)

    num_itr = 1
    for itr in range(num_itr):
        for t in tqdm(range(0, N-1)):
            tmp1 = features[t] - MISSOBS        # (M,4)
            tmp2 = np.sum(np.abs(tmp1), axis=1)  # (M,)
            validObsBool = (tmp2 > 0)            # (M,)
            validObsIdx = np.where(tmp2 > 0)[0]  # (Nt,)

            validFeat = features[t,validObsIdx,:]  # (Nt,4)
            Nt = validFeat.shape[0]

            meanLMHom = getHomoCoord(meanLM)            # (4M,)
            meanLMHom = np.reshape(meanLMHom, (-1,4))   # (M,4)

            IFcoord = np.matmul(meanPoseInv[t+1], meanLMHom[validObsIdx].T).T  # (Nt,4)
            OFcoord = np.matmul(CF_T_IF, IFcoord.T).T         # (Nt,4)
            predObs = np.matmul(Ks, projection(OFcoord).T).T   # (Nt,4)
            OFcoordJac = projectionJacobian(OFcoord)   # (Nt,4,4)

            tmp1 = np.matmul(Ks, np.matmul(OFcoordJac, CF_T_IF))         # (Nt,4,4)
            H = np.matmul(tmp1, np.matmul(meanPoseInv[t+1], P_const.T))   # (Nt,4,3)
            HLM = np.zeros((4*Nt,3*M))     # (4Nt,3M)
            for j in range(Nt):
                HLM[j*4:j*4+4, validObsIdx[j]*3:validObsIdx[j]*3+3] = H[j]

            B = np.matmul(HLM, covLM)     # (4Nt,3M)
            A = np.matmul(B, HLM.T) + IkroV[0:4*Nt,0:4*Nt]   # (4Nt,4Nt)
            KG = np.linalg.solve(A, B).T      # (3M,4Nt)

            row1 = validObsIdx*3     # (Nt,)
            rows = np.matrix.flatten(np.vstack((row1,row1+1,row1+2)).T)   # (3Nt)

            obsError = np.matrix.flatten(validFeat - predObs)     # (4Nt,)
            meanLM[rows] = meanLM[rows] + np.matmul(KG[rows], obsError)
            covLM[rows][:,rows] = covLM[rows][:,rows] - np.matmul(np.matmul(KG[rows], HLM[:,rows]), covLM[rows][:,rows])

    meanLMmat = np.reshape(meanLM, (-1,3))
    return meanLMmat