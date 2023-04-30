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
from myutils import *

MISSOBS = np.array([-1,-1,-1,-1])
P_const = np.zeros((3,4))
P_const[:3,:3] = np.eye(3)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ds', default=3, type=int)
    parser.add_argument('--featSkip', default=10, type=int)   # fs
    
    parser.add_argument('--initPoseCov', default=1e-4, type=float)  # ipv
    parser.add_argument('--initLMCov', default=0.1, type=float)    # ilv
    parser.add_argument('--vcov', default=1.0, type=float)           # vv
    parser.add_argument('--wcov', default=1e-2, type=float)         # wv
    
    parser.add_argument('--distThresh', default=200, type=float)    # dt
    parser.add_argument('--full', default=False, action='store_true')
    
    parser.add_argument('--mapping', default=False, action='store_true')
    parser.add_argument('--tosave', default=False, action='store_true')

    args = parser.parse_args()
    return args


def VISLAM(stamps, features, twistvel, meanLMInit, initPoseCov, initLMCov, option="part"):
    M = features.shape[1]
    timediff = np.squeeze(stamps[1:] - stamps[0:-1], axis=1)   # (N-1,)

    twistvelhat = axangle2twist(twistvel)   # (N,4,4)
    twistvelcurlyhat = axangle2adtwist(twistvel)  # (N,6,6)

    twistdisthat = twistvelhat[:-1] * timediff[:,None,None]           # (N-1,4,4)
    twistdistcurlyhat = twistvelcurlyhat[:-1] * timediff[:,None,None]   # (N-1,6,6)

    # initialize the gaussian mean and covariance matrix for M landmarks
    meanLM = np.matrix.flatten(meanLMInit)                        # (3M,)
    covLM = np.diag(np.full(shape=(3*M), fill_value=initLMCov))   # (3M,3M)
    IkroV = getIkroV(V, M)

    # initialize the state at t=0 as identity pose matrix in SE(3)
    meanPose = np.zeros((N,4,4))   # (N,4,4)
    meanPose[0] = np.eye(4)
    covPose = np.eye(6) * initPoseCov     # (6,6)

    covSLAM = np.zeros((3*M+6,3*M+6))   # (3M+6,3M+6)
    covSLAM[0:3*M,0:3*M] = covLM
    covSLAM[3*M:3*M+6,3*M:3*M+6] = covPose

    covdiagsum = []
    robotRowIdx = [3*M,3*M+1,3*M+2,3*M+3,3*M+4,3*M+5]
    
    num_itr = 1
    for itr in range(num_itr):
        for t in tqdm(range(0, N-1)):
            # Prediction step for the SE(3) pose
            expm_twistdisthat = scipy.linalg.expm(twistdisthat[t])               # (4,4)
            expm_twistdistcurlyhat = scipy.linalg.expm(-twistdistcurlyhat[t])    # (6,6)

            meanPose[t+1] = np.matmul(meanPose[t], expm_twistdisthat)
            meanPoseInv = inversePose(meanPose[None,t+1])[0]     # (4,4)

            F = expm_twistdistcurlyhat
            covSLAM[3*M:3*M+6,3*M:3*M+6] = np.matmul(np.matmul(F, covSLAM[3*M:3*M+6,3*M:3*M+6]), F.T) + W
            covSLAM[0:3*M,3*M:3*M+6] = np.matmul(covSLAM[0:3*M,3*M:3*M+6], F.T)
            covSLAM[3*M:3*M+6,0:3*M] = np.matmul(F, covSLAM[3*M:3*M+6,0:3*M])

            # Visual Map Update
            tmp1 = featuresk[t] - MISSOBS        # (M,4)
            tmp2 = np.sum(np.abs(tmp1), axis=1)  # (M,)
            validObsIdx = np.where(tmp2 > 0)[0]  # (Nt,)
            validFeat = featuresk[t,validObsIdx,:]  # (Nt,4)
            Nt = validFeat.shape[0]

            meanLMHom = getHomoCoord(meanLM)            # (4M,)
            meanLMHom = np.reshape(meanLMHom, (-1,4))   # (M,4)

            IFcoord = np.matmul(meanPoseInv, meanLMHom[validObsIdx].T).T  # (Nt,4)
            OFcoord = np.matmul(CF_T_IF, IFcoord.T).T         # (Nt,4)
            predObs = np.matmul(Ks, projection(OFcoord).T).T   # (Nt,4)
            OFcoordJac = projectionJacobian(OFcoord)   # (Nt,4,4)

            tmp1 = np.matmul(Ks, np.matmul(OFcoordJac, CF_T_IF))    # (Nt,4,4)
            tmp2 = computeCircDot(IFcoord)   # (Nt,4,6)
            H = -np.matmul(tmp1, tmp2)       # (Nt,4,6)
            Hrobot = np.reshape(H, (-1,6))   # (4Nt,6)

            H = np.matmul(tmp1, np.matmul(meanPoseInv, P_const.T))  # (Nt,4,3)

            row1 = validObsIdx*3     # (Nt,)
            rowsLM = np.matrix.flatten(np.vstack((row1,row1+1,row1+2)).T)     # (3Nt,)
            rowsLMRobot = np.append(rowsLM, robotRowIdx)                      # (3Nt+6,)

            if option == "full":
                # for full KG computation
                HLM = np.zeros((4*Nt,3*M))     # (4Nt,3M)
                for j in range(Nt):
                    HLM[j*4:j*4+4, validObsIdx[j]*3:validObsIdx[j]*3+3] = H[j]
                HSLAM = np.hstack((HLM, Hrobot))    # (4Nt,3M+6)

                B = np.matmul(HSLAM, covSLAM)       # (4Nt,3M+6)
                A = np.matmul(B, HSLAM.T) + IkroV[0:4*Nt,0:4*Nt]   # (4Nt,4Nt)
                KG = np.linalg.solve(A,B).T         # (3M+6,4Nt)
                KGLM = KG[0:3*M]         # (3M,4Nt)
                KGrobot = KG[3*M:3*M+6]  # (6,4Nt)

                # update the map landmarks mean and covariance
                obsError = np.matrix.flatten(validFeat - predObs)               # (4Nt,)
                obsError = np.clip(obsError, -50, 50)
                meanLM[rowsLM] = meanLM[rowsLM] + np.matmul(KGLM[rowsLM], obsError)

                # update the robot pose mean and covariance
                scaledError = np.matmul(KGrobot, obsError)   # (6,)
                meanPoseUpdate = axangle2pose(scaledError[None,:])[0]   # (4,4)
                meanPose[t+1] = np.matmul(meanPose[t+1], meanPoseUpdate)
                covSLAM[rowsLMRobot][:,rowsLMRobot] = np.matmul(np.eye(3*Nt+6) - np.matmul(KG[rowsLMRobot], HSLAM[:,rowsLMRobot]), 
                                                    covSLAM[rowsLMRobot][:,rowsLMRobot])
            elif option == "part":
                # for partial small KG computation
                HLM = np.zeros((4*Nt,3*Nt))     # (4Nt,3Nt)
                for j in range(Nt):
                    HLM[j*4:j*4+4, j*3:j*3+3] = H[j]
                HSLAM = np.hstack((HLM, Hrobot))    # (4Nt,3Nt+6)

                validcovSLAM = covSLAM[rowsLMRobot][:,rowsLMRobot]    # (3Nt+6,3Nt+6)
                B = np.matmul(HSLAM, validcovSLAM)       # (4Nt,3Nt+6)
                A = np.matmul(B, HSLAM.T) + IkroV[0:4*Nt,0:4*Nt]   # (4Nt,4Nt)
                KG = np.linalg.solve(A,B).T              # (3Nt+6,4Nt)
                KGLM = KG[0:3*Nt]          # (3Nt,4Nt)
                KGrobot = KG[3*Nt:3*Nt+6]  # (6,4Nt)

                obsError = np.matrix.flatten(validFeat - predObs)           # (4Nt,)
#                 obsError[np.where(obsError > 20)] = 0
#                 obsError[np.where(obsError < -20)] = 0
                obsError = np.clip(obsError, -50, 50)
                meanLM[rowsLM] = meanLM[rowsLM] + np.matmul(KGLM, obsError)

                # update the robot pose mean and covariance
                scaledError = np.matmul(KGrobot, obsError)              # (6,)
                meanPoseUpdate = axangle2pose(scaledError[None,:])[0]   # (4,4)
                meanPose[t+1] = np.matmul(meanPose[t+1], meanPoseUpdate)
                covSLAM[rowsLMRobot][:,rowsLMRobot] = np.matmul(np.eye(3*Nt+6) - np.matmul(KG, HSLAM), 
                                                                covSLAM[rowsLMRobot][:,rowsLMRobot])

            covdiagsum.append(np.sum(np.diag(covSLAM)))
            
    meanLMmat = np.reshape(meanLM, (-1,3))
    return meanPose, meanLMmat


if __name__ == '__main__':
    args = parse_args()
    
    # Load the measurements
    if args.ds == 3:
        filename = "../data/03.npz"
    elif args.ds == 10:
        filename = "../data/10.npz"
    
    if args.full == True:
        option = "full"
    else:
        option = "part"
    
#     os.makedirs("../plots/fs+"str(args.featSkip)+"_dt"+str(distThresh)+
    
    stamps, featOrig, linVel, angVel, K, stereoBaseline, imu_T_cam = load_data(filename)
    twistvel = np.hstack((linVel, angVel))  # (N,6)
    
    features = featOrig[:,::args.featSkip,:]
    N, M = stamps.shape[0], features.shape[1]
    Ks = getKsMat(K, stereoBaseline)
    
    W = np.diag([1,1,1,1,1,1]) * args.wcov     # (6,6)
    V = np.diag([1,1,1,1]) * args.vcov         # (4,4)

    IF_T_CF, RF_T_OF = getimu2cam(imu_T_cam)
    CF_T_IF = np.linalg.inv(IF_T_CF)

    print("Number of timestamps:", N)
    print("Total landmarks:", M)
    print("SE(3) transformation Left Cam to IMU\n", IF_T_CF)
    
    # IMU localization via EKF prediction
    print("Performing IMU Localization via EKF Prediction...")
    robotDeadReckPose = estPoseEKFPred(twistvel, stamps, args.initPoseCov, W)
    savepath = "../plots/imuPose_ds"+str(args.ds)+".png"
    visualize_trajectory_2d(robotDeadReckPose, show_ori=True, savepath=savepath)
    
    nearLM2RobotIdx, meanLMInit, deadReckLM = getValidLandmarks(features, robotDeadReckPose, IF_T_CF, Ks, distThresh=args.distThresh)
#     visualize_trajectory_2d(robotDeadReckPose, landmarks=deadReckLM, path_name="Unknown", show_ori=False)
    
    featuresk = features[:,nearLM2RobotIdx]   # (N,M,4)
    meanLMInit = meanLMInit[nearLM2RobotIdx]       # (3M,)

    
    if args.mapping == True:
        # mapping via EKF update
        print("Performing Landmark Mapping via EKF Update...")
        meanLMmat = mappingEKFUpdate(featuresk, robotDeadReckPose, meanLMInit, CF_T_IF, Ks, args.initLMCov, V)
        savepath = "../plots/map_ds"+str(args.ds)+"_fs"+str(args.featSkip)+"_dt"+str(args.distThresh)+"_ipv"+str(args.initPoseCov)+"_ilv"+str(args.initLMCov)+"_wv"+str(args.wcov)+"_vv"+str(args.vcov)+".png"
        visualize_trajectory_2d(robotDeadReckPose, landmarks=meanLMmat, show_ori=False, savepath=savepath)

    
    # VISLAM prediction and update
    print("Performing Visual-Inertial SLAM...")
    meanPose, meanLMmat = VISLAM(stamps, featuresk, twistvel, meanLMInit, args.initPoseCov, args.initLMCov, option=option)
    savepath = "../plots/slam_ds"+str(args.ds)+"_fs"+str(args.featSkip)+"_dt"+str(args.distThresh)+"_ipv"+str(args.initPoseCov)+"_ilv"+str(args.initLMCov)+"_wv"+str(args.wcov)+"_vv"+str(args.vcov)+".png"
    visualize_trajectory_2d(meanPose, landmarks=meanLMmat, show_ori=False, savepath=savepath)