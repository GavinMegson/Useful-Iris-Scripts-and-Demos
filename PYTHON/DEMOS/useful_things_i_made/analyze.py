### Read in data from h5df file and analyze
### Gavin Megson

import sys
sys.path.append('../IrisUtils/')

# prune imports sometime
import numpy as np
from optparse import OptionParser
import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import time
import os
import math
import datetime
import json
import signal
import pdb
import matplotlib
from scipy.linalg import hadamard
import scipy.io as sio
from array import array
matplotlib.rcParams.update({'font.size': 10})
import matplotlib.pyplot as plt
from matplotlib import animation
from csi_lib import *
from find_lts import *
from digital_rssi import *
from bandpower import *
from file_rdwr import *
from ofdmtxrx import *
from type_conv import *
from print_sensor import *
import h5py

# edited h5py recording class
from data_recorder3 import *

from time import sleep
import datetime

# for softnull code
from scipy import linalg as sp


plt.style.use('ggplot')  # customize your plots style

# debiggung - check which matplot backend is in use
#print(matplotlib.get_backend())




##################
# INITIALIZATION #
##################

running = True

def signal_handler(signum, frame):
    global running
    running = False

def init(ref_ant, cp, numSamps, prefix_length, postfix_length, both_channels, threshold, plotter, freq, \
        tx_antennas, rx_antennas, cl_antennas, final_plots, first_plots, effective, teststr, iterations, \
        phi, theta):

    # BS nodes, antennas, and client nodes
    R = len(tx_antennas) + len(rx_antennas)
#    ant = 2 if both_channels else 1
    M = len(tx_antennas) + len(rx_antennas) + len(cl_antennas)
    K = len(cl_antennas) if cl_antennas else 0
    print("base station antennas, all antennas, client antennas")
    print("(R,M,K) = (%d,%d,%d)"%(R,M,K))


    symSamp = numSamps + prefix_length + postfix_length
    print("numSamps = %d"%symSamp)


    # values hardcoded to work with Rice's library code
    fft_size = 64
    cp_len = 32 if cp else 0
    ofdm_len = 2*fft_size + cp_len
    zeros = np.array([0]*(numSamps-ofdm_len))
    ltsSym, lts_f = generate_training_seq(preamble_type='lts', cp=cp_len, upsample=1)
    pilot = np.concatenate((ltsSym, zeros)).astype(np.complex64)

    #a0, a1, corr = find_lts(ltsSym, flip=True)
    #print(len(corr))
    #plt.plot(corr)
    #plt.show()

    fft_offset = 6

    upsample = 1
    preambles_bs = generate_training_seq(preamble_type='gold_ifft', seq_length=128, cp=0, upsample=upsample)
    preambles = preambles_bs[:,::upsample] #the correlators can run at lower rates, so we only need the downsampled signal.
    beacon = preambles[0,:]
    pad1 = np.array([0]*(prefix_length), np.complex64) # to comprensate for front-end group delay
    pad2 = np.array([0]*(postfix_length), np.complex64) # to comprensate for rf path delay
    wbz = np.array([0]*(symSamp), np.complex64)
    bcnz = np.array([0]*(symSamp-prefix_length-len(beacon)), np.complex64)  
    beacon1 = np.concatenate([pad1,beacon*.5,bcnz]).astype(np.complex64)
    beacon2 = wbz #beacon1 if both_channels else wbz   

    wb_pilot = 0.25 * pilot
    wbz = np.array([0]*(symSamp), np.complex64)
    wb_pilot_pad = np.concatenate([pad1, wb_pilot, pad2]).astype(np.complex64)

    L = fft_size - 12
    lts_thresh = 0.8

    possible_dim = []
#    nRadios = 2*len(bsdrs) if both_channels else len(bsdrs)
    nRadios = M
    possible_dim.append(2**(np.ceil(np.log2(nRadios))))
    h_dim = min(possible_dim)
    hadamard_matrix = hadamard(h_dim)       #hadamard matrix : http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.hadamard.html
    beacon_weights = hadamard_matrix[0:nRadios, 0:nRadios]
    beacon_weights = beacon_weights.astype(np.uint32)

    # DEV: ueTrigTime = 153 (prefix_length=0), CBRS: ueTrigTime = 235 (prefix_length=82), tx_advance=prefix_length,
    # corr delay is 17 cycles <- this comment came from Rice and was not verified independently
    rf_roundtrip = prefix_length + len(beacon) + postfix_length + 17 + postfix_length



    zeroPlot = [[np.array([0]*symSamp).astype(np.complex64) for r in range(M)] for s in range(M)]
    peaks = [[np.array([0]*613, np.int32) for i in range(M)] for j in range(M)] 
    peaks1 = [[np.array([0]*613, np.int32) for i in range(2*K)] for j in range(M)] 


    if plotter and K > 0:
        fig1, axes1 = plt.subplots(nrows=M, ncols=2*K, figsize=(12,12))
        axes1[0,0].set_title('Downlink Pilot')
        axes1[0,K].set_title('Uplink Pilot')
        for m in range(M):
            for l in range(K):
                axes1[m,l].set_xlim(0, symSamp)
                axes1[m,l+K].set_xlim(0, symSamp)
                axes1[m,l].set_ylim(-1,1)
                axes1[m,l+K].set_ylim(-1,1)
                axes1[m,l].set_ylabel('Downlink BS Ant %d, Cl Ant %d'%(m,l))
                axes1[m,l+K].set_ylabel('Uplink BS Ant %d, Cl Ant %d'%(m,l))

        lines10 = [[axes1[m,l].plot(range(symSamp), np.real(zeroPlot[m][l]), label='Pilot TxAnt %d RxAnt %d (real)'%(m,l))[0] for l in range(2*K)] for m in range(M)]
        lines11 = [[axes1[m,l].plot(range(symSamp), np.imag(zeroPlot[m][l]), label='Pilot TxAnt %d RxAnt %d (imag)'%(m,l))[0] for l in range(2*K)] for m in range(M)]
        lines12 = [[axes1[m,l].plot(range(symSamp), symSamp*[lts_thresh])[0] for l in range(2*K)] for m in range(M)] 
        for m in range(M):
            for l in range(2*K):
                axes1[m,l].legend(fontsize=10)
        fig1.show()

    if plotter and recip_cal:
        fig2, axes2 = plt.subplots(nrows=M, ncols=M, figsize=(12,12))
        for m in range(M):
            for l in range(M):
                axes2[m,l].set_xlim(0, symSamp)
                axes2[m,l].set_ylim(-1,1)
                axes2[m,l].set_ylabel('Tx Ant %d, Rx Ant %d'%(m,l))
                axes2[m,l].legend(fontsize=10)

        lines20 = [[axes2[m,l].plot(range(symSamp), np.real(zeroPlot[m][l]), label='Pilot TxAnt %d RxAnt %d (real)'%(m,l))[0] for l in range(M)] for m in range(M)]
        lines21 = [[axes2[m,l].plot(range(symSamp), np.imag(zeroPlot[m][l]), label='Pilot TxAnt %d RxAnt %d (imag)'%(m,l))[0] for l in range(M)] for m in range(M)]
        lines22 = [[axes2[m,l].plot(range(symSamp), symSamp*[lts_thresh])[0] for m in range(M)] for l in range(M)] 
        #lines23 = [[axes2[m,l].plot(range(symSamp), peaks[m][l][:symSamp], label='Offset %d'%m)[0] for m in range(M)] for l in range(M)]  
        fig2.show()




    frame = 1000
    f = 0
    forward_sync = False
    reverse_sync = False
    offset = np.empty((M,M), np.int32) # bs array channels
    offset1 = np.empty((M,2*K), np.int32) # bs-client channels
    rssi = np.empty((M,M), np.float64)
    signal.signal(signal.SIGINT, signal_handler) # some keyboard interrupt magic


########################
# FILE INPUT TO ARRAYS #
########################

    # The file and datasets
    h5file = h5py.File("channel_data.hdf5", "r")

    file_H = h5file[teststr+"/all_H_measurements"]
    file_offs = h5file[teststr+"/all_offsets"]
    file_SNRs = h5file[teststr+"/all_SNRs"]
    file_noises = h5file[teststr+"/all_noises"]
    file_rssi = h5file[teststr+"/all_rssi"]

    # all array rssi measurements
    rssi_all = np.zeros((M,M,iterations), np.float64)

    # noises, SNRs
    noise_all = np.zeros((M,M,iterations), np.float64)
    SNRs = np.zeros((M,M,iterations), np.float64)

    # similar to offset (defunct, remove)
    timing = np.empty((M,M), np.int32)
    avg_timing = np.zeros((M,M),np.float64)

    # CSI values to/from client/bs (defunct/remove?)
    clCSI = np.zeros((M,K,symSamp),np.complex64)
    bsCSI = np.zeros((M,K,symSamp),np.complex64)

    # all measured CSI values
    arrCSI_all = np.zeros((M,M,iterations), np.complex64)

    # average after all frames
    avg_clCSI = np.zeros((M,K),np.complex64)
    avg_bsCSI = np.zeros((M,K),np.complex64)

    # bs array calibration CSI values (intra-loop reused)
    arrCSI = np.zeros((M,M,symSamp),np.complex64)
    avg_arrCSI = np.zeros((M,M),np.complex64)
    arrCSInoCFO = np.zeros((M,M,symSamp),np.complex64)

    # H matrix, bs array CSI values
    H = np.zeros((M,M,iterations),np.complex64) # using offset and coarse CFO
    H_2 = np.zeros((M,M,iterations),np.complex64) # check all rotations for lowest LSE value
    H_3 = np.zeros((M,M,iterations),np.complex64) # no CFO, for comparison
    # should really just be using H_3
    # CFO is commented out below; H is H_3


    # H matrix, bs<->cl
    H_BtoC = np.zeros((M,K,iterations),np.complex64)
    H_CtoB = np.zeros((M,K,iterations),np.complex64)

    # index matrix, bs array CSI values (debugging, find lowest lse by checking all values)
    indxs = np.zeros((M,M,iterations))
    # array offsets for debugging
    offs = np.zeros((M,M,iterations),int)
    offs_b2c = np.zeros((M,K,iterations),int)
    offs_c2b = np.zeros((M,K,iterations),int)
    # CFO estimates for debugging
    c_cfo_ests_arr = np.zeros((M,M,iterations),np.complex64) # array coarse CFO estimates
    c_cfo_ests_bsRx = np.zeros((M,K,iterations),np.complex64) # BS<- Cl
    c_cfo_ests_clRx = np.zeros((M,K,iterations),np.complex64) # BS ->Cl

    f_cfo_ests_arr = np.zeros((M,M,iterations),np.complex64) # array fine CFO estimates
    f_cfo_ests_bsRx = np.zeros((M,K,iterations),np.complex64) # BS<- Cl
    f_cfo_ests_clRx = np.zeros((M,K,iterations),np.complex64) # BS ->Cl

    # raw data
    arr_imp = np.zeros((M,M,iterations,symSamp),np.complex64)
    b2c_imp = np.zeros((M,K,iterations,symSamp),np.complex64)
    c2b_imp = np.zeros((M,K,iterations,symSamp),np.complex64)

    # raw samples, ul/dl
    bsRxSamps = np.copy(c2b_imp)
    clRxSamps = np.copy(b2c_imp)

    # date-time tracker
    dateTimes = []


    # read H and offs values from h5 file
    file_H.read_direct(H)
    file_offs.read_direct(offs)
    file_SNRs.read_direct(SNRs)
    file_noises.read_direct(noise_all)
    file_rssi.read_direct(rssi_all)


####################
# PLOTTING RESULTS #
####################

    # plot coloring options (will break if no bs self-measurements
    lab = np.zeros((M,M,iterations)) # label each base station antenna pair with its own color
    count = np.zeros((M,M,iterations)) # change color over time (check drift, etc.)
    for i in range(M):
        for j in range(M):
            lab[i][j] = (M*i+j) * np.ones((iterations))
            count[i][j] = np.arange(iterations)

    if K > 0:
        lab2 = np.zeros((M,K,iterations))
    else:
        lab2 = np.zeros((M,1,iterations))
    lab2[0][0] = np.ones((iterations))
    lab2[1][0] = 2*np.ones((iterations))

    if K > 0:
        count2 = np.zeros((M,K,iterations))
    else:
        count2 = np.zeros((M,1,iterations))
    count2[0][0] = np.arange(iterations)
    count2[1][0] = np.arange(iterations)


    foldstr = "./out/pics/"
    timestr = time.strftime("%m%d-%H%M")
    teststr = teststr

    # bs array CSI estimates, using offset
    X = [x.real for x in H]
    Y = [x.imag for x in H]
    Z = count
    plt.scatter(X,Y,c=Z,label=Z)
    print("Showing: Raw H values, colored over time")
    if first_plots:
        plt.show()
    plt.savefig(foldstr+teststr+"g1_"+timestr)

    X = [x.real for x in H]
    Y = [x.imag for x in H]
    Z = lab
    plt.scatter(X,Y,c=Z,label=Z)
    print("Showing: Raw H values, colored by antenna pair")
    if first_plots:
        plt.show()
    plt.savefig(foldstr+teststr+"g2_"+timestr)


    X = [x.real for x in H]
    Y = [x.imag for x in H]
    Z = offs
    plt.scatter(X,Y,c=Z,label=Z)
    print("Showing: Raw H values, colored by offset")
    if first_plots:
        plt.show()
    plt.savefig(foldstr+teststr+"g5_"+timestr)


    # impulse response of array (display every fifth sample for the first antenna pair)
    if first_plots:
        selections = range(0,iterations,5)

        order_lab = np.zeros((M,M,iterations),int)
        for s in selections:
            order_lab[0][1][s] = s

        # first, show the points, for reference
        f1 = plt.figure(1)

        plt.scatter([X[0][1][s] for s in selections],[Y[0][1][s] for s in selections],c=[order_lab[0][1][s] for s in selections])

        # also, show the impulse response, from board 1 to board 2, for some selections
        f2 = plt.figure(2)

        plt.subplot(len(selections)//2,2,1)
        X1 = np.arange(symSamp)
        Y1 = np.abs(np.roll(wb_pilot_pad,-1*prefix_length))
        plt.ylabel("pilot")
        plt.bar(X1,Y1)


        for i in range(len(selections)):
            plt.subplot(len(selections)//2,2,i+1)
            X = X1
            Y = Y1
            plt.ylabel(str(5*i))
            plt.bar(X,Y)
        for i, s in enumerate(selections):
#        plt.subplot(len(selections),2,i+2)
            plt.subplot(len(selections)//2,2,i+1)
            X = np.arange(symSamp)
            Y = np.abs(np.roll(arr_imp[0][1][s],-1*offs[0][1][s]))
            plt.bar(X,Y)

        plt.show()

    '''
    plt.subplot(4,1,1)
    X1 = np.arange(symSamp)
    Y1 = np.abs(np.roll(wb_pilot_pad,-1*prefix_length))
    plt.ylabel("pilot")
    plt.bar(X1,Y1)

    plt.subplot(4,1,2)
    X2 = np.arange(symSamp)
    Y2 = np.abs(arr_imp[0][1][10])
    plt.bar(X2,Y2)

    plt.subplot(4,1,3)
    X3 = np.arange(symSamp)
    Y3 = np.abs(arr_imp[0][1][25])
    plt.bar(X3,Y3)

    plt.subplot(4,1,4)
    X4 = np.arange(symSamp)
    Y4 = np.abs(arr_imp[0][1][40])
    plt.bar(X4,Y4)

    plt.show()
    '''


    if K > 0:
        # ul/dl links CSI, offset
        X = [x.real for x in H_BtoC]
        Y = [y.imag for y in H_BtoC]
        Z = lab2
        plt.scatter(X,Y,c=Z,label=Z)
        if first_plots:
            plt.show()
        plt.savefig(foldstr+teststr+"g3_"+timestr)

        X = [x.real for x in H_CtoB]
        Y = [y.imag for y in H_CtoB]
        Z = lab2
        plt.scatter(X,Y,c=Z,label=Z)
        if first_plots:
            plt.show()
        plt.savefig(foldstr+teststr+"g4_"+timestr)


        # impulse response of bs from cl
        selections = range(0,iterations,iterations//6)

        order_lab = np.zeros((M,K,iterations),int)
        for s in selections:
            order_lab[0][0][s] = s

        f1 = plt.figure(1)

        plt.scatter([X[0][0][s] for s in selections],[Y[0][0][s] for s in selections],c=[order_lab[0][0][s] for s in selections])

        f2 = plt.figure(2)

        plt.subplot(len(selections)//2,2,1)
        X1 = np.arange(symSamp)
        Y1 = np.abs(np.roll(wb_pilot_pad,-1*prefix_length))
        plt.ylabel("pilot")
        plt.bar(X1,Y1)

        for i in range(len(selections)):
            plt.subplot(len(selections)//2,2,i+1)
            X = X1
            Y = Y1
            plt.ylabel(str(5*i))
            plt.bar(X,Y)
        for i, s in enumerate(selections[1:len(selections)//2]):
            plt.subplot(len(selections)//2,2,i+2)
            X = np.arange(symSamp)
            Y = np.abs(np.roll(c2b_imp[0][0][s],-1*offs_c2b[0][0][s]))
            plt.bar(X,Y)
        for i, s in enumerate(selections[len(selections)//2:]):
            plt.subplot(len(selections)//2,2,len(selections)//2+i+1)
            X = np.arange(symSamp)
            Y = np.abs(np.roll(c2b_imp[0][0][s],-1*offs_c2b[0][0][s]))
            plt.bar(X,Y)


    '''
    plt.subplot(4,1,2)
    X2 = np.arange(symSamp)
    Y2 = np.abs(np.roll(c2b_imp[0][0][10],-1*offs_c2b[0][0][10]))
    plt.bar(X2,Y2)

    plt.subplot(4,1,3)
    X3 = np.arange(symSamp)
    Y3 = np.abs(np.roll(c2b_imp[0][0][25],-1*offs_c2b[0][0][25]))
    plt.bar(X3,Y3)

    plt.subplot(4,1,4)
    X4 = np.arange(symSamp)
    Y4 = np.abs(np.roll(c2b_imp[0][0][40],-1*offs_c2b[0][0][40]))
    plt.bar(X4,Y4)
    '''

    if first_plots:
        plt.show()

#################
# DATA ANALYSIS #
#################

    # multi-frame averages
    med_rssi = np.median(rssi_all,axis=(2))
#    avg_rssi = avg_rssi/iterations
#    print("avg rssi")
#    print(avg_rssi)
    med_CSI = np.median(H,axis=(2))
#    med_CSI2 = np.median((H[np.nonzero(H)], axis=(2))

    med_offs = np.median(offs,axis=(2))

    med_noise = np.median(noise_all,axis=(2))

#    nonzero_H = np.where(H != 0, 1, 0)
#    nonzero_count = np.sum(H, axis=(2))
    nonzero_offs = np.where(offs != 0, 1, 0)
    nonzero_count = np.sum(nonzero_offs, axis=(2))

    # median of nonzero csi
    temp_offs_csi = np.where(offs == 0, 0, H)
    masked_CSI = np.ma.masked_equal(temp_offs_csi, 0)
    med_nonzero_CSI = np.ma.median(masked_CSI, axis=(2))

    # median SNR values
    temp_SNRs = np.where(offs ==0, 0, SNRs)
    masked_SNRs = np.ma.masked_equal(temp_SNRs, 0)
    med_nonzero_SNRs = np.ma.median(masked_SNRs, axis=(2))

    # median RSSI, noise values
    temp_rssi = np.where(offs ==0, 0, rssi_all)
    masked_rssi = np.ma.masked_equal(temp_rssi, 0)
    med_nonzero_rssi = np.ma.median(masked_rssi, axis=(2))
    temp_noise = np.where(offs ==0, 0, noise_all)
    masked_noise = np.ma.masked_equal(temp_noise, 0)
    med_nonzero_noise = np.ma.median(masked_noise, axis=(2))

    print("avg array CSI")
    print(avg_arrCSI)
    print("avg bs-client CSI")
    print(avg_clCSI)
    print("avg client-bs CSI")
    print(avg_bsCSI)

    print("BASE STATION:")
    print("median offset")
    print(med_offs)
    print("median rssi")
    print(med_rssi)
    print("median CSI")
    print(med_CSI)
    print("median CSI of nonzero")
    print(med_nonzero_CSI)

    # get subset of median values between relevant antenna pairs
    rel_CSI = np.take(np.take(med_CSI, tx_antennas, 0), rx_antennas, 1)
    rel_rssi = np.take(np.take(med_rssi, tx_antennas, 0), rx_antennas, 1)
    rel_offs = np.take(np.take(med_offs, tx_antennas, 0), rx_antennas, 1)
    # offset is zero when it can't detect the lts signal that round
    rel_nonzero_CSI = np.take(np.take(med_nonzero_CSI, tx_antennas, 0), rx_antennas, 1)

    # paritions to/from client
    cl_tx = cl_antennas[0]
    cl_rx = cl_antennas[1]
    # CSIs to/from
    rel_cl_tx = np.take(np.reshape(np.take(med_nonzero_CSI, cl_tx, 0), (1,-1)), rx_antennas, 1)
    rel_cl_rx = np.reshape(np.take(np.take(med_nonzero_CSI, tx_antennas, 0), cl_rx, 1), (-1,1))
    # client noises, rssis to/from
    rel_cl_tx_rssi = np.take(np.reshape(np.take(med_nonzero_rssi, cl_tx, 0), (1,-1)), rx_antennas, 1)
    rel_cl_rx_rssi = np.reshape(np.take(np.take(med_nonzero_rssi, tx_antennas, 0), cl_rx, 1), (-1,1))
    rel_cl_tx_noise = np.take(np.reshape(np.take(med_nonzero_noise, cl_tx, 0), (1,-1)), rx_antennas, 1)
    rel_cl_rx_noise = np.reshape(np.take(np.take(med_nonzero_noise, tx_antennas, 0), cl_rx, 1), (-1,1))
    #SNRs to, from cl
    rel_SNRs_tx = np.take(np.reshape(np.take(med_nonzero_SNRs, cl_tx, 0), (1,-1)), rx_antennas, 1)
    rel_SNRs_rx = np.reshape(np.take(np.take(med_nonzero_SNRs, tx_antennas, 0), cl_rx, 1), (-1,1))

    print("relevant median CSI values recorded:")
#    print(rel_CSI)
    print(rel_nonzero_CSI)
    print("relevant median RSSI values recorded:")
    print(rel_rssi)
    print("relevant median offset values recorded:")
    print(rel_offs)
    print("medians to, from client:")
    print(rel_cl_rx)
    print(rel_cl_tx)
    print("Median SNRs to, from client:")
    print(rel_SNRs_tx)
    print(rel_SNRs_rx)


################################
# SOFTNULL AND PAFD COMPARISON #
################################

    # Number of Tx, Rx physical antennas
    M_tx = len(tx_antennas)
    M_rx = len(rx_antennas)

    # number of effective antennas to use:
    D_tx = effective

    # For simplicity, assume the message signal to send is all 1s (edit for beams)
    X_down = 0j + np.ones(M_tx)

    # MUMIMO precoder (above softnull) (all 1s here) (edit this to test actual beams)
    P_down = 0j + np.ones(D_tx)

    print("SOFTNULL:")

    # Perform Softnull on median CSIs:
#    U, E, V_H = np.linalg.svd(rel_CSI,full_matrices=False)

    # if not assuming X_down is all 1s, first multiply X_down and rel_nonzero_CSI)
    U, E, V_H = np.linalg.svd(rel_nonzero_CSI,full_matrices=False)
    print("SVD:")
    print("U:")
    print(U)
    print("E:")
    print(E)
    print("V_H:")
    print(V_H)

    # make E a matrix instead of ordered list
    sig = sp.diagsvd(E, M_tx, M_rx)

    # remove dimensions with greatest self-interference
    for i in range(0, M_tx - D_tx):
        sig[i][i] = 0

    # reconstruct new pysical-layer H matrix
    P_soft = np.matmul(U, np.matmul(sig, V_H))

    # rescale (because you're using "fewer" antennas,
    # you put more power in each antenna)
    P_soft = P_soft * (M_tx/D_tx)

    print("P_soft:\n", P_soft)



    
    ####
    # PAFD implementation for planar array
    import cvxopt

    print("PAFD")

    # Tx weights (what we're optimizing)
    W = np.zeros((M_tx, 1), np.complex64)
#    W = np.zeros((3,3),np.complex64) # hardcoded because
#    W = W.flatten()

    # For now, fixed . Rewrite this as command line input
#    theta = np.pi/4
#    phi = np.pi/4
    theta = math.radians(theta)
    phi = math.radians(phi)
    d = 1 # 10 cm, approximately, between every antenna
    wl = 1.0/freq # wavelength, or lambda
    d_g = 6 # "desired array factor gain"
    # assume power P = 1 for each antenna
    P = 1.0

    countup = np.zeros((3,3), dtype=np.complex64)
    for i in range(1,4):
        for j in range(1,4):
            countup[i-1][j-1] = i*j
#    a = countup * 1j*2*np.pi*d*np.cos(phi)*np.sin(theta)/wl redoing because maybe 1/freq is too small
    a = countup * 1j*2*np.pi*d*np.cos(phi)*np.sin(theta) * freq
    print("freq:",freq)
    print("test:",countup * 1j*2*np.pi*d*np.cos(phi)*np.sin(theta)/wl)
    a = np.exp(a)
    a = a.flatten()

    print("countup:\n",a)

    # AF = a*w

    # min: sum across all tx antennas m ( (|h_m*w|)^2 + (d_g - |a*w|)^2 )
    # s.t.: |w_i|^2 = P for all w_i in w

    # rewrite min:
    # min: sum across all m=1 ( (h_m*w)^2 + (d_g*exp(j*arg(a*w)) - a*w)^2
    # where theta_0 = arg(a*w)

    # rewrite, iterative method
    # solvle for fixed theta_0
    # min: sum_m=0 (d_m*exp(j*theta_m) - b_m*w)^2
    # with: m=0: b_m = a, d_m = d_g
    #       m>0: b_m = h_m, d_m = 0, theta_m = 0 (fixed)

    # d_m is mostly zeros? Because the first term only matters for d_0?

    # D is vector d
    D = np.zeros((M_tx+1,1), dtype=np.complex64)
    D[0] = d_g

#    B = np.zeros((M_tx+1,1), dtype=np.complex64)
    B = np.zeros((M_tx+1,M_rx), dtype=np.complex64)
    # REL_CSI ORDER NEEDS TO BE SAME AS a
    # OMG IT already IS BECAUSE I MADE ALL THAT EFFORT TO WRITE THEM IN ORDER IN THE FILES
    # I'm a genius
#    B_temp = np.copy(rel_CSI)
    B_temp = np.copy(rel_nonzero_CSI)
#    B_temp = np.reshape(B_temp, (M_tx,-1))
    for i in range(0,M_tx):
        for j in range(0,M_rx):
            B[i+1][j] = B_temp[i][j]
    B[0] = np.copy(a)

    print("D:\n",D)
    print("B:\n",B)
    print("B.shape:",B.shape)

    # rewrite as
    # min: ||D - B*w||_2 ^2

    # semidefinite relaxation:
    # min: ||D' - B'*w'||_2 ^2
    # s.t. w'.T * Q_n' * w' = P
    # where D' = [real(D), imag(D)],
    #       B' = [[real(B), -imag(B)], [imag(B), real(B)]]
    #       w' = [real(w), imag(w)]
    #       Q_n'(i,i) = 1 if i==n or i==n+M_rx, else 0
            # (Q_n' are M_rx diagonal matrices of dimension 2*M_rx by 2*M_rx)

    D_prime = np.zeros((2*(M_tx + 1), 1), dtype=np.float32)
    B_prime = np.zeros((2*(M_tx + 1), 2*M_rx), dtype=np.float32)
    w_prime = np.zeros((2*M_rx, 1), dtype=np.float32)

    for i in range(0, M_tx+1):
        D_prime[i] = np.real(D[i])
        D_prime[i+M_tx+1] = np.imag(D[i])

        for j in range(0, M_rx):
            B_prime[i][j] = np.real(B[i][j])
            B_prime[i][j+M_rx] = -1*np.imag(B[i][j])
            B_prime[i+M_tx+1][j] = np.imag(B[i][j])
            B_prime[i+M_tx+1][j+M_rx] = np.real(B[i][j])

    print("B_prime:\n", B_prime)

    for i in range(0,M_rx):
        w_prime[i] = np.real(W[i])
#        w_prime[i+M_rx+1] = np.imag(W[i])
        w_prime[i+M_rx] = np.imag(W[i])


    Q_prime = np.zeros((M_rx,2*M_rx,2*M_rx))
    for i in range(M_rx):
        for j in range(2*M_rx):
            if i == j or i + M_rx == j:
                Q_prime[i][j][j] = 1


    print("D prime:\n",D_prime)
    print("B prime:\n",B_prime)
    print("w prime:\n",w_prime)
    print("Q prime:\n",Q_prime)


    # P5:
    # min: ||t*D_prime - B_prime*w_prime||_2 ^2
    # s.t. t^2=1, stuff from before
    # t is real number
    # basically it's the same if t = 1 or -1

    # ||t*D_prime - B_prime*w_prime||_2 ^2
    # = ....

    w_prime_2 = np.zeros((2*M_rx+1, 1),dtype=np.float32)
    for i in range(2*M_rx):
        w_prime_2[i] = w_prime[i]
    w_prime_2[-1] = 1 # t

    print("w_prime size:",w_prime.size)
    print("w_prime_2 size:",w_prime_2.size)

    # quandrants of B_prime_2
    B_prime_2_UL = np.matmul(B_prime.T, B_prime) # if wrong shape, switch
    print("B_prime_2 UL should be 2*M_tx:", B_prime_2_UL.shape)
    B_prime_2_LL = -1*np.matmul(D_prime.T,B_prime)
    B_prime_2_UR = -1*np.matmul(B_prime.T,D_prime)
    B_prime_2_LR = np.linalg.norm(D_prime)**2
    print("LL, UR, LR shapes:", B_prime_2_LL.shape, B_prime_2_UR.shape,B_prime_2_LR.shape)


    B_prime_2 = np.zeros((2*M_rx + 1, 2*M_rx + 1), dtype=np.float32)
    for i in range(2*M_rx):
        for j in range(2*M_rx):
            B_prime_2[i][j] = B_prime_2_UL[i][j]
        B_prime_2[i][-1] = B_prime_2_UR[i][0]
        B_prime_2[-1][i] = B_prime_2_LL[0][i]
    B_prime_2[-1][-1] = B_prime_2_LR

    Q_prime_2 = np.zeros((M_rx+1, 2*M_rx+1, 2*M_rx+1),dtype=np.float32)
    for i in range(M_rx+1):
        for j in range(2*M_rx+1):
#            if (i != M_rx+1 and (j == i or j == i + M_rx)) or (i == M_rx+1 and j==2*M_rx+1):
            if (i != M_rx and (j == i or j == i + M_rx)) or (i == M_rx and j==2*M_rx):
                Q_prime_2[i][j][j] = 1

    print("B_prime_2:\n", B_prime_2)
    print("w_prime_2:\n", w_prime_2)
    print("Q_prime_2:\n", Q_prime_2)



    # rewrite
    # min: w_prime_2.T * B_prime_2 * w_prime_2
    # s.t. w_prime_2.T * Q_prime_2_n * w_prime_2 = P for all 1 <= n <= M_rx
    #      "                                     = 1 for n = M_rx+1


    print("B_prime_2 shape:", B_prime_2.shape)

    big_W = np.matmul(w_prime_2,w_prime_2.T)
    print("w_prime_2 shape:",w_prime_2.shape)
    print("Big W shape:", big_W.shape)

#    w_prime_2 = np.resize(w_prime_2,(w_prime_2.shape[1],1))
#    print("w_prime_2 resized shape:",w_prime_2.shape)
#    big_W = np.resize(big_W,(w_prime_2.shape[0],w_prime_2.shape[0]))
    print("Big W:", big_W)

    print("Q prime 2 shape:", Q_prime_2.shape)

    # min: tr(B_prime_2*W) for all real symmetric (2*M_rx+1 by 2*M_rx+1) matrices W
    # s.t. tr(Q_prime_2_n*W) = P for all 1 <= n <= M_rx
    #              "         = 1 for n = M_rx+1
    # and W >= 0, and rank(W) = 1 (W is positive semidefinite)

    # drop final rank constraint for final problem formulation P8.

    # program into CVXPY:
    import cvxpy as cp

    W_solve = cp.Variable(big_W.shape, symmetric=True)

    print("W_solve.shape:", W_solve.shape)

    constraints = []
    constraints += [W_solve >> 0]
    for i in range(0, M_rx):
        subarray = Q_prime_2[i]
        constraints += [cp.trace(subarray @ W_solve) == P]
    constraints += [cp.trace(Q_prime_2[M_rx] @ W_solve) == 1]

    prob = cp.Problem(cp.Minimize(cp.trace(B_prime_2 @ W_solve)), constraints)

    prob.solve()

    for variable in prob.variables():
        print("Variable %s: value %s" % (variable.name(), variable.value))
    prob_stat = prob.solver_stats
#    for stat in prob_stat:
#        print("stat %s: value %s" % (stat.name(), stat.value))
    print(prob_stat)
    print("status:", prob.status)


    print(W_solve.value)


    # work backwards
    W_solved = prob.value

    print("W_solved:\n", W_solved)



    # this is wrong
    '''
    w_p2_solve = cp.Variable(w_prime_2.size)
    prob2 = cp.Problem(cp.Minimize(w_p2_solve))
    constraints2 = [w_p2_solve @ w_p2_solve.T == W_solved]
    prob2.solve()
    '''

    # this is still wrong
    '''
    w_prime_2_solved = np.diagonal(np.copy(W_solve.value))
    w_prime_2_solved = np.sqrt(w_prime_2_solved) # returns non-negative square roots
    '''

    # maybe?
    '''
    print("w prime 2 shpae:",w_prime_2.shape)
    w_prime_2_solve = cp.Variable(w_prime_2.shape)

    constraints2 = []
    constraints2 += [w_prime_2_solve @ w_prime_2_solve.T >> 0]
    for i in range(0, M_rx):
        subarray = Q_prime_2[i]
        constraints2 += [cp.trace(subarray @ (w_prime_2_solve @ w_prime_2_solve.T)) == P]
    constraints2 += [cp.trace(Q_prime_2[M_rx] @ (w_prime_2_solve @ w_prime_2_solve.T)) == 1]

#    prob2 = cp.Problem(cp.Minimize(cp.trace(B_prime_2 @ W_solve)), constraints)
    prob2 = cp.Problem(cp.Minimize(cp.trace(B_prime_2 @ (w_prime_2_solve @ w_prime_2_solve.T))), constraints2)

    prob2.solve()
A
    print("prob2.value:", prob2.value)

    w_prime_2_solved = w_prime_2_solve.value
    print("solved w prime 2:\n", w_prime_2_solved)
    '''
    # cvxpy won't let me do it that way.

    # this is kind of ridiculous
    # assume a is positive, then assume a is negative, use diag, then get sign for the rest
    # look at both outcomes, hopefully one has good results
    w_prime_2_solved = np.diagonal(np.copy(W_solve.value))
    w_prime_2_solved = np.sqrt(w_prime_2_solved) # returns non-negative square roots

    for i in range(1,len(w_prime_2_solved)):
        if W_solve.value[0][i] / w_prime_2_solved[0] < 0:
            w_prime_2_solved[i] *= -1


    w_prime_solved = w_prime_2_solved[:-1]

#    print("w prime 2 solved, w prime prime:", w_prime_2_solved,w_prime_solved)

    w_solved = np.zeros(W.shape,np.complex64)
    for i in range(M_tx):
        w_solved[i] = w_prime_solved[i] + 1j * w_prime_solved[i+M_tx]

    print("Solved weights for PAFD:\n", w_solved)

    P_pafd = np.copy(rel_nonzero_CSI)
    for i in range(M_tx):
        for j in range(M_rx):
            P_pafd[i][j] *= w_solved[i]



    # compare SI of old vs new (add up signals by Rx antenna,
    # take absolute square, and sum total across all antennas)
    RxSums = np.sum(rel_nonzero_CSI,0)
    oldSI = np.sum(np.multiply(RxSums,RxSums.conjugate()))
    print("SI before:\n", oldSI)

    RxSums2 = np.sum(P_soft,0)
    softSI = np.sum(np.multiply(RxSums2,RxSums2.conjugate()))
    print("SI after SoftNull with",D_tx,"effective antennas:\n",softSI)

    RxSums3 = np.sum(P_pafd,0)
    pafdSI = np.sum(np.multiply(RxSums3,RxSums3.conjugate()))
    print("SI after PAFD with array factor of",d_g,":\n",pafdSI)




    # Client antennas

    # signals:
    # over the one client antenna
    cl_rx_sum = np.sum(rel_cl_rx,0)
    cl_rx_sig = np.sum(np.multiply(cl_rx_sum,cl_rx_sum.conjugate()))
    # over the rx partition from the one client tx antenna
    cl_tx_sum = np.sum(rel_cl_tx,0)
    cl_tx_sig = np.sum(np.multiply(cl_tx_sum,cl_tx_sum.conjugate()))

    # SNR weighted averages (more like SINR but I don't want to retype everything)
#    cl_tx_SNR = np.dot(np.ravel(X_down), rel_SNRs_tx)
#    cl_rx_SNR = np.dot(rel_SNRs_rx, np.ravel(X_down))
    # noises:
    cl_tx_sum_noise = np.sum(rel_cl_tx_noise,0)
    cl_tx_noise = np.sum(np.multiply(cl_tx_sum_noise,cl_tx_sum_noise.conjugate()))
    cl_rx_sum_noise = np.sum(rel_cl_rx_noise,0)
    cl_rx_noise = np.sum(np.multiply(cl_rx_sum_noise,cl_rx_sum_noise.conjugate()))

#    cl_tx_SNR = np.matmul(np.ravel(X_down).reshape((1,-1)), rel_SNRs_tx.reshape((-1,1)))
#    cl_rx_SNR = np.dot(rel_SNRs_rx.reshape((1,-1)), np.ravel(X_down).reshape((-1,1)))

    # unadjusted SNR (SINR): S/(N + I)
    cl_tx_SNR = cl_tx_sig / (cl_tx_noise + RxSums)
    cl_rx_SNR = cl_rx_sig / (cl_rx_noise + 0) # assume no SI from clients

    # Client capacities
    bandwidth = 22000000 # 22 MHz
    cl_tx_cap = bandwidth * np.log2(1+cl_tx_SNR)
    cl_rx_cap = bandwidth * np.log2(1+cl_rx_SNR)


    # client Signals after SoftNull:
    # method: adjust the CSI values with info from noise, rssi, and P_soft, then add up again
#    cl_rx_sum_soft = 
    # wait this is actually hard
    cl_rx_sig_soft = cl_rx_sig # 
    cl_tx_sig_soft = cl_tx_sig # because softnull doesn't affect BS Rx or Client

    # SNR after Softnull
#    cl_tx_SNR_soft = np.dot(np.ravel(P_soft), rel_SNRs_tx)
 #   cl_tx_SNR_soft = np.dot(np.ravel(P_soft).reshape(1,-1), rel_SNRs_tx.reshape((-1,1)))
#    cl_rx_SNR_soft = np.dot(rel_SNRs_rx, np.ravel(P_soft))
 #   cl_rx_SNR_soft = np.dot(rel_SNRs_rx.reshape((1,-1), np.ravel(P_soft).reshape(-1,1)))
  #  cl_tx_SNR_soft = np.matmul(P_soft, rel_SNRs_tx)

    cl_rx_SNR_soft = cl_rx_sig_soft / (cl_rx_noise + 0) #
    cl_tx_SNR_soft = cl_tx_sig_soft / (cl_tx_noise + RxSums2)

    # capacities after Softnull
    cl_tx_cap_soft = bandwidth * np.log2(1+cl_tx_SNR_soft)
    cl_rx_cap_soft = bandwidth * np.log2(1+cl_rx_SNR_soft)
    

    # Rx after softnull
#    P_soft_cl = 

    print("Client rx power:")
    print(cl_rx_sig)
    print("Client tx power:")
    print(cl_tx_sig)
    print("Client rx after softnull")
    
    print("Client Rx capacity (weighted average):")
    print(cl_rx_cap)
    print("Client Tx capacity: (average)")
    print(cl_tx_cap)

    print("Client Rx SNR after soft:")
    print(cl_rx_cap_soft)
    print("Client Tx SNR: (average)")
    print(cl_tx_cap_soft)

#############
# MORE INFO #
#############

    print("Possible errors:")
    print(np.where(rel_CSI==0,["CSI!"],["none"]))
    print(np.where(rel_rssi==0,["RSSI"],["none"]))
    print(np.where(rel_offs==0,["OFFS"],["none"]))

    print("Number of actual recorded valid CSIs:")
    print(nonzero_count)
    print("Number of relevant CSIs:")
    rel_nonzero_count = np.take(np.take(nonzero_count, tx_antennas, 0), rx_antennas, 1)
    print(rel_nonzero_count)


###############
# MORE GRAPHS #
###############

    # relevant CSI graphed only
    X = [x.real for x in rel_nonzero_CSI]
    Y = [x.imag for x in rel_nonzero_CSI]
#    Z = zeros(rel_nonzero_CSI.shape)
#    for i in range(len(X)):
#        for j in range(len(Y)):
#            Z[i][j] = len(Y)*i + j
#    plt.scatter(X,Y,c=Z)
    plt.scatter(X,Y)
    print("Showing: relevant nonzero median CSI values, colored over time")
    if final_plots:
        plt.show()
    plt.savefig(foldstr+teststr+"rel_CSI_"+timestr)

    '''
    # compare old vs softnull Rx sums by antenna pair
    X = RxSums
    Y = RxSums2
    plt.scatter(X,Y)
    print("Showing: Softnull vs old total SI by Rx antenna")
    if final_plots:
        plt.show()
    plt.savefig(foldstr+teststr+"soft_change_CSI_"+timestr)
    '''


###########
# CLEANUP #
###########

    # close streams


################
# COMMAND LINE #
################

def main():
    parser = OptionParser()
#    parser.add_option("--bnodes", type="string", dest="bnodes", help="file name containing serials on the base station", default="bs_serials.txt")
#    parser.add_option("--cnodes", type="string", dest="cnodes", help="file name containing serials to be used as clients", default="client_serials.txt")
#    parser.add_option("--hub", type="string", dest="hub", help="Hub node", default="")
    parser.add_option("--ref-ant", type="int", dest="ref_ant", help="Calibration reference antenna", default=0)
#    parser.add_option("--ampl", type="float", dest="ampl", help="Amplitude coefficient for downCal/upCal", default=5.0)
#    parser.add_option("--rate", type="float", dest="rate", help="Tx sample rate", default=5e6)
    parser.add_option("--freq", type="float", dest="freq", help="Optional Tx freq (Hz)", default=3.6e9)
#    parser.add_option("--txgain", type="float", dest="txgain", help="Optional Tx gain (dB)", default=40.0)
#    parser.add_option("--rxgain", type="float", dest="rxgain", help="Optional Rx gain (dB)", default=20.0)
#    parser.add_option("--bw", type="float", dest="bw", help="Optional Tx filter bw (Hz)", default=10e6)
    parser.add_option("--cp", action="store_true", dest="cp", help="adds cyclic prefix to tx symbols", default=False)
#    parser.add_option("--wait-trigger", action="store_true", dest="wait_trigger", help="wait for a trigger to start a frame",default=False)
    parser.add_option("--numSamps", type="int", dest="numSamps", help="Number of samples in Symbol", default=400)
    parser.add_option("--prefix-length", type="int", dest="prefix_length", help="prefix padding length for beacon and pilot", default=82)
    parser.add_option("--postfix-length", type="int", dest="postfix_length", help="postfix padding length for beacon and pilot", default=68)
#    parser.add_option("--tx-advance", type="int", dest="tx_advance", help="symbol advance for tx", default=2)
    parser.add_option("--both-channels", action="store_true", dest="both_channels", help="transmit from both channels",default=False)
    parser.add_option("--corr-threshold", type="int", dest="threshold", help="Correlator Threshold Value", default=1)
#    parser.add_option("--use-trig", action="store_true", dest="use_trig", help="uses chain triggers for synchronization",default=False)
#    parser.add_option("--recip-cal", action="store_true", dest="recip_cal", help="perform reciprocity calibration procedure",default=False)
#    parser.add_option("--sample-cal", action="store_true", dest="samp_cal", help="perform sample offset calibration",default=False)
    parser.add_option("--plotter", action="store_true", dest="plotter", help="continuously plots all signals and stats",default=False)

    parser.add_option("--iter", type="int", dest="iterations", help="number of times to run", default=1)
    parser.add_option("--show-first-plots", action="store_true", dest="first_plots", help="show first summary plots after data collection is finished",default=False)
    parser.add_option("--show-final-plots", action="store_true", dest="final_plots", help="show last summary plots after data collection is finished",default=False)

    # lists of relevant antennas
    parser.add_option("--tx-antennas", type="string", dest="tx_antennas", help="file name containing antenna indices on the base station in the Tx partition", default="tx_antennas.txt")
    parser.add_option("--rx-antennas", type="string", dest="rx_antennas", help="file name containing antenna indices on the base station in the Rx partition", default="rx_antennas.txt")
    parser.add_option("--cl-antennas", type="string", dest="cl_antennas", help="file name containing antenna indices on the base station acting as client antennas (instead of listing a board as a client board)", default="client_antennas.txt")

    # number of effective antennas to reduce to
    parser.add_option("--effective", type="int", dest="effective", help="Number of effective antennas to reduce down to for Softnull (D_tx)", default=6)

    # file name
    parser.add_option("--label", type="string", dest="teststr", help="label for filename", default="TEST")

    # angles; phi, theta (degrees)
    parser.add_option("--phi", type="float", dest="phi", help="angle phi, clockwise around normal, 0 is from Tx to Rx, in degrees", default=45)
    parser.add_option("--theta", type="float", dest="theta", help="angle theta, away from normal, 0 is normal from array, in degrees", default=45)

    (options, args) = parser.parse_args()

    '''
    # get lists of bs and client board serials
    bserials = []
    with open(options.bnodes, "r") as f:
        for line in f.read().split("\n"):
            if line == "":
                continue
            if line[0] != '#':
                bserials.append(line)
            else:
                continue      

    cserials = []
    with open(options.cnodes, "r") as f:
        for line in f.read().split("\n"):
            if line == "":
                continue
            if line[0] != '#':
                cserials.append(line)
            else:
                continue      
    '''


    # get lists of tx, rx, and client antenna indices on BS
    tx_indices = []
    with open(options.tx_antennas, "r") as f:
        for line in f.read().split("\n"):
            if line == "":
                continue
            if line[0] != '#':
                tx_indices.append(line)
            else:
                continue      

    print(tx_indices)

    rx_indices = []
    with open(options.rx_antennas, "r") as f:
        for line in f.read().split("\n"):
            if line == "":
                continue
            if line[0] != '#':
                rx_indices.append(line)
            else:
                continue      

    cl_indices = []
    with open(options.cl_antennas, "r") as f:
        for line in f.read().split("\n"):
            if line == "":
                continue
            if line[0] != '#':
                cl_indices.append(line)
            else:
                continue      

    init(
#	hub=options.hub,
#	bnodes=bserials,
#	cnodes=cserials,
	ref_ant=options.ref_ant,
#	ampl=options.ampl,
#	rate=options.rate,
	freq=options.freq,
#	txgain=options.txgain,
#	rxgain=options.rxgain,
        cp=options.cp,
#	wait_trigger=options.wait_trigger,
	numSamps=options.numSamps,
        prefix_length=options.prefix_length,
        postfix_length=options.postfix_length,
#        tx_advance=options.tx_advance,
        both_channels=options.both_channels,
        threshold=options.threshold,
#        use_trig=options.use_trig,
#        recip_cal=options.recip_cal,
#        samp_cal=options.samp_cal,
	plotter=options.plotter,

        iterations=options.iterations,
        final_plots=options.final_plots,
        first_plots=options.first_plots,
        tx_antennas=tx_indices,
        rx_antennas=rx_indices,
        cl_antennas=cl_indices,
        effective=options.effective,
        teststr=options.teststr,
#        save_raw=options.save_raw

        phi=options.phi,
        theta=options.theta
    )

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
#    except Exception as e:
#	print e
#	exit()

