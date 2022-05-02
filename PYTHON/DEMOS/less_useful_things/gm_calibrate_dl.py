import sys
sys.path.append('../IrisUtils/')

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
#import h5py

from time import sleep
import datetime


plt.style.use('ggplot')  # customize your plots style

# debiggung - check which matplot backend is in use
print(matplotlib.get_backend())



RF_RST_REG = 48
CORR_CONF = 60
CORR_RST = 64
CORR_THRESHOLD = 92
TDD_CONF_REG = 120
SCH_ADDR_REG = 136
SCH_MODE_REG = 140

running = True

def signal_handler(signum, frame):
    global running
    running = False

def init(hub, bnodes, cnodes, ref_ant, ampl, rate, freq, txgain, rxgain, cp, \
        wait_trigger, numSamps, prefix_length, postfix_length, tx_advance, \
        both_channels, threshold, use_trig, samp_cal, recip_cal, plotter, iterations):

    iterations = iterations

    R = len(bnodes)
    ant = 2 if both_channels else 1
    M = R * ant
    K = len(cnodes)
    if not recip_cal and K == 0:
        print("Either specify a client node or enable --recip-cal")
    else:
        print("(R,M,K) = (%d,%d,%d)"%(R,M,K))

    if hub != "": hub_dev = SoapySDR.Device(dict(driver="remote", serial = hub)) # device that triggers bnodes and ref_node
    bsdrs = [SoapySDR.Device(dict(driver="iris", serial = serial)) for serial in bnodes] # base station sdrs
    csdrs = [SoapySDR.Device(dict(driver="iris", serial = serial)) for serial in cnodes] # client sdrs

    # assume trig_sdr is part of the master nodes
    trig_dev = None
    if hub != "":
        trig_dev = hub_dev
    else:
        trig_dev = bsdrs[0]

    #set params on both channels
    for sdr in bsdrs+csdrs:
        info = sdr.getHardwareInfo()
        print("%s settings on device" % (info["frontend"]))
        for ch in [0, 1]:
            sdr.setSampleRate(SOAPY_SDR_TX, ch, rate)
            sdr.setSampleRate(SOAPY_SDR_RX, ch, rate)
            # sdr.setFrequency(SOAPY_SDR_TX, ch, freq)
            # sdr.setFrequency(SOAPY_SDR_RX, ch, freq)
            sdr.setFrequency(SOAPY_SDR_TX, ch, 'RF', freq-.75*rate)
            sdr.setFrequency(SOAPY_SDR_RX, ch, 'RF', freq-.75*rate)
            sdr.setFrequency(SOAPY_SDR_TX, ch, 'BB', .75*rate)
            sdr.setFrequency(SOAPY_SDR_RX, ch, 'BB', .75*rate)
            if "CBRS" in info["frontend"]:
                sdr.setGain(SOAPY_SDR_TX, ch, 'ATTN', -6)  # {-18,-12,-6,0}
                sdr.setGain(SOAPY_SDR_TX, ch, 'PA2', 0)    # LO: [0|17], HI:[0|14]
            #sdr.setGain(SOAPY_SDR_TX, ch, 'IAMP', 12)       # [-12,12]
            sdr.setGain(SOAPY_SDR_TX, ch, 'IAMP', 0)       # [-12,12]
            sdr.setGain(SOAPY_SDR_TX, ch, 'PAD', txgain)   # [0,52]

            if "CBRS" in info["frontend"]:
                if freq < 3e9: sdr.setGain(SOAPY_SDR_RX, ch, 'ATTN', -18)   # {-18,-12,-6,0}
                else: sdr.setGain(SOAPY_SDR_RX, ch, 'ATTN', 0)   # {-18,-12,-6,0}
                sdr.setGain(SOAPY_SDR_RX, ch, 'LNA1', 30)  # [0,33]
                sdr.setGain(SOAPY_SDR_RX, ch, 'LNA2', 17)  # LO: [0|17], HI:[0|14]

            sdr.setGain(SOAPY_SDR_RX, ch, 'LNA', rxgain)   # [0,30]
            sdr.setGain(SOAPY_SDR_RX, ch, 'TIA', 0)       # [0,12]
            sdr.setGain(SOAPY_SDR_RX, ch, 'PGA', 0)       # [-12,19]

            sdr.setAntenna(SOAPY_SDR_RX, ch, "TRX")
            sdr.setDCOffsetMode(SOAPY_SDR_RX, ch, True)

            sdr.writeSetting(SOAPY_SDR_RX, ch, "CALIBRATE", 'SKLK')
            sdr.writeSetting(SOAPY_SDR_TX, ch, "CALIBRATE", 'SKLK')

            # Read initial gain settings
            readLNA = sdr.getGain(SOAPY_SDR_RX, 0, 'LNA')
            readTIA = sdr.getGain(SOAPY_SDR_RX, 0, 'TIA')
            readPGA = sdr.getGain(SOAPY_SDR_RX, 0, 'PGA')
            print("INITIAL GAIN - LNA: {}, \t TIA:{}, \t PGA:{}".format(readLNA, readTIA, readPGA))

        sdr.writeRegister("IRIS30", RF_RST_REG, (1 << 29) | 0x1)
        sdr.writeRegister("IRIS30", RF_RST_REG, (1 << 29))
        sdr.writeRegister("IRIS30", RF_RST_REG, 0)
        if not both_channels and info["serial"].find("RF3E") < 0:
            print("SPI TDD MODE")
            sdr.writeSetting("SPI_TDD_MODE", "SISO")
            
        if not both_channels:
            regRbbA = sdr.readRegister("LMS7IC",0x0115)
            regTbbA = sdr.readRegister("LMS7IC",0x0105)

            sdr.writeRegisters("LMS7_PROG_SPI",16, [0xa1150000 | 0xe])
            sdr.writeRegisters("LMS7_PROG_SPI",16, [0xa1050000 | regTbbA])
            sdr.writeRegisters("LMS7_PROG_SPI",32, [0xa1150000 | regRbbA])
            sdr.writeRegisters("LMS7_PROG_SPI",32, [0xa1050000 | 0x1e])

            sdr.writeSetting(SOAPY_SDR_RX, 1, 'ENABLE_CHANNEL', 'false')
            sdr.writeSetting(SOAPY_SDR_TX, 1, 'ENABLE_CHANNEL', 'false')


    trig_dev.writeSetting("SYNC_DELAYS", "")

    symSamp = numSamps + prefix_length + postfix_length
    print("numSamps = %d"%symSamp)

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

    print("lts check")
    print(ltsSym)
    print(len(ltsSym))
    print(lts_f)

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
    nRadios = 2*len(bsdrs) if both_channels else len(bsdrs)
    possible_dim.append(2**(np.ceil(np.log2(nRadios))))
    h_dim = min(possible_dim)
    hadamard_matrix = hadamard(h_dim)       #hadamard matrix : http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.hadamard.html
    beacon_weights = hadamard_matrix[0:nRadios, 0:nRadios]
    beacon_weights = beacon_weights.astype(np.uint32)

    # DEV: ueTrigTime = 153 (prefix_length=0), CBRS: ueTrigTime = 235 (prefix_length=82), tx_advance=prefix_length,
    # corr delay is 17 cycles
    rf_roundtrip = prefix_length + len(beacon) + postfix_length + 17 + postfix_length

    # Create streams
    txBsStreams = [sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0, 1]) for sdr in bsdrs]
    rxBsStreams = [sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1]) for sdr in bsdrs]
    rxClStreams = [sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1]) for sdr in csdrs]

#marker1

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


    # CalibCSI and CSI objects (../IrisUtils/csi_lib.py)
    if recip_cal: calibObj = CalibCSI(bsdrs, trig_dev, txBsStreams, rxBsStreams, ant, symSamp, wb_pilot_pad)
    if K > 0: csiObj = CSI(bsdrs, csdrs, trig_dev, txBsStreams, rxBsStreams, rxClStreams, not use_trig, ant, rate, symSamp, wb_pilot_pad, beacon1, beacon, None if use_trig else beacon_weights, rf_roundtrip)





    frame = 1000
    f = 0
    forward_sync = False
    reverse_sync = False
    offset = np.empty((M,M), np.int32) # bs array channels
    offset1 = np.empty((M,2*K), np.int32) # bs-client channels
    rssi = np.empty((M,M), np.float64)
    signal.signal(signal.SIGINT, signal_handler)

##########################
# DATA COLLECTION ARRAYS #
##########################
    # BS array antenna measurements
    avg_rssi = np.zeros((M,M), np.float64)

    # similar to offset (defunct, remove)
    timing = np.empty((M,M), np.int32)
    avg_timing = np.zeros((M,M),np.float64)

    # CSI values to/from client/bs
    clCSI = np.zeros((M,K,symSamp),np.complex64)
    bsCSI = np.zeros((M,K,symSamp),np.complex64)
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


###############
# FILE OUTPUT #
###############
    '''
#marker4 file open/create
    h5file = h5py.File("mytestfile.hdf5",'w')

    client = h5file.create_group("cl")
    base = h5file.create_group("bs")

    cl_avgs = client.create_group("avg")
    bs_avgs = base.create_group("avg")
    '''



#    cl_dset = cl_avgs.create_dataset("cl_mean", (1,clShape[0],clShape[1]), numpy.complex64)
    cl_temp_data = np.empty((M,K,iterations),dtype=np.complex64)




#################
    # MAIN LOOP #
#################

    startDT = datetime.datetime.now()
    print(startDT.strftime("%I:%M:%S %p"))

    while(running and f < iterations):
        print("\nenter WHILE\n")

        # ARRAY CALIBRATION
        if recip_cal:
            print("\nenter RECIP_CAL\n")

            calibObj.setup()

            print("ONE")

            _, timeDiff, sampsRx = calibObj.collect_calib_pilots()

            print("time diff:", timeDiff)

            print("TWO")

            for m in range(M):
                for p in range(M):
                    if m != p:
                        # save for plotting later
                        arr_imp[m][p][f] = np.copy(sampsRx[m][p])
                        # save for CSI calc
                        arrCSI[m][p] = np.copy(sampsRx[m][p])

#                        print("arrCSI before roll",m,p)
 #                       print(arrCSI[m][p])

                        # normalize, removing DC component
                        sampsRx[m][p] -= np.mean(sampsRx[m][p])

                        # ../IrisUtils/find_lts.py, a0 is end of lts pilot or [] if not found
                        a0, a1, peaks[m][p] = find_lts(sampsRx[m][p], thresh=lts_thresh, flip=True)

                        # offset of beginning of lts pilot
                        offset[m,p] = 0 if not a0 else a0 - len(ltsSym) + cp_len
                        offs[m][p][f] = offset[m,p]

                        # "rssi" calculation
                        H1 = np.fft.fft(sampsRx[m][p][offset[m,p]:offset[m,p]+fft_size], fft_size, 0)
                        H2 = np.fft.fft(sampsRx[m][p][offset[m,p]+fft_size:offset[m,p]+2*fft_size], fft_size, 0)
                        rssi[m,p] = np.mean((np.abs(H1)+np.abs(H2))/2)

                        # timing vs offset - ignoring ltsSym length (defunct; remove)
                        timing[m,p] = 0 if not a0 else a0

                        '''
                        # carrier frequency offset, reference DEMOS/SISO_OFDM.py and IrisUtils/ofdmtxrs.py
                        # get received LTS pilot
                        lts = np.copy(sampsRx[m][p][offset[m,p] : offset[m,p]+len(ltsSym)])

                        # direct copy; magic numbers?
#                        lts_1 = lts[-fft_size + -fft_offset + np.array(range(97, 161))]
 #                       lts_2 = lts[-fft_offset + np.array(range(97, 161))]
                        lts_1 = lts[np.array(range(fft_size))]
                        lts_2 = lts[fft_size + np.array(range(fft_size))]

                        # nothing >360 degrees
                        tmp = np.unwrap(np.angle(lts_2 * np.conjugate(lts_1)))
                        coarse_cfo_est = np.mean(tmp)
                        coarse_cfo_est = coarse_cfo_est / (2 * np.pi * fft_size)

                        # save for analysis/debugging
                        c_cfo_ests_arr[m][p][f] = coarse_cfo_est

                        # array CSI
                        # roll to reposition lts signal and pilot at beginning
                        arrCSI[m][p] = np.roll(arrCSI[m][p],-1*offset[m,p])
                        rollPilot = np.roll(wb_pilot_pad,-1*prefix_length)
                        #arrCSI[m][p] = np.roll(sapsRx_cfo,-1*offset[m,p])

                        # apply CFO correction to data
                        correction_vec = np.exp(-1j * 2 * np.pi * coarse_cfo_est * np.array(range(0,len(arrCSI[m][p]))))

                        arrCSInoCFO[m][p] = np.copy(arrCSI[m][p])

                        arrCSI[m][p] = arrCSI[m][p] * correction_vec
                        '''

                        '''
                        print("CFO")
                        print(coarse_cfo_est)
                        print("correction vec")
                        print(correction_vec)
                        '''

                        # pilot and received
                        arrCSI[m][p] = np.roll(arrCSI[m][p],-1*offset[m,p])
                        rollPilot = np.roll(wb_pilot_pad,-1*prefix_length)
                        P = np.copy(rollPilot)
                        Y = np.copy(arrCSI[m][p])

                        # make it 2d for linalg package
                        P.shape = (-1,1)
                        Y.shape = (-1,1)

                        # solve via least square estimation 2 different ways
                        # test loop
                        lse = 999 # offset
                        lse2 = 999 # try everything
                        indx = 0
                        
                        for i in range(len(Y)):
                            test = np.linalg.lstsq(P[0:numSamps],np.roll(Y,i)[0:numSamps])[0]
                            if test < lse2:
                                lse2, indx = test, i
                        
                        lse = np.linalg.lstsq(P[0:ofdm_len],Y[0:ofdm_len])[0]
                        H[m][p][f] = lse
                        H_2[m][p][f] = lse2
                        indxs[m][p][f] = indx

#                        print("lse, min indx,",m,p)
 #                       print(lse, indx)
                        


#                        print("roll samps")
 #                       print(arrCSI[m][p])
  #                      print("magnitude samps")
   #                     print(np.absolute(arrCSI[m][p]))
#                        print("roll pilot")
 #                       print(rollPilot)
  #                      print("magnitude pilot")
   #                     print(np.absolute(rollPilot))
    #                    print("CSI (lstsq)",m,p)
     #                   print(H[m][p][f])
                        print("CSI (avg after",f+1,":",m,p)
                        print(np.sum(H[m][p])/(f+1))



                    # when loop is measuring the antenna with itself
                    else:
                        sampsRx[m][p] = wb_pilot_pad

                        a0, a1, peaks[m][p] = find_lts(sampsRx[m][p], thresh=lts_thresh, flip=True)

                        offset[m,p] = 0 if not a0 else a0 - len(ltsSym) + cp_len
                        rssi[m,p] = 0

                        timing[m,p] = 0

                        '''
                        # array CSI
                        # roll for lts length
                        arrCSI[m][p] = np.roll(arrCSI[m][p],-1*offset[m,p])
                        # compare against pilot
                        arrCSI[m][p] -= wb_pilot_pad
                        '''
                        '''
                        # print CSI
                        print("array CSI (all):",m,p)
                        print(arrCSI[m][p])
                        print("mean CSI",m,p)
                        print(np.mean(arrCSI[m][p]))
                        '''
#                    print("(%d, %d): %d"%(m,p,offset[m][p]))
 #                   print(peaks[m][p])

            # end array measure loop

            # mean CSI for array (per iteration)
            # update average CSI (will be mean of mean of means)
            avg_arr = np.mean(arrCSI,2)
            avg_arrCSI = avg_arrCSI*(f/(f+1)) + avg_arr/(f+1)

#            print("offset matrix")
 #           print(offset)
 #           print("rssi matrix")
#            print(rssi)
  #          print("CSI matrix")
   #         print(avg_arr)
    #        print("")

            print("avg CSI after",f+1,"rounds:")
            print(avg_arrCSI)


            # SAMPLE CALIBRATION
            if samp_cal:
                print("\nenter SAMP_CAL\n")

                print("rx offsets at ref_ant %d"%ref_ant)
                print(offset[:,ref_ant])
                print("tx offsets at ref_ant %d"%ref_ant)
                print(offset[ref_ant,:])

                ref_offset = 1 if ref_ant == 0 else 0

                if not forward_sync: forward_sync = calibObj.sample_cal(offset[ref_ant,:], ref_ant)
                if forward_sync and not reverse_sync: reverse_sync = calibObj.sample_cal(offset[:,ref_ant], ref_ant, offset[ref_ant, ref_offset], False) 

                print("\nleave SAMP_CAL\n")

            calibObj.close()

            # PLOTTING
            if plotter:
                for m in range(M):
                    for p in range(M):
                        lines20[m][p].set_ydata(np.real(sampsRx[m][p]))
                        lines21[m][p].set_ydata(np.imag(sampsRx[m][p]))
                        lines22[m][p].set_data(offset[m,p], np.linspace(-1.0, 1.0, num=100))
                        #lines23[m][p].set_ydata(np.real(peaks[m][p][:symSamp])/np.max(peaks[m][p][:symSamp]))


            print("\nleave RECIP_CAL\n")
        # end if recip cal




        # CHANNELS BETWEEN BS AND CLIENTS
        if K > 0:
            print("\nenter IF K > 0\n")

            # get data
            csiObj.setup() #../IrisUtils/csi_lib.py
            bsRxSamps, clRxSamps = csiObj.collectCSI()
            #tempBsRx, tempClRx = csiObj.collectCSI()
            csiObj.close()

            # with 1 antenna, received data is (M, K, symSamp) in dimension

            '''
            # enter data into (M,K,iterations,symSamp) structure
            for m in range(M):
                for k in range(K):
                    bsRxSamps[m][k][f] = np.copy(tempBsRx[m][k])
                    clRxSamps[m][k][f] = np.copy(tempClRx[m][k])
            '''

            print("RxSamps shape:")
            RxShape = np.array(bsRxSamps).shape # used later in testing
            print(RxShape)

#            print("rate:",rate,"samples (symbols+padding):",symSamp)
 #           print("TIME:",1/rate*symSamp/1000000,"microseconds")


            for m in range(M):
                # BASE STATION Rx FROM CLIENT
                for k in range(K):
                    # save for plotting later
                    c2b_imp[m][k][f] = np.copy(bsRxSamps[m][k])
                    # for purposes of CSI calc., save for later
                    bsCSI[m][k] = np.copy(bsRxSamps[m][k])

                    # remove DC portion (normalize around 0)
                    bsRxSamps[m][k] -= np.mean(bsRxSamps[m][k])

                    # find offset where lts signal ends
                    a0, a1, peaks1[m][k+K] = find_lts(bsRxSamps[m][k], thresh=lts_thresh, flip=False)
                    # save offset (a0 is either value or empty)
                    offset1[m,k+K] = 0 if not a0 else a0 - len(ltsSym) + cp_len
                    offs_c2b[m][k][f] = offset1[m,k+K]

                    print("offset:",m,k,"(bs Rx)")
                    print(offset1[m,k+K])

                    '''
                    # coarse carrier frequency offset, as above
                    # get received LTS pilot, beginning at offset
                    #lts = np.copy(bsRxSamps[m][k][offset1[m,k+K] : offset1[m,k+K]+len(ltsSym)])
                    lts = np.copy(bsRxSamps[m][k][offset1[m,k+K] : offset1[m,k+K]+2*fft_size])

                    #size = len(lts) // 2 if len(lts) > 0 else 0
                    size = len(lts) - fft_size
                    
                    #lts1 = lts[np.array(range(fft_size))]
                    #lts2 = lts[fft_size + np.array(range(fft_size))]
                    
                    print(len(lts))
                    print(size)

                    lts1 = lts[np.array(range(size)).astype(int)]
                    lts2 = lts[fft_size + np.array(range(size)).astype(int)]

                    tmp = np.unwrap(np.angle(lts2 * np.conjugate(lts1)))

                    coarse_cfo_est = np.mean(tmp)
                    #coarse_cfo_est = coarse_cfo_est / (2*np.pi*fft_size)
                    coarse_cfo_est = coarse_cfo_est / (2*np.pi*size)

                    # save for analysis/debugging
                    c_cfo_ests_bsRx[m][k][f] = coarse_cfo_est

                    # roll to put LTS signal at beginning
                    rollPilot = np.roll(wb_pilot_pad, -1*prefix_length)
                    bsCSI[m][k] = np.roll(bsCSI[m][k],-1*offset1[m,k+K])

                    # apply coarse CFO correction
                    correction_vec = np.exp(-1j *2 *np.pi * coarse_cfo_est * np.array(range(0,len(bsCSI[m][k]))))
                    bsCSI[m][k] = bsCSI[m][k] * correction_vec
                    '''

                    # Pilot and Received
                    bsCSI[m][k] = np.roll(bsCSI[m][k],-1*offset1[m,k+K])
                    rollPilot = np.roll(wb_pilot_pad, -1*prefix_length)
                    P = np.copy(rollPilot)
                    Y = np.copy(bsCSI[m][k])
                    P.shape = (-1,1)
                    Y.shape = (-1,1)

                    # LSE
                    lse = np.linalg.lstsq(P[0:ofdm_len],Y[0:ofdm_len])[0]

                    H_CtoB[m][k][f] = lse


                    # update graph
                    if plotter: lines10[m][k+K].set_ydata(np.real(bsRxSamps[m][k]))
                    if plotter: lines11[m][k+K].set_ydata(np.imag(bsRxSamps[m][k]))
                    if plotter: lines12[m][k+K].set_data(offset1[m,k+K], np.linspace(-1.0, 1.0, num=100))

                # CLIENT Rx FROM BASE STATION
                for k in range(K):
                    # save for plotting later
                    b2c_imp[m][k][f] = np.copy(clRxSamps[m][k])

                    # similar logic as above
                    clCSI[m][k] = np.copy(clRxSamps[m][k])

                    clRxSamps[m][k] -= np.mean(clRxSamps[m][k])

                    a0, a1, peaks1[m][k] = find_lts(clRxSamps[m][k], thresh=lts_thresh, flip=False)
                    offset1[m,k] = 0 if not a0 else a0 - len(ltsSym) + cp_len
                    offs_b2c[m][k][f] = offset1[m,k]

                    print("offset:",m,k,"(cl Rx)")
                    print(offset1[m,k])

                    '''
                    #lts = np.copy(clRxSamps[m][k][offset1[m,k] : offset1[m,k]+len(ltsSym)])
                    lts = np.copy(clRxSamps[m][k][offset1[m,k+K] : offset1[m,k+K]+2*fft_size])

                    size = len(lts) - fft_size
                    
                    #lts1 = lts[np.array(range(fft_size))]
                    #lts2 = lts[fft_size + np.array(range(fft_size))]
                    
                    print(len(lts))
                    print(size)

                    lts1 = lts[np.array(range(size)).astype(int)]
                    lts2 = lts[fft_size + np.array(range(size)).astype(int)]
#                    lts1 = lts[np.array(range(fft_size))]
 #                   lts2 = lts[fft_size + np.array(range(fft_size))]

                    tmp = np.unwrap(np.angle(lts2*np.conjugate(lts1)))
                    coarse_cfo_est = np.mean(tmp)
  #                  coarse_cfo_est = coarse_cfo_est / (2*np.pi*fft_size)
                    coarse_cfo_est = coarse_cfo_est / (2*np.pi*size)

                    c_cfo_ests_clRx[m][k][f] = coarse_cfo_est

                    rollPilot = np.roll(wb_pilot_pad, -1*prefix_length)
                    clCSI[m][k] = np.roll(clCSI[m][k],-1*offset1[m,k])

                    correction_vec = np.exp(-1j*2*np.pi*coarse_cfo_est*np.array(range(0,len(clCSI[m][k]))))
                    clCSI[m][k] = clCSI[m][k] * correction_vec
                    '''

                    clCSI[m][k] = np.roll(clCSI[m][k],-1*offset1[m,k])
                    rollPilot = np.roll(wb_pilot_pad, -1*prefix_length)
                    P = np.copy(rollPilot)
                    Y = np.copy(clCSI[m][k])
                    P.shape = (-1,1)
                    Y.shape = (-1,1)

                    lse = np.linalg.lstsq(P[0:ofdm_len],Y[0:ofdm_len])[0]

                    H_BtoC[m][k][f] = lse

#                    print("cl CSI:",m,k)
 #                   print(clCSI[m][k])
#                    print("avg CSI:")
 #                   print(np.mean(clCSI[m][k]))
                    print("CSI:")
                    print(lse)

                    if plotter: lines10[m][k].set_ydata(np.real(clRxSamps[m][k]))
                    if plotter: lines11[m][k].set_ydata(np.imag(clRxSamps[m][k]))
                    if plotter: lines12[m][k].set_data(offset1[m,k], np.linspace(-1.0, 1.0, num=100))

            if plotter:
                fig1.canvas.draw()
                #fig1.show()
                fig1.savefig('testimg1.png')

            print(offset1)


            print("\nleave IF K > 0\n")

        '''
            ####
            if adjust:
                int cal_ref_idx = 0
                for j in range(M-1):
                    peaks = {}
                    for peak in offset1:
                        if peak in peaks:
                            peaks[peak] += 1
                        else:
                            peaks[peak] = 1
                        if peaks[peak] > max_freq[j]:
                            max_freq[j] = peaks[peak]
                            most_freq[j] = peak
                    if max_freq[j] < pass_thresh or most_freq[j] == 0:
                        print("adjust FAILED at board",j,"MostFreq:",most_freq[j])
                        break
                    print("success")
                    ############


           # end if adjust
        '''
        # end if K

        if recip_cal:
            if plotter:
                fig2.canvas.draw()
                #fig2.show()
                fig2.savefig('testimg2.png')

        print("frame %d"%f)
        print("")
        f += 1

        # frame stats
        print("rssi, \ntiming (end of lts), \noffset (beginning of lts minus cyclic),\n")
        print(rssi)
        print(timing)
        avg_rssi += rssi
        avg_timing += timing

        print(offset)

    # end while

    print("Start/end:")
    endDT = datetime.datetime.now()
    print(startDT.strftime("%I:%M:%S %p"))
    print(endDT.strftime("%I:%M:%S %p"))


    # debug analysis
    '''
    print("H:")
    print(H)
    print("rollaround indices:")
    print(indxs)
    print("offset:")
    print(offs)

    for i in range(M):
        for j in range(M):
            if j == i:
                continue
            for k in range(iterations):
                print("H, indx, offs")
                print(H[i][j][k])
                print(indxs[i][j][k])
                print(offs[i][j][k])
    '''

####################
# PLOTTING RESULTS #
####################

    # plot coloring options (will break if not (M,K = 2,1)
    lab = np.zeros((M,M,iterations))
    lab[0][1] = np.ones((iterations))
    lab[1][0] = 2*np.ones((iterations))
#    print(lab)

    count = np.zeros((M,M,iterations))
    count[0][1] = np.arange(iterations)
    count[1][0] = np.arange(iterations)

    lab2 = np.zeros((M,K,iterations))
    lab2[0][0] = np.ones((iterations))
    lab2[1][0] = 2*np.ones((iterations))

    count2 = np.zeros((M,K,iterations))
    count2[0][0] = np.arange(iterations)
    count2[1][0] = np.arange(iterations)


    foldstr = "./pics/"
    timestr = time.strftime("%m%d-%H%M")

    # bs array CSI estimates, using offset
    X = [x.real for x in H]
    Y = [x.imag for x in H]
    Z = count
    plt.scatter(X,Y,c=Z,label=Z)
    plt.show()
    plt.savefig(foldstr+"g1_"+timestr)

    X = [x.real for x in H]
    Y = [x.imag for x in H]
    Z = lab
    plt.scatter(X,Y,c=Z,label=Z)
    plt.show()
    plt.savefig(foldstr+"g2_"+timestr)


    X = [x.real for x in H]
    Y = [x.imag for x in H]
    Z = offs
    plt.scatter(X,Y,c=Z,label=Z)
    plt.show()
    plt.savefig(foldstr+"g5_"+timestr)

    # impulse response of array
    selections = range(0,iterations,5)

    order_lab = np.zeros((M,M,iterations),int)
    for s in selections:
        order_lab[0][1][s] = s

    f1 = plt.figure(1)

    plt.scatter([X[0][1][s] for s in selections],[Y[0][1][s] for s in selections],c=[order_lab[0][1][s] for s in selections])

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
        plt.subplot(len(selections),2,i+2)
        X = np.arange(symSamp)
        Y = np.abs(np.roll(arr_imp[0][1][s],-1*offs[0][1][s]))
        plt.bar(X,Y)

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
    '''
    plt.show()


    # ul/dl links CSI, offset
    X = [x.real for x in H_BtoC]
    Y = [y.imag for y in H_BtoC]
    Z = lab2
    plt.scatter(X,Y,c=Z,label=Z)
    plt.show()
    plt.savefig(foldstr+"g3_"+timestr)

    X = [x.real for x in H_CtoB]
    Y = [y.imag for y in H_CtoB]
    Z = lab2
    plt.scatter(X,Y,c=Z,label=Z)
    plt.show()
    plt.savefig(foldstr+"g4_"+timestr)


    # impulse response of bs from cl
    selections = range(0,50,5)

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

    plt.show()


    # multi-frame averages
    avg_rssi = avg_rssi/iterations
    avg_timing = avg_timing/iterations
    print("avg rssi")
    print(avg_rssi)
    print("avg timing")
    print(avg_timing)
    print("avg array CSI")
    print(avg_arrCSI)
    print("avg bs-client CSI")
    print(avg_clCSI)
    print("avg client-bs CSI")
    print(avg_bsCSI)


    # close streams
    if recip_cal:
        calibObj.close()

    if K > 0:
        csiObj.close()
        [csdrs[r].closeStream(rxClStreams[r]) for r in range(len(csdrs))]

    [bsdrs[r].closeStream(txBsStreams[r]) for r in range(len(bsdrs))]
    [bsdrs[r].closeStream(rxBsStreams[r]) for r in range(len(bsdrs))]



    ####
#    cl_dset = cl_avgs.create_dataset("cl_mean", cl_temp_data.shape, cl_temp_data.dtype, data=cl_temp_data)



def main():
    parser = OptionParser()
    parser.add_option("--bnodes", type="string", dest="bnodes", help="file name containing serials on the base station", default="bs_serials.txt")
    parser.add_option("--cnodes", type="string", dest="cnodes", help="file name containing serials to be used as clients", default="client_serials.txt")
    parser.add_option("--hub", type="string", dest="hub", help="Hub node", default="")
    parser.add_option("--ref-ant", type="int", dest="ref_ant", help="Calibration reference antenna", default=0)
    parser.add_option("--ampl", type="float", dest="ampl", help="Amplitude coefficient for downCal/upCal", default=5.0)
    parser.add_option("--rate", type="float", dest="rate", help="Tx sample rate", default=5e6)
    parser.add_option("--freq", type="float", dest="freq", help="Optional Tx freq (Hz)", default=3.6e9)
    parser.add_option("--txgain", type="float", dest="txgain", help="Optional Tx gain (dB)", default=40.0)
    parser.add_option("--rxgain", type="float", dest="rxgain", help="Optional Rx gain (dB)", default=20.0)
    parser.add_option("--bw", type="float", dest="bw", help="Optional Tx filter bw (Hz)", default=10e6)
    parser.add_option("--cp", action="store_true", dest="cp", help="adds cyclic prefix to tx symbols", default=False)
    parser.add_option("--wait-trigger", action="store_true", dest="wait_trigger", help="wait for a trigger to start a frame",default=False)
    parser.add_option("--numSamps", type="int", dest="numSamps", help="Number of samples in Symbol", default=400)
    parser.add_option("--prefix-length", type="int", dest="prefix_length", help="prefix padding length for beacon and pilot", default=82)
    parser.add_option("--postfix-length", type="int", dest="postfix_length", help="postfix padding length for beacon and pilot", default=68)
    parser.add_option("--tx-advance", type="int", dest="tx_advance", help="symbol advance for tx", default=2)
    parser.add_option("--both-channels", action="store_true", dest="both_channels", help="transmit from both channels",default=False)
    parser.add_option("--corr-threshold", type="int", dest="threshold", help="Correlator Threshold Value", default=1)
    parser.add_option("--use-trig", action="store_true", dest="use_trig", help="uses chain triggers for synchronization",default=False)
    parser.add_option("--recip-cal", action="store_true", dest="recip_cal", help="perform reciprocity calibration procedure",default=False)
    parser.add_option("--sample-cal", action="store_true", dest="samp_cal", help="perform sample offset calibration",default=False)
    parser.add_option("--plotter", action="store_true", dest="plotter", help="continuously plots all signals and stats",default=False)

    parser.add_option("--iter", type="int", dest="iterations", help="number of times to run", default=1)

    (options, args) = parser.parse_args()

    bserials = []
    with open(options.bnodes, "r") as f:
        for line in f.read().split():
            if line[0] != '#':
                bserials.append(line)
            else:
                continue      

    cserials = []
    with open(options.cnodes, "r") as f:
        for line in f.read().split():
            if line[0] != '#':
                cserials.append(line)
            else:
                continue      

    init(
	hub=options.hub,
	bnodes=bserials,
	cnodes=cserials,
	ref_ant=options.ref_ant,
	ampl=options.ampl,
	rate=options.rate,
	freq=options.freq,
	txgain=options.txgain,
	rxgain=options.rxgain,
        cp=options.cp,
	wait_trigger=options.wait_trigger,
	numSamps=options.numSamps,
        prefix_length=options.prefix_length,
        postfix_length=options.postfix_length,
        tx_advance=options.tx_advance,
        both_channels=options.both_channels,
        threshold=options.threshold,
        use_trig=options.use_trig,
        recip_cal=options.recip_cal,
        samp_cal=options.samp_cal,
	plotter=options.plotter,

        iterations=options.iterations

    )

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
#    except Exception as e:
#	print e
#	exit()

