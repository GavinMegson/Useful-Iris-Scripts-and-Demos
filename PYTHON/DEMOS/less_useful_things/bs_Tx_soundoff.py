import sys
sys.path.append('../IrisUtils/')
import numpy as np
from optparse import OptionParser
import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import time
from time import sleep
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

matplotlib.rcParams.update({'font.size': 10})
plt.style.use('ggplot')



running = True

def signal_handler(signum, frame):
    global running
    running = False

def init(bnodes, cnodes, ref_ant, ampl, rate, freq, txgain, rxgain, cp, \
         wait_trigger, numSamps, prefix_length, postfix_length, tx_advance, \
         both_channels, threshold, use_trig, samp_cal, recip_cal, plotter, iterations):

    R = len(bnodes)
    ant = 2 # if both_channels else 1
    M = ant * R
    K = len(cnodes)

    if not recip_cal and K == 0:
        print("Either specify a client node or enable --recip-cal")
    else:
        print("(R,M,K) = (%d,%d,%d)"%(R,M,K))

    # list devices
    csdrs = [SoapySDR.Device(dict(driver="iris", serial = serial)) for serial in cnodes] # client sdrs
    bsdrs = [SoapySDR.Device(dict(driver="iris", serial = serial)) for serial in bnodes] # base station sdrs

    # trigger device is first board
    trig_dev = bsdrs[0]

    # synchronize delays
    trig_dev.writeSetting("SYNC_DELAYS", "")

    #set params on both channels
    for sdr in bsdrs+csdrs:
        info = sdr.getHardwareInfo()
        print("%s settings on device" % (info["frontend"]))
        for ch in [0, 1]:
            sdr.setSampleRate(SOAPY_SDR_TX, ch, rate)
            sdr.setSampleRate(SOAPY_SDR_RX, ch, rate)

            sdr.setFrequency(SOAPY_SDR_TX, ch, 'RF', freq-.75*rate)
            sdr.setFrequency(SOAPY_SDR_RX, ch, 'RF', freq-.75*rate)

            sdr.setFrequency(SOAPY_SDR_TX, ch, 'BB', .75*rate)
            sdr.setFrequency(SOAPY_SDR_RX, ch, 'BB', .75*rate)

            if "CBRS" in info["frontend"]:
                sdr.setGain(SOAPY_SDR_TX, ch, 'ATTN', -6)  # {-18,-12,-6,0}
                sdr.setGain(SOAPY_SDR_TX, ch, 'PA2', 0)    # LO: [0|17], HI:[0|14]

            sdr.setGain(SOAPY_SDR_TX, ch, 'IAMP', 0)       # [-12,12]
            sdr.setGain(SOAPY_SDR_TX, ch, 'PAD', txgain)   # [0,52]

            if "CBRS" in info["frontend"]:
                if freq< 3e9: sdr.setGain(SOAPY_SDR_RX, ch, 'ATTN', -18) #{-18, -12, -6, 0}
                else: sdr.setGain(SOAPY_SDR_RX, ch, 'ATTN', 0)

                sdr.setGain(SOAPY_SDR_RX, ch, 'LNA1', 30)  # [0,33]
                sdr.setGain(SOAPY_SDR_RX, ch, 'LNA2', 17)  # LO: [0|17], HI:[0|14]

            sdr.setGain(SOAPY_SDR_RX, ch, 'LNA', rxgain)   # [0,30]
            sdr.setGain(SOAPY_SDR_RX, ch, 'TIA', 0)       # [0,12]
            sdr.setGain(SOAPY_SDR_RX, ch, 'PGA', 0)       # [-12,19]

            sdr.setAntenna(SOAPY_SDR_RX, ch, "TRX")
            sdr.setDCOffsetMode(SOAPY_SDR_RX, ch, True)

#            sdr.writeSetting(SOAPY_SDR_TX, ch, "CALIBRATE", 'SKLK')
 #           sdr.writeSetting(SOAPY_SDR_RX, ch, "CALIBRATE", 'SKLK')

            readLNA = sdr.getGain(SOAPY_SDR_RX, 0, 'LNA')
            readTIA = sdr.getGain(SOAPY_SDR_RX, 0, 'TIA')
            readPGA = sdr.getGain(SOAPY_SDR_RX, 0, 'PGA')
            print("INITIAL GAIN - LNA: {}, \t TIA:{}, \t PGA:{}".format(readLNA, readTIA, readPGA))

        # SW delays
        sdr.writeSetting("TX_SW_DELAY", str(30))

        #sdr.writeSetting("RESET_DATA_LOGIC", "")

    # set up streams
    '''
    bsTxStreams = np.zeros[R][2]
    bsRxStreams = np.zeros[R][2]
    clRxStreams = np.zeros[K][2]
    for i, sdr in enumerate(bsdrs):
        for ch in [0,1]:
            bsTxStreams[i][ch] = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CS16, [ch])
            bsRxStreams[i][ch] = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [ch])
    for i, sdr in enumerate(csdrs):
        for ch in [0,1]:
            clRxStreams[i][ch] = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_C@16, [ch])
    '''
    # create pilot sequence
    fft_size = 64
    cp_len = 32 if cp else 0
    ofdm_len = 2*fft_size + cp_len
    zeros = np.zeros((numSamps - ofdm_len)) # filling out difference in lts length vs sigsamps
    # ../IrisUtils/generate_training_seq
    ltsSym, lts_f = generate_training_seq(preamble_type='lts', cp=cp_len, upsample=1)
    # making sure the 
    pilot = np.concatenate((ltsSym, zeros)).astype(np.complex64)

    print("lts Symbol:", ltsSym)
    print("lts Symbol length:", len(ltsSym))
    print("lts_f:", lts_f)
    print("pilot length:", len(pilot))


    # base station CSI (does not work as advertized due to hardware issues)
    bs_csi(iterations, bsdrs, pilot, cp, prefix_length, postfix_length)

    # bs to client/client to bs CSI
    #cl_csi(iterations, bsdrs, csdrs, pilot, cp, prefix_length, postfix_length)



def tdd_cl_csi(iterations, bsdrs, csdrs, pilot, cp, prefix_length, postfix_length):
    # set up schedules
    for i, sdr in enumerate(bsdrs):
        sched_main = "PG"+''.join("G"*i*self.sdr_ant)+"T"*self.sdr_ant+''.join("G"*(self.num_ants-(i+1)*self.sdr_ant))+"G"+''.join("R"*(len(self.csdrs)))+"G"
        bconf = {"tdd_enabled": True, "frame_mode": "free_running", "symbol_size" : len(pilot)+prefix_length+postfix_length, "frames": [sched_main], "max_frame" : 1}


def cl_csi(iterations, bsdrs, csdrs, pilot, cp, prefix_length, postfix_length):
    # BS board which will have trigger generated
    first_bsdr = bsdrs[0]

    M = len(bsdrs)
    K = len(csdrs)

    # create final pilot signal by adding padding
    # prefix padding
    pad1 = np.array([0]*(prefix_length), np.complex64)
    # postfix padding
    pad2 = np.array([0]*(postfix_length), np.complex64)
    # final signal
    pilotPadded = np.concatenate([pad1, pilot, pad2]).astype(np.complex64)

    numSamps = len(pilotPadded)

    # CSI matrix TO clients FROM bs
    bs2clH = np.zeros((M, 2, K, 1, iterations), np.complex64)
    # FROM clients TO bs
    cl2bsH = np.zeros((K, 1, M, 2, iterations), np.complex64)

    # markers indicating a valid CSI was interpreted
    cl2bsV = np.zeros((K, 1, M, 2, iterations), bool)
    bs2clV = np.zeros((M, 2, K, 1, iterations), bool)

    # buffer for storing raw samples (cleared and reused every iteration)
    cl2bsBuff = np.zeros((K, 1, M, 2, numSamps), np.complex64)
    bs2clBuff = np.zeros((M, 2, K, 1, numSamps), np.complex64)

    # set up and create arrays of streams ( (M, 2) and (K, 1) arrays (lists of lists) of streams)
    clRxStreams = []
    bsRxStreams = []
    clTxStreams = []
    bsTxStreams = []
    # base station streams
    '''
    for i, sdr in enumerate(bsdrs):
        bsTxSdrStreams = []
        bsRxSdrStreams = []
        for ch in [0, 1]:
            bsTxSdrStreams.append(sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [ch]))
            bsRxSdrStreams.append(sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [ch]))
        bsRxStreams.append(bsRxSdrStreams)
        bsTxStreams.append(bsTxSdrStreams)
    '''
    bsTxSdrStreams = []
    for i, sdr in enumerate(bsdrs): 
        bsTxSdrStreams.append(sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0, 1]))
    # client streams
    for i, sdr in enumerate(csdrs):
        clTxSdrStreams = []
        clRxSdrStreams = []
        for ch in [0]: #for now, assume only one client channel
            clTxSdrStreams.append(sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [ch]))
            clRxSdrStreams.append(sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [ch]))
        clRxStreams.append(clRxSdrStreams)
        clTxStreams.append(clTxSdrStreams)

    # Tx/Rx flags: Send finite burst of samples, and wait to send until triggered
    clTxFlags = SOAPY_SDR_END_BURST | SOAPY_SDR_WAIT_TRIGGER
    clRxFlags = SOAPY_SDR_END_BURST | SOAPY_SDR_WAIT_TRIGGER
    bsTxFlags = SOAPY_SDR_END_BURST #| SOAPY_SDR_WAIT_TRIGGER
    bsRxFlags = SOAPY_SDR_END_BURST | SOAPY_SDR_WAIT_TRIGGER


    # Main loop
    for iter in range(iterations):
        # first, BS to client
        for j, ref_sdr in enumerate(bsdrs):
            for ref_ch in [0, 1]:
                # specify the transmitting (reference) antenna
                ref_ant = (j, ref_ch)

                '''# activate Rx streams at clients
                for i, sdr in enumerate(csdrs):
                    for ch in [0]:
                        rc = sdr.activateStream(clRxStreams[i][ch], clRxFlags, 0, numSamps)
                        if rc < 0:
                            print("problem activating client rx stream", i, ch)
                '''
                # activate tx stream
                rc = ref_sdr.activateStream(bsTxStreams[j])
                if rc < 0:
                    print("problem activating bs tx stream", j, ref_ch)

                # write to tx stream (wait for trigger flag currently not enabled)
                print("writing bs stream", ref_ant)
                if ref_ch == 0:
                    ret = ref_sdr.writeStream(bsTxStreams[j], [pilotPadded,[0]*len(pilotPadded)], numSamps, bsTxFlags)
                else:
                    ret = ref_sdr.writeStream(bsTxStreams[j], [[0]*len(pilotPadded),pilotPadded], numSamps, bsTxFlags)
                   
                if ret.ret < 0:
                    print("problem writing bs tx stream", j, ref_ch)

                # start transmission and activate streams
#                first_bsdr.writeSetting("TRIGGER_GEN", "")

                # read client streams
                '''
                for i, sdr in enumerate(csdrs):
                    for ch in [0]:
                        rtrncd = sdr.readStream(clRxStreams[i][ch], [bs2clBuff[j][ref_ch][i][ch]], numSamps)
                        print("read stream from", j, ref_ch, "at bs to", i, ch, " at cl. rtncd:", rtrncd)
                '''

                # deactivate streams
                print("deactivating streams")
                '''
                for i, sdr in enumerate(csdrs):
                    for ch in [0]:
                        sdr.deactivateStream(clRxStreams[i][ch])
                '''
                ref_sdr.deactivateStream(bsTxStreams[j])

                # calculate csi from ref_ant to each client
                '''
                print("calculating CSIs from bs to cl")
                for i, sdr in enumerate(csdrs):
                    for ch in [0]:

                        print("cl Rx Samps", i, ch)
                        print(bs2clBuff[j][ref_ch][i][ch])

                        # normalzie, removing DC component
                        #bsRxSamps[i][ch] -= np.mean(bsRxSamps[i][ch])

                        # /IrisUtils/find_lts.py, a0 is index of end of lts pilot or [] if not found
                        a0, a1, _ = find_lts(bs2clBuff[j][ref_ch][i][ch], cp=cp, lts_seq=pilot)

                        # determine if a lts signal was found
                        print("a0:", a0)
                        if not a0:
                            a0 = 0
                            val = False
                        else:
                            val = True
                        bs2clV[j][ref_ch][i][ch][iter] = val

                        # channel state information (via least square error)
                        if bs2clV[j][ref_ch][i][ch][iter]:
                            # rotate the received samples backward to align with pilot (a0 is END of pilot)
                            tempSamps = np.copy(np.roll(bs2clBuff[j][ref_ch][i][ch],-1*(a0-len(pilot))))
                            tempSamps.shape = (-1, 1)
                            # cut off the prefix/postfix, leaving only samples corresponding to pilot
                            tempSamps2 = tempSamps[:len(pilot)]

                            # compare pilot to received samples via least squares
                            tempPilot = np.copy(pilot)
                            tempPilot.shape = (-1, 1)

                            print("tempSamps, tempPilot", tempSamps2.shape, tempPilot.shape)

                            bs2clH[j][ref_ch][i][ch][iter], _, _, _ = np.linalg.lstsq(tempPilot, tempSamps2)

                            print("bs2clH[j][ref_ch][i][ch][iter]:", j, ref_ch, i, ch, iter, bs2clH[j][ref_ch][i][ch][iter], \
                                "valid:", bs2clV[j][ref_ch][i][ch][iter])

                        # clear rx buffer
                        bs2clBuff[j][ref_ch][i][ch].fill(0)
                '''


###

def bs_csi(iterations, bsdrs, pilot, cp, prefix_length, postfix_length):
    # board which will have trigger generated
    first_sdr = bsdrs[0]

    M = len(bsdrs)

    # create final pilot signal by adding padding
    # prefix padding
    pad1 = np.array([0]*(prefix_length), np.complex64)
    # postfix padding
    pad2 = np.array([0]*(postfix_length), np.complex64)
    # final signal
    pilotPadded = np.concatenate([pad1, pilot, pad2]).astype(np.complex64)
    pilotZeros = np.array([0]*(len(pilotPadded)))

    numSamps = len(pilotPadded)

    # write pilot signal to each TX RAM for each channel (not necessary for streaming mode)
#    for sdr in bsdrs:
 #       replay_addr = 0 # sometimes used to add multiple signals into one RAM block
  #      sdr.writeRegisters("TX_RAM_A", replay_addr, cfloat2uint32(pilotPadded, order='QI').tolist())
   #     sdr.writeRegisters("TX_RAM_B", replay_addr, cfloat2uint32(pilotPadded, order='QI').tolist())

    # CSI matrix (BSDRS/2 channels to BSDRS/2 channes, iterations)
    H = np.zeros((M, 2, M, 2, iterations), np.complex64)
    # valid markers
    valid = np.zeros((M, 2, M, 2, iterations), bool)

    # Rx received raw samples
    # reusing the array between samples:
    bsRxSamps = np.zeros((M, 2, numSamps), np.complex64)

    '''
    # set up and create arrays of streams
    rxStreams = []
    txStreams = []
    for i, sdr in enumerate(bsdrs):
        txSdrStreams = []
        rxSdrStreams = []
        for ch in [0, 1]:
            txSdrStreams.append(sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [ch]))
            rxSdrStreams.append(sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [ch]))
        rxStreams.append(rxSdrStreams)
        txStreams.append(txSdrStreams)
    '''

    # set up and create arrays of streams
    rxStreams = []
    txStreams = []
    for i, sdr in enumerate(bsdrs):
        txSdrStreams = (sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0,1]))
        rxSdrStreams = (sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0,1]))
        rxStreams.append(rxSdrStreams)
        txStreams.append(txSdrStreams)

    # send finite burst of samples, and wait to send until triggered
    txFlags = SOAPY_SDR_END_BURST #| SOAPY_SDR_WAIT_TRIGGER
    rxFlags = SOAPY_SDR_END_BURST | SOAPY_SDR_WAIT_TRIGGER


    # Main loop
    for iter in range(iterations):
        # for each antenna, do the whole process
        for j, ref_sdr in enumerate(bsdrs):
            for ref_ch in [0, 1]:
                # specify the transmitting (reference) antenna
                ref_ant = (j, ref_ch)

                # Make sure pilot signal is on correct antenna and zeros on the other
                pilot_chan = [pilotPadded, pilotZeros] if ref_ch == 0 else [pilotZeros, pilotPadded]

                # activate Rx streams
                '''
                for i, sdr in enumerate(bsdrs):
                    for ch in [0, 1]:
                        if (i, ch) != ref_ant:
                            rc = sdr.activateStream(rxStreams[i][ch], rxFlags, 0, numSamps)
#                            rc = sdr.activateStream(rxStreams[i][ch], 0, 0, numSamps)
                            if rc < 0:
                                print("problem activating rx stream", i, ch)
                '''

                # activate tx stream
                rc = ref_sdr.activateStream(txStreams[j])
                if rc < 0:
                    print("problem activating tx stream", j, ref_ch)

                # write to tx stream (wait for trigger flag not enabled)
                print("writing stream", ref_ant)
#                ret = ref_sdr.writeStream(txStreams[j][ref_ch], pilot_chan, numSamps, txFlags)
                ret = ref_sdr.writeStream(txStreams[j], pilot_chan, numSamps, txFlags)
                if ret.ret < 0:
                    print("problem writing tx stream", j, ref_ch)

                # start transmission and activate streams
                '''
                first_sdr.writeSetting("TRIGGER_GEN", "")

                sleep(0.1)
                '''

                # read streams on trigger
                '''
                for i, sdr in enumerate(bsdrs):
                    for ch in [0, 1]:
                        if (i, ch) != ref_ant:
                            print("reading stream from", j, ref_ch, "to", i, ch)
                            rtrncd = sdr.readStream(rxStreams[i][ch], [bsRxSamps[i][ch]], numSamps)
#                            rtrncd = sdr.readStream(rxStreams[i][ch], [bsRxSamps[i][ch]], numSamps, rxFlags)
                            print("^^read stream from", j, ref_ch, "to", i, ch, "read:", rtrncd, "samples")
                '''

                # deactivate streams
                print("deactivating streams")
                for i, sdr in enumerate(bsdrs):
                    for ch in [0, 1]:
                        if (i, ch) != ref_ant:
                            #sdr.deactivateStream(rxStreams[i][ch])
                            pass
                        else:
                            sdr.deactivateStream(txStreams[i])

                # calculate csi from ref_ant to each antenna
                '''
                print("calculating CSIs")
                for i, sdr in enumerate(bsdrs):
                    for ch in [0, 1]:

                        print("bsRxSamps", i, ch)
                        print(bsRxSamps[i][ch])

                        # normalzie, removing DC component
                        #bsRxSamps[i][ch] -= np.mean(bsRxSamps[i][ch])

                        # /IrisUtils/find_lts.py, a0 is index of end of lts pilot or [] if not found
                        a0, a1, _ = find_lts(bsRxSamps[i][ch], cp=cp, lts_seq=pilot)

                        # determine if a lts signal was found
                        print("a0:", a0)
                        if not a0:
                            a0 = 0
                            val = False
                        else:
                            val = True
                        valid[j][ref_ch][i][ch][iter] = val

                        # channel state information (via least square error)
                        if valid[j][ref_ch][i][ch][iter]:
                            # rotate the received samples backward to align with pilot (a0 is END of pilot)
                            tempSamps = np.copy(np.roll(bsRxSamps[i][ch],-1*(a0-len(pilot))))
                            tempSamps.shape = (-1, 1)
                            # cut off the prefix/postfix, leaving only samples corresponding to pilot
                            tempSamps2 = tempSamps[:len(pilot)]

                            # compare pilot to received samples via least squares
                            tempPilot = np.copy(pilot)
                            tempPilot.shape = (-1, 1)

                            print("tempSamps, tempPilot", tempSamps2.shape, tempPilot.shape)

                            H[j][ref_ch][i][ch][iter], _, _, _ = np.linalg.lstsq(tempPilot, tempSamps2)

                            print("H[j][ref_ch][i][ch][iter]:", j, ref_ch, i, ch, iter, H[j][ref_ch][i][ch][iter], valid[j][ref_ch][i][ch][iter])

                        # clear rx buffer
                        bsRxSamps[i][ch] = np.copy(np.zeros((numSamps)))
                '''

    # close streams
    for i, sdr in enumerate(bsdrs):
        sdr.closeStream(txStreams[i])
#            sdr.closeStream(rxStreams[i][ch])
    '''
    # summarize channel data
    for i in range(M):
        for j in [0, 1]:
            for k in range(M):
                for l in [0, 1]:
                    # print valid csi readings
                    print("readings from", i, j, "to", k, l, "channel:", H[i][j][k][l][valid[i][j][k][l][iter]])
                    print("number of valid readings:", sum(valid[i][j][k][l]), "out of", iterations)
    '''

def main():
    # input options
    parser = OptionParser()
    parser.add_option("--bnodes", type="string", dest="bnodes", help="file name containing serials on the base station, default bs_serials.txt", default="bs_serials.txt")
    parser.add_option("--cnodes", type="string", dest="cnodes", help="file name containing serials to be used as clients, default client_serials.txt", default="client_serials.txt")
    parser.add_option("--hub", type="string", dest="hub", help="Hub node, default none", default="")
    parser.add_option("--ref-ant", type="int", dest="ref_ant", help="Calibration reference antenna, default 0", default=0)
    parser.add_option("--ampl", type="float", dest="ampl", help="Amplitude coefficient for downCal/upCal, default 5.0", default=5.0)
    parser.add_option("--rate", type="float", dest="rate", help="Tx sample rate (Hz), default 5e6", default=5e6)
    parser.add_option("--freq", type="float", dest="freq", help="Tx freq (Hz), default 3.6e9", default=3.6e9)
    parser.add_option("--txgain", type="float", dest="txgain", help="Tx gain (dB), default 40", default=40.0)
    parser.add_option("--rxgain", type="float", dest="rxgain", help="Rx gain (dB), default 20", default=20.0)
    parser.add_option("--bw", type="float", dest="bw", help="Tx filter bw (Hz), default 10e6", default=10e6)
    parser.add_option("--cp", action="store_true", dest="cp", help="adds cyclic prefix to tx symbols", default=False)
    parser.add_option("--wait-trigger", action="store_true", dest="wait_trigger", help="wait for a trigger to start a frame, this is actually defunct as in input",default=False)
    parser.add_option("--numSamps", type="int", dest="numSamps", help="Number of Samples in each sent Symbol (not including prefix/postfix guards), default 400", default=400)
    parser.add_option("--prefix-length", type="int", dest="prefix_length", help="prefix padding length for beacon and pilot, default 82", default=82)
    parser.add_option("--postfix-length", type="int", dest="postfix_length", help="postfix padding length for beacon and pilot, default 68", default=68)
    parser.add_option("--tx-advance", type="int", dest="tx_advance", help="symbol advance for tx, default 2, DEFUNCT", default=2)
    parser.add_option("--both-channels", action="store_true", dest="both_channels", help="transmit from both channels, DEFUNCT",default=False)
    parser.add_option("--corr-threshold", type="int", dest="threshold", help="Correlator Threshold Value, default 0.9", default=0.9)
    parser.add_option("--use-trig", action="store_true", dest="use_trig", help="uses chain triggers for synchronization",default=False)
    parser.add_option("--recip-cal", action="store_true", dest="recip_cal", help="perform reciprocity calibration procedure, DEFUNCT",default=False)
    parser.add_option("--sample-cal", action="store_true", dest="samp_cal", help="perform sample offset calibration, DEFUNCT",default=False)
    parser.add_option("--plotter", action="store_true", dest="plotter", help="continuously plots all signals and stats, DEFUNCT",default=False)
    parser.add_option("--iter", type="int", dest="iterations", help="number of iterations to get CSI, default 1", default=1)

    (options, args) = parser.parse_args()

    # create bnodes and cnodes lists
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

    # initial function
    init(
#        hub=options.hub,
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
