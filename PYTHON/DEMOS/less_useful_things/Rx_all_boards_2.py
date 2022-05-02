#!/usr/bin/python
"""
"""

import sys
sys.path.append('../IrisUtils/')
sys.path.append('../IrisUtils/data_in/')

import SoapySDR
import numpy as np
import time
import datetime
import os
import math
import signal
import threading
#import matplotlib
#import matplotlib.pyplot as plt
import collections
import logging
import pdb
from SoapySDR import *              # SOAPY_SDR_ constants
from optparse import OptionParser
#from matplotlib import animation
from data_recorder2 import *
from find_lts import *
from digital_rssi import *
from bandpower import *
from file_rdwr import *
from fft_power import *
from macros import *
from init_fncs import *
from MyFuncAnimation import *

#########################################
#            Global Parameters          #
#########################################
sdrs = []
rxStreams = []
recorder = None
FIG_LEN = 16384   
Rate = 5e6
fft_size = 2**12  # 1024
numBufferSamps = 1000
rssiPwrBuffer = collections.deque(maxlen=numBufferSamps)
timePwrBuffer = collections.deque(maxlen=numBufferSamps)
freqPwrBuffer = collections.deque(maxlen=numBufferSamps)
noisPwrBuffer = collections.deque(maxlen=numBufferSamps)
rssiPwrBuffer_fpga = collections.deque(maxlen=numBufferSamps)
frameCounter = 0

num_samps_circ_buff = 10
rssi_circ_buff = np.zeros(num_samps_circ_buff)
pwr_circ_buff = np.zeros(num_samps_circ_buff)

########################################
#               LOGGER                 #
########################################
# SOAPY_SDR_FATAL    = 1, //!< A fatal error. The application will most likely terminate. This is the highest priority.
# SOAPY_SDR_CRITICAL = 2, //!< A critical error. The application might not be able to continue running successfully.
# SOAPY_SDR_ERROR    = 3, //!< Error.An operation didn't complete successfully, but application as a whole not affected.
# SOAPY_SDR_WARNING  = 4, //!< A warning. An operation completed with an unexpected result.
# SOAPY_SDR_NOTICE   = 5, //!< A notice, which is an information with just a higher priority.
# SOAPY_SDR_INFO     = 6, //!< An informational message, usually denoting the successful completion of an operation.
# SOAPY_SDR_DEBUG    = 7, //!< A debugging message.
# SOAPY_SDR_TRACE    = 8, //!< A tracing message. This is the lowest priority.
# SOAPY_SDR_SSI      = 9, //!< Streaming status indicators such as "U" (underflow) and "O" (overflow).
logLevel = 3         # 4:WARNING, 6:WARNING+INFO, 7:WARNING+INFO+DEBUG...
SoapySDR.SoapySDR_setLogLevel(logLevel)
logging.basicConfig(filename='./data_out/debug_SISO_RX.log',
                    level=logging.DEBUG,
                    format='[%(levelname)s] (%(threadName)-10s) %(asctime)s %(message)s',)


#########################################
#              Functions                #
#########################################
def init():
    return


def rxsamples_app(srl, freq, gain, num_samps, recorder, agc_en, wait_trigger, iter):
    """
    Initialize IRIS parameters and animation kick-off
    """

    # Global declarations
    global sdrs, rxStreams, freqScale, Rate, numSdrs

    # Instantiate devices
#    sdr = SoapySDR.Device(dict(serial=srl))
    sdrs = [SoapySDR.Device(dict(driver='iris',serial=serial)) for serial in srl]
    info = sdrs[0].getHardwareInfo()
    print("First board info:")
    print(info)

    numSdrs = len(sdrs)

    # Set gains to very high value if AGC enabled (AGC only supports CBRS RF frontend at the moment).
    if agc_en and "CBRS" in info["frontend"]:
        gain = 100
        rssi_target_idx = 20
        agc_init(sdr, rssi_target_idx)
    else:
        # Make sure AGC is disabled if any of the previous checks fails
        agc_en = 0

    # Set params on both channels (both RF chains)
    for i, sdr in enumerate(sdrs):
        for ch in [0, 1]:
            sdr.setBandwidth(SOAPY_SDR_RX, ch, 2.5*Rate)
            sdr.setBandwidth(SOAPY_SDR_TX, ch, 2.5*Rate)
            sdr.setFrequency(SOAPY_SDR_RX, ch, freq)
            sdr.setSampleRate(SOAPY_SDR_RX, ch, Rate)
            sdr.setFrequency(SOAPY_SDR_TX, ch, freq)
            sdr.setSampleRate(SOAPY_SDR_TX, ch, Rate)
            sdr.setAntenna(SOAPY_SDR_RX, ch, "TRX")
            sdr.setDCOffsetMode(SOAPY_SDR_RX, ch, True)

            if "CBRS" in info["frontend"]:
                sdr.setGain(SOAPY_SDR_RX, ch, gain)
            else:
                # No CBRS board gains, only changing LMS7 gains
                sdr.setGain(SOAPY_SDR_RX, ch, "LNA", gain)  # [0:1:30]
                sdr.setGain(SOAPY_SDR_RX, ch, "TIA", 0)     # [0, 3, 9, 12]
                sdr.setGain(SOAPY_SDR_RX, ch, "PGA", -10)   # [-12:1:19]

        sdr.writeRegister("RFCORE", 120, 0)

        print("Number of Samples %d " % num_samps)
        print("Frequency has been set to %f" % sdr.getFrequency(SOAPY_SDR_RX, 0))

        # Setup RX stream
        rxStreams.append(sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1]))

        # RSSI read setup
        setUpDigitalRssiMode(sdr)




    testTime = [0] * len(sdrs)

    for i, sdr in enumerate(sdrs):
        # debug
        testTime[i]=sdr.getHardwareTime()
    for i, sdr in enumerate(sdrs):
        print("(before sync) board", i, testTime)



    # Sync timestamps with trigger
    sdrs[0].writeSetting('SYNC_DELAYS', "")
    for i, sdr in enumerate(sdrs):
        sdr.setHardwareTime(0, "TRIGGER")

        # debug
    for i, sdr in enumerate(sdrs):
        testTime[i]=sdr.getHardwareTime()
    for i, sdr in enumerate(sdrs):
        print("(before trigger) board", i, testTime)

    sdrs[0].writeSetting("TRIGGER_GEN", "")


        # debug
    for i, sdr in enumerate(sdrs):
        testTime[i]=sdr.getHardwareTime()

    for i, sdr in enumerate(sdrs):
        print("(after trigger) board", i, testTime)



    iterations = iter
    frames = 0
    if iter == 0:
        while 1:
            animate(num_samps, num_samps, recorder, agc_en, wait_trigger, "")
            frames += 1

    else:
        while frames < iterations:
            animate(num_samps, num_samps, recorder, agc_en, wait_trigger, "")
            frames += 1


def animate(i, num_samps, recorder, agc_en, wait_trigger, info):
    global sdrs, rxStreams, freqScale, sampsRx, frameCounter, fft_size, Rate, num_samps_circ_buff, rssi_circ_buff, pwr_circ_buff

    # Trigger AGC
    if agc_en:
        if frameCounter == 10:
            print(" *** ENABLE AGC/PKT DETECT *** ")
            sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_PKT_DET_ENABLE, 1)
            sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_AGC_ENABLE_FLAG, 1)
        if frameCounter == 20:
            print(" *** DONE WITH PREVIOUS FRAME, ASSUME NEW FRAME INCOMING *** ")
            sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_PKT_DET_NEW_FRAME, 1)
            sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_PKT_DET_NEW_FRAME, 0)  # DO NOT REMOVE ME!! Disable right away

    # Read samples into this buffer
    sampsRx = np.zeros((numSdrs, 2, num_samps), np.complex64)
#    timesRx = np.zeros((numSdrs, 

    flags = SOAPY_SDR_END_BURST
    if wait_trigger: flags |= SOAPY_SDR_WAIT_TRIGGER

    # Activate streams
    for i, sdr in enumerate(sdrs):
        sdr.activateStream(rxStreams[i],
            flags,    # flags
            0,                      # timeNs (dont care unless using SOAPY_SDR_HAS_TIME)
            sampsRx[0][0].size)             # numElems - this is the burst size

        test=sdr.readStream(rxStreams[i], [sampsRx[i][0], sampsRx[i][1]], 0)
        print("board:", i, test.timeNs)

    if wait_trigger:
        time.sleep(0.1)
        sdrs[0].writeSetting("TRIGGER_GEN", "")
        time.sleep(0.05)

    for i, sdr in enumerate(sdrs):
        sr = sdr.readStream(rxStreams[i], [sampsRx[i][0], sampsRx[i][1]], sampsRx[0][0].size)
        if sr.ret != sampsRx[0][0].size:
            print("Read RX burst of %d, requested %d" % (sr.ret, sampsRx[0][0].size))

        # DC removal
        for ch in [0, 1]:
            sampsRx[i][ch] -= np.mean(sampsRx[i][ch])

        # Find LTS peaks (in case LTSs were sent)
        lts_thresh = 0.8
        a, b, peaks0 = find_lts(sampsRx[i][0], thresh=lts_thresh)
        c, d, peaks1 = find_lts(sampsRx[i][1], thresh=lts_thresh)

        #print("Highest LTS peak, board", i, "channel 0:", a, "\nAll peaks:", b, "\nChannel 1:", c, "\nAll:", d)
        if (a != []) or (c != []):
            print("Found LTS:", a, c)

    # If recording samples
    if recorder is not None: 
        frame = sampsRx
        recorder.save_frame(frame, sr.timeNs) # sr.timeNs currently is not being collected by all boards

    # Store received samples in binary file (second method of storage)
    write_to_file('./data_out/rxsamps', sampsRx)

    print("Placeholder!")


#########################################
#                  Main                 #
#########################################
def main():
    parser = OptionParser()
    parser.add_option("--label", type="string", dest="label", help="Label for recorded file name. Default: 'TEST'", default="TEST")
    parser.add_option("--rxgain", type="float", dest="rxgain", help="RX GAIN (dB). Default: 65.", default=65.0)  # See documentation at top of file for info on gain range
    parser.add_option("--latitude", type="float", dest="latitude", help="Latitude. Default: 0.", default=0.0)
    parser.add_option("--longitude", type="float", dest="longitude", help="Longitude. Default: 0.", default=0.0)
    parser.add_option("--elevation", type="float", dest="elevation", help="Elevation. Defalut: 0.", default=0.0)
    parser.add_option("--freq", type="float", dest="freq", help="Rx freq (Hz). Default: 2.5e9", default=2.5e9)
    parser.add_option("--numSamps", type="int", dest="numSamps", help="Num samples to receive per frame. Default: 16384.", default=16384)
    parser.add_option("--serials", type="string", dest="serials", help="Serial numbers of the devices. Default: './rx_serials.txt'.", default="./rx_serials.txt")
    parser.add_option("--rxMode", type="string", dest="rxMode", help="RX Mode, Options:BASIC/REC/REPLAY", default="REC")
    parser.add_option("--AGCen", type="int", dest="AGCen", help="Enable AGC Flag. Options:0/1. Default: 0.", default=0)
    parser.add_option("--wait-trigger", action="store_true", dest="wait_trigger", help="Use this flag to wait for a trigger to start a frame.",default=False)
    parser.add_option("--iterations", type="int", dest="iterations", help="Number of times to run. Leave blank for continuous.", default=0)
    (options, args) = parser.parse_args()

    # Verify RX Mode
    if not (options.rxMode == "BASIC" or options.rxMode == "REC"):
        raise AssertionError("Invalid RX Mode")

    # parse serials
    bserials=[]
    with open(options.serials, "r") as f:
        for line in f.read().split():
            if line[0] != '#':
                bserials.append(line)
            else:
                continue

    # Current time
    now = datetime.datetime.now()
    print(now)

    # Display parameters
    print("\n")
    print("========== RX PARAMETERS =========")
    print("Receiving signal on boards {}".format(bserials))
    print("Sample Rate (sps): {}".format(Rate))
    print("Rx Gain (dB): {}".format(options.rxgain))
    print("Frequency (Hz): {}".format(options.freq))
    print("RX Mode: {}".format(options.rxMode))
    print("Number of Samples: {}".format(options.numSamps))
    if options.AGCen: print("** AGC ENABLED **")
    print("==================================")
    print("\n")

    # If recording file
    recorder = None
    if options.rxMode == "REC":
        filename = "./data_out/rx" + '%1.3f' % (float(options.freq)/1e9) + 'GHz_' + options.label + '.hdf5'
        recorder = DataRecorder(tag=options.label,
                                serial=bserials,
                                freq=options.freq,
                                numSamps=options.numSamps,
                                numBoards=len(bserials),
                                latitude=options.latitude,
                                longitude=options.longitude,
                                elevation=options.elevation,
##################################################
                                lna2=0, attn=0, lna=0, pga=0, tia=0, lna1=0)

        recorder.init_h5file(filename=filename)

    # IF REPLAY
    if options.rxMode == "REPLAY":
        pass

    else:
        rxsamples_app(
            srl=bserials,
            freq=options.freq,
            gain=options.rxgain,
            num_samps=options.numSamps,
            recorder=recorder,
            agc_en=options.AGCen,
            wait_trigger=options.wait_trigger,
            iter=options.iterations
        )


if __name__ == '__main__': 
    main()
