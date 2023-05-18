#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vibha Viswanathan
Copyright 2019-23 Vibha Viswanathan. All rights reserved.
"""

import mne
from anlffr.helper import biosemi2mne as bs
import numpy as np
from scipy.io import savemat, loadmat
import os
import pylab as pl
from preprocess_EEG import preprocess_EEG
from anlffr.tfr import tfr_multitaper, plot_tfr
import pandas as pd
from mne.viz import plot_topomap
from mne.channels import make_standard_montage
from mne.channels.layout import _pol_to_cart, _cart_to_sph
from scipy.stats import binom
import matplotlib.pyplot as plt

results_path = '/results/' 
behavior_data_path = '/percent_correct_scores/'
EEG_data_path = '/EEG_data/'
subjects = ['S046','S051','S059','S150','S064', 'S160']
nsubs = len(subjects)
numStimPerCond = 175 # number of stimuli per condition per subject
fs_EEG = 4096.
numchans = 32  
nSentPerCond = numStimPerCond * nsubs
conds = [2,3] # SiB -2 dB SNR, SiSSN -5 dB SNR 
condNames = ['SiB','SiSSN']
nconds = len(conds)
all_electrodes = np.arange(numchans)
# parieto-occipital electrodes 
po_electrodes = np.asarray([9,10,11,12,13,14,15,16,17,18,19,20,21,22]) - 1 
# frontal electrodes 
f_electrodes = np.asarray([1,2,3,4,5,6,25,26,27,28,29,30,31]) - 1

def processEEG(subj,trigger,whichelectrodes):
    data_path = EEG_data_path + subj + '/'
    ftags = os.listdir(data_path) 
    ftags.sort() 
    desiredSpeechLen = 2.5 # duration of speech stimuli in seconds
    reject = None
    # All times below are relative to audio stimulus onset
    t_speech_starts = 1.25 # time (in seconds) when the speech sentence starts
    listenNow_cue = -1 # time (in seconds) when the "listen now" visual cue is given
    t_min = listenNow_cue - 0.5 # to include baseline before "listen now" cue
    t_max = t_speech_starts + desiredSpeechLen
    numSamplesEEG = (t_max-t_min)*fs_EEG + 1
    cond_EEG = np.zeros((numStimPerCond,len(whichelectrodes),int(numSamplesEEG)))
    evStart = 0
    for k in np.arange(len(ftags)):
        fname_EEG = data_path + ftags[k]
        excludeChans = [] 
        for excChanNum in list(range(32)):
            excludeChans = excludeChans + ['B' + str(excChanNum+1)]
            excludeChans = excludeChans + ['C' + str(excChanNum+1)]
            excludeChans = excludeChans + ['D' + str(excChanNum+1)]
        #end
        raw_EEG, eves_EEG = bs.importbdf(fname_EEG, nchans=34,
                                 refchans=['EXG1','EXG2'], mask=255,
                                 exclude=excludeChans)
        raw_EEG, eves_EEG = preprocess_EEG(raw_EEG, eves_EEG, subj)
        if (subj == 'S046'):
            bad_trial_S046 = np.where(eves_EEG[:,0] == 2173803)
            eves_EEG[bad_trial_S046,2] = 500 # stimulus computer crashed
        #end
        epochs_EEG = mne.Epochs(raw_EEG, eves_EEG, trigger,
                                tmin=t_min, tmax=t_max,
                                proj=False, baseline=None,
                                picks=whichelectrodes, 
                                reject=reject)
        numEvents = (epochs_EEG.events).shape[0]
        cond_EEG[evStart:(evStart+numEvents),:,:] = epochs_EEG.get_data().squeeze()  # 32 channels
        evStart = evStart + numEvents
    # end
    return cond_EEG
#end

def computeSingletrialParietoOccipitalAlphaPower(subj):
    sub_results_path = results_path + subj + '/processed_EEG/'
    # Look up table for condition #
    # triggers: 
    # 1: SSN -2dB SNR
    # 2: SIB 4dB TMR (-2 dB SNR)
    # 3: SSN -5dB SNR
    # 4: SSN -8dB SNR
    # 5: ITFS 
    # 6: Reverb
    if subj in ['S046', 'S059']: # Type 1
        TrigLookup = [1,2,3,4]
    #end
    if subj in ['S051', 'S150']: # Type 2
        TrigLookup = [5,3,6,2]
    #end
    if subj in ['S064', 'S160']: # Type 3
        TrigLookup = [1,3,6,2]
    #end
    if subj in ['S190', 'S057']: # Type 4
        TrigLookup = [5,3,6,4]
    #end
    triggerset = TrigLookup # conditions that the subject performed
    for trigger in triggerset:
        cond_EEG = processEEG(subj,trigger,po_electrodes)
        cond_EEG = np.moveaxis(cond_EEG,[0,1,2],[1,0,2])
        freqs = np.arange(6,17,2) 
        mtgram = tfr_multitaper(cond_EEG,fs_EEG,freqs,time_bandwidth=2,use_fft=True,
                       n_cycles=5) # multitapered spectrogram 
        psd = mtgram[0]
        times = mtgram[2]
        WS = pd.read_excel(behavior_data_path+subj+'.xlsx')
        WS_np = np.array(WS)
        ind = np.where(WS_np[:,3] == trigger)
        ind = ind[0]
        pct_correct = 100*(WS_np[ind,2]/5)
        save_dict = dict(psd=psd, times=times, freqs=freqs, 
                         pct_correct=pct_correct, condition=trigger)
        savemat(sub_results_path + 'alpha_condition_' + str(trigger) + '.mat',save_dict)
    #end
#end       

def computeSingletrialFrontalBetaPower(subj):
    sub_results_path = results_path + subj + '/processed_EEG/'
    # Look up table for condition #
    # triggers: 
    # 1: SSN -2dB SNR
    # 2: SIB 4dB TMR (-2 dB SNR)
    # 3: SSN -5dB SNR
    # 4: SSN -8dB SNR
    # 5: ITFS 
    # 6: Reverb
    if subj in ['S046', 'S059']: # Type 1
        TrigLookup = [1,2,3,4]
    #end
    if subj in ['S051', 'S150']: # Type 2
        TrigLookup = [5,3,6,2]
    #end
    if subj in ['S064', 'S160']: # Type 3
        TrigLookup = [1,3,6,2]
    #end
    if subj in ['S190', 'S057']: # Type 4
        TrigLookup = [5,3,6,4]
    #end
    triggerset = TrigLookup # conditions that the subject performed
    for trigger in triggerset:
        cond_EEG = processEEG(subj,trigger,f_electrodes)
        cond_EEG = np.moveaxis(cond_EEG,[0,1,2],[1,0,2])
        freqs = np.arange(12,32,2) 
        mtgram = tfr_multitaper(cond_EEG,fs_EEG,freqs,time_bandwidth=2,use_fft=True,
                       n_cycles=5) # multitapered spectrogram 
        psd = mtgram[0]
        times = mtgram[2]
        WS = pd.read_excel(behavior_data_path+subj+'.xlsx')
        WS_np = np.array(WS)
        ind = np.where(WS_np[:,3] == trigger)
        ind = ind[0]
        pct_correct = 100*(WS_np[ind,2]/5)
        save_dict = dict(psd=psd, times=times, freqs=freqs, 
                         pct_correct=pct_correct, condition=trigger)
        savemat(sub_results_path + 'beta_condition_' + str(trigger) + '.mat',save_dict)
    #end
# end

def computeAverageSpectrogramAndTopomap(conds,subjects):
    freqs = np.arange(4,50,1) 
    nfreqs = freqs.shape[0]
    ntimes = 21505
    prestimulus = np.asarray([0.5,(0.5+1)]) # interval (in seconds) between "listen now" cue and stimulus onset
    prestim_samps = (prestimulus*fs_EEG).astype(int) # prestimulus interval (in samples)
    stiminterval = np.asarray([(0.5+1),(0.5+1)+(1.25+2.5)]) # stimulus interval (in seconds) 
    stim_samps = (stiminterval*fs_EEG).astype(int) # stimulus interval (in samples)      
    spect = np.zeros((1,nfreqs,ntimes))
    alpha_topodatpre = np.zeros(numchans)
    alpha_topodatdur = np.zeros(numchans)
    beta_topodatpre = np.zeros(numchans)
    beta_topodatdur = np.zeros(numchans)
    alpha_propChangeFromPreToDur = np.zeros(numchans)
    beta_propChangeFromPreToDur = np.zeros(numchans)
    nvals = 0
    for subj in subjects:
        for trigger in conds:
            if subj in ['S046', 'S059']: # Type 1
                TrigLookup = [1,2,3,4]
            #end
            if subj in ['S051', 'S150']: # Type 2
                TrigLookup = [5,3,6,2]
            #end
            if subj in ['S064', 'S160']: # Type 3
                TrigLookup = [1,3,6,2]
            #end
            if subj in ['S190', 'S057']: # Type 4
                TrigLookup = [5,3,6,4]
            #end
            if trigger not in TrigLookup:
                raise Exception("Subject did not perform condition")
            # end
            cond_EEG = processEEG(subj,trigger,all_electrodes)             
            mtgram = tfr_multitaper(cond_EEG,fs_EEG,freqs,
                                    time_bandwidth=2,use_fft=True,
                                    n_cycles=5) 
            psd = mtgram[0]
            times = mtgram[2]
            spect = spect + np.mean(psd,axis=0,keepdims=True) # average over channels
            nvals = nvals + 1
            alpharange = [6,16] 
            inda = np.where((freqs >= alpharange[0]) & (freqs <= alpharange[1]))
            alpha_topotemppre = np.squeeze(np.mean(np.mean(psd[:,inda,
                                    prestim_samps[0]:prestim_samps[1]],axis=-1),axis=-1))
            alpha_topodatpre = alpha_topodatpre + alpha_topotemppre
            alpha_topotempdur = np.squeeze(np.mean(np.mean(psd[:,inda,
                                    stim_samps[0]:stim_samps[1]],axis=-1),axis=-1))
            alpha_topodatdur = alpha_topodatdur + alpha_topotempdur
            alpha_propChangeFromPreToDur = alpha_propChangeFromPreToDur + alpha_topotempdur/alpha_topotemppre - 1
            betarange = [12, 31] 
            indb = np.where((freqs >= betarange[0]) & (freqs <= betarange[1]))
            beta_topotemppre = np.squeeze(np.mean(np.mean(psd[:,indb,
                                    prestim_samps[0]:prestim_samps[1]],axis=-1),axis=-1))
            beta_topodatpre = beta_topodatpre + beta_topotemppre
            beta_topotempdur = np.squeeze(np.mean(np.mean(psd[:,indb,
                                    stim_samps[0]:stim_samps[1]],axis=-1),axis=-1))
            beta_topodatdur = beta_topodatdur + beta_topotempdur 
            beta_propChangeFromPreToDur = beta_propChangeFromPreToDur + beta_topotempdur/beta_topotemppre - 1                         
        # end
    #end
    spect = spect/nvals
    alpha_topodatpre = alpha_topodatpre/nvals
    alpha_topodatdur = alpha_topodatdur/nvals
    beta_topodatpre = beta_topodatpre/nvals
    beta_topodatdur = beta_topodatdur/nvals
    alpha_propChangeFromPreToDur = alpha_propChangeFromPreToDur/nvals
    beta_propChangeFromPreToDur = beta_propChangeFromPreToDur/nvals
    # Normalize topomap values to add to one
    norm_alpha_topodatpre = alpha_topodatpre/alpha_topodatpre.max()
    norm_alpha_topodatdur = alpha_topodatdur/alpha_topodatdur.max()
    norm_beta_topodatpre = beta_topodatpre/beta_topodatpre.max()
    norm_beta_topodatdur = beta_topodatdur/beta_topodatdur.max()
    mdict = dict(spect=spect,times=times,freqs=freqs,
                 norm_alpha_topodatpre=norm_alpha_topodatpre,
                 norm_alpha_topodatdur=norm_alpha_topodatdur,
                 norm_beta_topodatpre=norm_beta_topodatpre,
                 norm_beta_topodatdur=norm_beta_topodatdur,
                 alpha_propChangeFromPreToDur=alpha_propChangeFromPreToDur,
                 beta_propChangeFromPreToDur=beta_propChangeFromPreToDur)
    savemat(results_path + 'averageSpectrogramAndTopomap.mat',mdict)
    pl.figure()
    plot_tfr(spect,np.squeeze(times),np.squeeze(freqs),ch_idx=0,cmap='YlGnBu')
    montage = make_standard_montage('biosemi32')
    chs = montage._get_ch_pos()
    ch_names, xyz = zip(*[(ich, ixyz) for ich, ixyz in chs.items()])
    xyz = np.vstack(xyz)
    sph = _cart_to_sph(xyz)
    pos2d = _pol_to_cart(sph[:, 1:][:, ::-1]) * 0.05
    pl.figure()
    plot_topomap(norm_alpha_topodatpre.squeeze(),pos2d,vmin=0,vmax=1)
    pl.title('Pre-stimulus alpha')
    pl.figure()
    plot_topomap(norm_alpha_topodatdur.squeeze(),pos2d,vmin=0,vmax=1)
    pl.title('During-stimulus alpha')
    pl.figure()
    plot_topomap(norm_beta_topodatpre.squeeze(),pos2d,vmin=0,vmax=1)
    pl.title('Pre-stimulus beta')
    pl.figure()
    im,cm = plot_topomap(norm_beta_topodatdur.squeeze(),pos2d,vmin=0,vmax=1)
    pl.title('During-stimulus beta')
    pl.colorbar(im, label='Normalized power', shrink=0.75)
#end

def computeAverageSpectrum():
    adict = loadmat(results_path + 'averageSpectrogramAndTopomap.mat')
    sgram = adict['spect']
    spectrm = (np.squeeze(sgram)).mean(axis=1)
    freqs = adict['freqs'].squeeze()
    pl.figure()
    pl.plot(freqs,spectrm)
    pl.xlabel('Frequency (Hz)')
    pl.ylabel('Average power (volt$^2$)')
#end

def computeTopomapDifferences():
    # Compute proportion change in alpha and beta from the pre- to the during-stimulus period
    mdict = loadmat(results_path + 'averageSpectrogramAndTopomap.mat')
    # Divide by mean over electrodes below to avoid blowing up low-power scalp areas
    topodiff_beta = (mdict['beta_topodatdur']-mdict['beta_topodatpre'])/mdict['beta_topodatpre'].mean()
    topodiff_alpha = (mdict['alpha_topodatdur']-mdict['alpha_topodatpre'])/mdict['alpha_topodatpre'].mean()
    montage = make_standard_montage('biosemi32')
    chs = montage._get_ch_pos()
    ch_names, xyz = zip(*[(ich, ixyz) for ich, ixyz in chs.items()])
    xyz = np.vstack(xyz)
    sph = _cart_to_sph(xyz)
    pos2d = _pol_to_cart(sph[:, 1:][:, ::-1]) * 0.05
    pl.figure()
    im,cm = plot_topomap(topodiff_alpha.squeeze(),pos2d)
    pl.colorbar(im, label='Power (proportion change)', shrink=0.75)
    pl.title('Alpha')
    pl.figure()
    im, cm = plot_topomap(topodiff_beta.squeeze(),pos2d)
    pl.colorbar(im, label='Power (proportion change)', shrink=0.75)
    pl.title('Beta')
#end

def histogramOfNumberOfWordsCorrect():
    numSubjsPerCond = 4
    pct_correct_allSubs = np.zeros((numSubjsPerCond,nconds))
    numWordsCorrect = np.zeros((numSubjsPerCond,nconds,numStimPerCond))
    # Look up table for condition # for different analysis types
    # triggers: 
    # 2: SIB 4dB TMR
    # 3: SSN -5dB SNR
    for whichCond in np.arange(conds):
        cond = conds[whichCond]
        whichSub = 0
        if (cond == 2):
            typeLookupForCond = [1,2]
        #end
        if (cond == 3):
            typeLookupForCond = [2,3]
        #end
        for exptType in typeLookupForCond:
            if (exptType == 1):
                subjlist = ['S046', 'S059']
            if (exptType == 2):
                subjlist = ['S051', 'S150']
            if (exptType == 3):
                subjlist = ['S064', 'S160']
            #end
            for subj in subjlist:
                subjpath = behavior_data_path + str(subj) + '/'
                fname =  subjpath + str(subj) + '.xlsx'
                x1 = pd.ExcelFile(fname)
                df1 = x1.parse('Sheet1')
                df2 = ((df1.loc[:]['Condition']) == cond)
                numWordsCorrect[whichSub,whichCond,:] = (df1.loc[df2]['Score'])
                pct_correct_subj_cond = 100.0*np.mean(df1.loc[df2]['Score'])/5.0
                savemat(subjpath + 'percent_correct_cond' + str(cond),
                    dict(pct_correct_subj_cond=pct_correct_subj_cond))
                pct_correct_allSubs[whichSub,whichCond] = pct_correct_subj_cond
                whichSub += 1                
            #end
        #end    
    #end
    pct_correct_avgAcrossSubs = pct_correct_allSubs.mean(axis=0)
    n = 5
    p = pct_correct_avgAcrossSubs/100. 
    x = np.arange(n+1)
    N = nSentPerCond # number of sentences per condition across all subjects
    M = np.arange(N+1)
    actualNumSentXWordsCorrect = np.zeros((nconds,n+1))
    pval = np.zeros((nconds,n+1))
    expectedNumSentXwordsCorrect = np.zeros((nconds,n+1))
    stdNumSentXwordsCorrect = np.zeros((nconds,n+1))
    for whichCond in np.arange(nconds):
        P2 = np.zeros((n+1,N+1))
        P_xWordsCorrectInOneSent = binom.pmf(x,n,p[whichCond]) # probability of getting x keywords correct in any one sentence
        # Calculate probability of getting x keywords correct in M sentences
        for xval in x:
            P2[xval,:] = binom.pmf(M,N,P_xWordsCorrectInOneSent[xval])
            expectedNumSentXwordsCorrect[whichCond,xval] = N*P_xWordsCorrectInOneSent[xval]
            stdNumSentXwordsCorrect[whichCond,xval] = ((N*P_xWordsCorrectInOneSent[xval]*
                                   (1-P_xWordsCorrectInOneSent[xval]))**0.5)        
        #end
        # Calculate pval 
        actualNumSentXWordsCorrect[whichCond,:],binedges = np.histogram(numWordsCorrect[:,whichCond,:],
                                            bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5])
        for whichX in x:
            if (actualNumSentXWordsCorrect[whichCond,whichX] < expectedNumSentXwordsCorrect[whichCond,whichX]):
                pval[whichCond,whichX] = P2[whichX,:actualNumSentXWordsCorrect[whichCond,whichX].astype('int')].sum()
            else:
                pval[whichCond,whichX] = P2[whichX,actualNumSentXWordsCorrect[whichCond,whichX].astype('int'):].sum()
            #end    
        #end
    #end
    for whichCond in np.arange(nconds):
        pl.figure()
        plt.hist(numWordsCorrect[:,whichCond,:].flatten(),
                 bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5],
                 linewidth=2, density=True,
                 histtype='step',label='Observed performance')
        pl.ylim(0,1)
        for whichX in x:
            pl.text(whichX-0.5, 0.5, 'p=' +
                    str(np.format_float_scientific(pval[whichCond,whichX], 
                                                   unique=False, precision=1)),
                    color='black',fontsize=8)
        #end
        pl.fill_between(x, 
                        (expectedNumSentXwordsCorrect[whichCond,:]-1.96*stdNumSentXwordsCorrect[whichCond,:])/nSentPerCond,
                        (expectedNumSentXwordsCorrect[whichCond,:]+1.96*stdNumSentXwordsCorrect[whichCond,:])/nSentPerCond,
                        color='k',alpha=0.3,
                        label='95% confidence interval for \nexpected distribution under \nindependent keyword outcomes')    
        condstr = condNames[whichCond]
        plt.title(condstr,fontsize=16)
        pl.xlabel('Number of keywords correct',fontsize=16)
        pl.ylabel('Proportion of sentences',fontsize=16)
        pl.legend(loc='best',fontsize=12)
        pl.xticks(fontsize=12)
        pl.yticks(fontsize=12)
    #end
#end


# Main script
for subj in subjects:
    computeSingletrialParietoOccipitalAlphaPower(subj)
    computeSingletrialFrontalBetaPower(subj)
# end
computeAverageSpectrogramAndTopomap(conds,subjects)
computeAverageSpectrum()
computeTopomapDifferences()




