# -*- coding: utf-8 -*-
"""
@author: Vibha Viswanathan
Copyright 2015-23 Vibha Viswanathan. All rights reserved.
"""

import mne
from anlffr.preproc import find_blinks
from mne.preprocessing.ssp import compute_proj_epochs

def preprocess_EEG(raw, eves, subj):
    numchans = 32    
    blinks = find_blinks(raw, ch_name=['A30',], l_trans_bandwidth=0.4)
    # Epoch around blink and derive spatial filter
    epochs_blinks = mne.Epochs(raw, blinks, 998, tmin=-0.25, tmax=0.25,
                               proj=True, baseline=(-0.25, 0.25),
                               reject=dict(eeg=500e-6))
    # Remove blinks and saccades
    blink_projs = compute_proj_epochs(epochs_blinks, n_grad=0, n_mag=0,
                                      n_eeg=3, verbose='DEBUG')
    raw.add_proj(blink_projs)
    raw.filter(l_freq=1., h_freq=400., picks=range(numchans))  
    raw.apply_proj()
    raw3 = raw.copy()
    return (raw3, eves)  
#end
    

