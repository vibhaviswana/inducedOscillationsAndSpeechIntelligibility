# inducedOscillationsAndSpeechIntelligibility

This repository contains custom code used for a project that compares induced brain oscillations (measured with EEG) with single-trial speech intelligibility in noise.

Copyright 2019-23 Vibha Viswanathan. All rights reserved.

## Basic usage

The code should be run using the following steps: 

- Run ```inducedRythmsAndIntelligibility.py```. This script calls ```preprocess_EEG.py``` to preprocess the EEG data and compute the average spectrogram, spectrum, topomaps in different bands, pre- and during-stimulus alpha power, and pre- and during-stimulus beta power. It also plots the histogram of number of keywords correct in a sentence.
