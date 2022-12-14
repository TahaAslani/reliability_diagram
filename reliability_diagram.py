#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 00:13:23 2022

@author: taha
"""

import matplotlib.pyplot as plt
import numpy as np



def plot_reliability_diagram(y_true, y_pred, raw_output=None, probs=None, nbins=20):
    
    if probs is None and raw_output is None:
        raise Exception('Either probabilities or raw outputs must be provided')
    
    if probs is None and raw_output is not None:
        
        # Only import torch if the user inputs raw valeus
        import torch
        
        probs = torch.softmax(torch.tensor(raw_output), dim=1).numpy()
    
    # Find the winnig class
    winning_class_prob = np.max(probs,axis=1)
    
    min_prob = 0
    max_prob = 1
    
    bin_boundry = np.linspace(min_prob, max_prob, num=nbins+1)
    
    # We need accuracy, confidence and frequency of each bin
    acc_bin = np.zeros(nbins)
    conf_bin = np.zeros(nbins)
    freq_bin = np.zeros(nbins)
    
    bin_centers = []
    
    for bin_num in range(nbins):
        
        # Bin boundries
        bin_min = bin_boundry[bin_num]
        bin_max = bin_boundry[bin_num+1]
        bin_centers.append(np.mean([bin_min,bin_max]))
        
        # Find which predictions are in this bin
        bin_ind = np.logical_and(bin_min <= winning_class_prob, winning_class_prob < bin_max)
        
        # Calcualte the accuracy of each bin
        acc_bin[bin_num] = np.sum(y_true[bin_ind]==y_pred[bin_ind])/len(bin_ind)
        
        # Calculate confidence of each bin (0 if the bin is empty)
        if len(winning_class_prob[bin_ind]) == 0:
            conf = 0
        else:
            conf = np.mean(winning_class_prob[bin_ind])
        
        conf_bin[bin_num] = conf
        
        # Count number of the predictions in the bin
        freq_bin[bin_num] = np.sum(bin_ind)
        
    # Expected Calibration Error
    ECE = np.sum(freq_bin * np.abs(conf_bin-acc_bin))/len(y_true)
    
    fig = plt.figure()
    
    # plot the bar
    plt.bar(x=bin_centers,height=acc_bin,width=1/nbins),
    
    plt.xticks(bin_boundry,[str(np.round(x,2)) for x in bin_boundry], rotation=90)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('ECE: '+str(np.round(ECE,4)))
    
    # Plot the diogonal line
    plt.plot([0, 1], [0, 1], color='k')
    
    # Tight layout
    plt.tight_layout()
    
    return fig
