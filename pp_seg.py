#-------------------------------------------------------------------------------
#Librairies
#-------------------------------------------------------------------------------

import glob, pickle, os, shutil
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

#-------------------------------------------------------------------------------
#Function
#-------------------------------------------------------------------------------

def compute_distribution(L_value, L_value_pp, L_n_value_pp, L_cum_n_value_pp):
    '''
    Compute the distribution (and its cumulative) of a list of values.
    '''
    # iterate on the list
    counter = 0
    for value in L_value:
        if L_value_pp[0]<=value and value<=L_value_pp[-1]:
            i_pp = 0
            while not(L_value_pp[i_pp]<=value and value<=L_value_pp[i_pp+1]):
                i_pp = i_pp + 1
            L_n_value_pp[i_pp] = L_n_value_pp[i_pp] + 1
        else :
            counter = counter + 1
    #print('number of values out of the range: ', counter, '/', len(L_value))
    # compute cumulative
    for i in range(len(L_n_value_pp)):
        L_cum_n_value_pp[i] = L_n_value_pp[i]/np.sum(L_n_value_pp)
        if i > 0:
            L_cum_n_value_pp[i] = L_cum_n_value_pp[i] + L_cum_n_value_pp[i-1]
    return L_n_value_pp, L_cum_n_value_pp

#---------------------------------------

def mk_new_dir(foldername):
    '''
    Create a new folder (erase the preexisting, if it exists).
    '''
    if Path(foldername).exists():
        shutil.rmtree(foldername)
    os.mkdir(foldername)

#-------------------------------------------------------------------------------
#User
#-------------------------------------------------------------------------------

# find all the data
L_seg = glob.glob('dict_seg_*')

# prepare the folder with the result
mk_new_dir('pp')

# prepare the plot
n_pp = 20
L_S_cement_pp = np.linspace(0, 150, n_pp)
L_S_cement_weighted_pp = np.linspace(0, 60, n_pp)
L_radius_pp = np.linspace(0, 30, n_pp)

# plot BSD
fig_bsd, ax1_bsd = plt.subplots(1,1,figsize=(16,9))

# plot w_BSD
fig_wbsd, ax1_wbsd = plt.subplots(1,1,figsize=(16,9))

# plot PSD
fig_psd, ax1_psd = plt.subplots(1,1,figsize=(16,9))

#-------------------------------------------------------------------------------
#Read data
#-------------------------------------------------------------------------------

# iterate on the segmentations 
for seg in L_seg:

    # read the name
    seg_name = seg[9:-5]

    # load the dict
    with open(seg, 'rb') as handle:
        dict_seg = pickle.load(handle)

    # compute the distribution of the cement area
    L_n_S_cement_pp, L_cum_n_S_cement_pp = compute_distribution(dict_seg['L_S_cement_pixel'], L_S_cement_pp, np.zeros((n_pp-1,)), np.zeros((n_pp-1,)))

    # compute the distribution of the cement area
    L_n_S_cement_weighted_pp, L_cum_n_S_cement_weighted_pp = compute_distribution(dict_seg['L_S_cement_weighted_pixel'], L_S_cement_weighted_pp, np.zeros((n_pp-1,)), np.zeros((n_pp-1,)))

    # compute the distribution of the particle size
    L_n_radius_pp, L_cum_n_radius_pp = compute_distribution(dict_seg['L_rad_pixel'], L_radius_pp, np.zeros((n_pp-1,)), np.zeros((n_pp-1,)))

    # plot BSD
    ax1_bsd.plot(L_S_cement_pp[:-1], L_cum_n_S_cement_pp, label=seg_name)

    # plot w_BSD
    ax1_wbsd.plot(L_S_cement_weighted_pp[:-1], L_cum_n_S_cement_weighted_pp, label=seg_name)

    # plot PSD
    ax1_psd.plot(L_radius_pp[:-1], L_cum_n_radius_pp, label=seg_name)


#-------------------------------------------------------------------------------
#Close plot
#-------------------------------------------------------------------------------

# plot BSD
ax1_bsd.set_xlabel('sectional surface (pixel^2)')
ax1_bsd.set_ylabel('cumulative probability (-)')
ax1_bsd.legend()
fig_bsd.tight_layout()
fig_bsd.savefig('pp/bond_size_distribution.png')
plt.close()

# plot w_BSD
ax1_wbsd.set_xlabel('weighted sectional surface (pixel^2)')
ax1_wbsd.set_ylabel('cumulative probability (-)')
ax1_wbsd.legend()
fig_wbsd.tight_layout()
fig_wbsd.savefig('pp/weighted_bond_size_distribution.png')
plt.close()

# plot PSD
ax1_psd.set_xlabel('grain radius (pixel)')
ax1_psd.set_ylabel('cumulative probability (-)')
ax1_psd.legend()
fig_psd.tight_layout()
fig_psd.savefig('pp/particle_size_distribution.png')
plt.close()
