#-------------------------------------------------------------------------------
#README
#-------------------------------------------------------------------------------

# this file is here to generate a synthetic ctscan image
# the image is then used to test the functions developped in pp_scans.py

#-------------------------------------------------------------------------------
#Librairies
#-------------------------------------------------------------------------------

import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from numpy_to_vtk import write_vtk_structured_points

#-------------------------------------------------------------------------------
#Functions
#-------------------------------------------------------------------------------

def new_folder(folderName):
    '''
    Create a new folder (erase it if it exists already).
    '''

    if Path(folderName).exists():
        shutil.rmtree(folderName)
    os.mkdir(folderName)

#-------------------------------------------------------------------------------

def ind_to_str(ind):
    '''
    Convert an index into a str with the following format 'XXX'.
    '''
    if ind < 10:
        return '00'+str(ind)
    elif ind < 100:
        return '0'+str(ind)
    else :
        return(str(ind))

#-------------------------------------------------------------------------------
#User
#-------------------------------------------------------------------------------

# name of the synthetic scans
folderName = 'synthetic'

# dimensions scans
n_x = 50
n_y = 50
n_z = 120

# grain 1 
g1_radius = 20
g1_pos_x = int(n_x/2)
g1_pos_y = int(n_y/2)
g1_pos_z = int(n_z/2) - g1_radius

# grain 2
g2_radius = 20
g2_pos_x = int(n_x/2)
g2_pos_y = int(n_y/2)
g2_pos_z = int(n_z/2) + g2_radius

# cement
c_radius = 20

#-------------------------------------------------------------------------------
#Generate data
#-------------------------------------------------------------------------------

# create the folder
new_folder(folderName)

# generate the image
M_scans = np.zeros((n_x, n_y, n_z))

# generate the position of g1 and g2
g1_pos = np.array([g1_pos_x, g1_pos_y, g1_pos_z])
g2_pos = np.array([g2_pos_x, g2_pos_y, g2_pos_z])

# iterate on the image
for i_x in range(n_x):
    for i_y in range(n_y):
        for i_z in range (n_z):
            # generate the position of the current point
            p_pos = np.array([i_x, i_y, i_z])

            # look for the g1
            if np.linalg.norm(p_pos-g1_pos) <= g1_radius:
                M_scans[i_x, i_y, i_z] = 255
            # look for the g2
            if np.linalg.norm(p_pos-g2_pos) <= g2_radius:
                M_scans[i_x, i_y, i_z] = 255

            # look for the cement
            if g1_pos_z <= i_z and i_z <= g2_pos_z:
                # compute the projected point
                p_pos_2d = np.array([i_x, i_y])
                c_pos_2d = np.array([g1_pos_x, g1_pos_y]) # point on the contact 1-2
                # verify the projected distance
                if np.linalg.norm(p_pos_2d-c_pos_2d) <= c_radius:
                    # do not erase grain
                    if M_scans[i_x, i_y, i_z] != 255:
                        M_scans[i_x, i_y, i_z] = 128
                        
                
#-------------------------------------------------------------------------------
#Generate vtk
#-------------------------------------------------------------------------------

# change the array structure to verify the function
M_scans_vtk = np.transpose(M_scans, (2, 1, 0))
write_vtk_structured_points(folderName+'/scans.vtk', M_scans_vtk, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), binary=False)  

#-------------------------------------------------------------------------------
#Generate images
#-------------------------------------------------------------------------------

# iterate on the z
for i_z in range(n_z):
    # convert index to str
    i_z_str = ind_to_str(i_z)
    # definition of the name of the file
    fileName = folderName + '/scans_'+i_z_str+'.png'
 
    # convert the numpy array into an image and save it
    plt.imsave(fileName, M_scans[:, :, i_z])
