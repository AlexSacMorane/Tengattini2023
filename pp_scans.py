#-------------------------------------------------------------------------------
#Librairies
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import math

#-------------------------------------------------------------------------------
#Function
#-------------------------------------------------------------------------------

def NCC(M1, M2, center, radius):
    '''
    Compute the normalized cross correlation coefficient between two numpy arrays.
    '''
    S_12 = 0
    S_11 = 0
    S_22 = 0
    for i in range(max(0, int(center[0]-1.2*radius)), min(M_bin_grain_i_test.shape[0], int(center[0]+1.2*radius))):
        for j in range(max(0, int(center[1]-1.2*radius)), min(M_bin_grain_i_test.shape[1], int(center[1]+1.2*radius))):
            for k in range(max(0, int(center[2]-1.2*radius)), min(M_bin_grain_i_test.shape[2], int(center[2]+1.2*radius))):
                S_12 = S_12 + M1[i,j,k] * M2[i,j,k]
                S_11 = S_11 + M1[i,j,k] * M1[i,j,k]
                S_22 = S_22 + M2[i,j,k] * M2[i,j,k]
    return S_12/(S_11*S_22)**(1/2)

#-------------------------------------------------------------------------------
#User
#-------------------------------------------------------------------------------

# size of sub_REV
size_subrev = 125

# extraction
i_x_min = 700 #450
i_x_max = 800 #i_x_min+size_subrev*4
i_y_min = 700 #475
i_y_max = 800 #i_y_min+size_subrev*4
i_z_min = 800 #150
i_z_max = 900 #i_z_min+size_subrev*10

#-------------------------------------------------------------------------------
#Read data
#-------------------------------------------------------------------------------

# init
M_bin = np.zeros((i_x_max-i_x_min, i_y_max-i_y_min, i_z_max-i_z_min))

for i_z in range(i_z_max-i_z_min):
    ind = i_z+i_z_min
    # convert in str index
    if ind < 1000:
        ind_str = '0'+str(ind)
    else :
        ind_str = str(ind)
    # name of the .png file
    z_section_file = 'Tengattini2023_scans/CGB29AT'+ind_str+'.png'

    # open the image and convert in numpy array
    z_section = np.array(Image.open(z_section_file))

    # define the window of study
    z_section = z_section[i_x_min:i_x_max, i_y_min:i_y_max]

    # add to the matrix
    M_bin[:, :, i_z] = z_section

# extract grain and cement
print('Extract phases')
M_bin_pore = M_bin == 0
M_bin_cement = M_bin == 128
M_bin_grain = M_bin == 255


# apply the watersheld algorithm to identify the grains
print('Apply the watershed')
distance = ndi.distance_transform_edt(M_bin_grain)
coords = peak_local_max(distance, labels=M_bin_grain, min_distance=3)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
M_bin_grain_labelled = watershed(-distance, markers, mask=M_bin_grain)

# shuffle labels for the plot
unique_labels = np.unique(M_bin_grain_labelled)[1:]
np.random.shuffle(unique_labels)
shuffled_labels = np.zeros_like(M_bin_grain_labelled)
for i, old_label in enumerate(unique_labels):
    shuffled_labels[M_bin_grain_labelled == old_label] = i + 1  # On évite le 0 pour le fond

L_radius_small = []
L_radius_large = []
L_NCC_small = []
L_NCC_large = []
# iterate on the labels
for i_label in np.unique(M_bin_grain_labelled)[1:]:
    # extract the grain
    M_bin_grain_i = M_bin_grain_labelled == i_label
    # compute the center of mass
    center = ndi.center_of_mass(M_bin_grain_i)
    # compute the volume 
    volume = np.sum(M_bin_grain_i)
    # compute the radius
    radius = (volume*3/4/math.pi)**(1/3)

    if radius < 5:
        # generate a approximation of the sphere
        M_bin_grain_i_test = np.zeros_like(M_bin_grain_i)
        for i_x in range(max(0, int(center[0]-1.2*radius)), min(M_bin_grain_i_test.shape[0], int(center[0]+1.2*radius))):
            for i_y in range(max(0, int(center[1]-1.2*radius)), min(M_bin_grain_i_test.shape[1], int(center[1]+1.2*radius))):
                for i_z in range(max(0, int(center[2]-1.2*radius)), min(M_bin_grain_i_test.shape[2], int(center[2]+1.2*radius))):
                    distance = ((center[0]-i_x)**2 + (center[1]-i_y)**2 + (center[2]-i_z)**2)**(1/2)
                    if distance <= radius:
                        M_bin_grain_i_test[i_x, i_y, i_z]=1
        # compute the NCC
        NCC_i=NCC(M_bin_grain_i, M_bin_grain_i_test, center, radius)
        if not math.isnan(NCC_i):
            L_NCC_small.append(NCC_i)
            L_radius_small.append(radius)

    if radius >= 5:
        # try to find a better fit 
        NCC_i_max = 0 
        L_radius_test = np.arange(radius-3, radius+3.1, 1)
        L_center_x_test = np.arange(center[0]-2, center[0]+2.1, 1)
        L_center_y_test = np.arange(center[1]-2, center[1]+2.1, 1)
        L_center_z_test = np.arange(center[2]-2, center[2]+2.1, 1)
        parameter_test_max = (L_radius_test[0], L_center_x_test[0], L_center_y_test[1], L_center_z_test[2])
        # introduce perturbation in the radius
        for radius_test in L_radius_test:
            # and in the position of the center
            for center_x_test in L_center_x_test:
                for center_y_test in L_center_y_test:
                    for center_z_test in L_center_z_test:
                        # generate a approximation of the sphere
                        M_bin_grain_i_test = np.zeros_like(M_bin_grain_i)
                        for i_x in range(max(0, int(center_x_test-1.5*radius_test)), min(M_bin_grain_i_test.shape[0], int(center_x_test+1.5*radius_test)+1)):
                            for i_y in range(max(0, int(center_y_test-1.5*radius_test)), min(M_bin_grain_i_test.shape[1], int(center_y_test+1.5*radius_test)+1)):
                                for i_z in range(max(0, int(center_z_test-1.5*radius_test)), min(M_bin_grain_i_test.shape[2], int(center_z_test+1.5*radius_test)+1)):
                                    distance = ((center_x_test-i_x)**2 + (center_y_test-i_y)**2 + (center_z_test-i_z)**2)**(1/2)
                                    if distance <= radius_test:
                                        M_bin_grain_i_test[i_x, i_y, i_z]=1
                        # compute the NCC
                        NCC_i=NCC(M_bin_grain_i, M_bin_grain_i_test, [center_x_test, center_y_test, center_z_test], radius_test)
                        if not math.isnan(NCC_i):
                            if NCC_i >= NCC_i_max:
                                NCC_i_max = NCC_i
                                parameter_test_max = (radius_test, center_x_test, center_y_test, center_z_test)
                                M_bin_grain_i_max = M_bin_grain_i_test.copy()
            
        # 
        L_NCC_large.append(NCC_i_max)
        L_radius_large.append(parameter_test_max[0])
        #plot
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(16,9), num=1)
        # labeled
        ax1.imshow(M_bin_grain_i[:,:,int(center[2])])
        ax2.imshow(M_bin_grain_i[:,int(center[1]),:])
        ax3.imshow(M_bin_grain_i[int(center[0]),:,:])
        # test
        ax4.imshow(M_bin_grain_i_max[:,:,int(center[2])])
        ax5.imshow(M_bin_grain_i_max[:,int(center[1]),:])
        ax6.imshow(M_bin_grain_i_max[int(center[0]),:,:])
        fig.suptitle('NCC='+str(round(NCC_i_max, 2)))
        plt.savefig('seg/'+str(i_label)+'.png')
        plt.close()

# plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,9), num=1)
ax1.imshow(shuffled_labels[:, :, int(shuffled_labels.shape[2]/2)])
ax2.imshow(shuffled_labels[:, int(shuffled_labels.shape[2]/2), :])
ax3.imshow(shuffled_labels[int(shuffled_labels.shape[2]/2), :, :])
ax4.scatter(L_radius_small, L_NCC_small, color='r')
ax4.scatter(L_radius_large, L_NCC_large, color='b')
plt.savefig('seg/resume.png')
plt.close()

