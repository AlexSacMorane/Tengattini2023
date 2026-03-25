#-------------------------------------------------------------------------------
#Librairies
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from pathlib import Path
import math, pickle

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

if not Path('seg/tempo_save.dict').exists():

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

    # determine the best sphere for each grain
    print('Segmentate spheres')
    # define a threshold value
    threshold_radius = 4
    # init
    L_radius_small = []
    L_radius_large = []
    L_NCC_small = []
    L_NCC_large = []
    L_M_bin_grain_i_max = []
    L_parameter_test_max = []
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

        if radius < threshold_radius:
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

        if radius >= threshold_radius:
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
            # save
            L_M_bin_grain_i_max.append(M_bin_grain_i_max)
            L_parameter_test_max.append(parameter_test_max)
            # save for plot
            L_NCC_large.append(NCC_i_max)
            L_radius_large.append(parameter_test_max[0])
            # plot
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(16,9), num=1)
            # labeled
            ax1.imshow(M_bin_grain_i[:,:,int(parameter_test_max[3])])
            ax2.imshow(M_bin_grain_i[:,int(parameter_test_max[2]),:])
            ax3.imshow(M_bin_grain_i[int(parameter_test_max[1]),:,:])
            # test
            ax4.imshow(M_bin_grain_i_max[:,:,int(parameter_test_max[3])])
            ax5.imshow(M_bin_grain_i_max[:,int(parameter_test_max[2]),:])
            ax6.imshow(M_bin_grain_i_max[int(parameter_test_max[1]),:,:])
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
    plt.savefig('seg/grain_resume.png')
    plt.close()

    # tempo save
    dict_save = {
        'L_M_bin_grain_i_max': L_M_bin_grain_i_max,
        'L_parameter_test_max': L_parameter_test_max,
    }
    with open('seg/tempo_save.dict', 'wb') as handle:
            pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL) 

else :
    # load save
    with open('seg/tempo_save.dict', 'rb') as handle:
            dict_save = pickle.load(handle)
    L_M_bin_grain_i_max = dict_save['L_M_bin_grain_i_max']
    L_parameter_test_max = dict_save['L_parameter_test_max']

# extract the radius
L_radius = []
# convert in µm
pixel_to_um = 14.8 # µm/pixel
for parameter in L_parameter_test_max:
    L_radius.append(parameter[0]*pixel_to_um)

# plot the PSD
n_pp = 20
L_radius_pp = np.linspace(min(L_radius), max(L_radius), n_pp)
L_n_radius_pp = np.zeros((n_pp-1,))
for radius in L_radius_pp:
    i_pp = 0
    while not(L_radius_pp[i_pp]<=radius and radius<=L_radius_pp[i_pp+1]):
        i_pp = i_pp + 1
    L_n_radius_pp[i_pp] = L_n_radius_pp[i_pp] + 1
# compute cumulative
L_cum_n_radius_pp = np.zeros((n_pp-1,))
for i in range(len(L_n_radius_pp)):
    L_cum_n_radius_pp[i] = L_n_radius_pp[i]/np.sum(L_n_radius_pp)
    if i > 0:
        L_cum_n_radius_pp[i] = L_cum_n_radius_pp[i] + L_cum_n_radius_pp[i-1]

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,9), num=1)
ax1.scatter(L_radius_pp[:-1], L_n_radius_pp)
ax2.scatter(L_radius_pp[:-1], L_cum_n_radius_pp)
plt.savefig('seg/radius_resume.png')
plt.close()

# rebuild the prediction of the microstructure (grain)
M_bin_grain_predicted = np.zeros_like(M_bin_grain)
for M_bin_grain_i_max in L_M_bin_grain_i_max:
    M_bin_grain_predicted = M_bin_grain_predicted + M_bin_grain_i_max
# characterize the segmentation of the grain
S_12 = 0
S_11 = 0
S_22 = 0
M_prediction_grain = np.ones_like(M_bin_grain)
for i_x in range(M_bin_grain.shape[0]):
    for i_y in range(M_bin_grain.shape[1]):
        for i_z in range(M_bin_grain.shape[2]):
            # NCC
            S_12 = S_12 + M_bin_grain[i_x,i_y,i_z] * M_bin_grain_predicted[i_x,i_y,i_z]
            S_11 = S_11 + M_bin_grain[i_x,i_y,i_z] * M_bin_grain[i_x,i_y,i_z]
            S_22 = S_22 + M_bin_grain_predicted[i_x,i_y,i_z] * M_bin_grain_predicted[i_x,i_y,i_z]
            # other comparison
            if M_bin_grain[i_x, i_y, i_z] == M_bin_grain_predicted[i_x, i_y, i_z]:
                M_prediction_grain[i_x, i_y, i_z] = True
            else : 
                M_prediction_grain[i_x, i_y, i_z] = False
print('Grain :', np.sum(M_prediction_grain)/M_prediction_grain.size, S_12/(S_11*S_22)**(1/2))

# determine contact with cement
print('Segmentate the cement')
L_ij_contact = []
L_L_xyz_contact = []
L_parameter_contact = []
M_n_active_cement = np.zeros_like(M_bin_cement, dtype=int)
for i_grain in range(len(L_M_bin_grain_i_max)-1):
    for j_grain in range(i_grain+1, len(L_M_bin_grain_i_max)):
        # compute the combination of the 2 grains and the cement phase
        M_bin_2grains_cement = M_bin_cement + L_M_bin_grain_i_max[i_grain] + L_M_bin_grain_i_max[j_grain]
        # label this map
        M_bin_2grains_cement_labelled, n_2grains_cement = ndi.label(M_bin_2grains_cement)
        # extract the data of the two grains
        center_i = np.array([L_parameter_test_max[i_grain][1], L_parameter_test_max[i_grain][2], L_parameter_test_max[i_grain][3]])
        radius_i = L_parameter_test_max[i_grain][0]
        center_j = np.array([L_parameter_test_max[j_grain][1], L_parameter_test_max[j_grain][2], L_parameter_test_max[j_grain][3]])
        radius_j = L_parameter_test_max[j_grain][0]
        # compute parameters of the segment C_i->C_j
        vector_ij = center_j-center_i
        distance_ij = np.linalg.norm(vector_ij)
        vector_ij = vector_ij/distance_ij # normalization
        # consider a criterion on the distance
        if distance_ij < radius_i*1.1 + radius_j*1.1:
            # check the contact with a common label
            if M_bin_2grains_cement_labelled[int(center_i[0]), int(center_i[1]), int(center_i[2])] == \
               M_bin_2grains_cement_labelled[int(center_j[0]), int(center_j[1]), int(center_j[2])]: # contact
                # determine the box of investigation
                x_min_box = int(max(0, min(center_i[0]-radius_i, center_j[0]-radius_j)))
                x_max_box = int(min(M_bin_2grains_cement_labelled.shape[0], max(center_i[0]+radius_i, center_j[0]+radius_j)+1))
                y_min_box = int(max(0, min(center_i[1]-radius_i, center_j[1]-radius_j)))
                y_max_box = int(min(M_bin_2grains_cement_labelled.shape[1], max(center_i[1]+radius_i, center_j[1]+radius_j)+1))
                z_min_box = int(max(0, min(center_i[2]-radius_i, center_j[2]-radius_j)))
                z_max_box = int(min(M_bin_2grains_cement_labelled.shape[2], max(center_i[2]+radius_i, center_j[2]+radius_j)+1))
                # iterate in the box of investigation
                L_yxz_contact = []
                for i_x in range(x_min_box, x_max_box):
                    for i_y in range(y_min_box, y_max_box):
                        for i_z in range(z_min_box, z_max_box):
                            # check if there is cement
                            if M_bin_cement[i_x, i_y, i_z]:   
                                point = np.array([i_x , i_y, i_z])
                                vector_ipoint = point-center_i
                                distance_projected_point = np.dot(vector_ipoint, vector_ij)
                                # determine if the cement is located between the centers
                                if 0 < distance_projected_point and distance_projected_point < distance_ij:
                                    # compute the position of the projected point
                                    projectedpoint = center_i + distance_projected_point*vector_ij
                                    # determine the distance between the projected point and the point
                                    vector_projectedpointpoint = point - projectedpoint
                                    distance_point = np.linalg.norm(vector_projectedpointpoint)
                                    # interpolate the radius of the truncated cylinder
                                    radius_cylinder = (1-distance_projected_point/distance_ij)*radius_i + distance_projected_point/distance_ij*radius_j
                                    # determine if the cement is located in the truncated cylinder
                                    if distance_point < radius_cylinder:
                                        M_n_active_cement[i_x, i_y, i_z] = M_n_active_cement[i_x, i_y, i_z] + 1
                                        L_yxz_contact.append((i_x, i_y, i_z))
                # save
                L_ij_contact.append((i_grain, j_grain))
                L_L_xyz_contact.append(L_yxz_contact)
                L_parameter_contact.append([radius_i, radius_j, distance_ij])
                # print
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,9), num=1)
                ax1.imshow(M_bin_2grains_cement[:, :, int((z_max_box+z_min_box)/2)])
                ax2.imshow(M_bin_2grains_cement[:, int((y_max_box+y_min_box)/2), :])
                ax3.imshow(M_bin_2grains_cement[int((x_max_box+x_min_box)/2), :, :])
                plt.savefig('seg/c_'+str(i_grain)+'_'+str(j_grain)+'_resume.png')
                plt.close()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,9), num=1)
ax1.imshow(M_bin_cement[:, :, int(M_bin_cement.shape[2]/2)])
ax2.imshow(M_bin_cement[:, int(M_bin_cement.shape[2]/2), :])
ax3.imshow(M_bin_cement[int(M_bin_cement.shape[2]/2), :, :])
plt.savefig('seg/cement_resume.png')
plt.close()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,9), num=1)
ax1.imshow(M_n_active_cement[:, :, int(M_n_active_cement.shape[2]/2)])
ax2.imshow(M_n_active_cement[:, int(M_n_active_cement.shape[2]/2), :])
ax3.imshow(M_n_active_cement[int(M_n_active_cement.shape[2]/2), :, :])
plt.savefig('seg/cement_n_active_resume.png')
plt.close()

# characterize the segmentation of the cement
S_12 = 0
S_11 = 0
S_22 = 0
M_prediction_cement = np.ones_like(M_bin_cement)
for i_x in range(M_bin_grain.shape[0]):
    for i_y in range(M_bin_grain.shape[1]):
        for i_z in range(M_bin_grain.shape[2]):
            if M_n_active_cement[i_x, i_y, i_z] > 0:
                # NCC
                S_12 = S_12 + M_bin_cement[i_x,i_y,i_z] * 1
                S_22 = S_22 + 1 * 1
            S_11 = S_11 + M_bin_cement[i_x,i_y,i_z] * M_bin_cement[i_x,i_y,i_z]
            # other comparison
            if M_bin_cement[i_x, i_y, i_z] == True and M_n_active_cement[i_x, i_y, i_z] > 0:
                M_prediction_cement[i_x, i_y, i_z] = True
            elif M_bin_cement[i_x, i_y, i_z] == False and M_n_active_cement[i_x, i_y, i_z] == 0:
                M_prediction_cement[i_x, i_y, i_z] = True
            else : 
                M_prediction_cement[i_x, i_y, i_z] = False
print('Cement :',np.sum(M_prediction_cement)/M_prediction_cement.size, S_12/(S_11*S_22)**(1/2))

# iterate on  the cement bridge
L_S_cement = []
for i_cement in range(len(L_L_xyz_contact)):
    # height of cement*
    H_cement = L_parameter_contact[i_cement][2] + 4/3*((-L_parameter_contact[i_cement][0]**3+\
                                                       2*L_parameter_contact[i_cement][0]**2*L_parameter_contact[i_cement][1]+\
                                                       2*L_parameter_contact[i_cement][0]*L_parameter_contact[i_cement][1]**2-\
                                                       L_parameter_contact[i_cement][1]**3)/\
                                                 (L_parameter_contact[i_cement][0]+L_parameter_contact[i_cement][1])**2)
    # compute the equivalent volume
    V_cement = 0
    for i_xyz in range(len(L_L_xyz_contact[i_cement])):
        # volume of cement
        V_cement = V_cement+1/M_n_active_cement[L_L_xyz_contact[i_cement][i_xyz][0], L_L_xyz_contact[i_cement][i_xyz][1], L_L_xyz_contact[i_cement][i_xyz][2]]
    # compute the equivalent section 
    S_cement = V_cement/H_cement
    L_S_cement.append(S_cement)

# compute the distribution of the cement area
n_pp = 20
L_S_cement_pp = np.linspace(min(L_S_cement), max(L_S_cement), n_pp)
L_n_S_cement_pp = np.zeros((n_pp-1,))
for S_cement in L_S_cement:
    i_pp = 0
    while not(L_S_cement_pp[i_pp]<=S_cement and S_cement<=L_S_cement_pp[i_pp+1]):
        i_pp = i_pp + 1
    L_n_S_cement_pp[i_pp] = L_n_S_cement_pp[i_pp] + 1
# compute cumulative
L_cum_n_S_cement_pp = np.zeros((n_pp-1,))
for i in range(len(L_n_S_cement_pp)):
    L_cum_n_S_cement_pp[i] = L_n_S_cement_pp[i]/np.sum(L_n_S_cement_pp)
    if i > 0:
        L_cum_n_S_cement_pp[i] = L_cum_n_S_cement_pp[i] + L_cum_n_S_cement_pp[i-1]

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,9), num=1)
ax1.scatter(L_S_cement_pp[:-1], L_n_S_cement_pp)
ax2.scatter(L_S_cement_pp[:-1], L_cum_n_S_cement_pp)
plt.savefig('seg/S_cement_resume.png')
plt.close()

# convert in µm
pixel_to_um = 14.8 # µm/pixel
for i_S_cement_pp in range(len(L_S_cement_pp)):
    L_S_cement_pp[i_S_cement_pp] = L_S_cement_pp[i_S_cement_pp]*pixel_to_um*pixel_to_um

# reference value
L_ref_size     = [0, 0.08e4, 0.21e4, 0.36e4, 0.51e4, 0.73e4, 0.89e4, 1.05e4, 1.27e4, 1.56e4, 1.94e4, 2.55e4, 3.2e4]
L_ref_cum_prob = [0,   0.01,   0.04,   0.10,   0.19,   0.33,   0.45,   0.56,   0.68,   0.81,   0.91,   0.97,     1]
size_min = 0.08e4 # µm2
size_max = 3.2e4 # µm2
n_size = 19
L_size = np.linspace(size_min, size_max, n_size)
# compute cumulative weight
L_cum_p_size = []
for size in L_size:
    # find interval
    i_ref_size = 0
    while not (L_ref_size[i_ref_size] <= size and size <= L_ref_size[i_ref_size+1]):
        i_ref_size = i_ref_size + 1
    # compute cumulative prob
    cum_p_size = L_ref_cum_prob[i_ref_size] + (L_ref_cum_prob[i_ref_size+1]-L_ref_cum_prob[i_ref_size])/(L_ref_size[i_ref_size+1]-L_ref_size[i_ref_size])*(size-L_ref_size[i_ref_size])
    L_cum_p_size.append(cum_p_size)
# compute the weight
L_p_size = []
for i_size in range(len(L_size)):
    if i_size == 0:
        p_size = (L_cum_p_size[i_size+1]-L_cum_p_size[i_size])/(L_size[i_size+1]-L_size[i_size])
    if 0 < i_size and i_size < len(L_size)-1:
        p_size = (L_cum_p_size[i_size+1]-L_cum_p_size[i_size-1])/(L_size[i_size+1]-L_size[i_size-1])
    if i_size == len(L_size)-1:
        p_size = (L_cum_p_size[i_size]-L_cum_p_size[i_size-1])/(L_size[i_size]-L_size[i_size-1])
    L_p_size.append(p_size*100)

# plot
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,9), num=1)
ax1.scatter(L_S_cement_pp[:-1], L_n_S_cement_pp)
#ax1.scatter(L_size, L_p_size)
ax2.scatter(L_S_cement_pp[:-1], L_cum_n_S_cement_pp)
#ax2.scatter(L_size, L_cum_p_size)
plt.savefig('seg/S_cement_resume2.png')
plt.close()