import numpy as np
import h5py
import matplotlib.pyplot as plt

import nibabel as nib
import cv2
from PIL import Image

# Creates quantitative maps based on anatomical tissue masks

def create_phantom(tissue_dict,
                   param_dict,
                   have_params,
                   data,
                   bound_dict=None,
                   proton_density=False,
                   print_process=False,
                   filling_type='from_distribution'):

    pil_data = Image.fromarray(data)
    resize_pil = pil_data.resize((250,250),resample = 0)
    resize_data = np.asarray(resize_pil)

    phantom = np.zeros(resize_data.shape)


    for tissue in have_params:
        #mean, std = param_dict[tissue_dict[tissue]]
        #matrix = np.full((resize_data.shape), mean)
        if proton_density is False:
            if filling_type == 'homogeneous':
                mean, std = param_dict[tissue_dict[tissue]]
                matrix = np.full((resize_data.shape), mean)
            if filling_type == 'boundary_homogenous':
                matrix = np.full((resize_data.shape), np.random.randint(bound_dict[tissue_dict[tissue]][0], bound_dict[tissue_dict[tissue]][1]))
            if filling_type == 'boundary_heterogenous':
                matrix = np.random.randint(bound_dict[tissue_dict[tissue]][0], bound_dict[tissue_dict[tissue]][1], (resize_data.shape))
            if filling_type == 'from_distribution':
                mean, std = param_dict[tissue_dict[tissue]]
                matrix = np.random.normal(loc=mean, scale=std, size=resize_data.shape)



        else:
            mean, std = param_dict[tissue_dict[tissue]]
            matrix = np.full((resize_data.shape), mean)

        matrix[resize_data != tissue] = 0

        #print(tissue_dict[tissue])

        if print_process:

            plt.imshow(matrix)
            plt.show()

        phantom += matrix

        if print_process:

            print('phantom')

            plt.imshow(phantom)
            plt.show()

    if proton_density is False:
        phantom = phantom * 1e-3

    return phantom

# Creates HDF5 file with phantom. This file you can use in KomaMRI for simulation

def create_h5_file(list_of_phantoms, phantom_name, dir_for_save):

    offset = np.array([0.0, 0.0, 0.0])
    offset = np.expand_dims(offset, 1)

    resolution = np.array([1.0, 1.0, 1.0])
    resolution = np.expand_dims(resolution, 1)

    rescaled_phantoms = []


    for i in range(len(list_of_phantoms)):
        if (i == 0) or (i == 4):
            rescaled_phantoms.append(list_of_phantoms[i])
        else:
            r_ph = list_of_phantoms[i] * 1e3
            r_ph = np.where(r_ph > 0, 1/r_ph, 0)
            rescaled_phantoms.append(r_ph)


    five_phant = np.array(rescaled_phantoms).transpose(1,2,0)


    with h5py.File(f'{dir_for_save}/{phantom_name}','w') as f:

        grp = f.create_group("sample")

        dset = grp.create_dataset('data', (250,250, 5), dtype="f8")
        dset[:,:, :] = five_phant[:,:, :]

        dset_1 = grp.create_dataset('offset', (3, 1), dtype="f8")
        dset_1[:, :] = offset[:, :]

        dset_2 = grp.create_dataset('resolution', (3, 1), dtype="f8")
        dset_2[:, :] = resolution[:, :]


numbers_of_train_subj = ['04', '05', '06']

numbers_of_test_subj = ['18']


def making_many_2_d_phantoms(tissue_dict,
                             param_dict_T1,
                             param_dict_T2,
                             param_dict_PD,
                             param_dict_T2_star,
                             have_params,
                             dir_for_save,
                             bound_dict_T1=None,
                             bound_dict_T2=None,
                             proton_density=False,
                             print_process=False,
                             filling_type='from_distribution',
                             path_to_anatomical_model=None):

    anatomical_model = nib.load(path_to_anatomical_model)
    anatomical_model_arr = anatomical_model.get_fdata()

    ph_name = path_to_anatomical_model.split('/')[-1][:-4]

    for i in range(anatomical_model_arr.shape[0]):
        if len(np.unique(anatomical_model_arr[i, :, :])) > 1:

            data = anatomical_model_arr[i, :, :]
            phantom_t_1 = create_phantom(tissue_dict,
                                         param_dict_T1,
                                         have_params,
                                         data,
                                         bound_dict_T1,
                                         print_process=False,
                                         filling_type='from_distribution')

            phantom_t_2 = create_phantom(tissue_dict,
                                         param_dict_T2,
                                         have_params,
                                         data,
                                         bound_dict_T2,
                                         print_process=False,
                                         filling_type=filling_type)

            phantom_t_2_s = create_phantom(tissue_dict,
                                           param_dict_T2_star,
                                           have_params,
                                           data,
                                           bound_dict_T2,
                                           print_process=False,
                                           filling_type=filling_type)

            phantom_pd = create_phantom(tissue_dict,
                                        param_dict_PD,
                                        have_params,
                                        data,
                                        proton_density=True,
                                        print_process=False,
                                        filling_type=filling_type)

            zer = np.zeros(phantom_pd.shape)

            list_of_phantoms = [phantom_pd, phantom_t_1, phantom_t_2, phantom_t_2_s, zer]

            create_h5_file(list_of_phantoms, f'{ph_name}_{i}.h5', dir_for_save)


tissue_dict = {1: 'CSF', 2: 'Gray Matter', 3: 'White Matter',
               4: 'Fat', 5: 'Muscle', 6: 'Muscle/Skin',
               7: 'Skull', 8: 'vessels', 9: 'around fat',
               10: 'dura matter', 11: 'bone marrow'}

# Dictionaries with different parameters. You can set your own values T1, T2 in ms.
# For example:
#param_dict_T1 = {'Gray Matter': (833, 83), ...}

#bound_dict_T1 = {}

#param_dict_T2 = {}

#param_dict_T2_star = {}

#param_dict_PD = {}

# Tissues for which there are parameters in dictionaries
have_params = [1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11]



if __name__ == "__main__":
    for num in numbers_of_train_subj:
        path_to_anatomical_model = f'anatomic_models/subject{num}_crisp_v.mnc'
        making_many_2_d_phantoms(tissue_dict,
                                 param_dict_T1,
                                 param_dict_T2,
                                 param_dict_PD,
                                 param_dict_T2_star,
                                 have_params,
                      'new_phantoms_train',
                                 bound_dict_T1=bound_dict_T1,
                                 bound_dict_T2=None,
                                 proton_density=False,
                                 print_process=False,
                                 filling_type='from_distribution',
                                 path_to_anatomical_model=path_to_anatomical_model)

    for num in numbers_of_test_subj:
        path_to_anatomical_model = f'anatomic_models/subject{num}_crisp_v.mnc'
        making_many_2_d_phantoms(tissue_dict,
                                 param_dict_T1,
                                 param_dict_T2,
                                 param_dict_PD,
                                 param_dict_T2_star,
                                 have_params,
                                 'new_phantoms_test',
                                 bound_dict_T1=bound_dict_T1,
                                 bound_dict_T2=None,
                                 proton_density=False,
                                 print_process=False,
                                 filling_type='from_distribution',
                                 path_to_anatomical_model=path_to_anatomical_model)