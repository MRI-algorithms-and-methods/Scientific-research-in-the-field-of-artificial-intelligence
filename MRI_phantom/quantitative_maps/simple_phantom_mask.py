import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import h5py
from PIL import Image
from sklearn.model_selection import train_test_split

# Creates one quantitative map (for one modality, T1 for example)

def create_phantom(tissue_dict,
                   param_dict,
                   have_params,
                   data,
                   bound_dict=None,
                   proton_density=False,
                   print_process=False,
                   filling_type='from_distribution'):

    pil_data = Image.fromarray(data)
    resize_pil = pil_data.resize((128,128),resample = 0)
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

    #if proton_density is False:
        #phantom = phantom * 1e-3

    return phantom


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


numbers_of_train_subj = ['04', '05', '06']

numbers_of_test_subj = ['18']

def create_train_test(numbers_of_train_subj, numbers_of_test_subj):
    for num in numbers_of_train_subj:
        path_to_anatomical_model = f'anatomic_models/subject{num}_crisp_v.mnc'

        anatomical_model = nib.load(path_to_anatomical_model)
        data_mask = anatomical_model.get_fdata()

        for i in range(data_mask.shape[0]):
            if len(np.unique(data_mask[i, :, :])) > 1:
                ph = create_phantom(tissue_dict,
                                    param_dict_T1,
                                    have_params,
                                    data_mask[i, :, :],
                                    bound_dict=bound_dict_T1,
                                    proton_density=False,
                                    print_process=False,
                                    filling_type='from_distribution')
                name = path_to_anatomical_model.split('/')[-1][:-4]
                np.save(f'train_phantoms/{name}_{i}.npy', ph)


    for num in numbers_of_test_subj:
        path_to_anatomical_model = f'anatomic_models/subject{num}_crisp_v.mnc'

        anatomical_model = nib.load(path_to_anatomical_model)
        data_mask = anatomical_model.get_fdata()

        for i in range(data_mask.shape[0]):
            if len(np.unique(data_mask[i, :, :])) > 1:
                ph = create_phantom(tissue_dict,
                                    param_dict_T1,
                                    have_params,
                                    data_mask[i, :, :],
                                    bound_dict=bound_dict_T1,
                                    proton_density=False,
                                    print_process=False,
                                    filling_type='from_distribution')
                name = path_to_anatomical_model.split('/')[-1][:-4]
                np.save(f'test_phantoms/{name}_{i}.npy', ph)




create_train_test(numbers_of_train_subj, numbers_of_test_subj)