# -*- coding:utf-8 -*-

import glob
import cv2
import numpy as np
import random
from torch.utils.data import Dataset
import torch
import scipy.io as io
from data_tools import sigma_estimate, gaussian_kernel
import netCDF4 as nc
import matplotlib.pyplot as plt
import os



patch_size, stride = 20, 10
batch_size = 128


class ReconstructingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        Label (Tensor): clean image patches
        Data: label with mask
        Mask: mask
    """
    def __init__(self, Label, Data, Mask):
        super(ReconstructingDataset, self).__init__()
        self.Label = Label
        self.Data = Data
        self.Mask = Mask

    def __getitem__(self, index):
        batch_label = self.Label[index]
        batch_data = self.Data[index]
        batch_mask = self.Mask[index]
        return batch_label, batch_data, batch_mask

    def __len__(self):
        return self.Label.size(0)



def datagenerator(data_dir='D:/AMSR2/Data/2019'):
    mask_dir = 'D:/AMSR2/Mask/2019'
    Label = []
    Mask = []
    Data = []
    files = os.listdir(data_dir)
    Mask_global = io.loadmat('D:/AMSR2/Land_mask_float.mat')
    Mask_land = Mask_global['Land_mask_float']
    num_land_pixels = Mask_land.sum()

    patch_size = 40
    stride = 10
    patch_num = 0
    x1 = np.empty([patch_size, patch_size, 9], dtype='float32')
    x2 = np.empty([patch_size, patch_size, 9], dtype='float32')

    for i in range(4, 365 - 4, 1):
        Temp_mask_cur = io.loadmat(mask_dir + '/Mask_' + str(i + 1) + '.mat')
        Mask_cur = Temp_mask_cur['mask']

        Temp_mask_1 = io.loadmat(mask_dir + '/Mask_' + str(i + 1 - 4) + '.mat')
        Mask_1 = Temp_mask_1['mask']
        Temp_mask_2 = io.loadmat(mask_dir + '/Mask_' + str(i + 1 - 3) + '.mat')
        Mask_2 = Temp_mask_2['mask']
        Temp_mask_3 = io.loadmat(mask_dir + '/Mask_' + str(i + 1 - 2) + '.mat')
        Mask_3 = Temp_mask_3['mask']
        Temp_mask_4 = io.loadmat(mask_dir + '/Mask_' + str(i + 1 - 1) + '.mat')
        Mask_4 = Temp_mask_4['mask']
        Temp_mask_5 = io.loadmat(mask_dir + '/Mask_' + str(i + 1 + 1) + '.mat')
        Mask_5 = Temp_mask_5['mask']
        Temp_mask_6 = io.loadmat(mask_dir + '/Mask_' + str(i + 1 + 2) + '.mat')
        Mask_6 = Temp_mask_6['mask']
        Temp_mask_7 = io.loadmat(mask_dir + '/Mask_' + str(i + 1 + 3) + '.mat')
        Mask_7 = Temp_mask_7['mask']
        Temp_mask_8 = io.loadmat(mask_dir + '/Mask_' + str(i + 1 + 4) + '.mat')
        Mask_8 = Temp_mask_8['mask']
        Mask_all = Mask_1 + Mask_2 + Mask_3 + Mask_4 + Mask_5 + Mask_6 + Mask_7 + Mask_8
        Mask_all_final = Mask_all >= 1
        Mask_all_final_float = Mask_all_final.astype(float)
        num_cur_land_pixels = Mask_all_final_float.sum()
        coverage = num_cur_land_pixels / num_land_pixels

        Position_cur = data_dir + '/' + files[i]
        Data_cur = nc.Dataset(Position_cur)
        SM_cur = Data_cur.variables['soil_moisture_c1']
        SM_cur = np.transpose(SM_cur)


        Position_1 = data_dir + '/' + files[i - 4]
        Data_1 = nc.Dataset(Position_1)
        SM_1 = Data_1.variables['soil_moisture_c1']
        SM_1 = np.transpose(SM_1)
        Position_2 = data_dir + '/' + files[i - 3]
        Data_2 = nc.Dataset(Position_2)
        SM_2 = Data_2.variables['soil_moisture_c1']
        SM_2 = np.transpose(SM_2)
        Position_3 = data_dir + '/' + files[i - 2]
        Data_3 = nc.Dataset(Position_3)
        SM_3 = Data_3.variables['soil_moisture_c1']
        SM_3 = np.transpose(SM_3)
        Position_4 = data_dir + '/' + files[i - 1]
        Data_4 = nc.Dataset(Position_4)
        SM_4 = Data_4.variables['soil_moisture_c1']
        SM_4 = np.transpose(SM_4)
        Position_5 = data_dir + '/' + files[i + 1]
        Data_5 = nc.Dataset(Position_5)
        SM_5 = Data_5.variables['soil_moisture_c1']
        SM_5 = np.transpose(SM_5)
        Position_6 = data_dir + '/' + files[i + 2]
        Data_6 = nc.Dataset(Position_6)
        SM_6 = Data_6.variables['soil_moisture_c1']
        SM_6 = np.transpose(SM_6)
        Position_7 = data_dir + '/' + files[i + 3]
        Data_7 = nc.Dataset(Position_7)
        SM_7 = Data_7.variables['soil_moisture_c1']
        SM_7 = np.transpose(SM_7)
        Position_8 = data_dir + '/' + files[i + 4]
        Data_8 = nc.Dataset(Position_8)
        SM_8 = Data_8.variables['soil_moisture_c1']
        SM_8 = np.transpose(SM_8)

        for w in range(0, 600, stride):
            for h in range(0, 1440, stride):
                cur_patch_land = Mask_land[w:w + patch_size, h:h + patch_size]
                cur_num_land_pixels = cur_patch_land.sum()
                cur_patch_mask = Mask_cur[w:w + patch_size, h:h + patch_size]
                cur_num_patch_pixels = cur_patch_mask.sum()
                cur_all_patch_mask = Mask_all_final_float[w:w + patch_size, h:h + patch_size]
                cur_num_all_patch_pixels = cur_all_patch_mask.sum()

                if cur_num_land_pixels == patch_size * patch_size:
                    if cur_num_patch_pixels == patch_size * patch_size:
                        if cur_num_all_patch_pixels >= patch_size * patch_size * 0.9:
                            patch_num = patch_num + 1
                            simulated_mask_name = 'D:/AMSR2/Simulated_Mask_40/Mask_' + str(patch_num % 4946 + 1)
                            Temp_mat = io.loadmat(simulated_mask_name)
                            Mask_simulated = Temp_mat['simulated_mask']

                            SM_label = SM_cur[w:w + patch_size, h:h + patch_size] / 100.0
                            SM_data = Mask_simulated.copy() * SM_label.copy()
                            SM_1_patch = Mask_1[w:w + patch_size, h:h + patch_size] * SM_1[w:w + patch_size,
                                                                                      h:h + patch_size] / 100.0
                            SM_2_patch = Mask_2[w:w + patch_size, h:h + patch_size] * SM_2[w:w + patch_size,
                                                                                      h:h + patch_size] / 100.0
                            SM_3_patch = Mask_3[w:w + patch_size, h:h + patch_size] * SM_3[w:w + patch_size,
                                                                                      h:h + patch_size] / 100.0
                            SM_4_patch = Mask_4[w:w + patch_size, h:h + patch_size] * SM_4[w:w + patch_size,
                                                                                      h:h + patch_size] / 100.0
                            SM_5_patch = Mask_5[w:w + patch_size, h:h + patch_size] * SM_5[w:w + patch_size,
                                                                                      h:h + patch_size] / 100.0
                            SM_6_patch = Mask_6[w:w + patch_size, h:h + patch_size] * SM_6[w:w + patch_size,
                                                                                      h:h + patch_size] / 100.0
                            SM_7_patch = Mask_7[w:w + patch_size, h:h + patch_size] * SM_7[w:w + patch_size,
                                                                                      h:h + patch_size] / 100.0
                            SM_8_patch = Mask_8[w:w + patch_size, h:h + patch_size] * SM_8[w:w + patch_size,
                                                                                      h:h + patch_size] / 100.0

                            x1[:, :, 0] = SM_1_patch
                            x1[:, :, 1] = SM_2_patch
                            x1[:, :, 2] = SM_3_patch
                            x1[:, :, 3] = SM_4_patch
                            x1[:, :, 4] = SM_data.copy()
                            x1[:, :, 5] = SM_5_patch
                            x1[:, :, 6] = SM_6_patch
                            x1[:, :, 7] = SM_7_patch
                            x1[:, :, 8] = SM_8_patch

                            x2[:, :, 0] = Mask_1[w:w + patch_size, h:h + patch_size]
                            x2[:, :, 1] = Mask_2[w:w + patch_size, h:h + patch_size]
                            x2[:, :, 2] = Mask_3[w:w + patch_size, h:h + patch_size]
                            x2[:, :, 3] = Mask_4[w:w + patch_size, h:h + patch_size]
                            x2[:, :, 4] = Mask_simulated.copy()
                            x2[:, :, 5] = Mask_5[w:w + patch_size, h:h + patch_size]
                            x2[:, :, 6] = Mask_6[w:w + patch_size, h:h + patch_size]
                            x2[:, :, 7] = Mask_7[w:w + patch_size, h:h + patch_size]
                            x2[:, :, 8] = Mask_8[w:w + patch_size, h:h + patch_size]

                            Label.append(SM_label.copy())
                            Data.append(x1.copy())
                            Mask.append(x2.copy())

        print('Date: ' + str(i + 1) + ';      Coverage: ' + str(coverage) + ';      Patch Numbers: ' + str(patch_num))

    Label = np.array(Label)
    Data = np.array(Data)
    Mask = np.array(Mask)
    discard_n = len(Label) - len(Label) // batch_size * batch_size  # because of batch namalization
    Label = np.delete(Label, range(discard_n), axis=0)
    Data = np.delete(Data, range(discard_n), axis=0)
    Mask = np.delete(Mask, range(discard_n), axis=0)
    # print(Label.shape)
    # print(Data.shape)
    # print(Mask.shape)

    Label = Label[:, :, :, np.newaxis]
    Label = torch.from_numpy(Label.transpose((0, 3, 1, 2)))
    Data = torch.from_numpy(Data.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
    Mask = torch.from_numpy(Mask.transpose((0, 3, 1, 2)))
    # print(Label.shape)
    # print(Data.shape)
    # print(Mask.shape)
    print('- - Generating Patch Finished! - -')
    return Label, Data, Mask


if __name__ == '__main__':
    data_dir = 'D:/AMSR2/Data/2019/'
    mask_dir = 'D:/AMSR2/Mask/2019/'
    Label = []
    Mask = []
    Data = []
    files = os.listdir(data_dir)
    Mask_global = io.loadmat('D:/AMSR2/Land_mask_float.mat')
    Mask_land = Mask_global['Land_mask_float']
    num_land_pixels = Mask_land.sum()

    patch_size = 20
    stride = 10
    patch_num = 0
    x1 = np.empty([patch_size, patch_size, 9], dtype='float32')
    x2 = np.empty([patch_size, patch_size, 9], dtype='float32')

    for i in range(4, 365 - 4, 400):
        Temp_mask_cur = io.loadmat(mask_dir + '/Mask_' + str(i + 1) + '.mat')
        Mask_cur = Temp_mask_cur['mask']

        Temp_mask_1 = io.loadmat(mask_dir + '/Mask_' + str(i + 1 - 4) + '.mat')
        Mask_1 = Temp_mask_1['mask']
        Temp_mask_2 = io.loadmat(mask_dir + '/Mask_' + str(i + 1 - 3) + '.mat')
        Mask_2 = Temp_mask_2['mask']
        Temp_mask_3 = io.loadmat(mask_dir + '/Mask_' + str(i + 1 - 2) + '.mat')
        Mask_3 = Temp_mask_3['mask']
        Temp_mask_4 = io.loadmat(mask_dir + '/Mask_' + str(i + 1 - 1) + '.mat')
        Mask_4 = Temp_mask_4['mask']
        Temp_mask_5 = io.loadmat(mask_dir + '/Mask_' + str(i + 1 + 1) + '.mat')
        Mask_5 = Temp_mask_5['mask']
        Temp_mask_6 = io.loadmat(mask_dir + '/Mask_' + str(i + 1 + 2) + '.mat')
        Mask_6 = Temp_mask_6['mask']
        Temp_mask_7 = io.loadmat(mask_dir + '/Mask_' + str(i + 1 + 3) + '.mat')
        Mask_7 = Temp_mask_7['mask']
        Temp_mask_8 = io.loadmat(mask_dir + '/Mask_' + str(i + 1 + 4) + '.mat')
        Mask_8 = Temp_mask_8['mask']
        Mask_all = Mask_1 + Mask_2 + Mask_3 + Mask_4 + Mask_5 + Mask_6 + Mask_7 + Mask_8
        Mask_all_final = Mask_all >= 1
        Mask_all_final_float = Mask_all_final.astype(float)
        num_cur_land_pixels = Mask_all_final_float.sum()
        coverage = num_cur_land_pixels / num_land_pixels

        Position_cur = data_dir + '/' + files[i]
        Data_cur = nc.Dataset(Position_cur)
        SM_cur = Data_cur.variables['soil_moisture_c1']
        SM_cur = np.transpose(SM_cur)

        Position_1 = data_dir + '/' + files[i - 4]
        Data_1 = nc.Dataset(Position_1)
        SM_1 = Data_1.variables['soil_moisture_c1']
        SM_1 = np.transpose(SM_1)
        Position_2 = data_dir + '/' + files[i - 3]
        Data_2 = nc.Dataset(Position_2)
        SM_2 = Data_2.variables['soil_moisture_c1']
        SM_2 = np.transpose(SM_2)
        Position_3 = data_dir + '/' + files[i - 2]
        Data_3 = nc.Dataset(Position_3)
        SM_3 = Data_3.variables['soil_moisture_c1']
        SM_3 = np.transpose(SM_3)
        Position_4 = data_dir + '/' + files[i - 1]
        Data_4 = nc.Dataset(Position_4)
        SM_4 = Data_4.variables['soil_moisture_c1']
        SM_4 = np.transpose(SM_4)
        Position_5 = data_dir + '/' + files[i + 1]
        Data_5 = nc.Dataset(Position_5)
        SM_5 = Data_5.variables['soil_moisture_c1']
        SM_5 = np.transpose(SM_5)
        Position_6 = data_dir + '/' + files[i + 2]
        Data_6 = nc.Dataset(Position_6)
        SM_6 = Data_6.variables['soil_moisture_c1']
        SM_6 = np.transpose(SM_6)
        Position_7 = data_dir + '/' + files[i + 3]
        Data_7 = nc.Dataset(Position_7)
        SM_7 = Data_7.variables['soil_moisture_c1']
        SM_7 = np.transpose(SM_7)
        Position_8 = data_dir + '/' + files[i + 4]
        Data_8 = nc.Dataset(Position_8)
        SM_8 = Data_8.variables['soil_moisture_c1']
        SM_8 = np.transpose(SM_8)

        for w in range(0, 720, stride):
            for h in range(0, 1440, stride):
                cur_patch_land = Mask_land[w:w + patch_size, h:h + patch_size]
                cur_num_land_pixels = cur_patch_land.sum()
                cur_patch_mask = Mask_cur[w:w + patch_size, h:h + patch_size]
                cur_num_patch_pixels = cur_patch_mask.sum()
                cur_all_patch_mask = Mask_all_final_float[w:w + patch_size, h:h + patch_size]
                cur_num_all_patch_pixels = cur_all_patch_mask.sum()

                if cur_num_land_pixels == patch_size * patch_size:
                    if cur_num_patch_pixels == patch_size * patch_size:
                        if cur_num_all_patch_pixels == patch_size * patch_size:
                            patch_num = patch_num + 1
                            simulated_mask_name = 'D:/AMSR2/Simulated_Mask/Mask_' + str(patch_num % 4946 + 1)
                            Temp_mat = io.loadmat(simulated_mask_name)
                            Mask_simulated = Temp_mat['simulated_mask']

                            SM_label = SM_cur[w:w + patch_size, h:h + patch_size] / 100.0
                            SM_data = Mask_simulated.copy() * SM_label.copy()
                            SM_1_patch = Mask_1[w:w + patch_size, h:h + patch_size] * SM_1[w:w + patch_size,
                                                                                      h:h + patch_size] / 100.0
                            SM_2_patch = Mask_2[w:w + patch_size, h:h + patch_size] * SM_2[w:w + patch_size,
                                                                                      h:h + patch_size] / 100.0
                            SM_3_patch = Mask_3[w:w + patch_size, h:h + patch_size] * SM_3[w:w + patch_size,
                                                                                      h:h + patch_size] / 100.0
                            SM_4_patch = Mask_4[w:w + patch_size, h:h + patch_size] * SM_4[w:w + patch_size,
                                                                                      h:h + patch_size] / 100.0
                            SM_5_patch = Mask_5[w:w + patch_size, h:h + patch_size] * SM_5[w:w + patch_size,
                                                                                      h:h + patch_size] / 100.0
                            SM_6_patch = Mask_6[w:w + patch_size, h:h + patch_size] * SM_6[w:w + patch_size,
                                                                                      h:h + patch_size] / 100.0
                            SM_7_patch = Mask_7[w:w + patch_size, h:h + patch_size] * SM_7[w:w + patch_size,
                                                                                      h:h + patch_size] / 100.0
                            SM_8_patch = Mask_8[w:w + patch_size, h:h + patch_size] * SM_8[w:w + patch_size,
                                                                                      h:h + patch_size] / 100.0

                            x1[:, :, 0] = SM_1_patch
                            x1[:, :, 1] = SM_2_patch
                            x1[:, :, 2] = SM_3_patch
                            x1[:, :, 3] = SM_4_patch
                            x1[:, :, 4] = SM_data.copy()
                            x1[:, :, 5] = SM_5_patch
                            x1[:, :, 6] = SM_6_patch
                            x1[:, :, 7] = SM_7_patch
                            x1[:, :, 8] = SM_8_patch

                            x2[:, :, 0] = Mask_1[w:w + patch_size, h:h + patch_size]
                            x2[:, :, 1] = Mask_2[w:w + patch_size, h:h + patch_size]
                            x2[:, :, 2] = Mask_3[w:w + patch_size, h:h + patch_size]
                            x2[:, :, 3] = Mask_4[w:w + patch_size, h:h + patch_size]
                            x2[:, :, 4] = Mask_simulated.copy()
                            x2[:, :, 5] = Mask_5[w:w + patch_size, h:h + patch_size]
                            x2[:, :, 6] = Mask_6[w:w + patch_size, h:h + patch_size]
                            x2[:, :, 7] = Mask_7[w:w + patch_size, h:h + patch_size]
                            x2[:, :, 8] = Mask_8[w:w + patch_size, h:h + patch_size]



                            Label.append(SM_label.copy())
                            Data.append(x1.copy())
                            Mask.append(x2.copy())



        print('Date: ' + str(i + 1) + ';      Coverage: ' + str(coverage) + ';      Patch Numbers: ' + str(patch_num))



    Label2 = np.array(Label)
    Data2 = np.array(Data)
    Mask2 = np.array(Mask)



    discard_n = len(Label2) - len(Label2) // batch_size * batch_size  # because of batch namalization
    Label2 = np.delete(Label2, range(discard_n), axis=0)
    Data2 = np.delete(Data2, range(discard_n), axis=0)
    Mask2 = np.delete(Mask2, range(discard_n), axis=0)
    # print(Label.shape)
    # print(Data.shape)
    # print(Mask.shape)
    plt.matshow(Label2[100, :, :])
    plt.matshow(Data2[100, :, :, 3])
    plt.matshow(Mask2[100, :, :, 3])
    plt.show()

    Label3 = Label2[:, :, :, np.newaxis]
    Label4 = torch.from_numpy(Label3.transpose((0, 3, 1, 2)))
    Data3 = torch.from_numpy(Data2.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
    Mask3 = torch.from_numpy(Mask2.transpose((0, 3, 1, 2)))
    # print(Label.shape)
    # print(Data.shape)
    # print(Mask.shape)
    print('- - Generating Patch Finished! - -')
