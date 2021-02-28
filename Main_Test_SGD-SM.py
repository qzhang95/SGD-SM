
# Run this to output the SGD-SM Products

import argparse
import os, time, datetime
import PIL.Image as Image
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
import scipy.io as io
from matplotlib import pyplot as plt
from skimage import img_as_float, img_as_ubyte
import random
from numpy.linalg import norm
import netCDF4 as nc
from Save_nc import Save_to_NC
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='D:/AMSR2/Data/', type=str, help='directory of test dataset')
    parser.add_argument('--model_dir', default=os.path.join('models', 'SGD_3DCNN'), help='directory of the model')
    parser.add_argument('--model_name', default='model_500.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=0, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()



class SGD(nn.Module):
    def __init__(self, depth=7, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(SGD, self).__init__()
        kernel_size = 3
        padding = 1

        self.spatial_conv = nn.PConv3d(1, 32, 3, 3, 3, 1, 1)
        self.spectral_conv = nn.PConv3d(9, 32, 3, 3, 3, 1, 1)
        self.mid_conv = nn.PConv3d(64, 64, 3, 3, 3, 1, 1)
        self.last_conv = nn.PConv3d(64, 1, 3, 3, 3, 1, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x1, x2, mask):
        x1 = x1.unsqueeze(1)
        mask = 1 - mask.unsqueeze(1)
        f1 = self.spatial_conv(x1)
        f2 = self.spectral_conv(x2)
        x = torch.cat([f1, f2], 1)
        x = self.relu(x)
        x = self.mid_conv(x)
        x = self.relu(x)
        x = self.mid_conv(x)
        x = self.relu(x)
        x = self.mid_conv(x)
        x = self.relu(x)
        x = self.mid_conv(x)
        x = self.relu(x)
        x = self.mid_conv(x)
        x = self.relu(x)
        x = self.mid_conv(x)
        x = self.relu(x)
        x = self.mid_conv(x)
        x = self.relu(x)
        x = self.mid_conv(x)
        x = self.relu(x)
        res = self.last_conv(x)
        final = mask * res

        return final

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.PConv3d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

				
				
				
if __name__ == '__main__':

    args = parse_args()

    # model
    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):

        model = torch.load(os.path.join(args.model_dir, 'model_500.pth'))
        # load weights into new model
        log('load trained model on AMSR2 dataset')
    else:
        # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
        model = torch.load(os.path.join(args.model_dir, args.model_name))
        log('Load model...')

    model.eval()  # evaluation mode

    if torch.cuda.is_available():
        model = model.cuda()

    year = 2013
    data_dir = args.data_dir + str(year)
    data_dir_before = 'D:/AMSR2/Data/' + str(year-1)
    data_dir_later = 'D:/AMSR2/Data/' + str(year+1)

    mask_dir = 'D:/AMSR2/Mask/' + str(year)
    mask_dir_before = 'D:/AMSR2/Mask/' + str(year-1)
    mask_dir_later = 'D:/AMSR2/Mask/' + str(year+1)

    save_dir = 'D:/AMSR2/Code/results/' + str(year) + '/'
    files = os.listdir(args.data_dir + str(year))
    files_before =  os.listdir(data_dir_before)
    files_later = os.listdir(data_dir_later)

    Mask_global = io.loadmat('D:/AMSR2/Land_mask_float.mat')
    Mask_land = Mask_global['Land_mask_float']
    num_land_pixels = Mask_land.sum()

    patch_size= 600
    patch_size2 = 840
    stride = 300
    patch_num = 0
    x1 = np.empty([patch_size, patch_size, 9], dtype='float32')
    x2 = np.empty([patch_size, patch_size, 9], dtype='float32')
    x3 = np.empty([patch_size, patch_size2, 9], dtype='float32')
    x4 = np.empty([patch_size, patch_size2, 9], dtype='float32')
    Final_global = np.zeros([720, 1440], dtype='float32')
    Temp_mask_all = np.empty([720, 1440, 12], dtype='float32')
    Temp_data_all = np.empty([720, 1440, 12], dtype='float32')

    print('--Test Begin...')

    #Day: 1-4
    m_1228 =  io.loadmat(mask_dir_before + '/Mask_' + str(files_before.__len__()-3) + '.mat')
    Temp_mask_all[:, :, 0] = m_1228['mask']
    m_1229 =  io.loadmat(mask_dir_before + '/Mask_' + str(files_before.__len__()-2) + '.mat')
    Temp_mask_all[:, :, 1] = m_1229['mask']
    m_1230 =  io.loadmat(mask_dir_before + '/Mask_' + str(files_before.__len__()-1) + '.mat')
    Temp_mask_all[:, :, 2] = m_1230['mask']
    m_1231 =  io.loadmat(mask_dir_before + '/Mask_' + str(files_before.__len__()) + '.mat')
    Temp_mask_all[:, :, 3] = m_1231['mask']
    m_0101 =  io.loadmat(mask_dir + '/Mask_1' + '.mat')
    Temp_mask_all[:, :, 4] = m_0101['mask']
    m_0102 =  io.loadmat(mask_dir + '/Mask_2' + '.mat')
    Temp_mask_all[:, :, 5] = m_0102['mask']
    m_0103 =  io.loadmat(mask_dir + '/Mask_3' + '.mat')
    Temp_mask_all[:, :, 6] = m_0103['mask']
    m_0104 =  io.loadmat(mask_dir + '/Mask_4' + '.mat')
    Temp_mask_all[:, :, 7] = m_0104['mask']
    m_0105 =  io.loadmat(mask_dir + '/Mask_5' + '.mat')
    Temp_mask_all[:, :, 8] = m_0105['mask']
    m_0106 =  io.loadmat(mask_dir + '/Mask_6' + '.mat')
    Temp_mask_all[:, :, 9] = m_0106['mask']
    m_0107 =  io.loadmat(mask_dir + '/Mask_7' + '.mat')
    Temp_mask_all[:, :,10] = m_0107['mask']
    m_0108 =  io.loadmat(mask_dir + '/Mask_8' + '.mat')
    Temp_mask_all[:, :,11] = m_0108['mask']

    Position_d_1228 = data_dir_before + '/' + files_before[files_before.__len__()-4]
    Data_1228 = nc.Dataset(Position_d_1228)
    SM_1228 = Data_1228.variables['soil_moisture_c1']
    Temp_data_all[:, :, 0] = np.transpose(SM_1228)
    Position_d_1229 = data_dir_before + '/' + files_before[files_before.__len__()-3]
    Data_1229 = nc.Dataset(Position_d_1229)
    SM_1229 = Data_1229.variables['soil_moisture_c1']
    Temp_data_all[:, :, 1] = np.transpose(SM_1229)
    Position_d_1230 = data_dir_before + '/' + files_before[files_before.__len__()-2]
    Data_1230 = nc.Dataset(Position_d_1230)
    SM_1230 = Data_1230.variables['soil_moisture_c1']
    Temp_data_all[:, :, 2] = np.transpose(SM_1230)
    Position_d_1231 = data_dir_before + '/' + files_before[files_before.__len__()-1]
    Data_1231 = nc.Dataset(Position_d_1231)
    SM_1231 = Data_1231.variables['soil_moisture_c1']
    Temp_data_all[:, :, 3] = np.transpose(SM_1231)
    Position_d_0101 = data_dir + '/' + files[0]
    Data_0101 = nc.Dataset(Position_d_0101)
    SM_0101 = Data_0101.variables['soil_moisture_c1']
    Temp_data_all[:, :, 4] = np.transpose(SM_0101)
    Position_d_0102 = data_dir + '/' + files[1]
    Data_0102 = nc.Dataset(Position_d_0102)
    SM_0102 = Data_0102.variables['soil_moisture_c1']
    Temp_data_all[:, :, 5] = np.transpose(SM_0102)
    Position_d_0103 = data_dir + '/' + files[2]
    Data_0103 = nc.Dataset(Position_d_0103)
    SM_0103 = Data_0103.variables['soil_moisture_c1']
    Temp_data_all[:, :, 6] = np.transpose(SM_0103)
    Position_d_0104 = data_dir + '/' + files[3]
    Data_0104 = nc.Dataset(Position_d_0104)
    SM_0104 = Data_0104.variables['soil_moisture_c1']
    Temp_data_all[:, :, 7] = np.transpose(SM_0104)
    Position_d_0105 = data_dir + '/' + files[4]
    Data_0105 = nc.Dataset(Position_d_0105)
    SM_0105 = Data_0105.variables['soil_moisture_c1']
    Temp_data_all[:, :, 8] = np.transpose(SM_0105)
    Position_d_0106 = data_dir + '/' + files[5]
    Data_0106 = nc.Dataset(Position_d_0106)
    SM_0106 = Data_0106.variables['soil_moisture_c1']
    Temp_data_all[:, :, 9] = np.transpose(SM_0106)
    Position_d_0107 = data_dir + '/' + files[6]
    Data_0107 = nc.Dataset(Position_d_0107)
    SM_0107 = Data_0107.variables['soil_moisture_c1']
    Temp_data_all[:, :, 10] = np.transpose(SM_0107)
    Position_d_0108 = data_dir + '/' + files[7]
    Data_0108 = nc.Dataset(Position_d_0108)
    SM_0108 = Data_0108.variables['soil_moisture_c1']
    Temp_data_all[:, :, 11] = np.transpose(SM_0108)

    # start
    for i in range(0, 4, 1):
        Temp_mask_cur = io.loadmat(mask_dir + '/Mask_' + str(i + 1) + '.mat')
        Mask_cur = Temp_mask_cur['mask']

        Mask_1 = Temp_mask_all[:, :, i + 4 - 4]
        Mask_2 = Temp_mask_all[:, :, i + 4 - 3]
        Mask_3 = Temp_mask_all[:, :, i + 4 - 2]
        Mask_4 = Temp_mask_all[:, :, i + 4 - 1]
        Mask_5 = Temp_mask_all[:, :, i + 4 + 1]
        Mask_6 = Temp_mask_all[:, :, i + 4 + 2]
        Mask_7 = Temp_mask_all[:, :, i + 4 + 3]
        Mask_8 = Temp_mask_all[:, :, i + 4 + 4]

        Mask_all = Mask_1 + Mask_2 + Mask_3 + Mask_4 + Mask_5 + Mask_6 + Mask_7 + Mask_8
        Mask_all_final = Mask_all >= 1
        Mask_all_final_float = Mask_all_final.astype(float)
        num_cur_land_pixels = Mask_all_final_float.sum()
        coverage = num_cur_land_pixels / num_land_pixels
        print('Temporal Coverage: '+str(coverage))

        Position_cur = data_dir + '/' + files[i]
        Data_cur = nc.Dataset(Position_cur)
        SM_cur = Data_cur.variables['soil_moisture_c1']
        SM_cur = np.transpose(SM_cur)

        SM_1 = Temp_data_all[:, :, i + 4 - 4]
        SM_2 = Temp_data_all[:, :, i + 4 - 3]
        SM_3 = Temp_data_all[:, :, i + 4 - 2]
        SM_4 = Temp_data_all[:, :, i + 4 - 1]
        SM_5 = Temp_data_all[:, :, i + 4 + 1]
        SM_6 = Temp_data_all[:, :, i + 4 + 2]
        SM_7 = Temp_data_all[:, :, i + 4 + 3]
        SM_8 = Temp_data_all[:, :, i + 4 + 4]


        # Patch [0:600, 0:600] (South+North America)
        w, h = 0, 0
        cur_patch_land = Mask_land[w:w + patch_size, h:h + patch_size]
        cur_num_land_pixels = cur_patch_land.sum()
        cur_patch_mask = Mask_cur[w:w + patch_size, h:h + patch_size]
        cur_num_patch_pixels = cur_patch_mask.sum()
        cur_all_patch_mask = Mask_all_final_float[w:w + patch_size, h:h + patch_size]
        cur_num_all_patch_pixels = cur_all_patch_mask.sum()

        SM_data = Mask_cur[w:w + patch_size, h:h + patch_size] * SM_cur[w:w + patch_size, h:h + patch_size] / 100.0
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
        x2[:, :, 4] = Mask_cur[w:w + patch_size, h:h + patch_size]
        x2[:, :, 5] = Mask_5[w:w + patch_size, h:h + patch_size]
        x2[:, :, 6] = Mask_6[w:w + patch_size, h:h + patch_size]
        x2[:, :, 7] = Mask_7[w:w + patch_size, h:h + patch_size]
        x2[:, :, 8] = Mask_8[w:w + patch_size, h:h + patch_size]
        mask_cur = Mask_cur[w:w + patch_size, h:h + patch_size].astype(np.float32)

        y1_ = torch.from_numpy(x1.transpose((2, 0, 1))[np.newaxis,])
        mask_cur_ = torch.from_numpy((mask_cur)[np.newaxis,])

        torch.cuda.synchronize()
        start_time = time.time()
        y1_ = y1_.cuda()
        mask_cur_ = mask_cur_.cuda()
        y1_cur_ = y1_[:, 4, :, :]

        x_ = model(y1_cur_, y1_, mask_cur_)  # inference
        res_ = x_ + y1_cur_
        res_ = torch.clamp(res_, 0.0, 0.99)
        res_ = res_.squeeze()
        res_ = res_.cpu()
        res_ = res_.detach().numpy().astype(np.float32)
        res_ = res_ * (1 - Mask_cur[w:w + patch_size, h:h + patch_size]) + SM_data.copy()
        final = res_ * Mask_land[w:w + patch_size, h:h + patch_size]
        Final_global[0:600, 0:600] = final.copy()*100.0


        # Patch [0:600, 600:1440] (others)
        w, h = 0, 600
        cur_patch_land = Mask_land[w:w + patch_size, h:h + patch_size2]
        cur_num_land_pixels = cur_patch_land.sum()
        cur_patch_mask = Mask_cur[w:w + patch_size, h:h + patch_size2]
        cur_num_patch_pixels = cur_patch_mask.sum()
        cur_all_patch_mask = Mask_all_final_float[w:w + patch_size, h:h + patch_size2]
        cur_num_all_patch_pixels = cur_all_patch_mask.sum()

        SM_data = Mask_cur[w:w + patch_size, h:h + patch_size2] * SM_cur[w:w + patch_size, h:h + patch_size2] / 100.0
        SM_1_patch = Mask_1[w:w + patch_size, h:h + patch_size2] * SM_1[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_2_patch = Mask_2[w:w + patch_size, h:h + patch_size2] * SM_2[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_3_patch = Mask_3[w:w + patch_size, h:h + patch_size2] * SM_3[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_4_patch = Mask_4[w:w + patch_size, h:h + patch_size2] * SM_4[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_5_patch = Mask_5[w:w + patch_size, h:h + patch_size2] * SM_5[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_6_patch = Mask_6[w:w + patch_size, h:h + patch_size2] * SM_6[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_7_patch = Mask_7[w:w + patch_size, h:h + patch_size2] * SM_7[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_8_patch = Mask_8[w:w + patch_size, h:h + patch_size2] * SM_8[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0

        x3[:, :, 0] = SM_1_patch
        x3[:, :, 1] = SM_2_patch
        x3[:, :, 2] = SM_3_patch
        x3[:, :, 3] = SM_4_patch
        x3[:, :, 4] = SM_data.copy()
        x3[:, :, 5] = SM_5_patch
        x3[:, :, 6] = SM_6_patch
        x3[:, :, 7] = SM_7_patch
        x3[:, :, 8] = SM_8_patch

        x4[:, :, 0] = Mask_1[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 1] = Mask_2[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 2] = Mask_3[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 3] = Mask_4[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 4] = Mask_cur[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 5] = Mask_5[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 6] = Mask_6[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 7] = Mask_7[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 8] = Mask_8[w:w + patch_size, h:h + patch_size2]
        mask_cur = Mask_cur[w:w + patch_size, h:h + patch_size2].astype(np.float32)

        y1_ = torch.from_numpy(x3.transpose((2, 0, 1))[np.newaxis,])
        mask_cur_ = torch.from_numpy((mask_cur)[np.newaxis,])

        torch.cuda.synchronize()
        y1_ = y1_.cuda()
        mask_cur_ = mask_cur_.cuda()
        y1_cur_ = y1_[:, 4, :, :]

        x_ = model(y1_cur_, y1_, mask_cur_)  # inference
        res_ = x_ + y1_cur_
        res_ = torch.clamp(res_, 0.0, 0.99)
        res_ = res_.squeeze()
        res_ = res_.cpu()
        res_ = res_.detach().numpy().astype(np.float32)
        res_ = res_ * (1 - Mask_cur[w:w + patch_size, h:h + patch_size2]) + SM_data.copy()
        final = res_ * Mask_land[w:w + patch_size, h:h + patch_size2]
        Final_global[0:600, 600:1440] = final.copy()*100.0

        filename_all = files[i]
        nc_filename = save_dir + 'LPRM_AMSR2_'+ filename_all[28:36] + '.nc'
        res_all = Final_global + (1-Mask_land)* -32767 / Mask_land.astype(np.float32)
        ori_all = SM_cur / Mask_cur
        Save_to_NC(nc_filename, ori_all, res_all)
        log('Finished Date: ' + filename_all[28:36])

    # Day:  4 ~ Last-4
    for i in range(4, files.__len__()-4, 1):
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
        print('Temporal Coverage: '+str(coverage))

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

        # Patch [0:600, 0:600] (South+North America)
        w, h = 0, 0
        cur_patch_land = Mask_land[w:w + patch_size, h:h + patch_size]
        cur_num_land_pixels = cur_patch_land.sum()
        cur_patch_mask = Mask_cur[w:w + patch_size, h:h + patch_size]
        cur_num_patch_pixels = cur_patch_mask.sum()
        cur_all_patch_mask = Mask_all_final_float[w:w + patch_size, h:h + patch_size]
        cur_num_all_patch_pixels = cur_all_patch_mask.sum()

        SM_data = Mask_cur[w:w + patch_size, h:h + patch_size] * SM_cur[w:w + patch_size, h:h + patch_size] / 100.0
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
        x2[:, :, 4] = Mask_cur[w:w + patch_size, h:h + patch_size]
        x2[:, :, 5] = Mask_5[w:w + patch_size, h:h + patch_size]
        x2[:, :, 6] = Mask_6[w:w + patch_size, h:h + patch_size]
        x2[:, :, 7] = Mask_7[w:w + patch_size, h:h + patch_size]
        x2[:, :, 8] = Mask_8[w:w + patch_size, h:h + patch_size]
        mask_cur = Mask_cur[w:w + patch_size, h:h + patch_size].astype(np.float32)

        y1_ = torch.from_numpy(x1.transpose((2, 0, 1))[np.newaxis,])
        mask_cur_ = torch.from_numpy((mask_cur)[np.newaxis,])

        torch.cuda.synchronize()
        start_time = time.time()
        y1_ = y1_.cuda()
        mask_cur_ = mask_cur_.cuda()
        y1_cur_ = y1_[:, 4, :, :]

        x_ = model(y1_cur_, y1_, mask_cur_)  # inference
        res_ = x_ + y1_cur_
        res_ = torch.clamp(res_, 0.0, 0.99)
        res_ = res_.squeeze()
        res_ = res_.cpu()
        res_ = res_.detach().numpy().astype(np.float32)
        res_ = res_ * (1 - Mask_cur[w:w + patch_size, h:h + patch_size]) + SM_data.copy()
        final = res_ * Mask_land[w:w + patch_size, h:h + patch_size]
        Final_global[0:600, 0:600] = final.copy()*100.0


        # Patch [0:600, 600:1440] (others)
        w, h = 0, 600
        cur_patch_land = Mask_land[w:w + patch_size, h:h + patch_size2]
        cur_num_land_pixels = cur_patch_land.sum()
        cur_patch_mask = Mask_cur[w:w + patch_size, h:h + patch_size2]
        cur_num_patch_pixels = cur_patch_mask.sum()
        cur_all_patch_mask = Mask_all_final_float[w:w + patch_size, h:h + patch_size2]
        cur_num_all_patch_pixels = cur_all_patch_mask.sum()

        SM_data = Mask_cur[w:w + patch_size, h:h + patch_size2] * SM_cur[w:w + patch_size, h:h + patch_size2] / 100.0
        SM_1_patch = Mask_1[w:w + patch_size, h:h + patch_size2] * SM_1[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_2_patch = Mask_2[w:w + patch_size, h:h + patch_size2] * SM_2[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_3_patch = Mask_3[w:w + patch_size, h:h + patch_size2] * SM_3[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_4_patch = Mask_4[w:w + patch_size, h:h + patch_size2] * SM_4[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_5_patch = Mask_5[w:w + patch_size, h:h + patch_size2] * SM_5[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_6_patch = Mask_6[w:w + patch_size, h:h + patch_size2] * SM_6[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_7_patch = Mask_7[w:w + patch_size, h:h + patch_size2] * SM_7[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_8_patch = Mask_8[w:w + patch_size, h:h + patch_size2] * SM_8[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0

        x3[:, :, 0] = SM_1_patch
        x3[:, :, 1] = SM_2_patch
        x3[:, :, 2] = SM_3_patch
        x3[:, :, 3] = SM_4_patch
        x3[:, :, 4] = SM_data.copy()
        x3[:, :, 5] = SM_5_patch
        x3[:, :, 6] = SM_6_patch
        x3[:, :, 7] = SM_7_patch
        x3[:, :, 8] = SM_8_patch

        x4[:, :, 0] = Mask_1[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 1] = Mask_2[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 2] = Mask_3[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 3] = Mask_4[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 4] = Mask_cur[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 5] = Mask_5[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 6] = Mask_6[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 7] = Mask_7[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 8] = Mask_8[w:w + patch_size, h:h + patch_size2]
        mask_cur = Mask_cur[w:w + patch_size, h:h + patch_size2].astype(np.float32)

        y1_ = torch.from_numpy(x3.transpose((2, 0, 1))[np.newaxis,])
        mask_cur_ = torch.from_numpy((mask_cur)[np.newaxis,])

        torch.cuda.synchronize()
        y1_ = y1_.cuda()
        mask_cur_ = mask_cur_.cuda()
        y1_cur_ = y1_[:, 4, :, :]

        x_ = model(y1_cur_, y1_, mask_cur_)  # inference
        res_ = x_ + y1_cur_
        res_ = torch.clamp(res_, 0.0, 0.99)
        res_ = res_.squeeze()
        res_ = res_.cpu()
        res_ = res_.detach().numpy().astype(np.float32)
        res_ = res_ * (1 - Mask_cur[w:w + patch_size, h:h + patch_size2]) + SM_data.copy()
        final = res_ * Mask_land[w:w + patch_size, h:h + patch_size2]
        Final_global[0:600, 600:1440] = final.copy()*100.0

        filename_all = files[i]
        nc_filename = save_dir + 'LPRM_AMSR2_'+ filename_all[28:36] + '.nc'
        res_all = Final_global + (1-Mask_land)* -32767 / Mask_land.astype(np.float32)

        ori_all = SM_cur / Mask_cur

        Save_to_NC(nc_filename, ori_all, res_all)

        log('Finished Date: ' + filename_all[28:36])

    # Day: last-4 ~ last
    m_1224 =  io.loadmat(mask_dir + '/Mask_' + str(files.__len__()-7) + '.mat')
    Temp_mask_all[:, :, 0] = m_1224['mask']
    m_1225 =  io.loadmat(mask_dir + '/Mask_' + str(files.__len__()-6) + '.mat')
    Temp_mask_all[:, :, 1] = m_1225['mask']
    m_1226 =  io.loadmat(mask_dir + '/Mask_' + str(files.__len__()-5) + '.mat')
    Temp_mask_all[:, :, 2] = m_1226['mask']
    m_1227 =  io.loadmat(mask_dir + '/Mask_' + str(files.__len__()-4) + '.mat')
    Temp_mask_all[:, :, 3] = m_1227['mask']
    m_1228 =  io.loadmat(mask_dir + '/Mask_' + str(files.__len__()-3) + '.mat')
    Temp_mask_all[:, :, 4] = m_1228['mask']
    m_1229 =  io.loadmat(mask_dir + '/Mask_' + str(files.__len__()-2) + '.mat')
    Temp_mask_all[:, :, 5] = m_1229['mask']
    m_1230 =  io.loadmat(mask_dir + '/Mask_' + str(files.__len__()-1) + '.mat')
    Temp_mask_all[:, :, 6] = m_1230['mask']
    m_1231 =  io.loadmat(mask_dir + '/Mask_' + str(files.__len__()) + '.mat')
    Temp_mask_all[:, :, 7] = m_1231['mask']
    m_0101 =  io.loadmat(mask_dir_later + '/Mask_1' + '.mat')
    Temp_mask_all[:, :, 8] = m_0101['mask']
    m_0102 =  io.loadmat(mask_dir_later + '/Mask_2' + '.mat')
    Temp_mask_all[:, :, 9] = m_0102['mask']
    m_0103 =  io.loadmat(mask_dir_later + '/Mask_3' + '.mat')
    Temp_mask_all[:, :,10] = m_0103['mask']
    m_0104 =  io.loadmat(mask_dir_later + '/Mask_4' + '.mat')
    Temp_mask_all[:, :,11] = m_0104['mask']

    Position_d_1224 = data_dir + '/' + files[files.__len__()-8]
    Data_1224 = nc.Dataset(Position_d_1224)
    SM_1224 = Data_1224.variables['soil_moisture_c1']
    Temp_data_all[:, :, 0] = np.transpose(SM_1224)
    Position_d_1225 = data_dir + '/' + files[files.__len__()-7]
    Data_1225 = nc.Dataset(Position_d_1225)
    SM_1225 = Data_1225.variables['soil_moisture_c1']
    Temp_data_all[:, :, 1] = np.transpose(SM_1225)
    Position_d_1226 = data_dir + '/' + files[files.__len__()-6]
    Data_1226 = nc.Dataset(Position_d_1226)
    SM_1226 = Data_1226.variables['soil_moisture_c1']
    Temp_data_all[:, :, 2] = np.transpose(SM_1226)
    Position_d_1227 = data_dir + '/' + files[files.__len__()-5]
    Data_1227 = nc.Dataset(Position_d_1227)
    SM_1227 = Data_1227.variables['soil_moisture_c1']
    Temp_data_all[:, :, 3] = np.transpose(SM_1227)
    Position_d_1228 = data_dir + '/' + files[files.__len__()-4]
    Data_1228 = nc.Dataset(Position_d_1228)
    SM_1228 = Data_1228.variables['soil_moisture_c1']
    Temp_data_all[:, :, 4] = np.transpose(SM_1228)
    Position_d_1229 = data_dir + '/' + files[files.__len__()-3]
    Data_1229 = nc.Dataset(Position_d_1229)
    SM_1229 = Data_1229.variables['soil_moisture_c1']
    Temp_data_all[:, :, 5] = np.transpose(SM_1229)
    Position_d_1230 = data_dir + '/' + files[files.__len__()-2]
    Data_1230 = nc.Dataset(Position_d_1230)
    SM_1230 = Data_1230.variables['soil_moisture_c1']
    Temp_data_all[:, :, 6] = np.transpose(SM_1230)
    Position_d_1231 = data_dir + '/' + files[files.__len__()-1]
    Data_1231 = nc.Dataset(Position_d_1231)
    SM_1231 = Data_1231.variables['soil_moisture_c1']
    Temp_data_all[:, :, 7] = np.transpose(SM_1231)
    Position_d_0101 = data_dir_later + '/' + files_later[0]
    Data_0101 = nc.Dataset(Position_d_0101)
    SM_0101 = Data_0101.variables['soil_moisture_c1']
    Temp_data_all[:, :, 8] = np.transpose(SM_0101)
    Position_d_0102 = data_dir_later + '/' + files_later[1]
    Data_0102 = nc.Dataset(Position_d_0102)
    SM_0102 = Data_0102.variables['soil_moisture_c1']
    Temp_data_all[:, :, 9] = np.transpose(SM_0102)
    Position_d_0103 = data_dir_later + '/' + files_later[2]
    Data_0103 = nc.Dataset(Position_d_0103)
    SM_0103 = Data_0103.variables['soil_moisture_c1']
    Temp_data_all[:, :, 10] = np.transpose(SM_0103)
    Position_d_0104 = data_dir_later + '/' + files_later[3]
    Data_0104 = nc.Dataset(Position_d_0104)
    SM_0104 = Data_0104.variables['soil_moisture_c1']
    Temp_data_all[:, :, 11] = np.transpose(SM_0104)

    # start
    for i in range(files.__len__()-4, files.__len__(), 1):
        Temp_mask_cur = io.loadmat(mask_dir + '/Mask_' + str(i + 1) + '.mat')
        Mask_cur = Temp_mask_cur['mask']

        Mask_1 = Temp_mask_all[:, :, i - files.__len__() + 4]
        Mask_2 = Temp_mask_all[:, :, i - files.__len__() + 5]
        Mask_3 = Temp_mask_all[:, :, i - files.__len__() + 6]
        Mask_4 = Temp_mask_all[:, :, i - files.__len__() + 7]
        Mask_5 = Temp_mask_all[:, :, i - files.__len__() + 9]
        Mask_6 = Temp_mask_all[:, :, i - files.__len__() + 10]
        Mask_7 = Temp_mask_all[:, :, i - files.__len__() + 11]
        Mask_8 = Temp_mask_all[:, :, i - files.__len__() + 12]

        Mask_all = Mask_1 + Mask_2 + Mask_3 + Mask_4 + Mask_5 + Mask_6 + Mask_7 + Mask_8
        Mask_all_final = Mask_all >= 1
        Mask_all_final_float = Mask_all_final.astype(float)
        num_cur_land_pixels = Mask_all_final_float.sum()
        coverage = num_cur_land_pixels / num_land_pixels
        print('Temporal Coverage: '+str(coverage))

        Position_cur = data_dir + '/' + files[i]
        Data_cur = nc.Dataset(Position_cur)
        SM_cur = Data_cur.variables['soil_moisture_c1']
        SM_cur = np.transpose(SM_cur)

        SM_1 = Temp_data_all[:, :, i - files.__len__() + 4]
        SM_2 = Temp_data_all[:, :, i - files.__len__() + 5]
        SM_3 = Temp_data_all[:, :, i - files.__len__() + 6]
        SM_4 = Temp_data_all[:, :, i - files.__len__() + 7]
        SM_5 = Temp_data_all[:, :, i - files.__len__() + 9]
        SM_6 = Temp_data_all[:, :, i - files.__len__() + 10]
        SM_7 = Temp_data_all[:, :, i - files.__len__() + 11]
        SM_8 = Temp_data_all[:, :, i - files.__len__() + 12]


        # Patch [0:600, 0:600] (South+North America)
        w, h = 0, 0
        cur_patch_land = Mask_land[w:w + patch_size, h:h + patch_size]
        cur_num_land_pixels = cur_patch_land.sum()
        cur_patch_mask = Mask_cur[w:w + patch_size, h:h + patch_size]
        cur_num_patch_pixels = cur_patch_mask.sum()
        cur_all_patch_mask = Mask_all_final_float[w:w + patch_size, h:h + patch_size]
        cur_num_all_patch_pixels = cur_all_patch_mask.sum()

        SM_data = Mask_cur[w:w + patch_size, h:h + patch_size] * SM_cur[w:w + patch_size, h:h + patch_size] / 100.0
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
        x2[:, :, 4] = Mask_cur[w:w + patch_size, h:h + patch_size]
        x2[:, :, 5] = Mask_5[w:w + patch_size, h:h + patch_size]
        x2[:, :, 6] = Mask_6[w:w + patch_size, h:h + patch_size]
        x2[:, :, 7] = Mask_7[w:w + patch_size, h:h + patch_size]
        x2[:, :, 8] = Mask_8[w:w + patch_size, h:h + patch_size]
        mask_cur = Mask_cur[w:w + patch_size, h:h + patch_size].astype(np.float32)

        y1_ = torch.from_numpy(x1.transpose((2, 0, 1))[np.newaxis,])
        mask_cur_ = torch.from_numpy((mask_cur)[np.newaxis,])

        torch.cuda.synchronize()
        start_time = time.time()
        y1_ = y1_.cuda()
        mask_cur_ = mask_cur_.cuda()
        y1_cur_ = y1_[:, 4, :, :]

        x_ = model(y1_cur_, y1_, mask_cur_)  # inference
        res_ = x_ + y1_cur_
        res_ = torch.clamp(res_, 0.0, 0.99)
        res_ = res_.squeeze()
        res_ = res_.cpu()
        res_ = res_.detach().numpy().astype(np.float32)
        res_ = res_ * (1 - Mask_cur[w:w + patch_size, h:h + patch_size]) + SM_data.copy()
        final = res_ * Mask_land[w:w + patch_size, h:h + patch_size]
        Final_global[0:600, 0:600] = final.copy()*100.0


        # Patch [0:600, 600:1440] (others)
        w, h = 0, 600
        cur_patch_land = Mask_land[w:w + patch_size, h:h + patch_size2]
        cur_num_land_pixels = cur_patch_land.sum()
        cur_patch_mask = Mask_cur[w:w + patch_size, h:h + patch_size2]
        cur_num_patch_pixels = cur_patch_mask.sum()
        cur_all_patch_mask = Mask_all_final_float[w:w + patch_size, h:h + patch_size2]
        cur_num_all_patch_pixels = cur_all_patch_mask.sum()

        SM_data = Mask_cur[w:w + patch_size, h:h + patch_size2] * SM_cur[w:w + patch_size, h:h + patch_size2] / 100.0
        SM_1_patch = Mask_1[w:w + patch_size, h:h + patch_size2] * SM_1[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_2_patch = Mask_2[w:w + patch_size, h:h + patch_size2] * SM_2[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_3_patch = Mask_3[w:w + patch_size, h:h + patch_size2] * SM_3[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_4_patch = Mask_4[w:w + patch_size, h:h + patch_size2] * SM_4[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_5_patch = Mask_5[w:w + patch_size, h:h + patch_size2] * SM_5[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_6_patch = Mask_6[w:w + patch_size, h:h + patch_size2] * SM_6[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_7_patch = Mask_7[w:w + patch_size, h:h + patch_size2] * SM_7[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0
        SM_8_patch = Mask_8[w:w + patch_size, h:h + patch_size2] * SM_8[w:w + patch_size,
                                                                  h:h + patch_size2] / 100.0

        x3[:, :, 0] = SM_1_patch
        x3[:, :, 1] = SM_2_patch
        x3[:, :, 2] = SM_3_patch
        x3[:, :, 3] = SM_4_patch
        x3[:, :, 4] = SM_data.copy()
        x3[:, :, 5] = SM_5_patch
        x3[:, :, 6] = SM_6_patch
        x3[:, :, 7] = SM_7_patch
        x3[:, :, 8] = SM_8_patch

        x4[:, :, 0] = Mask_1[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 1] = Mask_2[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 2] = Mask_3[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 3] = Mask_4[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 4] = Mask_cur[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 5] = Mask_5[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 6] = Mask_6[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 7] = Mask_7[w:w + patch_size, h:h + patch_size2]
        x4[:, :, 8] = Mask_8[w:w + patch_size, h:h + patch_size2]
        mask_cur = Mask_cur[w:w + patch_size, h:h + patch_size2].astype(np.float32)

        y1_ = torch.from_numpy(x3.transpose((2, 0, 1))[np.newaxis,])
        mask_cur_ = torch.from_numpy((mask_cur)[np.newaxis,])

        torch.cuda.synchronize()
        y1_ = y1_.cuda()
        mask_cur_ = mask_cur_.cuda()
        y1_cur_ = y1_[:, 4, :, :]

        x_ = model(y1_cur_, y1_, mask_cur_)  # inference
        res_ = x_ + y1_cur_
        res_ = torch.clamp(res_, 0.0, 0.99)
        res_ = res_.squeeze()
        res_ = res_.cpu()
        res_ = res_.detach().numpy().astype(np.float32)
        res_ = res_ * (1 - Mask_cur[w:w + patch_size, h:h + patch_size2]) + SM_data.copy()
        final = res_ * Mask_land[w:w + patch_size, h:h + patch_size2]
        Final_global[0:600, 600:1440] = final.copy()*100.0

        filename_all = files[i]
        nc_filename = save_dir + 'LPRM_AMSR2_'+ filename_all[28:36] + '.nc'
        res_all = Final_global + (1-Mask_land)* -32767 / Mask_land.astype(np.float32)
        ori_all = SM_cur / Mask_cur
        Save_to_NC(nc_filename, ori_all, res_all)
        log('Finished Date: ' + filename_all[28:36])







