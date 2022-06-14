
# -*- coding:utf-8 -*-
# This code is a example of reading our seamless global daily 
# AMSR2 soil moisture long-term productions from 2013 to 2019.

# Project website: https://qzhang95.github.io/Projects/Global-Daily-Seamless-AMSR2/
# Power by Qiang Zhang. (whuqzhang@gmail.com)


import netCDF4 as nc
import numpy as np
import os


year = 2019  # 2013 to 2019
file_dir = 'D:/results/' + str(year) + '/'  # you can replace this dataset record position with yours.
files = os.listdir(file_dir)


for i in range(0, files.__len__(), 1):
    Position_cur = file_dir + '/' + files[i]
    Data = nc.Dataset(Position_cur)
	
	# Get original and reconstructed soil moisture data
    Ori_data = Data.variables['original_sm_c1']
    Rec_data = Data.variables['reconstructed_sm_c1']
    Ori = Ori_data[0:720, 0:1440]
    Rec = Rec_data[0:720, 0:1440]
	
	# Get mask information
    Mask_ori = np.ma.getmask(Ori)

    if i%30==0:
        print(i)
