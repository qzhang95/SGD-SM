
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as io
import time, datetime

def Save_to_NC(filename, cur_date, data_ori, data_rec):
    data_ori[data_ori < -1] = -32767
    data_rec[data_rec < -1] = -32767

    gridspi = nc.Dataset(filename, 'w', format='NETCDF4')
    # dimensions
    gridspi.createDimension('Latitude', 720)
    gridspi.createDimension('Longitude', 1440)
    # Create coordinate variables for dimensions
    latitudes = gridspi.createVariable('Latitude', np.float32, ('Latitude',))
    longitudes = gridspi.createVariable('Longitude', np.float32, ('Longitude',))

    # Variable Attributes
    latitudes.units = 'degree_north'
    longitudes.units = 'degree_east'
    
	# data
    lats = np.arange(90 - 0.25 / 2, -90, -0.25)  # notice: the last numb is not included
    lons = np.arange(-180 + 0.25 / 2, 180, 0.25)  # notice: the last numb is not included
    latitudes[:] = lats
    longitudes[:] = lons

    data1 = gridspi.createVariable('original_sm_c1', np.int, ('Latitude', 'Longitude'), fill_value = -32767)
    data1[:] = np.floor(data_ori).astype(int)
    # data1[:] = data_ori
    data1.long_name = "Original SM_C1 from 6.9 GHZ"
    data1.units = 'percent'
    data1.coordinates = 'Longitude Latitude'
    data1.scale_factor = 1.0
    data1.add_offset = 0.0
    data1.origname = 'original_sm_c1'
    data1.fullnamepath = '/original_sm_c1'

    data2 = gridspi.createVariable('reconstructed_sm_c1', np.int, ('Latitude', 'Longitude'), fill_value = -32767)
    data2[:] = np.floor(data_rec).astype(int)
    # data2[:] = data_rec
    data2.long_name = "Reconstructed SM_C1 from 6.9 GHZ"
    data2.units = 'percent'
    data2.coordinates = 'Longitude Latitude'
    data2.scale_factor = 1.0
    data2.add_offset = 0.0
    data2.origname = 'reconstructed_sm_c1'
    data2.fullnamepath = '/reconstructed_sm_c1'

    # data[data == np.NAN] = -32767
    gridspi.date = cur_date
    gridspi.source = 'netCDF4 python module tutorial'
    gridspi.reference = 'SGD-SM: Generating Seamless Global Daily AMSR2 Soil Moisture Long-term Products (2013–2019)'
    gridspi.url = 'https://doi.org/10.5281/zenodo.3960425'
    gridspi.time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gridspi.author = 'Processed by Qiang Zhang, Wuhan University'
    gridspi.close()

    return