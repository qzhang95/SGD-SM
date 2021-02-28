# SGD-SM
* A seamless global daily (SGD) AMSR2 soil moisture long-term (2013-2019) dataset is generated through the proposed model. This daily products include 2553 global soil moisture NetCDF4 files, starting from Jan 01, 2013 to Dec 31, 2019 (about 20GB memory after uncompressing this zip file).

* To further validate the effectiveness of these productions, three verification ways are employed as follow: 1) In-situ validation; 2) Time-series validation; And 3) simulated missing regions validation. More validation results can be viewed at [SGD-SM](https://qzhang95.github.io/Projects/Global-Daily-Seamless-AMSR2/).

<div align=center><img src="./figures/ori.gif" align=center width="360px"/><img src="./figures/rec.gif" align=center width="360px"/></div>



# Dataset Download

* **Link 1**: [Baidu Yun](https://pan.baidu.com/s/1SGdKmfgUgUBmcWse-cDsWg) (Extracting Code: fu8f)

* **Link 2**: [Google Drive](https://drive.google.com/file/d/1pGoX12Va3k6o9ybIMBjpDDHLbcUShM1P/view?usp=sharing)

* **Link 3**: [Zenodo](http://doi.org/10.5281/zenodo.4417458)


# Environments and Dependencies
* Windows 10
* Python 3.7.4
* netCDF4
* numpy


# Toolkit Installation
This soil moisture dataset is comprised of netCDF4 (\*.nc) files. Therefore, users need to install netCDF4 toolkit before reading the data:
```
    pip install netCDF4
    pip install numpy
```

# Data Reading
It should be noted that the original and reconstructed soil moisture data are both recorded in a NC file. 
User can read the original data, reconstructed data, and mask data as follows (more details can be viewed in [Example.py](Example.py)):
```
    Data = nc.Dataset(NC_file_position)
    Ori_data = Data.variables['original_sm_c1']
    Rec_data = Data.variables['reconstructed_sm_c1']
    Ori = Ori_data[0:720, 0:1440]
    Rec = Rec_data[0:720, 0:1440]
    Mask_ori = np.ma.getmask(Ori)
```

# Code Running
```
    run Main_Test_SGD-SM.py
```



# Data Visualization
Users can visualize \*.nc format file through [Panoply](https://www.giss.nasa.gov/tools/panoply/download/) software. Before visualizing, you must install [Java SE Development Kit](https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html).


# Contact Information
If you have any query for this work, please directly contact me.

Author: Qiang Zhang, Wuhan Unviversity.

E-mail: whuqzhang@gmail.com

Homepage: [qzhang95.github.io](https://qzhang95.github.io/)

