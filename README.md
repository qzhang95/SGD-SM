# SGD-SM
# Generating Seamless Global Daily AMSR2 Soil Moisture Long-term Productions (2013-2019)

High quality and long-term soil moisture productions are significant for hydrologic monitoring and agricultural management. 
However, the acquired daily soil moisture productions are incomplete in global land (**just about 30%~80% coverage ratio**),
 due to the satellite orbit coverage and the limitations of soil moisture retrieving algorithms. To solve this inevitable problem, 
 we develop a novel 3-D spatio-temporal partial convolutional neural network for Advanced Microwave Scanning Radiometer 2 (AMSR2) soil moisture productions gap-filling. 
 Through the proposed framework, we generate the [**seamless global daily AMSR2 soil moisture long-term productions from 2013 to 2019**](https://qzhang95.github.io/Projects/Global-Daily-Seamless-AMSR2/).

Original/reconstructed AMSR2 global daily 0.25° soil moisture time-series productions in 2019.6.1 to 6.30:

<img src="./figures/ori.gif" align=left width="256px"/>
<img src="./figures/rec.gif" align=right width="256px/>


# Dataset Download Links
We provide **three ways** to download this dataset:

**Link 1**: [Baidu Yun](https://pan.baidu.com/s/1SGdKmfgUgUBmcWse-cDsWg) (Extracting Code: fu8f)

**Link 2**: [Google Drive](https://drive.google.com/file/d/1pGoX12Va3k6o9ybIMBjpDDHLbcUShM1P/view?usp=sharing)

**Link 3**: [Zenodo](http://doi.org/10.5281/zenodo.3960425)


# Environments and Dependencies
* Windows 10
* Python 3.7.4
* netCDF4
* numpy


# Toolkit Installation
This soil moisture dataset is comprised of NetCDF (.nc) files. Therefore, users need to install netCDF4 toolkit before reading the data:
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

# Data visualization
Users can visualize nc format file through [Panoply](https://www.giss.nasa.gov/tools/panoply/download/) software. Before visualizing, you must install [Java SE Development Kit](https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html).

