# imaging
## Setup
Currently, the [nx application from NoMachine](https://www.nomachine.com/) [supported by NIH HPC](https://hpc.nih.gov/docs/connect.html) is not working. To get around this and visually inspect images remotely, follow these steps:

1. Install XQuartz for mac (https://www.xquartz.org/), logout and log back into NIH laptop, then use the -Y flag when connecting via ssh to Biowulf. 
2. Install [conda](https://hpc.nih.gov/apps/python.html#envs) with settings storing envs on your biowulf /data folder so that large custom packages packages can be installed without root. 
3. Install libvips 

```bash
conda install -c conda-forge libvips
```

4. Load conda env from your /data folder. Consider adding this to your bash_rc file so that you can use conda everytime you logon to an interactive session, otherwise run this command each time that you log on. 

```bash
source /data/$USER/conda/etc/profile.d/conda.sh
```

## Pilot Data
A dataset of [Breast Metastases to Axillary Lymph Nodes](https://wiki.cancerimagingarchive.net/display/Public/Breast+Metastases+to+Axillary+Lymph+Nodes) is being used for the initial pilot. 
GDC H&E data can be downloaded using gdc command line tools and a manifest generated on the [GDC Website](https://portal.gdc.cancer.gov/):
   
```bash 
module load gdc-client/1.5.0
gdc-client download -m  /home/luberjm/gdc_manifest_20201105_184746.txt
```

The manifest is also included in the docs folder.

## Fast Conversion of SVS to JPG 

## Examples of Slides That Will Be Challenging For Autoencoder
This is an ongoing list of challenging training examples that we will have to account for via preprocessing. 
![Challenging Image](examples/chal1.jpg)
