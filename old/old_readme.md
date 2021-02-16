for x in *.svs; do echo "sh /home/luberjm/code/imaging/scripts/svs_conversion.sh /data/luberjm/data/lungadeno/svs/test/$x" >> image_processing.swarm; done
swarm -f image_processing.swarm -g 10 -t 1 -p 2 -b 10 --partition=ccr --time=00:15:00 --merge-output
http://manpages.ubuntu.com/manpages/bionic/man1/openslide-write-png.1.html

# imaging
## Setup
Currently, the [nx application from NoMachine](https://www.nomachine.com/) is not working. See Biowulf instructions [here](https://hpc.nih.gov/docs/connect.html). To get around this and visually inspect images remotely, follow these steps:

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

## Fast Conversion of SVS to PNG 
The subdirectory structure is annoying from the gdc-client output, so we will first fix that.

```bash
mv */*.svs .
rm -R -- */
ls * > images.txt
```

Now, we will convert everything to PNG. Note that libvips has really efficient C integrations with OpenSlide, and that we can set a level "flag" to determine the resolution of our output. After benchmarking, it was shown that the pipeline requires the maximum resolution (level 0). The conversion script is located in scripts.

Some insight into what is happening here: SVS images are a propietary format from Aperio digital pathology instruments. They are essentially TIFFs at heart, but have some modifications (such as layers with different resolution) and extra tags that make them unreadable without conversion. 

### Convert 1 Image

```bash 
sbatch --partition=ccr --mem=10gb --export=F='/data/luberjm/images/TCGA-XM-A8RC-01A-01-TSA.E8BB705F-15D0-41F3-8A37-1F25964A5BBB.svs' /home/luberjm/code/imaging/scripts/svs_conversion.sbatch
```

sbatch doesn't support command line args in bash, so we are exporting what would be the command line arg so that this can be parallelized with a sbatch swarm: 

### Convert All Images In A Batch 

```bash 
for x in *.svs;do echo "sbatch --partition=ccr --mem=10gb --export=F='/data/luberjm/images/$x' /home/luberjm/code/imaging/scripts/svs_conversion.sbatch" >> batch_jobs.sh;done
sh batch_jobs.sh 
```

## Viewing Converted Output

```bash
module load fiji/2.0.0-pre-7
Fiji
```

This is an ImageJ wrapper on Biowulf where we can look to check that the image conversions were succesful. 

## Breaking Down Converted Output Into Patches For Training
The script "patch_sampling.py" in the scripts folder generates patches for training the autoencoder. It takes as arguments a jpg post svs conversion and x_slide and y_slide, which are how much the window slides in pixels. All the code is based around a cartesian coordinate system with 0,0 at the upper left of the images. 

### Running on 1 image

```bash 
python3 /home/luberjm/code/imaging/scripts/patch_sampling.py /data/luberjm/images/TCGA-ZT-A8OM-01A-01-TS1.98208A8F-FE2D-4967-B31D-B66C188A2F66.jpg 50 50
```

### Running on a set of images via a swarm 

```bash
for x in *.jpg;
do echo "python3 /home/luberjm/code/imaging/scripts/patch_sampling.py /data/luberjm/images/$x 50 50" >> image_processing.swarm;
done
```

Run the swarmfile where you want the 10k-100ks of output image partitions for training to be stored. 
```bash
mv image_processing.swarm output
swarm -f image_processing.swarm --partition ccr
```

For 137 aperio svs images in the pilot at relatively low resolution, this has generated an initial training set of 470,298 250x250 pixel images for the inital training, which are located at /data/luberjm/images/output. 

## Autoencoder Training 
Current autoencoder work is located in scripts, there are currently some issues that are being resolved. 

## Examples of Slides That Will Be Challenging For Autoencoder
This is an ongoing list of challenging training examples that we will have to account for via preprocessing. 
![Challenging Image](examples/chal1.jpg)
As you can see in the above image, most "patches" will be useless, so we likely need to do some sort of edge detection preprocessing. 
