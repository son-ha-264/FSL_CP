# FSL_CP
Few-shot learning for bioassay prediction with Cell Painting dataset.

## Installation
1. Clone this repository to your HOME folder.
2. To install dependencies, simply run:

    ```
    cd FSL_CP

    conda env create -f environment.yaml
    conda activate fslcp
    ```

Note 1: It is advised to run mamba instead of conda to save ~20 mins of your life.

Note 2: CUDA is not available on MacOS. If you use MacOS, delete 'nvidia' from the channel list in environment.yaml.

## Downloading data 

Click on the hyperlink to download the [images](https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/fsl_cp_images.zip?ticket=l2P9J6ConqQOLNF), [csv files](https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/output.zip?ticket=zLF2wINy8vpK6oK), and [weights of multiask model](https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/weights.zip?ticket=vkjYyYjMGvLIQOh).

Since the images are fairly big (~300G), curl or wget can also be used to download:

> curl https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/fsl_cp_images.zip?ticket=l2P9J6ConqQOLNF --output fsl_cp_images.zip

> wget https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/fsl_cp_images.zip?ticket=l2P9J6ConqQOLNF -O fsl_cp_images.zip


#### Sample dataset

In addition, we supply a small [sample](https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/fsl_cp_sample.zip?ticket=WUTctNZlyRc6Mdw) of the dataset. It is useful for those who are curious what the dataset looks like, but cannot be used to run the scripts.


## Setting up
1. Create a *data* folder, place the downloaded *output* folder (csv files) into it.
2. Place the downloaded *weights* folder.
3. Create an empty *logs* folder.
4. Place the images folder anywhere you like.

The folder hierachy should look like the Screenshot_repo.png file.

## Benchmark models
The codes for all models are placed in the *fsl_cp* folder. 

## Using the dataset
There are tutorial notebooks available (soon). 
