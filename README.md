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

Since the images are fairly big (~300G), it is a possibility to run curl or wget:

> curl https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/fsl_cp_images.zip?ticket=l2P9J6ConqQOLNF --output fsl_cp_images.zip

> wget https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/fsl_cp_images.zip?ticket=l2P9J6ConqQOLNF -O fsl_cp_images.zip


#### Sample dataset

In addition, we supply a small [sample](https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/fsl_cp_sample.zip?ticket=WUTctNZlyRc6Mdw) of the dataset. It is useful for those who are curious what the dataset looks like, but cannot be used to run the scripts.


## Additional folders
A **data**, **logs** and **weights** folder are also needed to run the necessary scripts. 
