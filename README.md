# FSL-CP: Few-shot Prediction of Small Molecule Activity using Cell Microscopy Images 

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.13-red.svg)](https://pytorch.org/get-started/previous-versions/)

Few-shot learning for bioassay prediction with Cell Painting dataset.



## Installation
1. Clone this repository to your HOME folder.
2. To install dependencies, simply run:

    ```
    cd FSL_CP

    conda env create -f environment.yaml
    conda activate fslcp
    ```

Note: It is advised to run mamba instead of conda to save ~20 mins of your life.

## Results
All results are available in the *result* folder. In there, subfolder *notebook* is where all of the graphs come from.

## Downloading data 
Click on the hyperlink to download the [images](https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/fsl_cp_images.zip?ticket=l2P9J6ConqQOLNF), [csv files](https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/output.zip?ticket=zLF2wINy8vpK6oK), and [weights of multitask model](https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/weights.zip?ticket=vkjYyYjMGvLIQOh).

Since the images are fairly big (~300G), you can also download them via curl or wget:

> curl https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/fsl_cp_images.zip?ticket=l2P9J6ConqQOLNF --output fsl_cp_images.zip

> wget https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/fsl_cp_images.zip?ticket=l2P9J6ConqQOLNF -O fsl_cp_images.zip


#### Sample dataset

In addition, we supply a small [sample](https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/fsl_cp_sample.zip?ticket=WUTctNZlyRc6Mdw) of the dataset. It is useful for those who are curious what the dataset looks like, but cannot be used to run the scripts.


## Setting up
1. Create a *data* folder, place the downloaded *output* folder (csv files) into it.
2. Place the downloaded *weights* folder.
3. Create an empty *logs* folder.
4. Place the images folder anywhere you like.

The folder hierachy should look like [this](./Screenshot_repo.png).


## Benchmark models
The codes for all models are placed in the *fsl_cp* folder. Simply run:

> python fsl_cp/desired_file.py

You might need to change some of the flags to get it run properly on your system. So retrieve a list of flags, run:

> python fsl_cp/desired_file.py -h


## Generate ResNet50 embeddings
Run the script:
> python fsl_cp/generate_cnn_embeddings.py -p path/to/image/folder

#### If you want to generate your own embeddings 
The base dataset class supports concatenating features from different CSV files. But if you generate new embeddings from the data, please save them to a CSV file (like norm_CP_feature_df.csv), and make sure:
1. The first 3 columns are 'INCHIKEY', 'CPD_SMILES', 'SAMPLE_KEY'. The rest of the columns are embeddings.
2. the 'SAMPLE_KEY'column is **in the same order** as in the norm_CP_feature_df.csv.


## Tutorials
There are tutorial notebooks available in the *notebook* folder.

## Contacts
Son Ha | [LinkedIn](https://linkedin.com/in/son-ha-479909159) | [Twitter](https://twitter.com/sonha1999)