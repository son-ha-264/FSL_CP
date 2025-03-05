# FSL-CP: A Benchmark for Small Molecule Activity Few-Shot Prediction using Cell Microscopy Images 

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.13-red.svg)](https://pytorch.org/get-started/previous-versions/)

Dataset and benchmark for activity few-shot prediction with Cell Painting dataset.



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
Data can be downloaded via curl or wget:

CSV files:
> curl https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/project/m2_jgu-fsl-cp/output.zip?ticket=YX8NDNVBw9aUfzQ --output output.zip

> wget https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/project/m2_jgu-fsl-cp/output.zip?ticket=YX8NDNVBw9aUfzQ -O output.zip

Weights of the Multitask model
> curl https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/project/m2_jgu-fsl-cp/weights.zip?ticket=NAJlnUkDvS8yY6G --output weights.zip

> wget https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/project/m2_jgu-fsl-cp/weights.zip?ticket=NAJlnUkDvS8yY6G -O weights.zip

#### NOTE: Images file
Currently, there are problems with our data storage server and the images cannot be downloaded. We are actively trying to get it online as soon as possible.

#### Sample dataset
In addition, we supply a small sample of the dataset. It is useful for those who are curious what the dataset looks like, but cannot be used to run the scripts.These can be downloaded by running:
> curl https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/project/m2_jgu-fsl-cp/fsl_cp_sample.zip?ticket=Vvu3IGRYbRtvWAG --output fsl_cp_sample.zip

> wget https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/project/m2_jgu-fsl-cp/fsl_cp_sample.zip?ticket=Vvu3IGRYbRtvWAG -O fsl_cp_sample.zip


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
