# FSL_CP
Few-shot learning for bioassay prediction with Cell Painting dataset.

Need a 'data', an 'output' and a 'logs' folder. Inside 'data' is 'output' folder.

## Downloading data 

Click on the hyperlink to download the [images](https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/fsl_cp_images.zip?ticket=l2P9J6ConqQOLNF), [csv files](https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/output.zip?ticket=zLF2wINy8vpK6oK), and [weights of multiask model](https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/weights.zip?ticket=vkjYyYjMGvLIQOh).

You can also download using curl or wget like below:

#### Images
> curl https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/fsl_cp_images.zip?ticket=l2P9J6ConqQOLNF --output fsl_cp_images.zip

> wget https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/fsl_cp_images.zip?ticket=l2P9J6ConqQOLNF -O fsl_cp_images.zip

#### Csv files
> curl https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/output.zip?ticket=zLF2wINy8vpK6oK --output output.zip

> wget https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/output.zip?ticket=zLF2wINy8vpK6oK -O output.zip

#### Weights of multitask model
> curl https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/weights.zip?ticket=vkjYyYjMGvLIQOh --output weights.zip

> wget https://irods-web.zdv.uni-mainz.de/irods-rest/rest/fileContents/zdv/home/sonha/weights.zip?ticket=vkjYyYjMGvLIQOh -O weights.zip