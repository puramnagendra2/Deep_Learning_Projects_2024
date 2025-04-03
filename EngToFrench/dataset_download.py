# This function is used to download dataset from kaggle
# Dataset Link https://www.kaggle.com/datasets/rajpulapakura/english-to-french-small-dataset

import kaggle

def download_dataset(dataset_source, dataset_dest):
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_source, path=dataset_dest, unzip=True)

    return "Done"

dataset_source = r'rajpulapakura/english-to-french-small-dataset'
datatset_dest = r'.'

print("Download", download_dataset(dataset_source, datatset_dest))
