# Image Colorizer

COS 429 Final Project

This project uses Python 3.9.16. Install the required packages with `pip install -r requirements.txt`.

## Dataset extraction

Download the train and validation ImageNet-1K 64x64 dataset from [here](https://image-net.org). Place the `.npz` files in the `image-colorizer/data` directory. Then run the `imagenet_extract.ipynb` notebook. This will extract the images from the `.npz` files and save them as `.png` files in the `image-colorizer/data` directory.

## Models

The autoencoder model described in the paper is available in Jupyter notebook form in `model3raw.ipynb`. The U-Net model described in the paper is available in Jupyter nootebook form in `model4.ipynb`. There are also Python script versions of these models in `model3raw.py` and `model4.py`. The Python scripts will also create TensorBoard logs in the `runs` directory and will place epoch-end checkpoints in the `checkpoints` directory. Model training will use up to 20 GB of GPU memory.