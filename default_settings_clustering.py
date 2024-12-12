import os
import sys
import pickle
import json
import numpy as np 
import torch
from PIL import Image
from torchvision import transforms
from clustpy.deep import DEC
from clustpy.metrics.clustering_metrics import unsupervised_clustering_accuracy as uca
from clustpy.data import (
    load_usps, load_mnist, load_cifar10, load_organ_c_mnist, load_fmnist, load_stl10, load_svhn, load_semeion, load_imagenet10,
    load_imagenet_dog, load_gtsrb, load_coil100, load_coil20, load_organ_s_mnist, load_oct_mnist, load_derma_mnist, load_breast_mnist,
    load_organ_a_mnist, load_kmnist, load_blood_mnist
)

data_loaders = [
    load_usps,
    load_mnist,
    load_cifar10,
    load_organ_c_mnist,
    load_fmnist,
    load_stl10,
    load_svhn,
    load_semeion,
    load_imagenet10,
    load_imagenet_dog,
    load_gtsrb,
    load_coil100,
    load_coil20,
    load_organ_s_mnist,
    load_oct_mnist,
    load_derma_mnist,
    load_breast_mnist,
    load_organ_a_mnist,
    load_kmnist,
    load_blood_mnist
]
#results = {}
index = int(sys.argv[1])

def transform_data(images):
    """
    Transform images to 3x32x32 if RGB or 32x32 if grayscale.
    This function checks the data type of the images and normalizes
    them if they are in float format.
    """
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    # Check if images are in float format and normalize them to uint8
    if images.dtype == np.float32 or images.dtype == np.float64:
        images = (images * 255).astype(np.uint8)

    # Handle case where images have more than 3 dimensions
    if len(images.shape) > 3:
        images = images.transpose(0, 2, 3, 1)

    transformed_images = [transform(Image.fromarray(i)) for i in images]
    print("Transformation done")
    return torch.stack(transformed_images)

data_loader = data_loaders[index]
d = data_loader()
data, labels, dataset_name = d.images, d.target, d.dataset_name.lower()
data = transform_data(data)
data = data.view(data.shape[0], -1)
data_shape = data.shape[1]
pseudo_labels = []
dataset_results = {}

print(f'Clustering {dataset_name}')
for i in range(10):
    dec = DEC(np.unique(labels).shape[0])
    dec.fit(data)
    dataset_results[i] = dec.labels_
    dataset_results[f'uca_{i}'] = uca(labels, dec.labels_)
dataset_results['mean_uca'] = np.mean([dataset_results[f'uca_{i}'] for i in range(10)])
with open("/home/wiss/aljoud/DCMatch/config.json", "r") as f:
    config = json.load(f)
    
os.makedirs(config["baseline_scores_path"], exist_ok=True)
with open(f'{config["baseline_scores_path"]}{dataset_name}_results.pkl', 'wb') as f:
    pickle.dump(dataset_results, f)
    
        
