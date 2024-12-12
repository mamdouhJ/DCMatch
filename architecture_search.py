import sys
import os
import pickle 
import json 
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from clustpy.deep import DEC
from clustpy.metrics.clustering_metrics import unsupervised_clustering_accuracy as uca
from clustpy.data import (
    load_usps, load_mnist, load_cifar10, load_organ_c_mnist, load_fmnist, load_stl10, load_svhn, load_semeion, load_imagenet10,
    load_imagenet_dog, load_gtsrb, load_coil100, load_coil20, load_organ_s_mnist, load_oct_mnist, load_derma_mnist, load_breast_mnist,
    load_organ_a_mnist, load_kmnist, load_blood_mnist
)

class AutoEncoder: 
    """
    Create an architecture based on the grid search parameters and pass it
    to the FeedforwardAutoencoder class.
    """
    def __init__(self, data_shape:int) -> None:
        self.data_shape = data_shape
        
    def get_architecture(self, num_layers:int, bottleneck_size:int) -> list:
        layers = [125, 250, 500, 1000, 2000]
        
        l = [bottleneck_size]
        if num_layers >= 1:
            l.extend(layers[:num_layers])
            l.append(self.data_shape)
            l.reverse()
        return l
    
    def get_autoencoder(self, num_layers:int, bottleneck_size:int) -> FeedforwardAutoencoder:
        layers = self.get_architecture(num_layers, bottleneck_size)
        return FeedforwardAutoencoder(layers=layers)
    
class ClusterModel:
    """
    Clusterer model that uses the DEC class from clustpy.deep to cluster the data.
    The data_loader is a function that returns the data and labels in the form of a Bunch object.
    The autoencoder is an instance of the FeedforwardAutoencoder class.
    """
    def __init__(self,clustering_algorithm, data:np.ndarray, labels:np.ndarray, autoencoder:FeedforwardAutoencoder,architecture:list) -> None:
        self.clustering_algorithm = clustering_algorithm
        self.autoencoder = autoencoder
        self.data, self.labels = data, labels
        self.architecture = architecture
        self.no_clusters = len(np.unique(self.labels))
        self.clusterer = self.initialize_clusterer()
        self.fit_clusterer()
        
    def initialize_clusterer(self):
        return self.clustering_algorithm(neural_network=self.autoencoder, n_clusters=self.no_clusters, device=torch.device('cuda'))
    
    def fit_clusterer(self):
        self.clusterer.fit(self.data)
        
    def predict(self):
        return self.clusterer.labels_
    
    def evaluate(self):
        return uca(self.labels, self.predict())

class ResultsParser:
    """
    Results class that takes as input a labels list and a list of pseudo labels lists and calculates the unsupervised clustering accuracy
    for each pseudo labels list and returns the results in a dictionary.
    """
    def __init__(self, labels, pseudo_labels, architecture):
        self.labels = labels
        self.pseudo_labels = pseudo_labels
        self.architecture = architecture

    def get_uca(self, pseudo_labels):
        return uca(self.labels, pseudo_labels)

    def parse_results(self):
        results = {'architecture': self.architecture, 'labels': self.labels, 'pseudo_labels': self.pseudo_labels}
        for i, pseudo_label in enumerate(self.pseudo_labels):
            results[f'uca_{i}'] = self.get_uca(pseudo_label)
        results['mean_uca'] = np.mean([self.get_uca(pseudo_label) for pseudo_label in self.pseudo_labels])
        return results

    def save_results(self, results, filename):
        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(results, f)
    
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
    return torch.stack(transformed_images)

def main():
    datasets_loaders = [
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

    data_loader = datasets_loaders[int(sys.argv[1])]
    d = data_loader()
    data, labels, dataset_name = d.images, d.target, d.dataset_name
    data = transform_data(data)
    data = data.view(data.shape[0], -1)
    data_shape = data.shape[1]
    with open("config.json", "r") as f:
        config = json.load(f)
    architecture_path = config["architecture_results_path"]
    ae = AutoEncoder(data_shape)
    for num_layers in range(1,6):
        for bottleneck_size in range(2,51,2):
            #check if the architecture already exists
            print(f'{architecture_path}{dataset_name.lower()}_layers{num_layers}_bottleneck{bottleneck_size}.pkl')
            os.makedirs(architecture_path, exist_ok=True)
            if os.path.exists(f'{architecture_path}{dataset_name.lower()}_layers{num_layers}_bottleneck{bottleneck_size}.pkl'):
                continue
            autoencoder = ae.get_autoencoder(num_layers, bottleneck_size)
            architecture = ae.get_architecture(num_layers, bottleneck_size)
            print(f'Architecture: {architecture}')
            pseudo_labels = []
            for j in range(10):
                cluster_model = ClusterModel(DEC,data, labels, autoencoder, architecture)
                pseudo_labels.append(cluster_model.predict())
                print(f'UCA {j}: {cluster_model.evaluate()}')
            result_parser = ResultsParser(labels, pseudo_labels, architecture)
            results = result_parser.parse_results()
            result_parser.save_results(results, f'{architecture_path}{dataset_name.lower()}_layers{num_layers}_bottleneck{bottleneck_size}')
    
if __name__ == "__main__":
    main()
    
    