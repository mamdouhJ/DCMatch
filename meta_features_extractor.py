import json
import os
import sys
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import open_clip
from clustpy.data import (
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
)

class MetaFeatureExtractor:
    """
    Extracts meta-features from datasets.
    """

    def __init__(self, config_path, dataset_loader):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_config(config_path)
        self._initialize_clip_model()
        self.dataset = self._load_dataset(dataset_loader)
        self.meta_features = self.extract_meta_features()

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def _initialize_clip_model(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.config['clip_model_name'],
            pretrained=self.config['pretrained']
        )
        self.model.eval()
        self.model.to(self.device)

    def _load_dataset(self, dataset_loader):
        dataset = dataset_loader()
        dataset.images = self._normalize_images(dataset.images)
        return dataset

    def _normalize_images(self, images):
        if images.dtype in [np.float32, np.float64]:
            images = (images * 255).astype(np.uint8)
        return images

    def transform_images(self, images):
        """
        Transform images to 3x32x32 (RGB).
        """
        transform = transforms.Compose([transforms.Resize((32, 32))])
        transformed_images = [transform(Image.fromarray(img).convert('RGB')) for img in images]
        return transformed_images

    def get_feature_vector(self, img):
        img = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(img)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def extract_meta_features(self):
        """
        Extract and save feature vectors for all dataset samples.
        """
        save_path = f'{self.config["meta_features_path"]}/{self.dataset.dataset_name}_features.npy'
        if os.path.exists(save_path):
            return np.load(save_path)

        transformed_images = self.transform_images(self.dataset.images)
        features = [self.get_feature_vector(img) for img in transformed_images]
        os.makedirs(self.config["meta_features_path"], exist_ok=True)
        np.save(save_path, np.array(features))
        return features

    def get_random_indices(self, num_meta_samples, num_samples):
        return [
            np.random.choice(len(self.dataset.images), num_samples, replace=False)
            for _ in range(num_meta_samples)
        ]

    def extract_random_sample_features(self, num_meta_samples, num_samples):
        """
        Extract meta-features for random subsets of the dataset.
        """
        feature_vectors = self.meta_features
        random_indices = self.get_random_indices(num_meta_samples, num_samples)
        meta_dataset = []
        for meta_sample in random_indices:
            meta_dataset.append(feature_vectors[meta_sample]) 

        save_path = f'{self.config["meta_features_path"]}/{self.dataset.dataset_name}_meta_features_{num_samples}.npy'
        os.makedirs(self.config["meta_features_path"], exist_ok=True)
        meta_dataset = np.array(meta_dataset)
        meta_dataset = np.squeeze(meta_dataset)
        np.save(save_path, meta_dataset)
        return meta_dataset

class DatasetLoader:
    """
    Provides access to dataset loaders.
    """
    LOADERS = [
        load_usps, load_mnist, load_cifar10, load_organ_c_mnist, load_fmnist, load_stl10,
        load_svhn, load_semeion, load_imagenet10, load_imagenet_dog, load_gtsrb,
        load_coil100, load_coil20, load_organ_s_mnist, load_oct_mnist, load_derma_mnist,
        load_breast_mnist, load_organ_a_mnist, load_kmnist, load_blood_mnist
    ]

    @staticmethod
    def get_loader(index):
        return DatasetLoader.LOADERS[index]

def main(config_path, dataset_index, num_meta_samples, num_samples):
    dataset_loader = DatasetLoader.get_loader(dataset_index)
    extractor = MetaFeatureExtractor(config_path, dataset_loader)
    extractor.extract_random_sample_features(num_meta_samples, num_samples)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python script.py <dataset_index> <num_meta_samples> <num_samples>")
        sys.exit(1)

    config_path = 'config.json'
    dataset_index = int(sys.argv[1])
    num_meta_samples = int(sys.argv[2])
    num_samples = int(sys.argv[3])

    main(config_path, dataset_index, num_meta_samples, num_samples)
