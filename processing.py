import os
import random
import pickle
import json
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage import measure
from scipy import stats
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from joblib import Parallel, delayed
import multiprocessing
import logging
from meta_features_extractor import MetaFeatureExtractor

from clustpy.data import (
    load_usps, load_mnist, load_cifar10, load_organ_c_mnist, load_fmnist, load_stl10, load_svhn,
    load_semeion, load_imagenet10, load_imagenet_dog, load_gtsrb, load_coil100, load_coil20,
    load_organ_s_mnist, load_oct_mnist, load_derma_mnist, load_breast_mnist, load_organ_a_mnist,
    load_kmnist, load_blood_mnist
)

# Configure logging
logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)


class DatasetsDicts:
    """
    Process the architectures and scores from pickled files in the path.
    The excluded dataset is not included in the analysis.
    """
    def __init__(self, config_path: str, excluded_dataset=None) -> None:
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.excluded_dataset = excluded_dataset
        self.path = self.config['architecture_results_path']
        self.rename_files()

    def rename_files(self):
        """
        Rename files in the path to lowercase for consistency.
        """
        for f in os.listdir(self.path):
            os.rename(os.path.join(self.path, f), os.path.join(self.path, f.lower()))

    def get_dataset_names(self):
        """
        Retrieve the dataset names from the files in the path.
        """
        dataset_names = {f.split('_')[0].lower() for f in os.listdir(self.path)}
        if self.excluded_dataset:
            dataset_names.discard(self.excluded_dataset)
        return dataset_names

    def get_archs_scores(self, dataset: str):
        """
        Get the architectures and scores for the dataset.
        """
        file_names = [f for f in os.listdir(self.path) if f.startswith(dataset)]
        architectures, scores = [], []
        for file in file_names:
            with open(os.path.join(self.path, file), 'rb') as f:
                data = pickle.load(f)
            architectures.append(data['architecture'])
            scores.append(data['mean_uca'])
        return architectures, scores

    def build_dataset_dict(self):
        """
        Build a dictionary with the dataset names as keys and the architectures and scores as values.
        """
        dataset_dict = {}
        for dataset in self.get_dataset_names():
            architectures, scores = self.get_archs_scores(dataset)
            dataset_dict[dataset] = {'architectures': architectures, 'scores': scores, 'dataset_name': dataset}
        return dataset_dict


class TrainingSamplesCreator:
    """
    Process architectures and scores, along with meta features, to create training samples.
    """
    def __init__(self, dataset_loaders, datasets_dict: dict, config_path: str) -> None:

        self.datasets_dict = datasets_dict
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.meta_features_path = self.config['meta_features_path']
        self.baseline_path = self.config['baseline_scores_path']
        self.dataset_loader = dataset_loaders

    def get_meta_features(self, dataset_name):
        """
        Retrieve the meta features for the dataset.
        """
        try:
            feature_vectors = np.load(os.path.join(self.meta_features_path, f'{dataset_name}_features.npy'))
        except:
            mfe = MetaFeatureExtractor(self.config, self.dataset_loader[dataset_name])
            feature_vectors = mfe.get_feature_vectors_for_all_samples()
        return feature_vectors

    def get_baseline_scores(self, dataset_name):
        """
        Load baseline_scores.
        """
        with open(os.path.join(self.baseline_path, f'{dataset_name}_results.pkl'), 'rb') as f:
            baseline_scores = pickle.load(f)
        return baseline_scores['mean_uca']

    def normalize_scores(self, scores):
        """
        Normalize scores between 0 and 1.
        """
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    def transform_data(self, images):
        """
        Transform images to 3x32x32 if RGB or 32x32 if grayscale.
        This function checks the data type of the images and normalizes them if they are in float format.
        """
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

        # Check if images are in float format and normalize them to uint8
        if images.dtype in [np.float32, np.float64]:
            images = (images * 255).astype(np.uint8)

        # Handle case where images have more than 3 dimensions
        if len(images.shape) == 4 and images.shape[1] == 3:
            images = images.transpose(0, 2, 3, 1)
        transformed_images = [transform(Image.fromarray(i)) for i in images]
        transformed_images = torch.stack(transformed_images)
        return transformed_images

    def precompute_statistical_features(self, dataset_name, images, n_clusters, dataset_size,mode='classic'):
        """
        Precompute statistical features for all images in the dataset and store them.
        mode : 'classic' or 'sift' def = 'classic'
        """
        print("checking if features for exists:", dataset_name)
        feature_file = os.path.join(self.meta_features_path, f"{dataset_name}_stat_features_{mode}.npy")
        valid_indices_file = os.path.join(self.meta_features_path, f"{dataset_name}_valid_indices_{mode}.npy")
        if os.path.exists(feature_file) and os.path.exists(valid_indices_file):
            statistical_features = np.load(feature_file)
            valid_indices = np.load(valid_indices_file)
        else:
            print("Precomputing statistical features for dataset:", dataset_name)
            # Transform images
            images = self.transform_data(images)
            # Extract features
            statistical_features, valid_indices = self.extract_statistical_meta_features_batch(images, n_clusters, dataset_size,mode=mode)
            # Save features and valid indices
            np.save(feature_file, statistical_features)
            np.save(valid_indices_file, valid_indices)
        return statistical_features, valid_indices

    def extract_statistical_meta_features_batch(self, images, num_classes: int, num_samples: int,mode='classic') -> (np.ndarray, np.ndarray):
        """
        Extract statistical meta-features from images and store them as a feature vector.
        Processes images in parallel using joblib for efficiency.
        Returns statistical features and indices of valid images.
        mode : 'classic' or 'sift' def = 'classic'
        """
        if isinstance(images, torch.Tensor):
            images = images.numpy()

        if images.dtype in (np.float32, np.float64):
            images = (images * 255).astype(np.uint8)

        # Handle grayscale images
        if len(images.shape) == 4 and images.shape[1] == 1:
            images = images.squeeze(1)
        # Handle images with channels first
        if len(images.shape) == 4 and images.shape[1] == 3:
            images = images.transpose(0, 2, 3, 1)

        num_cores = multiprocessing.cpu_count()
        if mode == 'classic':
            results = Parallel(n_jobs=num_cores)(
                delayed(self.extract_features_single_image)(img, num_classes, num_samples, idx)
                for idx, img in enumerate(images)
            )
        elif mode == 'sift':
            results = Parallel(n_jobs=num_cores)(
                delayed(self.extract_sift_features_single_image)(img, num_classes, num_samples,idx)
                for idx,img in enumerate(images)
            )
        #results = Parallel(n_jobs=num_cores)(
        #    delayed(self.extract_features_single_image)(img, num_classes, num_samples, idx)
        #    for idx, img in enumerate(images)
        #)

        # Separate features and valid indices
        features_list = []
        valid_indices = []
        for res in results:
            if res is not None:
                idx, features = res
                features_list.append(features)
                valid_indices.append(idx)

        features_array = np.array(features_list)
        valid_indices = np.array(valid_indices)
        return features_array, valid_indices

    def extract_features_single_image(self, img, num_classes: int, num_samples: int, idx: int):
        """
        Extract features for a single image.
        """
        try:
            # Convert to grayscale if the image is RGB
            if img.ndim == 3 and img.shape[-1] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img  # Already grayscale

            # Basic statistical features
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            skewness = stats.skew(gray.ravel())
            kurtosis = stats.kurtosis(gray.ravel())
            entropy = measure.shannon_entropy(gray)

            # Edge density
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.mean(edges)

            # GLCM features
            glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

            # Local Binary Pattern (LBP)
            lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
            lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), density=True)

            # Color features
            if img.ndim == 3 and img.shape[-1] == 3:
                mean_color = np.mean(img, axis=(0, 1))
                std_color = np.std(img, axis=(0, 1))
                color_features = np.concatenate([mean_color, std_color])
            else:
                color_features = np.zeros(6)  # No color features for grayscale images

            # HOG features
            hog_features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
            hog_mean = np.mean(hog_features)
            hog_std = np.std(hog_features)

            # Shape descriptors - Hu moments
            moments = cv2.moments(gray)
            hu_moments = cv2.HuMoments(moments).flatten()

            # Global dataset features
            global_features = [
                num_classes,         # Number of classes
                num_samples          # Number of samples
            ]

            # Combine all features into one vector
            image_features = [
                mean_intensity, std_intensity, skewness, kurtosis, entropy, edge_density,
                contrast, correlation, energy, homogeneity,  # GLCM features
                *lbp_hist,                                   # LBP histogram
                *color_features,                             # Color features
                hog_mean, hog_std,                           # HOG features
                *hu_moments                                  # Hu moments
            ]
            

            # Check for NaN or infinite values
            image_features = np.array(image_features)
            if np.isnan(image_features).any() or np.isinf(image_features).any():
                #logger.warning(f"Invalid features detected for image index {idx}. Skipping this image.")
                return None  # Exclude this image
            else:
                return idx, np.concatenate([global_features, image_features])

        except Exception as e:
           # logger.warning(f"Error processing image index {idx}: {e}")
            return None  # Exclude this image

    def extract_sift_features_single_image(self, img, num_classes: int, num_samples: int,idx: int):
        try:
            # Convert to grayscale if the image is RGB
            if img.ndim == 3 and img.shape[-1] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img  # Already grayscale
            sift = cv2.SIFT_create()
            _,des = sift.detectAndCompute(gray,None)
            des = np.mean(des, axis=0)
            return idx,np.concatenate([np.array([num_classes, num_samples]), des])
        except:
            return None
        
    
    def get_dataset_based_sample_precomputed(self, meta_feature_dataset:np.ndarray, stat_feature_dataset:np.ndarray, valid_indices:np.ndarray, mode: str = "both", num_samples: int = 10) -> list:
        """
        Return a training sample by sampling num_samples features from precomputed meta and stat features.
        """
        indices = random.sample(range(len(valid_indices)), num_samples)
        valid_sample_indices = valid_indices[indices]
        if mode == 'meta':
            sample_meta_features = meta_feature_dataset[valid_sample_indices]
            sample_meta_features = np.mean(sample_meta_features, axis=0)
            return [sample_meta_features]
        elif mode == 'stat':
            sample_stat_features = stat_feature_dataset[indices]
            sample_stat_features = np.mean(sample_stat_features, axis=0)
            return [sample_stat_features]
        elif mode == 'both':
            sample_meta_features = meta_feature_dataset[valid_sample_indices]
            sample_meta_features = np.mean(sample_meta_features, axis=0)
            sample_stat_features = stat_feature_dataset[indices]
            sample_stat_features = np.mean(sample_stat_features, axis=0)
            return [sample_meta_features, sample_stat_features]

    def get_arch_based_sample(self, arch_dict: dict,
                              mode: str = "triplet",arch_mode:str = "meta") -> list:
        """
        Return the architecture based part of the training sample.
        modes : 'triplet', 'score' def = 'triplet'
        arch_mode : 'meta','full' def = 'meta'
        arch_mode = 'meta' : return only the meta features of the architecture
        arch_mode = 'full' : return the full architecture
        """
        good_arch = [0]*2 
        bad_arch = [0]*2
        archs = arch_dict['architectures']
        scores = arch_dict['scores']
        dataset_name = arch_dict['dataset_name']
        baseline_score = self.get_baseline_scores(dataset_name)
        good_scores_idx = np.where(scores > baseline_score + (baseline_score * 0.05))[0]
        bad_scores_idx = np.where(scores < baseline_score - (baseline_score * 0.05))[0]
        if len(good_scores_idx) == 0 or len(bad_scores_idx) == 0:
            print("The length of good scores is ",len(good_scores_idx))
            print("The length of bad scores is ",len(bad_scores_idx))
            print(dataset_name)
           # logger.warning(f"Not enough good or bad architectures for dataset {dataset_name}. Skipping this dataset.")
            return None  # Cannot create a sample without both good and bad architectures

        if mode == 'triplet':
            good_idx = random.choice(good_scores_idx)
            bad_idx = random.choice(bad_scores_idx)
            good_a = archs[good_idx]
            bad_a = archs[bad_idx]
            if arch_mode == 'meta':
                good_arch[0] = len(good_a)-2
                bad_arch[0] = len(bad_a)-2
                good_arch[1] = good_a[-1]
                bad_arch[1] =  bad_a[-1]


            return [good_arch, bad_arch]
        if mode == 'score':
            i = random.choice(range(len(archs)))
            return [archs[i], scores[i]]

    def create_training_samples(self, num_samples_per_dataset: int, mode: str = "both",num_samples: int=10,stat_mode='classic') -> list:
        """
        Create training samples for the datasets.
        """
        training_samples = []
        for dataset_name in self.datasets_dict.keys():
            print(f"Processing dataset: {dataset_name}")
            #logger.info(f"Processing dataset: {dataset_name}")
            meta_features = self.get_meta_features(dataset_name)
            arch_dict = self.datasets_dict[dataset_name]
            dataset = self.dataset_loader[dataset_name]()

            images = dataset['images']
            labels = dataset['target']
            n_clusters = len(np.unique(labels))
            dataset_size = len(images)

            # Precompute statistical features and get valid indices
            statistical_features, valid_indices = self.precompute_statistical_features(dataset_name, images, n_clusters, dataset_size,mode=stat_mode)
            valid_indices_numbers = np.load(f"{self.meta_features_path}/{dataset_name}_valid_indices_{stat_mode}.npy").shape[0]
            if valid_indices_numbers <= num_samples + 500:
                print(f"Dataset {dataset_name} has less than {num_samples + 500} valid images. Skipping this dataset.")
                continue
            if valid_indices_numbers == 0:
              #  logger.warning(f"No valid images found for dataset {dataset_name}. Skipping this dataset.")
                continue  # Skip this dataset if no valid images

            for i in range(num_samples_per_dataset):
                dataset_sample = self.get_dataset_based_sample_precomputed(meta_features, statistical_features, valid_indices, mode,num_samples=num_samples)
                if dataset_sample is None:
                    continue  # Skip if dataset sample couldn't be created

                arch_sample = self.get_arch_based_sample(arch_dict)
                if arch_sample is None:
                    break  # Skip if architecture sample couldn't be created

                training_samples.append(dataset_sample + arch_sample)
        print(len(training_samples))
        return training_samples


class NASDataset(Dataset):
    """
    Custom dataset class for Neural Architecture Search (NAS) training samples.
    """
    def __init__(self, training_samples):
        self.training_samples = training_samples

    def __len__(self):
        return len(self.training_samples)

    def __getitem__(self, idx):
        sample = self.training_samples[idx]
        original_meta_feature = torch.tensor(sample[0], dtype=torch.float32)
        original_stat_feature = torch.tensor(sample[1], dtype=torch.float32)
        good_arch = torch.tensor(sample[2], dtype=torch.float32)
        bad_arch = torch.tensor(sample[3], dtype=torch.float32)
        return (original_meta_feature, original_stat_feature, good_arch, bad_arch)


def main():
    config_path = 'config.json'
    excluded_dataset = ''
    datasets_loaders = {
        'usps': load_usps,
        'mnist': load_mnist,
        'cifar10': load_cifar10,
        'organcmnist': load_organ_c_mnist,
        'fashionmnist': load_fmnist,
        'stl10': load_stl10,
        'svhn': load_svhn,
        'semeion': load_semeion,
        'imagenet10': load_imagenet10,
        'imagenetdog': load_imagenet_dog,
        'gtsrb': load_gtsrb,
        'coil100': load_coil100,
        'coil20': load_coil20,
        'organsmnist': load_organ_s_mnist,
        'octmnist': load_oct_mnist,
        'dermamnist': load_derma_mnist,
        'breastmnist': load_breast_mnist,
        'organamnist': load_organ_a_mnist,
        'kmnist': load_kmnist,
        'bloodmnist': load_blood_mnist
    }
    d = DatasetsDicts(config_path, excluded_dataset)
    datasets_dict = d.build_dataset_dict()
    t = TrainingSamplesCreator(datasets_loaders, datasets_dict, config_path)
    combs = t.create_training_samples(1,stat_mode='classic')
    
    if not combs:
        #logger.warning("No training samples were created.")
        return
    ds = NASDataset(combs)
    
    for i in range(len(ds)):
        sample = ds[i]
        #print(sample)
        print(f"Sample {i}:")
        print("Meta Features:", sample[0].shape)
        print("Statistical Features:", sample[1].shape)
        print("Good Architecture:", sample[2])
        print("Bad Architecture:", sample[3])
        
        break  # Remove this line if you want to print all samples

    print(combs[0][3:])
    return ds

if __name__ == '__main__':
    main()
