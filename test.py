import pickle
import json
import numpy as np
import joblib
import torch
from train import BaseMetaFeaturesEncoder, StatisticalMetaFeaturesEncoder, ArchitectureEncoder
from processing import DatasetsDicts
from clustpy.data import (
    load_usps, load_mnist, load_cifar10, load_organ_c_mnist, load_fmnist, load_stl10, load_svhn,
    load_semeion, load_imagenet10, load_imagenet_dog, load_gtsrb, load_coil100, load_coil20,
    load_organ_s_mnist, load_oct_mnist, load_derma_mnist, load_breast_mnist, load_organ_a_mnist,
    load_kmnist, load_blood_mnist
)
import matplotlib.pyplot as plt

class Tester:
    """
    Given a dataset name, load the model weights for architecture and meta-feature encoder and get a recommendation for a good architecture.
    """
    def __init__(self, dataset_name, config_path, num_samples=100, stat_mode='classic', alg='dec'):
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.stat_mode = stat_mode
        self.alg = alg

        with open(config_path, 'rb') as f:
            self.config = json.load(f)

        self.meta_features = np.load(f"{self.config['meta_features_path']}/{dataset_name}_features.npy")
        self.stat_features = np.load(f"{self.config['meta_features_path']}/{dataset_name}_stat_features_{stat_mode}.npy")
        self.valid_indices = np.load(f"{self.config['meta_features_path']}/{dataset_name}_valid_indices_{stat_mode}.npy")
        self.baseline_score = self.get_baseline_score()

        self.arch_encoder, self.base_meta_encoder, self.stat_meta_encoder = self.load_model()
        self.arch_encoder.eval()
        self.base_meta_encoder.eval()
        self.stat_meta_encoder.eval()

    def get_baseline_score(self):
        """Get the baseline score for the dataset."""
        with open(f"{self.config['baseline_scores_path']}/{self.dataset_name}_results_{self.alg}.pkl", 'rb') as f:
            baseline_score = pickle.load(f)
        return baseline_score['mean_uca']

    def load_model(self):
        """Load the model weights for architecture and meta-feature encoders."""
        arch_encoder = ArchitectureEncoder()
        base_meta_encoder = BaseMetaFeaturesEncoder()
        stat_meta_encoder = StatisticalMetaFeaturesEncoder(no_features=37)

        arch_encoder.load_state_dict(torch.load(f"{self.config['models_path']}/{self.dataset_name}arch_{self.num_samples}_{self.stat_mode}_{self.alg}.pth"))
        base_meta_encoder.load_state_dict(torch.load(f"{self.config['models_path']}/{self.dataset_name}meta_{self.num_samples}_{self.stat_mode}_{self.alg}.pth"))
        stat_meta_encoder.load_state_dict(torch.load(f"{self.config['models_path']}/{self.dataset_name}stat_{self.num_samples}_{self.stat_mode}_{self.alg}.pth"))

        return arch_encoder, base_meta_encoder, stat_meta_encoder

    def get_scaler(self):
        """Load the saved scaler."""
        return joblib.load(f"{self.config['models_path']}/{self.dataset_name}scaler_{200}_{self.stat_mode}_{self.alg}.pkl")

    def get_sample(self):
        """Get a sample of the meta and statistical features."""
        random_idx = np.random.choice(range(len(self.valid_indices)), self.num_samples)
        idx = self.valid_indices[random_idx]
        scaler = self.get_scaler()

        sampled_meta_features = np.mean(self.meta_features[idx], axis=0)
        sampled_stat_features = scaler.transform(self.stat_features[random_idx])
        sampled_stat_features = np.mean(sampled_stat_features, axis=0)

        return torch.tensor(sampled_meta_features).float(), torch.tensor(sampled_stat_features).float()

    def get_recommendation(self):
        """Get architecture recommendations based on meta and statistical features."""
        meta_features, stat_features = self.get_sample()

        base_meta_embeddings = self.base_meta_encoder(meta_features).squeeze()
        stat_embeddings = self.stat_meta_encoder(stat_features)
        meta_embedding = torch.cat([base_meta_embeddings, stat_embeddings])

        d_dict = DatasetsDicts("config.json", alg=self.alg)
        archs, scores, _ = d_dict.get_archs_scores(self.dataset_name)
        archs_tensor = torch.tensor([[len(a)-2, a[-1]] for a in archs]).float()
        arch_embeddings = self.arch_encoder(archs_tensor)

        similarities = torch.cosine_similarity(arch_embeddings, meta_embedding.unsqueeze(0))
        top_indices = torch.topk(similarities, 5).indices

        top_archs = [archs[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]

        return top_archs, top_scores

    def plot_results(self, results, output_path="./figures/results.png"):
        """Plot baseline vs recommended performance."""
        datasets = list(results.keys())
        baseline = [results[d]['baseline'] for d in datasets]
        recommended = [results[d]['recommended'] for d in datasets]

        x = np.arange(len(datasets))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, baseline, width, label="Baseline")
        ax.bar(x + width/2, recommended, width, label="Recommended")

        ax.set_xlabel("Datasets")
        ax.set_ylabel("Performance")
        ax.set_title("Baseline vs Recommended Performance")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()

if __name__ == "__main__":
    dataset_names = [
        'usps', 'mnist', 'cifar10', 'organcmnist', 'fashionmnist', 'stl10', 'svhn', 'semeion',
        'imagenet10', 'imagenetdog', 'gtsrb', 'coil100', 'coil20', 'organsmnist', 'octmnist',
        'dermamnist', 'breastmnist', 'organamnist', 'kmnist', 'bloodmnist'
    ]

    results = {}
    for dataset_name in dataset_names:
        try:
            tester = Tester(dataset_name, "config.json")
            recommended_archs, recommended_scores = tester.get_recommendation()
            baseline_score = tester.get_baseline_score()

            results[dataset_name] = {
                "baseline": baseline_score,
                "recommended": np.mean(recommended_scores)
            }
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")

    tester.plot_results(results)
