import sys
import json
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import TripletMarginLoss
from torch.optim.lr_scheduler import StepLR
from processing import DatasetsDicts, TrainingSamplesCreator, NASDataset
import wandb
from clustpy.data import (
    load_usps, load_mnist, load_cifar10, load_organ_c_mnist, load_fmnist, load_stl10, load_svhn,
    load_semeion, load_imagenet10, load_imagenet_dog, load_gtsrb, load_coil100, load_coil20,
    load_organ_s_mnist, load_oct_mnist, load_derma_mnist, load_breast_mnist, load_organ_a_mnist,
    load_kmnist, load_blood_mnist
)

class BaseMetaFeaturesEncoder(nn.Module):
    def __init__(self, input_dim=768, output_dim=32):
        super(BaseMetaFeaturesEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class StatisticalMetaFeaturesEncoder(nn.Module):
    def __init__(self, input_dim=32, output_dim=32):
        super(StatisticalMetaFeaturesEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class ArchitectureEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=64):
        super(ArchitectureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def train(config_path, stat_mode='classic', alg='dec', n_samples=1000, training_set=100000):
    with open(config_path + "config.json", "r") as f:
        config = json.load(f)

    dataset_names = [
        'usps', 'mnist', 'cifar10', 'organcmnist', 'fashionmnist', 'stl10', 'svhn', 'semeion',
        'imagenet10', 'imagenetdog', 'gtsrb', 'coil100', 'coil20', 'organsmnist', 'octmnist',
        'dermamnist', 'breastmnist', 'organamnist', 'kmnist', 'bloodmnist'
    ]
    datasets_loaders = {
        name: globals()[f'load_{name}'] for name in dataset_names
    }

    dataset_of_interest = dataset_names[int(sys.argv[1])]
    print(f"Training for {dataset_of_interest}")

    models_path = config["models_path"]
    d_dict = DatasetsDicts(config_path + "config.json", dataset_of_interest, alg=alg)
    dataset_dict = d_dict.build_dataset_dict()

    t = TrainingSamplesCreator(datasets_loaders, dataset_dict, config_path + "config.json", alg=alg)
    combs = t.create_training_samples(training_set, num_samples=n_samples, stat_mode=stat_mode)

    stat_features = np.array([c[1][5:] for c in combs])
    scaler = StandardScaler()
    stat_features = scaler.fit_transform(stat_features)
    combs = [(c[0], torch.Tensor(stat_features[i]), c[2], c[3]) for i, c in enumerate(combs)]

    joblib.dump(scaler, f"{models_path}/{dataset_of_interest}_scaler_{n_samples}_{stat_mode}_{alg}.pkl")

    nas_dataset = NASDataset(combs)
    dataloader = DataLoader(nas_dataset, batch_size=256, shuffle=True)

    wandb.init(project=config['project_name'], name=f"{dataset_of_interest}_encoder_{stat_mode}_{alg}")

    base_meta_features_encoder = BaseMetaFeaturesEncoder().to('cuda')
    stat_features_encoder = StatisticalMetaFeaturesEncoder().to('cuda')
    arch_encoder = ArchitectureEncoder().to('cuda')
    
    criterion = TripletMarginLoss().to('cuda')
    optimizer = torch.optim.Adam(
        list(base_meta_features_encoder.parameters()) +
        list(stat_features_encoder.parameters()) +
        list(arch_encoder.parameters()),
        lr=0.01
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 50
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            base_meta_features, stat_features, pos_arch, neg_arch = batch
            base_meta_features = base_meta_features.to('cuda').float()
            stat_features = stat_features.to('cuda').float()
            pos_arch = pos_arch.to('cuda').float()
            neg_arch = neg_arch.to('cuda').float()

            base_meta_enc = base_meta_features_encoder(base_meta_features)
            stat_features_enc = stat_features_encoder(stat_features)
            combined_meta_enc = torch.cat([base_meta_enc, stat_features_enc], dim=1)

            pos_arch_enc = arch_encoder(pos_arch)
            neg_arch_enc = arch_encoder(neg_arch)

            loss = criterion(combined_meta_enc, pos_arch_enc, neg_arch_enc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                wandb.log({"loss": loss.item()})

        scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    save_model(base_meta_features_encoder, f"{models_path}/{dataset_of_interest}_meta_encoder.pth")
    save_model(stat_features_encoder, f"{models_path}/{dataset_of_interest}_stat_encoder.pth")
    save_model(arch_encoder, f"{models_path}/{dataset_of_interest}_arch_encoder.pth")

if __name__ == "__main__":
    train(
        config_path="/home/wiss/aljoud/AutomatedDEC/",
        alg=sys.argv[2],
        n_samples=int(sys.argv[3]),
        training_set=100000
    )
