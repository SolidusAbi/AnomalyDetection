import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt

class AnomalyMNIST(Dataset):
    ''' 
        Anomaly detection dataset for MNIST. The dataset is composed by the normal samples (1) and
        the anomaly samples (7). The known anomalies are anotated by the label 1, while the unknown
        anomalies are anotated by the label 0.

        Args:
            root: str
                Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            download: bool, optional
                If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.

            transform: callable, optional
                A function/transform that takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``. By default,
                ``torchvision.transforms.ToTensor`` is applied.
            
            n_known_anomalies: int, optional
                Number of known anomalies to use in the dataset. The number of unknown anomalies
                is 5% of the normal samples.

            seed: int, optional
                Seed to use for the random permutation of the anomaly indices.
    '''

    def __init__(self, root, download=False, transform = ToTensor(), n_known_anomalies=512, seed=None):
        self.transform = transform
        self.n_known_anomalies = n_known_anomalies
        mnist = MNIST(root = root, train=True, download=download, transform = transform)

        rng = torch.Generator() if not seed else torch.Generator().manual_seed(seed)
        normal_idx, anomaly_idx = self.create_subset(mnist, rng)

        mnist.targets = -1 * torch.ones(len(mnist.targets))
        mnist.targets[normal_idx] = 0
        mnist.targets[anomaly_idx] = 1
        self.subset = Subset(mnist, torch.cat([normal_idx, anomaly_idx]))

    def create_subset(self, dataset:MNIST, rng:torch.Generator) -> tuple:
        '''
            Create a subset of the dataset for anomaly detection. The indices of the normal
            samples are contaimnated by the 5% of unknown anomalies.

            Args:
                rng: torch.Generator
                    Random number generator to use for the random permutation of the anomaly indices.

            Returns:
                normal_idx: list
                    List of indices of the normal samples.
                anomaly_idx: list
                    List of indices of the anomaly samples.
        '''
        normal_idx = torch.where((dataset.targets == 1))[0]        
        anomaly_idx = torch.where((dataset.targets == 7))[0]
        rnd_perm = torch.randperm(len(anomaly_idx), generator=rng)

        n_unknown_anomalies = int(len(normal_idx)*0.01)

        normal_idx = torch.cat([normal_idx, anomaly_idx[rnd_perm[:n_unknown_anomalies]]])
        anomaly_idx = anomaly_idx[rnd_perm[n_unknown_anomalies:n_unknown_anomalies+self.n_known_anomalies]]

        return (normal_idx, anomaly_idx)
    
    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx) -> tuple:
        return self.subset[idx]
    
    def montage(self, n_row = 5, n_col= 5, seed=None):
        '''
            Plot a montage of the dataset.
            
            Args:
                n_row: int, optional
                    Number of rows of the montage.
                n_col: int, optional
                    Number of columns of the montage.
                seed: int, optional
                    Seed to use for the random permutation of the anomaly indices.

            Returns:
                fig: matplotlib.pyplot.figure
                    Figure of the montage.
        '''
        rng = torch.Generator() if not seed else torch.Generator().manual_seed(seed)
        rng = torch.randperm(len(self.subset), generator=rng)
        fig = plt.figure(figsize=(10,10))
        length = n_row*n_col

        plt.subplots_adjust(hspace=0.5)
        for i in range(length):
            plt.subplot(n_row,n_col,i+1)
            x, y = self.subset[rng[i]]
            plt.imshow(x.squeeze(), cmap='gray')
           
            if y == 1:
                plt.gca().spines['top'].set_color('red')
                plt.gca().spines['bottom'].set_color('red')
                plt.gca().spines['left'].set_color('red')
                plt.gca().spines['right'].set_color('red')

                plt.gca().spines['top'].set_linewidth(5)
                plt.gca().spines['bottom'].set_linewidth(5)
                plt.gca().spines['left'].set_linewidth(5)
                plt.gca().spines['right'].set_linewidth(5)

                plt.xticks([]), plt.yticks([])
            else:
                 plt.axis('off')

        return fig
