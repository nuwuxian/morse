import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from noise_utils import noisify
import torch

def get_dataset(root, dataset, num_classes=10):
    # noise-data
    data = np.load(root + '/' + dataset + '.npz')
    train_data = data['X_train']
    train_labels = data['y_train']
    test_data = data['X_test']
    test_labels = data['y_test']
    # clean-data
    clean_data = np.load(root + '/' + dataset +'_true.npz')
    clean_labels = clean_data['y_train']
    
    dataset_train = Train_Dataset(train_data, train_labels, num_classes=10, noise_type='none')
    dataset_test = Test_Dataset(test_data, test_labels)


    return dataset_train, dataset_test, train_data, train_labels, clean_labels


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * 12
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
            
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.FloatTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples


class Train_Dataset(Dataset):

    def __init__(self, data, label, transform = None, num_classes = 10,
                 noise_type='symmetric', noise_rate=0.5, select_class=-1):

        self.num_classes = num_classes

        self.train_data = data
        self.train_labels = label

        self.gt = self.train_labels.copy()

        self.transform = transform
        self.train_noisy_labels = self.train_labels.copy()
        self.noise_or_not = np.array([True for _ in range(len(self.train_labels))])
        self.P = np.zeros((num_classes, num_classes))

        if noise_type !='none':
            # noisify train data
            self.train_labels = np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])

            self.train_noisy_labels, self.actual_noise_rate, self.P = noisify(dataset=None, train_labels=self.train_labels,
                             noise_type=noise_type, noise_rate=noise_rate, random_state=0, nb_classes=self.num_classes,
                             select_class=select_class)

            self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]

            _train_labels=[i[0] for i in self.train_labels]
            self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(_train_labels)

    def __getitem__(self, index):

        feat, gt = self.train_data[index], int(self.train_noisy_labels[index])
        if self.transform is not None:
            feat = self.transform(feat)

        return feat, gt, index

    def __len__(self):
        return len(self.train_data)


class Test_Dataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.data = np.array(data)
        self.targets = np.array(labels)
        self.length = len(self.targets)

        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.data, self.targets


class Semi_Labeled_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = np.array(data)
        self.targets = np.array(labels)
        self.length = len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.data, self.targets


class Semi_Unlabeled_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = np.array(data)
        self.targets = np.array(labels)
        self.length = self.data.shape[0]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.data, self.targets