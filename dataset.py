import numpy as np 



def get_dataset(root, dataset):
	# noise-data
    data = np.load(root + '/' + dataset + '.npz')
    train_data = data['X_train']
    train_labels = data['y_train']
    test_data = data['X_test']
    test_labels = data['y_test']
    # clean-data
    clean_data = np.load(root + '/' + dataset +'_clean.npz')
    clean_labels = clean_data['y_train']

    dataset_train, dataset_test = Train_Dataset(train_data, train_labels), \
    		Train_Dataset(test_data, test_labels)
    return dataset_train, dataset_test, train_data, train_labels, clean_labels 


class Train_Dataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.train_data = np.array(data)
        self.train_labels = np.array(labels)
        self.length = len(self.train_labels)
        self.target_transform = target_transform

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.train_data, self.train_labels


class Semi_Labeled_Dataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.train_data = np.array(data)
        self.train_labels = np.array(labels)
        self.length = len(self.train_labels)
        self.target_transform = target_transform

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]

        if self.transform is not None:
            out1 = self.transform(img)
            out2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return out1, out2, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.train_data, self.train_labels


class Semi_Unlabeled_Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.train_data = np.array(data)
        self.length = self.train_data.shape[0]

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        img = self.train_data[index]

        if self.transform is not None:
            out1 = self.transform(img)
            out2 = self.transform(img)

        return out1, out2

    def __len__(self):
        return self.length

    def getData(self):
        return self.train_data
