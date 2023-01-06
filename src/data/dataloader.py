from pathlib import Path

import torch
from torch.utils.data import Dataset


class PlantVillage(Dataset):
    """
    For defining the PlantVillage dataset as a class (to be used when creating a torch.data.utils.DataLoader).

    ...

    Attributes
    ----------
        data_path: str
            Path to datafiles to be loaded
    """

    def __init__(
        self,
        dtype: str = "train",
        data_path: str = "data/processed",
        process_type: str = "color",
    ):
        """
        Initialize Dataset-class by loading datafiles.

        Parameters
        ----------
            dtype: str
                Type of dataset, i.e. either 'train', 'val' or 'test'
            data_path: str
                Path to datafiles to be loaded
            process_type: str
                type of dataset, i.e. either 'color', 'grayscale' or 'segmented'.
        """
        # Create as Path-object
        data_path = Path(data_path) / process_type / dtype

        # Load processed data
        #self.images = torch.load(data_path / "images.pth")
        #self.images = torch.stack(self.images).view(len(self.images), -1)

        #self.labels = torch.LongTensor(torch.load(Path(data_path) / "labels.pth"))
        self.images = torch.stack(torch.load(data_path/ "images.pth"))
        self.labels = torch.LongTensor(torch.load(Path(data_path) / "labels.pth"))
        #self.images = torch.load(data_path/dtype/ "images.pth")
        #self.images = self.images.view(self.images.shape[0], -1)
        #self.labels = torch.LongTensor(torch.load(data_path / dtype / "labels.pth"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item, :]
        label = self.labels[item]
        return {"data": image, "label": label}

    def get_loaders(self,dtype: str,data_path: str, batch_size: int, shuffle: bool, num_workers: int = 4, process_type: str = 'color'):
        """
        Exploits the MNIST-class for creating torch.Dataloaders.

        Parameters
        ----------
            data_path: str
                Path to datafiles to be loaded
            batch_size: int
                Size of the batch used for train and test set
            shuffle: bool
                Whether to shuffle the dataset upon loading
            process_type: str
                The preprocessed version to use, i.e. 'color', 'grayscaled' or 'segmented'
                
        Returns
        -------
            loaders: dict
                A dictionary of the dataloaders with keys being the dataset type and values being the
                torch.Dataloader-class.
        """
        if dtype == 'train':
            trainClass = PlantVillage(dtype='train', data_path=data_path, process_type=process_type)
            train_loader = torch.utils.data.DataLoader(
                trainClass, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            )
            return train_loader
        elif dtype == 'val':
            valClass = PlantVillage(dtype='val', data_path=data_path, process_type=process_type)
            val_loader = torch.utils.data.DataLoader(
                valClass, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            )
            return val_loader
        elif dtype=='test':
            testClass = PlantVillage(dtype='test', data_path=data_path, process_type=process_type)
            test_loader = torch.utils.data.DataLoader(
                testClass, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            )
            return test_loader
