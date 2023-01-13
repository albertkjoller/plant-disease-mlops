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
        # Setup data class
        self.dtype = dtype

        # Create as Path-object
        self.data_path = Path(data_path) / process_type / dtype

        # Load processed data
        self.images = torch.stack(torch.load(self.data_path / "images.pth"))
        self.labels = torch.LongTensor(torch.load(Path(self.data_path) / "labels.pth"))

        # Define number of classes and points
        self.n_classes = self.labels.unique().__len__()
        self.N_points = self.labels.__len__()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item, :]
        label = self.labels[item]
        return {"data": image, "label": label}

    def get_loader(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 4,
    ):
        """
        Exploits the MNIST-class for creating torch.Dataloaders.

        Parameters
        ----------
            batch_size: int
                Size of the batch used for train and test set
            shuffle: bool
                Whether to shuffle the dataset upon loading
            process_type: str
                The preprocessed version to use, i.e. 'color', 'grayscaled' or 'segmented'

        Returns
        -------
            dataloader: torch.utils.data.DataLoader
                A torch DataLoader to be used when training models.
        """

        dataloader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        return dataloader
