
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

    def __init__(self, data_path: str, process_type: str = 'color'):
        """
        Initialize Dataset-class by loading datafiles.

        Parameters
        ----------
            data_path: str
                Path to datafiles to be loaded
            process_type: str
                type of dataset, i.e. either 'color', 'grayscale' or 'segmented'.
        """
        # Create as Path-object
        data_path = Path(data_path) / process_type

        # Load processed data
        self.images = torch.load(data_path / "images.pth")
        self.images = self.images.view(self.images.shape[0], -1)

        self.labels = torch.load(Path(data_path) / dtype / "labels.pth").type(torch.LongTensor)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item, :]
        label = self.labels[item]
        return {"data": image, "label": label}

