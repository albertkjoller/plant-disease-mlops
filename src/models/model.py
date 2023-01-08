import timm
import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F


class ImageClassification(LightningModule):
    """
    Defining the model used for classifying plant diseases.

    ...

    Attributes
    ----------
        lr: float
            Learning rate
        n_classes: int
            Number of classes to be outputted
    """

    def __init__(self, lr: float = 1e-3, n_classes: int = 38):
        # Initialize as Lightning Module
        super().__init__()

        # Load pre-train ResNet network
        self.model = timm.create_model(
            "resnet50", pretrained=True, num_classes=n_classes
        )

        # Define parameters
        self.params = self.parameters()
        self.lr = lr

    def configure_optimizers(self):
        return torch.optim.Adam(self.params, self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch["data"], batch["label"]
        z = self.model(x)
        loss = F.cross_entropy(z, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["data"], batch["label"]
        z = self.model(x)
        loss = F.cross_entropy(z, y)
        return loss
