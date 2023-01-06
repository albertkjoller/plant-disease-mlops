import timm
import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F


class ImageClassification(LightningModule):
    def __init__(self, lr=1e-3, n_classes=38):
        super().__init__()
        self.model = timm.create_model(
            "resnet50", pretrained=True, num_classes=n_classes
        )
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
        loss = self.criterion(z, y)
        return loss
