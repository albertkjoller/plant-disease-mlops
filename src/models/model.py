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

        # Compute loss
        loss = F.cross_entropy(z, y)

        # Get predictions and compute accuracy
        probs, pred_class = torch.topk(F.softmax(z, dim=0), k=1)
        pred_class = torch.reshape(pred_class, (torch.tensor(y.size()).item(),))
        train_acc = (torch.sum(pred_class == y) / torch.tensor(y.size())).item()

        # Log to W&B dashboard
        self.log("train_loss", loss)
        self.log("train_acc", train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["data"], batch["label"]
        z = self.model(x)

        # Compute loss
        loss = F.cross_entropy(z, y)

        # Get predictions and compute accuracy
        probs, pred_class = torch.topk(F.softmax(z, dim=0), k=1)
        pred_class = torch.reshape(pred_class, (torch.tensor(y.size()).item(),))
        val_acc = (torch.sum(pred_class == y) / torch.tensor(y.size())).item()

        # Log to W&B dashboard
        self.log("val_loss", loss)
        self.log("val_acc", val_acc)

        return loss
