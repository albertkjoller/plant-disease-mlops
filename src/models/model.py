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

        # Freeze ResNet weights 
        for param in self.model.parameters():
            param.requires_grad = False
        # except for the last, fully connected output layer
        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True

        # Setup learning rate
        self.lr = lr

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch["data"], batch["label"]
        z = self.model(x)
        loss = F.cross_entropy(z, y)
        probs,pred_class = torch.topk(F.softmax(z,dim=0),k=1)
        pred_class=torch.reshape(pred_class,(torch.tensor(y.size()).item(),))
        train_acc = (torch.sum(pred_class==y)/torch.tensor(y.size())).item()
        # Log to W&B dashboard
        self.log("train_loss", loss)
        self.log("train_acc",train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["data"], batch["label"]
        z = self.model(x)
        loss = F.cross_entropy(z, y)
        probs,pred_class = torch.topk(F.softmax(z,dim=0),k=1)
        pred_class=torch.reshape(pred_class,(torch.tensor(y.size()).item(),))
        val_acc = (torch.sum(pred_class==y)/torch.tensor(y.size())).item()
        # Log to W&B dashboard
        self.log("val_loss", loss)
        self.log("val_acc",val_acc)
        return loss
