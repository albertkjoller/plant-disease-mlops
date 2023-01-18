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

    def __init__(self, lr: float = 1e-3, batch_size: int = 64, n_classes: int = 38):
        # Initialize as Lightning Module
        super().__init__()
        self.save_hyperparameters()

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

        # Include LogSoftmax for numerical stability
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

        # Setup learning rate
        self.lr = lr
        self.batch_size = batch_size
        self.n_classes = n_classes

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch["data"], batch["label"]
        z = self.log_softmax(self.model(x))

        # Compute loss and probability of prediction
        loss = F.nll_loss(z, y)
        log_prob, pred_class = torch.topk(z, k=1)
        prob = torch.exp(log_prob)

        # Compute accuracy
        pred_class = pred_class.reshape(y.shape)
        train_acc = torch.mean((pred_class == y).float())

        # Log to W&B dashboard
        self.log("train_loss", loss)
        self.log("train_acc", train_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["data"], batch["label"]
        z = self.log_softmax(self.model(x))

        # Compute loss and probability of prediction
        loss = F.nll_loss(z, y)
        log_prob, pred_class = torch.topk(z, k=1)
        prob = torch.exp(log_prob)

        # Compute accuracy
        pred_class = pred_class.reshape(y.shape)
        val_acc = torch.mean((pred_class == y).float())

        # Log to W&B dashboard
        self.log("val_loss", loss)
        self.log("val_acc", val_acc)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch["data"], batch["label"]
        z = self.log_softmax(self.model(x))

        if batch_idx != -1:
            # Compute loss and probability of prediction
            loss = F.nll_loss(z, y)
            log_prob, pred_class = torch.topk(z, k=1)
            prob = torch.exp(log_prob)

            # Compute accuracy
            pred_class = pred_class.reshape(y.shape)
            test_acc = torch.mean((pred_class == y).float())

            # Log to W&B dashboard
            # self.log("test_loss", loss)
            # self.log("test_acc", test_acc)
            return pred_class

        else:  # consistently set to -1 in deployment mode
            from src.models.model_utils import get_labels  # get labels dictionary here

            # Get label dictionary
            self.idx2class = get_labels()
            self.class2idx = {v: int(k) for (k, v) in self.idx2class.items()}

            # Get topK predictions
            K = 5
            log_prob, pred_class = torch.topk(z, k=K)
            prob = torch.exp(log_prob)

            output = {
                y[i]: {
                    i: {
                        "pred": pred_[i].item(),
                        "prob": round(prob_[i].item(), 3),
                        "label": self.idx2class[pred_[i].item()],
                    }
                    for i in range(K)
                }
                for i, (pred_, prob_) in enumerate(zip(pred_class, prob))
            }
            return output
