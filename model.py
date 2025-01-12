import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
import torchvision.models as models

class ImageClassifier(pl.LightningModule):
  def __init__(self, num_classes):
    super().__init__()
    self.model = models.resnet18(pretrained=True)
    self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    self.criterion = nn.CrossEntropyLoss()
    self.num_classes = num_classes

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    images, labels = batch
    outputs = self(images)
    loss = self.criterion(outputs, labels)
    acc = accuracy(outputs, labels, task='multiclass', num_classes=self.num_classes)
    self.log('train_loss', loss)
    self.log('train_acc', acc)
    return loss
  

  def validation_step(self, batch, batch_idx):
    images, labels = batch

    
    outputs = self(images)
    loss = self.criterion(outputs, labels)
    acc = accuracy(outputs, labels, task='multiclass', num_classes=self.num_classes)
    self.log('val_loss', loss)
    self.log('val_acc', acc)

    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    return optimizer