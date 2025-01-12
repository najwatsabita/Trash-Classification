import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import wandb
from pytorch_lightning.loggers import WandbLogger
from dataset import get_dataloader
from model import ImageClassifier

wandb.login()
data_dir = 'dataset/'
num_classes = 6
batch_size = 16


train_loader, val_loader = get_dataloader(data_dir=data_dir, batch_size=batch_size)

model = ImageClassifier(num_classes=num_classes)

checkpoint_callback = ModelCheckpoint(
    monitor= 'val_loss',
    dirpath = 'checkpoint/',
    filename ='Trash-Classification-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min'
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

wandb_logger = WandbLogger(
    project='Trash-Classification',
    name = 'training-run',
    save_dir = 'logs/'
)

wandb_logger.experiment.config.update({
    "data_dir": data_dir,
    "num_classes":num_classes,
    "batch_size":batch_size,
    "max_epochs":20
})

trainer = pl.Trainer(
    max_epochs=20,
    callbacks=[checkpoint_callback, early_stopping],
    logger=wandb_logger
)
# Resume training dari checkpoint
#trainer.fit(
   # model, 
    #train_loader, 
    #val_loader,
    #ckpt_path='checkpoint/Trash-Classification-epoch=16-val_loss=0.26.ckpt'  # Ganti dengan path checkpoint terakhir
#)