import pytorch_lightning as pl
from utils.marker import place_marker

class DoneMarkerCallback(pl.Callback):
  def __init__(self, path):
    self.path = path
  def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    place_marker(self.path)
    print(f"TRAIN END CALLBACK")
    