from typing import List, Tuple
import torch
from pytorch_lightning import Callback, LightningModule, Trainer

class GraphCallback(Callback):
    """
        A callback to get the model as a graph in tensorboard. Must specifiy the input type as
        a Tuple ((size input 1), (size input 2), ...)
    """
    def __init__(self, dims: List[Tuple]):
        super().__init__()
        self.dims = dims
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        with torch.no_grad():
            pl_module.eval()
            tensors = [torch.normal(mean=0.0, std=1.0, size=v, device=pl_module.device) for v in self.dims]
            trainer.logger.experiment.add_graph(pl_module, tensors)
            pl_module.train()