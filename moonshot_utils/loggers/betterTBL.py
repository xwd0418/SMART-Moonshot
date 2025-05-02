from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import logging

class BetterTBL(TensorBoardLogger):
  def __init__(self, best_metric=None, **kwargs):
    super().__init__(**kwargs)
    self.logger = logging.getLogger("lightning")
    self.best_metric = best_metric
    self.logger.info(
        f"Using {best_metric} as auto-logging hyperparameter value")

  @rank_zero_only
  def log_metrics(self, metrics, step):
    # super().log_metrics(metrics, step)
    for k, v in metrics.items():
      if self.best_metric in k:
        super().log_metrics({"hp_metric": v}, step)
