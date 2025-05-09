from pytorch_lightning.callbacks import Callback
import torch

class GraphCallback(Callback):
    def __init__(self, nmr_shape, nmr_type_shape, selfie_shape):
        super().__init__()
        self.nmr_shape = nmr_shape
        self.nmr_type_shape = nmr_type_shape
        self.selfie_shape = selfie_shape
        # self.device = device

    def on_fit_start(self, trainer, pl_module):
        self.device = pl_module.device
        # try:
        NMR = torch.zeros(self.nmr_shape).to(self.device)
        NMR_type_indicator = torch.zeros(self.nmr_type_shape).to(self.device).long()
        tgt_selfie = torch.zeros(self.selfie_shape, dtype=torch.long).to(self.device)

        # Wrap the forward call using a lambda to support multiple inputs
        trainer.logger.experiment.add_graph(
            pl_module,
            input_to_model=(NMR, NMR_type_indicator, tgt_selfie)
                            
        )
        print("[TensorBoard] Model graph added.")
        # except Exception as e:
        #     print(f"[TensorBoard] Failed to add graph: {e}")
