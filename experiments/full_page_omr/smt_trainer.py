import torch
import random
import numpy as np
import wandb
import lightning.pytorch as L

from .eval.eval_functions import compute_poliphony_metrics

from . import _globals

class SMTPP_Trainer(L.LightningModule):
    def __init__(self, smt_config, smt_model):
        super().__init__()
        self.model = smt_model
        self.padding_token = smt_config.padding_token

        self.preds = []
        self.grtrs = []

        self.worst_loss = -1
        self.worst_image = None
        self.best_loss = np.inf
        self.best_image = None

        self.save_hyperparameters()
        self.unfrozen_vision = False
    
    def configure_optimizers(self):
        return torch.optim.Adam(list(self.model.parameters()), lr=_globals.learning_rate, amsgrad=False)
    
    def forward(self, input, last_preds):
        return self.model(input, last_preds)
    
    def training_step(self, batch):
        
        if not self.unfrozen_vision and self.global_step > 200000:
            self.model.unfreeze_encoder()
            self.unfrozen_vision = True
            
        x, di, y, = batch
        outputs = self.model(x, di[:, :-1], labels=y)
        loss = outputs.loss
        self.log('loss', loss, on_epoch=True, batch_size=1, prog_bar=True)
        
        if loss.item() > self.worst_loss:
            self.worst_image = x
            self.worst_loss = loss.item()
        
        if loss.item() < self.best_loss:
            self.best_image = x
            self.best_loss = loss.item()
        
        return loss
        
    
    def validation_step(self, val_batch):
        x, dec_in, y = val_batch
        predicted_sequence, _ = self.model.predict(input=x)
        
        dec = "".join(predicted_sequence)
        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")
        dec = dec.replace("<s>", " ")

        gt = "".join([self.model.i2w[token.item()] for token in y.squeeze(0)[:-1]])
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")
        gt = gt.replace("<s>", " ")

        self.preds.append(dec)
        self.grtrs.append(gt)
        
    def on_validation_epoch_end(self, metric_name="val"):
        cer, ser, ler = compute_poliphony_metrics(self.preds, self.grtrs)
        
        random_index = random.randint(0, len(self.preds)-1)
        predtoshow = self.preds[random_index]
        gttoshow = self.grtrs[random_index]
        print(f"[Prediction] - {predtoshow}")
        print(f"[GT] - {gttoshow}")
        
        self.log(f'{metric_name}_CER', cer, on_epoch=True, prog_bar=True)
        self.log(f'{metric_name}_SER', ser, on_epoch=True, prog_bar=True)
        self.log(f'{metric_name}_LER', ler, on_epoch=True, prog_bar=True)
        
        self.preds = []
        self.grtrs = []
        
        return ser
        
    
    def test_step(self, test_batch):
        return self.validation_step(test_batch)
    
    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end("test")
