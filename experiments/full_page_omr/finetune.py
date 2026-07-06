import json

from fire import Fire
from loguru import logger

from . import _globals
from .config.ExperimentConfigWrapper import ExperimentConfig, experiment_config_from_dict
from .data import SyntheticGrandStaffDataset, CLFinetuningDataset, SynthRealFinetuningDataset
from .smt_foundation import SMTFoundationConfig, SMTFoundationModelForCausalLM
from .smt_trainer import SMTPP_Trainer
from .data_augmentation.data_augmentation import set_up_processor

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


DATASETS_TYPE = {
    "CL": CLFinetuningDataset,
    "SR": SynthRealFinetuningDataset,
    "CL1": SyntheticGrandStaffDataset,
    "R": None
}


def main(config: ExperimentConfig, experiment_name,
         foundation_architecture="ViTMAEBase", foundation_weights="carlospm12/LSMT-MAE-Base-1024-16",
         finetuning_technique="CL", from_checkpoint: str | None = None, resolution: int | None = None,
         max_steps: int = -1, train: bool = True, starting_weights: str | None = None):
    if resolution is None:
        _globals.resolution = 1024
    else:
        _globals.resolution = resolution
    resolution = _globals.resolution

    # Credentials (WANDB_API_KEY / HUGGINGFACE_KEY) are loaded once by the `musvit`
    # launcher (see musvit/env.py) before this function runs.

    logger.info(f"Using {finetuning_technique} technique, implementing {DATASETS_TYPE[finetuning_technique]}")

    data = DATASETS_TYPE[finetuning_technique](config)
    print("data_module_type:", type(data))

    set_up_processor(model=foundation_weights)

    logger.info(f"Creating MuSViT ({foundation_architecture}) from the weights: {foundation_weights}")
    config = SMTFoundationConfig(foundation_architecture=foundation_architecture, foundation_weights=foundation_weights,
                                 maxh=resolution, maxw=resolution, maxlen=7512, out_categories=len(data.train_dataset.w2i),
                                 padding_token=0, in_channels=3, w2i=data.train_dataset.w2i, i2w=data.train_dataset.i2w,
                                 d_model=256, dim_ff=256, num_dec_layers=8)
    model = SMTFoundationModelForCausalLM(config)

    if starting_weights is None:
        model_wrapper = SMTPP_Trainer(config, model)
    else:
        model_wrapper = SMTPP_Trainer.load_from_checkpoint(starting_weights, smt_config=config, smt_model=model)

    early_stopping = EarlyStopping(monitor="val_SER", min_delta=0.01, patience=3, mode="min", verbose=True)

    print(f"Checkpoints will be saved to \"{experiment_name}_{finetuning_technique}\"")
    epoch_checkpointer = ModelCheckpoint(dirpath="weights/", filename=f"{experiment_name}_{finetuning_technique}-epoch", every_n_epochs=1, save_on_train_epoch_end=True, enable_version_counter=False, save_top_k=1, verbose=True)
    checkpointer = ModelCheckpoint(dirpath="weights/", filename=f"{experiment_name}_{finetuning_technique}",
                                   monitor="val_SER", mode='min',
                                   save_top_k=1, verbose=True)

    wandb_logger = WandbLogger(project='Foundation_SMT',
                               name=f"{experiment_name}",
                               log_model=False, save_dir="wandb_logs/")

    trainer = Trainer(max_epochs=100000, max_steps=max_steps,
                      check_val_every_n_epoch=3500,
                      callbacks=[epoch_checkpointer, checkpointer, early_stopping], logger=wandb_logger,
                      precision='16-mixed')

    data.train_dataset.set_trainer_data(trainer)

    if train:
        trainer.fit(model_wrapper, datamodule=data, ckpt_path=from_checkpoint)

        try:
            from_checkpoint = checkpointer.best_model_path
            model_wrapper = SMTPP_Trainer.load_from_checkpoint(checkpoint_path=from_checkpoint)
        except:
            from_checkpoint = f"weights/{experiment_name}_{finetuning_technique}-end.ckpt"
            trainer.save_checkpoint(from_checkpoint, weights_only=True)

    if from_checkpoint == "":
        from_checkpoint = None

    trainer.test(model_wrapper, datamodule=data, ckpt_path=from_checkpoint)


def launch(config_path: str, experiment_name: str,
           foundation_architecture="ViTMAEBase", foundation_weights="carlospm12/LSMT-MAE-Base-1024-16",
           finetuning: str = "CL", from_checkpoint: str | None = None, resolution: int | None = None,
           max_steps: int = -1, train: bool = True, starting_weights: str | None = None, learning_rate: float | None = None):
    with open(config_path, 'r') as file:
        config_dict = json.load(file)
        config = experiment_config_from_dict(config_dict)

    if learning_rate is not None:
        _globals.learning_rate = learning_rate

    main(config=config, experiment_name=experiment_name,
         foundation_architecture=foundation_architecture,
         foundation_weights=foundation_weights, finetuning_technique=finetuning, from_checkpoint=from_checkpoint,
         resolution=resolution, max_steps=max_steps, train=train, starting_weights=starting_weights)


if __name__ == "__main__":
    # When run directly (instead of through `musvit`), load the root .env and log in.
    try:
        from musvit.env import setup as _setup_env

        _setup_env()
    except Exception:
        pass
    Fire(launch)