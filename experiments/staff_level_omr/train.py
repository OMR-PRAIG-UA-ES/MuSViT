"""Training entry point for staff-level Optical Music Recognition.

End-to-end flow:
  1. Parse CLI args (see arguments.py) and seed all RNGs for reproducibility.
  2. Load (image, symbol-sequence) pairs and encode + pad the sequences.
  3. Split into train / val / test and drop training samples whose sequence is
     too long for the chosen CTC time-axis.
  4. Build the pre-processing transforms, datasets and dataloaders.
  5. Instantiate ViTRNN and configure the backbone (LoRA adapters or frozen).
  6. Train with CTC loss, evaluating CER on validation and early-stopping on it.
  7. Reload the best checkpoint and report train / test CER.

Run individual experiments via executions.sh.
"""

import torch
import random
import numpy as np
from utils.data_utils import encode_sequences, pad_sequences, filter_max_len, load_image_gt_pairs
from datasets import CTC_ds
from torch.utils.data import DataLoader
from config import data_paths, data_models
from peft import LoraConfig, LoraModel
from model import ViTRNN
from torch.optim import Adam
from utils.utils import engine_test
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import time
from torchvision import transforms as T
from augments import get_ssl_transform
from arguments import parser_train
import os

# CTC blank symbol id. Kept at 0 throughout; the label encoder shifts real
# classes up by 1 (bias=1) so that id 0 stays reserved for the blank.
BLANK = 0

def _worker_init_fn(worker_id):
    """Seed all RNGs per worker so augmentations differ across images and epochs.

    albumentations 2.x uses its own numpy Generator (not np.random.seed), so
    we call set_random_seed() on the pipeline directly via the dataset handle.
    torch.utils.data.get_worker_info().seed is unique per worker per epoch.
    """
    worker_info = torch.utils.data.get_worker_info()
    # Derive a 32-bit seed from the worker's per-epoch seed.
    seed = worker_info.seed % 2**32
    np.random.seed(seed)
    random.seed(seed)
    # Reach into this worker's copy of the dataset and seed its augmentation
    # pipeline explicitly (albumentations keeps its own RNG).
    augment = worker_info.dataset.augment
    if augment is not None and hasattr(augment, 'set_random_seed'):
        augment.set_random_seed(seed)

if __name__ == '__main__':

    # --- Reproducibility ----------------------------------------------------
    # Seed the main process RNGs. Worker RNGs are seeded separately above.
    seed = 7
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # --- Arguments ----------------------------------------------------------
    args = parser_train.parse_args()
    print(args)

    ds_name = args.ds_name
    model_name = args.model_name

    # --- Data loading & label preparation -----------------------------------
    # load_image_gt_pairs -> list of (image_path, list_of_symbols).
    data = load_image_gt_pairs(data_paths[ds_name])
    images = [x[0] for x in data]
    sequences = [x[1] for x in data]

    # Fit a LabelEncoder over all symbols and map each sequence to integer ids.
    # bias=1 shifts every class id up by one, leaving 0 free for the CTC blank.
    le, sequences = encode_sequences(sequences, bias=1)

    # Right-pad all sequences to a common length with the blank id, and record
    # each sequence's true (unpadded) length for the CTC loss.
    sequences, sequences_len = pad_sequences(sequences, BLANK)

    # 80/20 split, then split that 20% into equal validation and test halves
    # (=> 80% train / 10% val / 10% test). The fixed random_state keeps splits
    # reproducible and aligned across images / seqs / lengths.
    train_imgs, test_imgs, train_seqs, test_seqs, train_len_seqs, test_len_seqs = train_test_split(images, sequences, sequences_len, test_size=0.2, random_state=seed)
    test_imgs, val_imgs, test_seqs, val_seqs, test_len_seqs, val_len_seqs = train_test_split(test_imgs, test_seqs, test_len_seqs, test_size=0.5, random_state=seed)

    # CTC requires input length >= target length. Drop *training* samples whose
    # symbol sequence is longer than the number of column frames (cols).
    train_imgs, train_seqs, train_len_seqs = filter_max_len(train_imgs, train_seqs, train_len_seqs, args.shape_patches[1])
    print('Partitions:', len(train_imgs), len(val_imgs), len(test_imgs))

    # --- Input pre-processing (resize / pad / tensorise) --------------------
    # Target input size is patch_size * (rows, cols) so the ViT sees exactly the
    # intended patch grid.
    if args.method == 'lora':
        # LoRA path: resize to the exact grid; positional embeddings will be
        # interpolated inside the model, so no extra padding is needed.
        processor = T.Compose([
            T.Resize((data_models[model_name]['patch_size']*args.shape_patches[0], data_models[model_name]['patch_size']*args.shape_patches[1])),
            T.ToTensor()
        ])
    else:
        # Linear-probing path: the model assumes a fixed 64x64 patch grid, so
        # pad the height up to 1024 px (=64*16) with white (fill=255) and keep
        # only the real rows later inside the model.
        print(data_models[model_name]['patch_size']*args.shape_patches[0], data_models[model_name]['patch_size']*args.shape_patches[1])
        processor = T.Compose([
            T.Resize((data_models[model_name]['patch_size']*args.shape_patches[0], data_models[model_name]['patch_size']*args.shape_patches[1])),
            T.Pad((0, 0, 0, 1024-data_models[model_name]['patch_size']*args.shape_patches[0]), fill=255),
            T.ToTensor()
        ])

    # --- Datasets & dataloaders ---------------------------------------------
    # Training set uses augmentation; val/test do not.
    train_ds = CTC_ds(train_imgs, train_seqs, train_len_seqs, processor, augment=get_ssl_transform())
    train_dl = DataLoader(train_ds, args.batch_size, num_workers=6, shuffle=True, worker_init_fn=_worker_init_fn)
    val_ds = CTC_ds(val_imgs, val_seqs, val_len_seqs, processor)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=6, shuffle=False)
    test_ds = CTC_ds(test_imgs, test_seqs, test_len_seqs, processor)
    test_dl = DataLoader(test_ds, args.batch_size, num_workers=6, shuffle=False)


    # --- Model & fine-tuning strategy ---------------------------------------
    # clases = number of encoded symbols + 1 (for the blank).
    model = ViTRNN(model_name, len(le.classes_)+1, args.shape_patches)
    print(model)

    if args.method == 'lora':
        # Inject low-rank adapters into the attention query/key/value projections.
        # Only these small adapters (plus the head) are trained; the ViT weights
        # stay frozen. use_rslora enables rank-stabilised scaling.
        lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", target_modules=["query", "key", "value"], use_rslora=True)
        model.backbone = LoraModel(model.backbone, lora_cfg, adapter_name='default')
    else:
        # Linear probing: freeze the entire backbone, train only the head.
        model.freeze_all()

    model = model.cuda().train()

    # Sanity print of how many parameters are actually being optimised.
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {num_params}")
    optimizer = Adam(model.parameters(), lr = args.lr, weight_decay=0.0)

    #print(model.backbone)

    # --- Training loop ------------------------------------------------------
    global_step=0
    losses = []
    best_cer = float('inf')   # best validation CER seen so far
    patience = 0              # epochs since the last improvement (early stop)
    for epoch in range(1, 1001):
        model.train()
        losses = []
        start_time = time.time()
        for idx, (imgs, seqs, seqs_lens) in enumerate(train_dl):
            optimizer.zero_grad()
            imgs, seqs, seqs_lens = imgs.cuda(), seqs.cuda(), seqs_lens.cuda()

            # Forward pass -> log-probs (B, cols, classes).
            # interpolate_pos_encoding is only enabled for the LoRA method.
            out_ctc = model(imgs, interpolate_pos_encoding=args.method=='lora')

            # CTC loss expects (T, B, C), so permute to put time first.
            # Input lengths are all equal to the number of columns (out shape[1]);
            # target lengths are the true unpadded sequence lengths.
            # zero_infinity guards against inf losses from impossible alignments.
            loss = F.ctc_loss(out_ctc.permute(1, 0, 2), seqs, torch.tensor([out_ctc.shape[1]]*out_ctc.shape[0]).cuda(), seqs_lens, BLANK, zero_infinity=True)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            global_step+=1

        # Report average loss and wall-clock time (minutes) for the epoch.
        epoch_time = (time.time() - start_time) / 60.0
        print(f"Epoch [{epoch}/{1000}], time {epoch_time}, avg loss {np.array(losses).mean()}")

        # --- Validation & checkpointing -------------------------------------
        if epoch>=args.start_eval:
            model.eval()
            with torch.no_grad():
                # Character Error Rate on the validation split.
                cer = engine_test(model, val_dl, interpolate_pos_encoding=args.method=='lora', blank= BLANK)
                print('Validation CER', cer)
                if cer < best_cer:
                    # New best -> save weights and reset the patience counter.
                    best_cer = cer
                    torch.save(model.state_dict(), model_name+'_'+ds_name+'_'+args.method+'_'+str(args.shape_patches[1])+'_ctc.pt')
                    patience = 0
                else:
                    patience += 1

        # Early stopping: stop after 30 evaluations without improvement.
        if patience >= 30:
            break

    # --- Final evaluation ---------------------------------------------------
    # Reload the best checkpoint (move to CPU to load, then back to GPU/eval).
    model = model.cpu()
    model.load_state_dict(torch.load(model_name+'_'+ds_name+'_'+args.method+'_'+str(args.shape_patches[1])+'_ctc.pt', map_location="cpu"))
    model = model.cuda().eval()
    with torch.no_grad():
        print('Results for', ds_name)
        cer = engine_test(model, test_dl, interpolate_pos_encoding=args.method=='lora', blank= BLANK)
        print('Test CER', cer)

