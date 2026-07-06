"""Model definition: a ViT backbone followed by a BiLSTM + CTC head.

Pipeline for one staff-region image:

    image
      -> MusViT (frozen or LoRA-adapted) produces patch tokens
      -> keep spatial tokens, reshape into a (rows x cols) grid
      -> linear projection to 256-d
      -> collapse the vertical (staff-height) axis by mean pooling
         => a left-to-right sequence of column features
      -> 2-layer bidirectional LSTM models horizontal context
      -> linear classifier emits per-column class logits
      -> log-softmax for CTC training / greedy CTC decoding for inference

CTC (Connectionist Temporal Classification) lets the network align a sequence
of ``cols`` column-frames to a shorter sequence of music symbols without needing
explicit per-symbol positions.
"""

import torch
from torch import nn
import torch.nn.functional as F
import itertools
from config import data_models
from utils.utils import get_pretrained_model




class ViTRNN(nn.Module):
    """ViT feature extractor + recurrent CTC head for staff-level OMR.

    Args:
        model_name (str): key into ``config.data_models`` selecting the
            pre-trained backbone (e.g. 'musvit').
        clases (int): number of output classes for the CTC classifier,
            including the blank symbol.
        shape_patches (Sequence[int]): (rows, cols) patch grid. ``cols`` becomes
            the CTC time-axis length; ``rows`` spans the staff height and is
            pooled away.
    """

    def __init__(self, model_name, clases, shape_patches):
        super(ViTRNN, self).__init__()
        self.model_name = model_name
        self.shape_patches = shape_patches

        # Load the pre-trained MusViT backbone (see utils.get_pretrained_model).
        self.backbone = get_pretrained_model(model_name)
        self.dropout = nn.Dropout(0.25)

        # Project each patch token from the ViT hidden size down to 256-d.
        self.projection = nn.Linear(data_models[self.model_name]['dim'], 256, bias=False)

        # 2-layer bidirectional LSTM over the column sequence.
        # input=256, hidden=256 per direction, dropout between layers.
        self.rnn = nn.LSTM(256, 256, 2, batch_first=True, dropout=0.5, bidirectional=True)

        # Final classifier: 256*2 (bidirectional) -> number of classes.
        self.classifier_ctc = nn.Linear(256*2, clases)

    def forward(self, images, interpolate_pos_encoding= False, only_logits= False):
        """Compute per-column class scores for a batch of images.

        Args:
            images (Tensor): batch of shape (B, C, H, W).
            interpolate_pos_encoding (bool): if True, let the ViT interpolate
                its positional embeddings to the actual patch grid (used with
                LoRA, where the input grid can differ from the pre-training
                resolution). If False, a fixed 64x64 grid is assumed and the
                needed rows are sliced out.
            only_logits (bool): if True return raw logits (used for decoding);
                otherwise return log-probabilities (used for CTC loss).

        Returns:
            Tensor: shape (B, cols, clases). Either logits or log-softmax
            depending on ``only_logits``.
        """
        model_info = data_models[self.model_name]

        # Run the backbone and drop non-spatial tokens (e.g. the [CLS] token)
        # by slicing from ``start_patch`` onward. Result: (B, num_tokens, dim).
        out = self.backbone(images, interpolate_pos_encoding=interpolate_pos_encoding).last_hidden_state[:, model_info['start_patch']:, :]

        num_rows = self.shape_patches[0]
        num_cols = self.shape_patches[1]

        if interpolate_pos_encoding:
            # Positional embeddings were interpolated to our exact grid, so the
            # token count already equals num_rows * num_cols.
            out = out.reshape(out.shape[0], num_rows, num_cols, -1)
        else:
            # Fixed-resolution path: the ViT saw a padded 64x64 patch grid.
            # Reshape to that full grid, then keep only the top ``num_rows``
            # rows that actually contain the staff (the rest was padding).
            out = out.reshape(out.shape[0], 64, 64, -1)[:, :num_rows, :, :]

        # Project every patch token to 256-d (dropout applied first).
        # Shape: (B, num_rows, num_cols, 256).
        out = self.projection(self.dropout(out))

        # Collapse the vertical (staff-height) axis: average over rows so each
        # column becomes a single feature vector -> (B, num_cols, 256).
        # This yields the left-to-right frame sequence CTC consumes.
        out = out.mean(dim=1)

        # Initial hidden/cell states for the LSTM:
        # (num_layers * num_directions, batch, hidden) = (2*2, B, 256).
        h0 = torch.zeros(2*2, out.size(0), 256).to(out.device)
        c0 = torch.zeros(2*2, out.size(0), 256).to(out.device)
        out, _ = self.rnn(out, (h0, c0))

        # Per-column class logits: (B, num_cols, clases).
        logs = self.classifier_ctc(out)

        if only_logits is True:
            return logs
        else:
            # CTC loss expects log-probabilities over the class axis.
            return F.log_softmax(logs, dim=-1)

    def ctc_decode(self, images, interpolate_pos_encoding= False, blank= 0):
        """Greedy CTC decoding: convert model output to symbol id sequences.

        Standard best-path decoding: take the argmax class at every column,
        collapse consecutive duplicates, and drop the blank symbol.

        Args:
            images (Tensor): batch of input images.
            interpolate_pos_encoding (bool): forwarded to ``forward``.
            blank (int): id of the CTC blank symbol to remove.

        Returns:
            list[list[int]]: one decoded id sequence per image in the batch.
        """
        log_probs = self(images, interpolate_pos_encoding, only_logits=True)

        # Best class per column: (B, cols).
        _, max_index = torch.max(log_probs, dim=-1)
        max_index = max_index.detach().cpu().numpy().tolist()

        # For each sequence: collapse runs of identical labels (itertools.groupby)
        # and then remove blanks -> the CTC-decoded symbol sequence.
        predicts = list(map(lambda sentence: [elem for elem, _ in itertools.groupby(sentence) if elem!=blank], max_index))
        return predicts

    def freeze_all(self):
        """Freeze every backbone parameter (used for linear probing).

        Only the projection / LSTM / classifier remain trainable after this.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
