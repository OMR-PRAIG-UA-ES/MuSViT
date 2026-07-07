"""Evaluation metric, test loop, and backbone loader.

Groups three small utilities used across the project:
  * calc_cer_metric  -> sequence-level error rate,
  * engine_test      -> run the model over a dataloader and return that metric,
  * get_pretrained_model -> load the requested MusViT checkpoint.
"""

import editdistance
from transformers import ViTModel
from config import data_models



def calc_cer_metric(seqs_gt, preds):
    """Character/symbol Error Rate over a set of sequences.

    Args:
        seqs_gt (list[list[int]]): ground-truth id sequences.
        preds (list[list[int]]): predicted id sequences (same order).

    Returns:
        float: aggregate error rate.
    """
    total_dist = 0
    total_len = 0
    for pred, real in zip(preds, seqs_gt):
        # Edit distance between one prediction and its reference.
        eddist = editdistance.distance(pred, real)
        total_dist += eddist
        total_len += len(real)

    return float(total_dist)/float(total_len)


def engine_test(model, dl, interpolate_pos_encoding= False, blank= 0):
    """Run greedy CTC decoding over a dataloader and return the CER.

    Args:
        model (ViTRNN): the trained model (should be in eval mode).
        dl (DataLoader): validation or test dataloader.
        interpolate_pos_encoding (bool): forwarded to the model (True for LoRA).
        blank (int): CTC blank id, also used as the padding id to trim targets.

    Returns:
        float: CER over the whole dataloader.
    """
    seqs_gt = []
    preds_ctc = []
    for idx, (imgs, seqs, seqs_lens) in enumerate(dl):
        imgs, seqs = imgs.cuda(), seqs.cuda()

        # Greedy CTC decode -> list of predicted id sequences.
        pred_ctc = model.ctc_decode(imgs, interpolate_pos_encoding, blank)

        # Trim each target at the first blank/pad id to recover its true length.
        seqs = list(map(lambda sec: sec[0:sec.index(blank)] if blank in sec else sec, seqs.detach().cpu().numpy().tolist()))

        preds_ctc.extend(pred_ctc)
        seqs_gt.extend(seqs)

    cer = calc_cer_metric(seqs_gt, preds_ctc)

    return cer

def get_pretrained_model(model_name):
    """Load a pre-trained MusViT backbone from the Hugging Face Hub.

    Args:
        model_name (str): 'musvit' or 'musvit_light'.

    Returns:
        transformers.ViTModel: the loaded backbone.

    Raises:
        NotImplementedError: for any unrecognised model name.
    """
    if model_name == 'musvit':
        return ViTModel.from_pretrained(data_models[model_name]['link'], trust_remote_code=True)
    elif model_name == 'musvit_light':
        return ViTModel.from_pretrained(data_models[model_name]['link'], trust_remote_code=True)
    else:
        raise NotImplementedError()
