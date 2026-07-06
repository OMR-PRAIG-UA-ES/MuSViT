import argparse
import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from datasets import load_dataset
from huggingface_hub import login
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from transformers import (
    AutoImageProcessor,
    AutoModel,
    BeitImageProcessor,
    Kosmos2_5Model,
    PaliGemmaForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from . import data_generator
from . import distances
from . import metrics_utils
from .classifier_sklearn import ClassifierSKL

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None



def login_to_huggingface_if_available():
    """Authenticate with Hugging Face when HF_TOKEN is defined.

    Public models do not always require authentication. Gated/private models do.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("HF_TOKEN is not set. Continuing without Hugging Face authentication.")



def check_gpu(device_index):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        print("GPU is available")
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")

        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        # Select the requested GPU
        device = torch.device("cuda:"+str(device_index))
        print(f"Using device: {device}")

    else:
        print("GPU is not available")
        device = torch.device("cpu")
    return device

def export_accumulated_gt(metric_mat, ks, out_csv, use_mean=True):
    """
    GT-only baseline:
      - near: k smallest distances (excluding self)
      - far : k largest distances  (excluding self)
    Writes CSV columns: k=K_near, k=K_far
    """
    import numpy as np
    import pandas as pd

    N = metric_mat.shape[0]
    results = []

    for i in range(N):
        d = metric_mat[i].astype(np.float64).copy()

        # remove self-distance robustly
        idx_sorted = np.argsort(d)
        idx_sorted = idx_sorted[idx_sorted != i]

        idx_near_all = idx_sorted
        idx_far_all  = idx_sorted[::-1]

        row = {"row": i}
        for k in ks:
            if k <= len(idx_near_all):
                near_idx = idx_near_all[:k]
                far_idx  = idx_far_all[:k]

                near_vals = metric_mat[i, near_idx].astype(np.float64)
                far_vals  = metric_mat[i, far_idx].astype(np.float64)

                if use_mean:
                    row[f"k={k}_near"] = float(np.mean(near_vals))
                    row[f"k={k}_far"]  = float(np.mean(far_vals))
                else:
                    row[f"k={k}_near"] = float(np.sum(near_vals))
                    row[f"k={k}_far"]  = float(np.sum(far_vals))
            else:
                row[f"k={k}_near"] = None
                row[f"k={k}_far"]  = None

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False, sep=';', decimal=',')
    print("GT CSV written ->", out_csv)

# ----------------------------------------------------------------------------
def menu():
    parser = argparse.ArgumentParser(
        description=(
            "Compute image embeddings with vision models and compare embedding "
            "distances against transcription-based distances."
        )
    )
    parser.add_argument("-dev", "--device", default=0, type=int, help="GPU device index. Uses CPU if CUDA is unavailable.")
    parser.add_argument("--dataset", default="PRAIG/polish-scores", help="Hugging Face dataset name.")
    parser.add_argument("--split", default="train", help="Dataset split to process.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", dest="weights_encoder", help="Vision encoder model ID.")
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size used for embedding extraction.")
    parser.add_argument("--n-neighbors", default=1, type=int, help="Number of neighbors for the KNN classifier wrapper.")
    parser.add_argument("--embeddings-dir", default="embeddings", help="Directory used to cache embeddings.")
    parser.add_argument("--results-dir", default="results", help="Directory used to write CSV outputs.")
    parser.add_argument("--no-save-gt", action="store_true", help="Do not export the ground-truth nearest/farthest baseline CSV.")
    args = parser.parse_args()

    print("CONFIG:\n -", str(args).replace("Namespace(", "").replace(")", "").replace(", ", "\n - "))
    return args


def run_classifier_machine_learning(classifier, encoder, images_train, images_test, labels_train, labels_test):
    class_name = classifier.__class__.__name__
    
    emb_images_train = encoder(images_train)
    emb_images_test = encoder(images_test)

    classifier.fit(emb_images_train, labels_train)
    y_pred = classifier.predict(emb_images_test)
    P, R, F1, _ = precision_recall_fscore_support(y_true = labels_test, y_pred = y_pred, average = 'macro')

    results = classification_report(y_true = labels_test, y_pred = y_pred)

    print("{} - F1: {:.2f}%".format(class_name,100*F1))
    print(results)

    return P, R, F1


def get_path_and_class(row):
    img = row["image"]  # SMB usually provides PIL images with .filename; mozarteum can provide PIL images without a filename

    path = getattr(img, "filename", None)
    if isinstance(path, str) and path.strip():
        image_ref = path          # SMB: path
    else:
        image_ref = img           # mozarteum: PIL image without a path

    texture = row["page_texture"] if "page_texture" in row else "texture"

    # transcription (SMB uses row["page"] with ekern; mozarteum usually provides "transcription")
    if "page" in row:
        row_dict = ast.literal_eval(row["page"])
        transcription = row_dict["ekern"]
    else:
        transcription = row["transcription"]

    return image_ref, texture, transcription


def get_embeddings(processor, encoder, decoder, train_generator, preprocess_mode=None, target_size=1024):
    """
    - Qwen: one embedding per image = mean of all tokens (tamaño fijo D)
    - Other models: one embedding per image = flattened tokens/patches (legacy mode, no average pooling)
    """
    TARGET_LAYER_MODEL = "carlospm12/LSMT-MAE-Large-1024-16"
    TARGET_HIDDEN_LAYER_IDX = 15

    import torch
    import torchvision.transforms as T

    list_output_encoder = []
    list_labels = []
    idx = 0
    idx_batch = 1
    is_qwen = hasattr(encoder, "get_image_features")

    # ------------------------------------------------------------
    # Manual preprocessing (the pil_resize_1024)
    if preprocess_mode == "pil_resize_1024":
        tfm = T.Compose([
            T.ToPILImage(),
            T.Resize((target_size, target_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

    def process_img(img):
        if preprocess_mode == "pil_resize_1024":
            # manual resize path
            if isinstance(img, np.ndarray):
                pil = Image.fromarray(img)
            else:
                pil = img
            pil = pil.convert("RGB").resize((target_size, target_size), resample=Image.BILINEAR)

            # Qwen: use AutoImageProcessor to obtain image_grid_thw
            if is_qwen and (processor is not None):
                try:
                    # avoid resizing again when supported by the installed version
                    return processor(images=pil, return_tensors="pt", do_resize=False, do_center_crop=False)
                except TypeError:
                    return processor(images=pil, return_tensors="pt")

            # other manually preprocessed models
            x = T.ToTensor()(pil).unsqueeze(0)
            return {"pixel_values": x}

        # modo normal
        return processor(images=img, return_tensors="pt")

    # ------------------------------------------------------------
    def _extract_last_hidden(out, prefer_hidden_layer_idx=None):
        # If the model returns a tuple/list
        if isinstance(out, (tuple, list)):
            # If an intermediate layer is requested, intentamos leer hidden_states si viene en dict/obj
            # pero en tupla normalmente out[0] es last_hidden_state
            return out[0]

        # If the model returns a ModelOutput-like object (lo normal en HF)
        if prefer_hidden_layer_idx is not None:
            if hasattr(out, "hidden_states") and out.hidden_states is not None:
                hs = out.hidden_states
                if prefer_hidden_layer_idx < 0 or prefer_hidden_layer_idx >= len(hs):
                    raise RuntimeError(
                        f"Hidden layer idx {prefer_hidden_layer_idx} out of range. "
                        f"len(hidden_states)={len(hs)}"
                    )
                return hs[prefer_hidden_layer_idx]

        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state

        # If the model returns a dict
        if isinstance(out, dict):
            if prefer_hidden_layer_idx is not None and "hidden_states" in out and out["hidden_states"] is not None:
                hs = out["hidden_states"]
                if prefer_hidden_layer_idx < 0 or prefer_hidden_layer_idx >= len(hs):
                    raise RuntimeError(
                        f"Hidden layer idx {prefer_hidden_layer_idx} out of range. "
                        f"len(hidden_states)={len(hs)}"
                    )
                return hs[prefer_hidden_layer_idx]
            if "last_hidden_state" in out:
                return out["last_hidden_state"]

        # If it is already a tensor
        if torch.is_tensor(out):
            return out

        return out


    # ------------------------------------------------------------
    def _flatten_tokens_per_image(x: torch.Tensor, B: int) -> torch.Tensor:
        """
        (legacy mode, no average pooling)
          (B,H,W,D) -> (B, HW*D)
          (B,N,D)   -> (B, N*D)
          (B,D)     -> (B, D)
          (N,D) con B=1 -> (1, N*D)
        """
        if x is None:
            raise RuntimeError("Encoder output is None")

        if x.dim() == 4:  # (B,H,W,D)
            return x.reshape(x.size(0), -1)

        if x.dim() == 3:  # (B,N,D)
            return x.reshape(x.size(0), -1)

        if x.dim() == 2:  # (B,D) o (N,D)
            if x.size(0) == B:
                return x
            if B == 1:
                return x.reshape(1, -1)
            # last fallback when the first dimension is divisible by the batch size
            if x.size(0) % B == 0:
                return x.reshape(B, -1)
            raise RuntimeError(f"Cannot reshape x {tuple(x.shape)} a B={B}")

        if x.dim() == 1:
            return x.unsqueeze(0)

        raise RuntimeError(f"Unexpected output shape: {tuple(x.shape)}")


    # ------------------------------------------------------------
    encoder.eval()
    with torch.no_grad():
        for images, labels in train_generator:
            print("Processing batch", idx_batch)
            processed = [process_img(img) for img in images]

            # -----------------------------
            # Qwen3-VL case (AutoImageProcessor + get_image_features)
            # -----------------------------
            if hasattr(encoder, "get_image_features") and ("image_grid_thw" in processed[0]):
                pixel_values = torch.cat([p["pixel_values"] for p in processed], dim=0).to(config.device)
                image_grid_thw = torch.cat([p["image_grid_thw"] for p in processed], dim=0).to(config.device)
                B = image_grid_thw.size(0)

                out_feats = encoder.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)

                # 1) extract image embeddings robustly
                if isinstance(out_feats, (tuple, list)):
                    image_embeds = out_feats[0]
                elif isinstance(out_feats, dict):
                    image_embeds = out_feats.get("image_embeds", None)
                    if image_embeds is None:
                        image_embeds = out_feats.get("last_hidden_state", None)
                    if image_embeds is None:
                        image_embeds = out_feats.get("image_features", None)
                    if image_embeds is None:
                        raise RuntimeError(f"Qwen: dict without expected keys. keys={list(out_feats.keys())}")
                else:
                    image_embeds = out_feats

                if isinstance(image_embeds, (tuple, list)):
                    image_embeds = image_embeds[0]

                # 2) pooling -> (B, D)
                if torch.is_tensor(image_embeds):
                    if image_embeds.dim() == 3:                 # (B, T, D)
                        feats = image_embeds.mean(dim=1)        # (B, D)
                    elif image_embeds.dim() == 2:
                        if image_embeds.size(0) == B:           # (B, D)
                            feats = image_embeds
                        elif B == 1:                            # (T, D)
                            feats = image_embeds.mean(dim=0, keepdim=True)
                        else:
                            raise RuntimeError(f"Qwen: unexpected 2D tensor {tuple(image_embeds.shape)} con B={B}")
                    elif image_embeds.dim() == 4:               # (B, H, W, D)
                        feats = image_embeds.mean(dim=(1, 2))   # (B, D)
                    else:
                        raise RuntimeError(f"Qwen: tensor with unexpected shape {tuple(image_embeds.shape)}")
                else:
                    # list of tensors (Ti, D)
                    feats = torch.stack([t.mean(dim=0) for t in image_embeds], dim=0)  # (B, D) 

            # -----------------------------
            # Kosmos-2.5 case (processor devuelve flattened_patches/height/width)
            elif "flattened_patches" in processed[0]:
                flattened = torch.cat([p["flattened_patches"] for p in processed], dim=0).to(config.device)
                height    = torch.cat([p["height"] for p in processed], dim=0).to(config.device)
                width     = torch.cat([p["width"] for p in processed], dim=0).to(config.device)
                B = flattened.size(0)

                out = encoder(flattened_patches=flattened, height=height, width=width)
                x = _extract_last_hidden(out)
                feats = _flatten_tokens_per_image(x, B=B)  # (B, D') modo antiguo

            # -----------------------------
            # General case (BEiT/DiT/DINO/SigLIP/PaliGemma tower/etc.)
            else:
                pixel_values = torch.cat([p["pixel_values"] for p in processed], dim=0).to(config.device)
                B = pixel_values.size(0)

                # ------------------------------------------------------------
                # Special case: LSMT-MAE-Large-1024-16 -> extract embeddings from layer 15.
                if getattr(config, "weights_encoder", None) == TARGET_LAYER_MODEL:
                    out = encoder(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
                    x = _extract_last_hidden(out, prefer_hidden_layer_idx=TARGET_HIDDEN_LAYER_IDX)
                else:
                    out = encoder(pixel_values=pixel_values)
                    x = _extract_last_hidden(out)

                feats = _flatten_tokens_per_image(x, B=B)  # (B, D') modo antiguo

            # Store one embedding per image
            feats = feats.detach().to(torch.float32).cpu()
            if feats.dim() == 1:
                feats = feats.unsqueeze(0)

            for vec in feats:
                list_output_encoder.append(vec.numpy())

            list_labels.extend(labels)
            idx += len(images)
            print("[Generator] encoded images:", idx, "/", train_generator.getNumberImages())
            idx_batch += 1

    assert len(list_output_encoder) == train_generator.getNumberImages(), (
        f"Embeddings ({len(list_output_encoder)}) != num imágenes ({train_generator.getNumberImages()})."
    )
    assert list_output_encoder[0].ndim == 1, "Each embedding must be a 1D vector (D,)"

    return list_output_encoder, list_labels



def _to_device_dict(d, device):
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _extract_tensor_output(out):
    """
    Convert a Hugging Face output to a tensor so fvcore can trace it.
    """
    if torch.is_tensor(out):
        return out

    if isinstance(out, (tuple, list)):
        return _extract_tensor_output(out[0])

    if isinstance(out, dict):
        for key in ["last_hidden_state", "image_embeds", "image_features", "pooler_output"]:
            if key in out and out[key] is not None:
                return _extract_tensor_output(out[key])
        first_key = list(out.keys())[0]
        return _extract_tensor_output(out[first_key])

    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state

    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds

    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output

    raise RuntimeError(f"Cannot extract a tensor output from type {type(out)}")


class PixelValuesFlopsWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values):
        out = self.encoder(pixel_values=pixel_values)
        return _extract_tensor_output(out)


class KosmosFlopsWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, flattened_patches, height, width):
        out = self.encoder(
            flattened_patches=flattened_patches,
            height=height,
            width=width
        )
        return _extract_tensor_output(out)


class QwenFlopsWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values, image_grid_thw):
        out = self.encoder.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw
        )
        return _extract_tensor_output(out)


def load_image_for_flops(image_ref):
    """
    image_ref can be:
      - path string
      - PIL.Image
      - numpy array
    """
    if isinstance(image_ref, str):
        return Image.open(image_ref).convert("RGB")

    if isinstance(image_ref, Image.Image):
        return image_ref.convert("RGB")

    if isinstance(image_ref, np.ndarray):
        return Image.fromarray(image_ref).convert("RGB")

    raise RuntimeError(f"Unsupported image type for FLOPs: {type(image_ref)}")


def process_one_image_for_flops(img, processor, encoder, preprocess_mode=None, target_size=1024):
    is_qwen = hasattr(encoder, "get_image_features")

    if preprocess_mode == "pil_resize_1024":
        pil = load_image_for_flops(img)
        pil = pil.resize((target_size, target_size), resample=Image.BILINEAR)

        if is_qwen and processor is not None:
            try:
                return processor(
                    images=pil,
                    return_tensors="pt",
                    do_resize=False,
                    do_center_crop=False
                )
            except TypeError:
                return processor(images=pil, return_tensors="pt")

        x = T.ToTensor()(pil).unsqueeze(0)
        return {"pixel_values": x}

    return processor(images=img, return_tensors="pt")


def estimate_encoder_gflops(
    encoder,
    processor,
    image_ref,
    device,
    preprocess_mode=None,
    target_size=1024,
    print_details=True
):
    """
    Estimate approximate GFLOPs for one image.

    Returns:
      gflops, flops_totales
    """
    if FlopCountAnalysis is None:
        print("Could not calculate GFLOPs: fvcore is missing. Install it with: pip install fvcore")
        return None, None

    encoder.eval()

    img = load_image_for_flops(image_ref)

    processed = process_one_image_for_flops(
        img=img,
        processor=processor,
        encoder=encoder,
        preprocess_mode=preprocess_mode,
        target_size=target_size
    )

    processed = _to_device_dict(processed, device)

    try:
        with torch.no_grad():
            if hasattr(encoder, "get_image_features") and "image_grid_thw" in processed:
                wrapper = QwenFlopsWrapper(encoder).to(device).eval()

                inputs = (
                    processed["pixel_values"],
                    processed["image_grid_thw"]
                )

            elif "flattened_patches" in processed:
                wrapper = KosmosFlopsWrapper(encoder).to(device).eval()

                inputs = (
                    processed["flattened_patches"],
                    processed["height"],
                    processed["width"]
                )

            else:
                wrapper = PixelValuesFlopsWrapper(encoder).to(device).eval()

                inputs = (
                    processed["pixel_values"],
                )

            flops = FlopCountAnalysis(wrapper, inputs)
            total_flops = flops.total()
            gflops = total_flops / 1e9

            if print_details:
                print("\n=== FLOPs ===")
                print(f"Encoder: {getattr(config, 'weights_encoder', 'unknown')}")
                print(f"GFLOPs por imagen: {gflops:.4f}")
                print(f"FLOPs totales: {total_flops:.0f}")

                unsupported = flops.unsupported_ops()
                if len(unsupported) > 0:
                    print("Operaciones no contabilizadas por fvcore:")
                    print(unsupported)

            return gflops, total_flops

    except Exception as e:
        print("\nCould not calculate GFLOPs for this model.")
        print("Reason:", repr(e))
        return None, None
    

def get_model_for_cost_stats(model):
    """
    Returns el submodelo sobre el que tiene más sentido contar parámetros.

    For most models, 'model' ya es el encoder visual.
    For Qwen this can be the full model, so we try to extract the visual component.
    """
    for attr in ["visual", "vision_model", "vision_tower"]:
        if hasattr(model, attr):
            submodel = getattr(model, attr)
            if submodel is not None:
                return submodel

    if hasattr(model, "model"):
        inner = model.model
        for attr in ["visual", "vision_model", "vision_tower"]:
            if hasattr(inner, attr):
                submodel = getattr(inner, attr)
                if submodel is not None:
                    return submodel

    return model

def count_model_parameters_and_buffers(model):
    params_total = sum(p.numel() for p in model.parameters())
    buffers_total = sum(b.numel() for b in model.buffers())

    state_total = params_total + buffers_total

    return params_total, buffers_total, state_total


def model_state_memory_mb(model):
    total_bytes = 0

    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()

    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()

    return total_bytes / (1024 ** 2)


def get_model_cost_stats(model):
    cost_model = get_model_for_cost_stats(model)

    params_total, buffers_total, state_total = count_model_parameters_and_buffers(cost_model)
    state_memory_mb = model_state_memory_mb(cost_model)

    return {
        "params_total": params_total,
        "buffers_total": buffers_total,
        "state_total": state_total,
        "params_total_m": params_total / 1e6,
        "buffers_total_m": buffers_total / 1e6,
        "state_total_m": state_total / 1e6,
        "state_memory_mb": state_memory_mb,
    }


def print_model_cost_stats(model_name, input_size, cost_stats):
    print("\n=== MODEL COST STATS ===")
    print("Model:", model_name)
    print("Input size:", input_size)
    print(f"Params total: {cost_stats['params_total']:,} ({cost_stats['params_total_m']:.2f} M)")
    print(f"Buffers total: {cost_stats['buffers_total']:,} ({cost_stats['buffers_total_m']:.2f} M)")
    print(f"State total: {cost_stats['state_total']:,} ({cost_stats['state_total_m']:.2f} M)")
    print(f"State memory: {cost_stats['state_memory_mb']:.2f} MB")

def run_analysis(config):
    """Run the embedding-distance analysis for a single vision encoder.

    `config` is an argparse-style namespace carrying the fields produced by
    ``menu()`` (device, dataset, split, weights_encoder, batch_size,
    n_neighbors, embeddings_dir, results_dir, no_save_gt).
    """

    login_to_huggingface_if_available()

    name_dataset = config.dataset
    ds = load_dataset(name_dataset)

    # Select the requested dataset split.
    ds_train = ds[config.split]
    
    dict_images_train = {}
    for row in ds_train:
        image_ref, texture, ekern_transcription = get_path_and_class(row)
        if texture not in dict_images_train:
            dict_images_train[texture] = []    
        
        sample = (image_ref, ekern_transcription)
        dict_images_train[texture].append(sample)
    
    
    dict_models = {
        "microsoft/beit-base-patch16-224-pt22k": 
        {
            "size":224, 
            "image_processor": lambda model_weights: BeitImageProcessor.from_pretrained(model_weights),
            "model": lambda model_weights: AutoModel.from_pretrained(model_weights)
        },
        "microsoft/beit-large-patch16-512": 
        {
            "size":224,
            "image_processor": lambda model_weights: BeitImageProcessor.from_pretrained(model_weights),
            "model": lambda model_weights: AutoModel.from_pretrained(model_weights)
        },
        "microsoft/dit-base": 
        {
            "size":224, 
            "image_processor": lambda model_weights: BeitImageProcessor.from_pretrained(model_weights),
            "model": lambda model_weights: AutoModel.from_pretrained(model_weights)
        },
        "models/MAE-8X8-Small": 
        {
            "size":512, 
            "processor_class": BeitImageProcessor,
            "model": lambda model_weights: AutoModel.from_pretrained(model_weights)
        },
        "google/siglip2-so400m-patch14-384":
        {
            "size":384, 
            "image_processor": lambda model_weights: AutoImageProcessor.from_pretrained(model_weights),
            "model": lambda model_weights: AutoModel.from_pretrained(model_weights).vision_model
        },
        "google/siglip-so400m-patch14-384":
        {
            "size":384, 
            "image_processor": lambda model_weights: AutoImageProcessor.from_pretrained(model_weights),
            "model": lambda model_weights: AutoModel.from_pretrained(model_weights).vision_model
        },
        "google/siglip2-so400m-patch14-224":
        {
            "size":224, 
            "image_processor": lambda model_weights: AutoImageProcessor.from_pretrained(model_weights),
            "model": lambda model_weights: AutoModel.from_pretrained(model_weights).vision_model
        },
        "google/siglip-so400m-patch14-224":
        {
            "size":224, 
            "image_processor": lambda model_weights: AutoImageProcessor.from_pretrained(model_weights),
            "model": lambda model_weights: AutoModel.from_pretrained(model_weights).vision_model
        },
        "facebook/dinov2-base": 
        {
            "size":224, 
            "image_processor": lambda model_weights: AutoImageProcessor.from_pretrained(model_weights),
            "model": lambda model_weights: AutoModel.from_pretrained(model_weights)
        },
        "facebook/dinov2-large": 
        {
            "size":224, 
            "image_processor": lambda model_weights: AutoImageProcessor.from_pretrained(model_weights),
            "model": lambda model_weights: AutoModel.from_pretrained(model_weights)
        },
        "facebook/dinov2-giant": 
        {
            "size":224, 
            "image_processor": lambda model_weights: AutoImageProcessor.from_pretrained(model_weights),
            "model": lambda model_weights: AutoModel.from_pretrained(model_weights)
        },
        "google/paligemma2-3b-pt-224": 
        {
            "size": 224, 
            "image_processor": lambda model_weights: AutoImageProcessor.from_pretrained(model_weights),
            "model": lambda model_weights: PaliGemmaForConditionalGeneration.from_pretrained(model_weights).model.vision_tower
        },
        "google/paligemma2-3b-pt-448": 
        {
            "size": 448, 
            "image_processor": lambda model_weights: AutoImageProcessor.from_pretrained(model_weights),
            "model": lambda model_weights: PaliGemmaForConditionalGeneration.from_pretrained(model_weights).model.vision_tower
        },
        "google/paligemma2-3b-pt-896": 
        {
            "size": 896, 
            "image_processor": lambda model_weights: AutoImageProcessor.from_pretrained(model_weights),
            "model": lambda model_weights: PaliGemmaForConditionalGeneration.from_pretrained(model_weights).model.vision_tower
        },
        "Qwen/Qwen3-VL-8B-Instruct": {
            "size": 1024,
            "preprocess": "pil_resize_1024",
            "image_processor": lambda w: AutoImageProcessor.from_pretrained(w),
            "model": lambda w: Qwen3VLForConditionalGeneration.from_pretrained(w),
        },

        "facebook/dinov3-vit7b16-pretrain-lvd1689m": {
            "size": 1024,
            "preprocess": "pil_resize_1024",
            "image_processor": lambda model_weights: AutoImageProcessor.from_pretrained(model_weights),
            "model": lambda model_weights: AutoModel.from_pretrained(model_weights),
        },
        "facebook/dinov3-vitb16-pretrain-lvd1689m": {
            "size": None,
            "image_processor": lambda model_weights: AutoImageProcessor.from_pretrained(model_weights),
            "model": lambda model_weights: AutoModel.from_pretrained(model_weights),
        },
        "microsoft/kosmos-2.5" : {
            "size": None,
            "image_processor": lambda model_weights: AutoImageProcessor.from_pretrained(model_weights),
            "model": lambda model_weights: Kosmos2_5Model.from_pretrained(model_weights).vision_model,
        },
        "carlospm12/LSMT-MAE-Small-1024-16": {
            "size": 1024,
            "preprocess": "pil_resize_1024",
            "model": lambda model_weights: AutoModel.from_pretrained(model_weights)
        },
        "carlospm12/LSMT-MAE-Base-1024-16": {
            "size": 1024,
            "preprocess": "pil_resize_1024",
            "model": lambda model_weights: AutoModel.from_pretrained(model_weights)
        },
        "carlospm12/LSMT-MAE-Large-1024-16": {
            "size": 1024,
            "preprocess": "pil_resize_1024",
            "model": lambda model_weights: AutoModel.from_pretrained(model_weights)
        },
        
    }
        
    config.device = check_gpu(config.device)
    config.freeze_encoder = True
    # config.weights_encoder is provided by --model.


    config.weights_decoder = "facebook/detr-resnet-50"
    config.lr = 0.001
    config.epochs = 100
    config.CLS = True
    config.batch_size = config.batch_size
    config.patience = 20
    config.metric = metrics_utils.Accuracy
    config.validation_monitor = "max"
    config.path_model = "models/model.pt"   
    config.n_neighbors = config.n_neighbors
    config.n_train_pages = 5
    config.outputEncoder_Token = True
    config.saveGT = not config.no_save_gt

    #----------------------------------------------------------------
    list_filenames_train = [element[0] for key in dict_images_train.keys() for element in dict_images_train[key]]
    list_transcriptions_train = [element[1] for key in dict_images_train.keys() for element in dict_images_train[key]]
    list_labels_train = [key for key in dict_images_train.keys() for element in dict_images_train[key]]

    height = dict_models[config.weights_encoder]["size"]
    width = dict_models[config.weights_encoder]["size"]

    if height is None or width is None:
        config.input_size_str = "-"
    else:
        config.input_size_str = f"{height}x{width}"

    preprocess_mode = dict_models[config.weights_encoder].get("preprocess", None)
    is_manual = (preprocess_mode is not None)

    if is_manual:
        if "image_processor" in dict_models[config.weights_encoder]:
            processor = dict_models[config.weights_encoder]["image_processor"](config.weights_encoder)
        else:
            processor = None
    else:
        if ("processor_class" in dict_models[config.weights_encoder]):
            assert("image_processor" not in dict_models[config.weights_encoder])
            processorClass = dict_models[config.weights_encoder]["processor_class"]
            processor = processorClass(size={"height": height, "width": width},
                                    do_center_crop=False, do_normalize=False)
        else:
            assert("image_processor" in dict_models[config.weights_encoder])
            processor = dict_models[config.weights_encoder]["image_processor"](config.weights_encoder)
            if height is not None:
                try:
                    height_model = processor.crop_size["height"]
                    width_model  = processor.crop_size["width"]
                except:
                    height_model = processor.size["height"]
                    width_model  = processor.size["width"]
                assert(height == height_model and width == width_model)

    encoder = dict_models[config.weights_encoder]["model"](config.weights_encoder).to(config.device)

    cost_stats = get_model_cost_stats(encoder)

    config.encoder_params_total = cost_stats["params_total"]
    config.encoder_buffers_total = cost_stats["buffers_total"]
    config.encoder_state_total = cost_stats["state_total"]

    config.encoder_params_total_m = cost_stats["params_total_m"]
    config.encoder_buffers_total_m = cost_stats["buffers_total_m"]
    config.encoder_state_total_m = cost_stats["state_total_m"]

    config.encoder_state_memory_mb = cost_stats["state_memory_mb"]

    print_model_cost_stats(
        model_name=config.weights_encoder,
        input_size=config.input_size_str,
        cost_stats=cost_stats
    )

    # GFLOPs por imagen
    target_size_for_flops = dict_models[config.weights_encoder]["size"] if preprocess_mode else None

    config.encoder_gflops, config.encoder_flops = estimate_encoder_gflops(
        encoder=encoder,
        processor=processor,
        image_ref=list_filenames_train[0],
        device=config.device,
        preprocess_mode=preprocess_mode,
        target_size=(target_size_for_flops or 0)
    )

    #encoder = AutoModel.from_pretrained(config.weights_encoder).to(config.device)
    '''
    try:
        encoder_vision = encoder.vision_model
        encoder = encoder_vision
    except:
        pass
    '''

    #utilIO.checkNotRepeatedFiles(list_filenames_train_flatten, list_filenames_test_flatten)

    trainDataGenerator = data_generator.DataGenerator(list_filenames_train, list_labels_train, config, None)
    #images_train, labels_train = trainDataGenerator.getCompleteListData()
    
    svm = SVC()
    knn = KNeighborsClassifier(n_neighbors=config.n_neighbors)
    raf = RandomForestClassifier()

    knnClass = ClassifierSKL (knn)

    
    name_id = config.weights_encoder.replace("/", "_") + "__" + name_dataset.replace("/", "_")
    embeddings_dir = Path(config.embeddings_dir)
    results_dir = Path(config.results_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    str_name_file_embeddings = str(embeddings_dir / f"{name_id}.npz")
    
    if os.path.exists(str_name_file_embeddings):
        print ("Loading embeddings from file " + str_name_file_embeddings)
        data = np.load(str_name_file_embeddings, allow_pickle=True)
        list_embeddings = list(data["arrays"])
        list_labels = list(data["labels"])
        list_transcriptions_train = list(data["transcriptions"])
        print("Embeddings loaded: " + str(len(list_embeddings)))
    else:
        print ("Generating embeddings")
        
        target_size = dict_models[config.weights_encoder]["size"] if preprocess_mode else None

        list_embeddings, list_labels = get_embeddings(
            processor, encoder, knnClass, trainDataGenerator,
            preprocess_mode=preprocess_mode,
            target_size=(target_size or 0)
        )
        
        matriz_embeddings = np.stack(list_embeddings, axis=0)
        np.savez(str_name_file_embeddings, arrays=matriz_embeddings, labels=np.array(list_labels), transcriptions=np.array(list_transcriptions_train))
        print("Embeddings generated: " + str(len(list_embeddings)))
    
    
    #model_builder.fitDecoderWithGenerator(trainDataGenerator)
    #results = model_builder.evaluate(images_test, labels_test)

    
    edit_distance = distances.calculateEditDistance(list_transcriptions_train)
    hist_distance = distances.calculateHistogramDistanceWords(list_transcriptions_train)

    euclidean_distance = distances.calculateEuclideanDistance(list_embeddings)

    # Normalized (0..1):
    # - Edit distance: dist / max(len_i, len_j)
    # Minimal compatibility fallback: use a dedicated function when available; otherwise fall back to normalize_distance_matrix(mode="max").
    if hasattr(distances, "normalize_edit_by_maxlen"):
        edit_distance_norm = distances.normalize_edit_by_maxlen(edit_distance, list_transcriptions_train)
    else:
        edit_distance_norm = distances.normalize_distance_matrix(
            edit_distance, list_transcriptions_train, mode="max"
        )

    # - Histogram L1: dist / (len_i + len_j)
    # Minimal compatibility fallback: use a dedicated function when available; otherwise fall back to normalize_distance_matrix(mode="sum").
    if hasattr(distances, "normalize_hist_by_sumlen"):
        hist_distance_norm = distances.normalize_hist_by_sumlen(hist_distance, list_transcriptions_train)
    else:
        hist_distance_norm = distances.normalize_distance_matrix(
            hist_distance, list_transcriptions_train, mode="sum"
        )

    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

    if config.saveGT:
        # GT baseline con ED crudo (media por k)
        export_accumulated_gt(edit_distance, ks, str(results_dir / "accumulated_GT.csv"), use_mean=True)

    # Choose which metric to accumulate:
    # 1) Raw edit distance (enteros)
    metric_mat = edit_distance

    # 2) "SER" tipo tokens (0..1) => descomenta esta línea y comenta la anterior:
    # metric_mat = edit_distance_norm

    results = []
    N = metric_mat.shape[0]

    for i in range(N):
        # Neighbors according to embeddings (this is what changes per model)
        d = euclidean_distance[i].astype(np.float64).copy()

        idx_sorted = np.argsort(d)      # from nearest to farthest
        idx_sorted = idx_sorted[idx_sorted != i]
        
        idx_near_all = idx_sorted
        idx_far_all  = idx_sorted[::-1]

        row_result = {"row": i}

        for k in ks:
            if k <= len(idx_near_all):
                near_idx = idx_near_all[:k]
                far_idx  = idx_far_all[:k]

                # Store the accumulated mean in the original output format.
                row_result[f"k={k}_near"] = float(metric_mat[i, near_idx].mean())
                row_result[f"k={k}_far"]  = float(metric_mat[i, far_idx].mean())
            else:
                row_result[f"k={k}_near"] = None
                row_result[f"k={k}_far"]  = None

        results.append(row_result)

    df = pd.DataFrame(results)

    output_path = str(results_dir / ("accumulated_" + name_id + ".csv"))
    df.to_csv(output_path, index=False, sep=';', decimal=',')

    # Export with comma as decimal separator
    output_path = str(results_dir / ("accumulated_" + name_id + ".csv"))
    df.to_csv(output_path, index=False, sep=';', decimal=',')

    pass

    # ------------------------------------------------------------
    # Correlations using the full matrix (all i != j)
    # ------------------------------------------------------------
    N = edit_distance.shape[0]
    mask = ~np.eye(N, dtype=bool)  # True for i != j

    ed_raw_vec   = edit_distance[mask].astype(np.float64)
    hist_raw_vec = hist_distance[mask].astype(np.float64)

    ed_norm_vec   = edit_distance_norm[mask].astype(np.float64)
    hist_norm_vec = hist_distance_norm[mask].astype(np.float64)

    euc_vec = euclidean_distance[mask].astype(np.float64)

    # Pearson
    r_ed, p_ed = pearsonr(ed_raw_vec, euc_vec)
    r_hist, p_hist = pearsonr(hist_raw_vec, euc_vec)
    r_ed_norm, p_ed_norm = pearsonr(ed_norm_vec, euc_vec)
    r_hist_norm, p_hist_norm = pearsonr(hist_norm_vec, euc_vec)

    # Spearman
    rho_ed, p_s_ed = spearmanr(ed_raw_vec, euc_vec)
    rho_hist, p_s_hist = spearmanr(hist_raw_vec, euc_vec)
    rho_ed_norm, p_s_ed_norm = spearmanr(ed_norm_vec, euc_vec)
    rho_hist_norm, p_s_hist_norm = spearmanr(hist_norm_vec, euc_vec)
    
    modelName = config.weights_encoder
    
    print(modelName + f": Pearson (EDIT TOKENS RAW): r = {r_ed:.4f}, p = {p_ed:.4e}".replace('.', ','))
    print(modelName + f": Pearson (HIST TOKENS L1 RAW): r = {r_hist:.4f}, p = {p_hist:.4e}".replace('.', ','))
    print(modelName + f": Pearson (EDIT TOKENS NORM): r = {r_ed_norm:.4f}, p = {p_ed_norm:.4e}".replace('.', ','))
    print(modelName + f": Pearson (HIST TOKENS NORM): r = {r_hist_norm:.4f}, p = {p_hist_norm:.4e}".replace('.', ','))

    print(modelName + f": Spearman (EDIT TOKENS RAW): r = {rho_ed:.4f}, p = {p_s_ed:.4e}".replace('.', ','))
    print(modelName + f": Spearman (HIST TOKENS L1 RAW): r = {rho_hist:.4f}, p = {p_s_hist:.4e}".replace('.', ','))
    print(modelName + f": Spearman (EDIT TOKENS NORM): r = {rho_ed_norm:.4f}, p = {p_s_ed_norm:.4e}".replace('.', ','))
    print(modelName + f": Spearman (HIST TOKENS NORM): r = {rho_hist_norm:.4f}, p = {p_s_hist_norm:.4e}".replace('.', ','))


    # ------------------------------------------------------------
    # Excel: cabecera + row (original format)
    # ------------------------------------------------------------
    assert len(list_embeddings[0].shape) == 1
    size_str = "-" if height is None else str(int(height))
    emb_dim = list_embeddings[0].shape[0]

    excel_header = (
        "Modelo;input_size;"
        "correl;p-value;"
        "correl;p-value;"
        "correl;p-value;"
        "correl;p-value;;"
        "correl;p-value;"
        "correl;p-value;"
        "correl;p-value;"
        "correl;p-value;;"
        "embedding_dim;"
        "GFLOPs;"
        "params_total;"
        "buffers_total;"
        "state_total;"
        "params_total_M;"
        "buffers_total_M;"
        "state_total_M;"
        "state_memory_MB"
    )

    gflops_str = "-" if config.encoder_gflops is None else f"{config.encoder_gflops:.4f}"

    params_total_str = str(config.encoder_params_total)
    buffers_total_str = str(config.encoder_buffers_total)
    state_total_str = str(config.encoder_state_total)

    params_total_m_str = f"{config.encoder_params_total_m:.2f}"
    buffers_total_m_str = f"{config.encoder_buffers_total_m:.2f}"
    state_total_m_str = f"{config.encoder_state_total_m:.2f}"

    state_memory_mb_str = f"{config.encoder_state_memory_mb:.2f}"

    excel_row = (
        f"{config.weights_encoder};{config.input_size_str};"
        f"{r_ed:.4f};{p_ed:.4e};"
        f"{r_hist:.4f};{p_hist:.4e};"
        f"{r_ed_norm:.4f};{p_ed_norm:.4e};"
        f"{r_hist_norm:.4f};{p_hist_norm:.4e};;"
        f"{rho_ed:.4f};{p_s_ed:.4e};"
        f"{rho_hist:.4f};{p_s_hist:.4e};"
        f"{rho_ed_norm:.4f};{p_s_ed_norm:.4e};"
        f"{rho_hist_norm:.4f};{p_s_hist_norm:.4e};;"
        f"{emb_dim};"
        f"{gflops_str};"
        f"{params_total_str};"
        f"{buffers_total_str};"
        f"{state_total_str};"
        f"{params_total_m_str};"
        f"{buffers_total_m_str};"
        f"{state_total_m_str};"
        f"{state_memory_mb_str}"
    ).replace(".", ",")

    print("\n=== EXCEL HEADER ===")
    print(excel_header)
    print("=== EXCEL ROW ===")
    print(excel_row)


if __name__ == "__main__":
    run_analysis(menu())