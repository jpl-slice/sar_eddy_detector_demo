import os

import torch
import torch.nn as nn

from models.simclr_resnet import get_simclr_resnet

# Extracted helper function from original model.py


class SimCLRLoader:
    @staticmethod
    def load(args) -> tuple:
        print(f"Loading SimCLR model: {args.arch}")
        device = torch.device(args.device)
        num_classes = args.num_classes
        num_channels = args.num_channels
        checkpoint_path = args.pretrain
        if args.arch != "r50_1x_sk0":
            raise NotImplementedError(
                f"Model architecture: {args.arch} is still unsupported"
            )
        resnet, _ = get_simclr_resnet(depth=50, width_multiplier=1, sk_ratio=0)
        resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)

        if num_channels != 3:
            resnet.net[0][0].in_channels = num_channels
            resnet.net[0][0].weight = torch.nn.Parameter(
                resnet.net[0][0].weight[:, :num_channels, :, :]
            )

        if not hasattr(args, "pretrain") or not args.pretrain:
            raise ValueError(
                "Pretrained model path (--pretrain) required for r50_1x_sk0."
            )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained model not found at: {checkpoint_path}")

        load_state_dict_subset(
            resnet, torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        model = resnet.to(device).eval()
        # input size and interpolation mode don't matter here; we just keep it to maintain compatibility with the TIMM loader
        return model, (224, 224), "bilinear"


def load_state_dict_subset(model, state: dict, verbose=True):
    # only keep weights whose shapes match the corresponding layers in the model
    if "state_dict" in state:
        state = state["state_dict"]
    elif "resnet" in state:  # simclrv2 weights split into "resnet" and "head"
        state = state["resnet"]
    elif "head" in state:
        state = state["head"]
    elif "model" in state:
        state = state["model"]
    m_s = model.state_dict()
    if not verbose:
        subset = {
            k: v for k, v in state.items() if k in m_s and m_s[k].shape == v.shape
        }
    else:
        subset = dict()
        print(f"Loading state dict for {model.__class__.__name__}:")
        loaded_keys = 0
        skipped_keys = 0
        mismatched_keys = 0
        not_found_keys = 0
        for k, v in state.items():
            if k in m_s and m_s[k].shape == v.shape:
                subset[k] = v
                print(f"  Loading: {k} (Shape: {tuple(v.shape)})")
                loaded_keys += 1
            elif k not in m_s:
                print(f"  Skipping (Not in model): {k}")
                not_found_keys += 1
                skipped_keys += 1
            elif k in m_s and m_s[k].shape != v.shape:
                print(
                    f"  Shape mismatch: {k} (Model: {m_s[k].shape}, Checkpoint: {v.shape})"
                )
                mismatched_keys += 1
                skipped_keys += 1
            else:
                print(f"  Skipping (Unknown reason): {k}")
                skipped_keys += 1
    print(
        f"  Loaded {loaded_keys}/{len(state)} weights for {model.__class__.__name__} (which has {len(m_s)} parameters)."
    )
    missing_keys, unexpected_keys = model.load_state_dict(subset, strict=False)
    if missing_keys:
        print(f"  Warning: Missing keys in model state_dict: {len(missing_keys)}")
    if unexpected_keys:
        print(
            f"  Warning: Unexpected keys in checkpoint state_dict: {len(unexpected_keys)}"
        )
    return subset
