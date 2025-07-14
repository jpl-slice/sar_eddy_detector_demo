import timm
import torch
import torch.nn as nn


class TimmLoader:
    @staticmethod
    def load(args) -> tuple:
        """Loads a TIMM model as a feature extractor."""
        model_name = args.arch
        device = torch.device(args.device)
        num_channels = args.num_channels  # Configured number of channels
        print(f"Loading TIMM model: {model_name}")

        try:
            feature_extractor = timm.create_model(
                model_name, pretrained=True, num_classes=0
            )

            # Get model config *before* moving to device (sometimes needed)
            config = feature_extractor.default_cfg
            input_size = (config["input_size"][1], config["input_size"][2])  # (H, W)
            interpolation_mode = config["interpolation"]
            timm_in_channels = config["input_size"][0]  # Channels TIMM expects

            print(
                f"  Model details: Input size={input_size}, Interpolation='{interpolation_mode}', Expects Channels={timm_in_channels}"
            )

            # --- Optional: Adapt input layer if channel mismatch ---
            # This is complex and often better handled by adapting data loading (repeating channels)
            if timm_in_channels != num_channels:
                print(
                    f"Warning: TIMM model expects {timm_in_channels} channels, config specifies {num_channels}. Data loading should handle channel adaptation."
                )
                # Example adaptation (if you *really* need to modify the model):
                # first_conv_layer_name = 'conv_stem' # Or model specific first layer name
                # if hasattr(feature_extractor, first_conv_layer_name):
                #      original_layer = getattr(feature_extractor, first_conv_layer_name)
                #      if isinstance(original_layer, nn.Conv2d):
                #           print(f"Attempting to adapt input layer '{first_conv_layer_name}' for {num_channels} channels.")
                #           # Create new layer, copy weights carefully
                #           new_layer = nn.Conv2d(num_channels, original_layer.out_channels, ...) # Match params
                #           # Weight copying logic needed here...
                #           setattr(feature_extractor, first_conv_layer_name, new_layer)
                # else:
                #      print(f"Warning: Could not find standard input conv layer to adapt channels.")

            # try:
            #     feature_extractor = torch.compile(feature_extractor)
            # except Exception as e:
            #     print(f"Tried to compile model but failed: {e}")
            model = feature_extractor.to(device)
            model.eval()

            return model, input_size, interpolation_mode

        except Exception as e:
            raise RuntimeError(f"Could not load TIMM model '{model_name}'. Error: {e}")
