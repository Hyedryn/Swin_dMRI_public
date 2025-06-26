import torch
import torch.nn as nn
from torchvision.models import swin_t
from peft import LoraConfig, get_peft_model

class SwinTST(nn.Module):
    def __init__(self, num_classes, size, freeze_mode="allButStage4", lora_r=4, lora_alpha=1.0, lora_dropout=0.15):
        super(SwinTST, self).__init__()

        # SwinTransformer
        if size == "swint":
            self.swin = swin_t(weights='IMAGENET1K_V1')
        else:
            raise Exception("unknown size")

        # Freeze the backbone if required
        if freeze_mode in ["allButStage4", "all", "LoRA"] or "LoRA" in freeze_mode:
            for param in self.swin.parameters():
                param.requires_grad = False

        # Unfreeze the last stage of the Swin Transformer
        if freeze_mode in ["allButStage4"]:
            last_stage = self.swin.features[-1]  # The last sequential block
            for param in last_stage.parameters():
                param.requires_grad = True

        # Modify the classification head
        num_features = self.swin.head.in_features
        print("swin num_features", num_features)
        self.swin.head = nn.Linear(num_features, num_classes)

        if "LoRA" in freeze_mode and "+" in freeze_mode:
            lora_params = freeze_mode.split("+")
            lora_r = int(lora_params[1].replace("r", ""))
            lora_alpha = float(lora_params[2].replace("a", ""))
            lora_dropout = float(lora_params[3].replace("d", ""))
            if lora_params[4] == "attn":
                lora_modules = ["attn.qkv"]
            elif lora_params[4] == "attnmlp":
                lora_modules = ["attn.qkv", "mlp.fc1"]
            else:
                raise Exception("Unknown module")

        if freeze_mode in ["LoRA"] or "LoRA" in freeze_mode:
            # Create the LoRA configuration
            print("Lora R: ", lora_r, " Lora Alpha: ", lora_alpha, " Lora dropout: ", lora_dropout, " Lora modules: ",
                  lora_modules)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=lora_modules
            )
            # Apply LoRA to all attention layers within the Swin Transformer
            self.swin = get_peft_model(self.swin, lora_config)

        # Ensure the classification head's parameters are trainable
        for param in self.swin.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.swin(x)
