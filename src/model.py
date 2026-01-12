import timm 
import torch

def get_model(num_classes, pretrained=True, model_name="mobilevit_s"):
    """
    Function to create MobileViT-S model.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model

def load_weights(model, weight_path, device="cpu"):
    """
    Safely loads weights handling both state_dict and full ckpts dicts.
    """
    checkpoint = torch.load(weight_path, map_location=device)
    
    # handle case where ckpts is a dict containing 'model_state'
    if "model_state" in checkpoint:
        state_dict= checkpoint["model_state"]
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=True)
    return model