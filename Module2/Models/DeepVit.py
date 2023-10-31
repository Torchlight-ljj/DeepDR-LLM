from functools import partial
import torch
import torch.nn as nn
from Models.vision_transformer import VisionTransformer as VIT
from Models.pos_embed import interpolate_pos_embed

def vit_base_patch16(**kwargs):
    model = VIT(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

if __name__ == "__main__":
    model = vit_base_patch16(img_size=448, weight_init="nlhb",  cla_num_classes=2)
    checkpoint_model = torch.load('/home3/ljj/Retrospective/ViT-DME/vit_base.pth')['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    checkpoint_model = interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=False)
    img = torch.randn(1, 3, 448, 448).cpu()
    results= model(img)
    if isinstance(results,list):
        print(results[-1].shape)
    else:
        print(results.shape)
