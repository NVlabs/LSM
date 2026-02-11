import torch
import torch.nn as nn
from submodules.lang_seg.modules.models.lseg_net import LSegNet, clip

class LSegFeatureExtractor(LSegNet):
    def __init__(self, half_res=True):
        super().__init__(
            labels='', 
            backbone='clip_vitl16_384', 
            features=256, 
            crop_size=224, 
            arch_option=0, 
            block_depth=0, 
            activation='lrelu'
        )

        self.half_res = half_res

    @torch.no_grad()
    def extract_features(self, x):
        layer_1, layer_2, layer_3, layer_4 = forward_layers(self.pretrained, x)
        # layer:(b, 1024, h//16, w//16)
        # image_features = torch.cat([layer_1, layer_2, layer_3, layer_4], dim=1)
        # # image_features:(b, 4096, h//16, w//16)
        
        # dense feature
        # DPT head
        pretrained = self.pretrained
        layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
        layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
        layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
        layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)
        
        # refinenet
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        # (b, 512, h//2, w//2)
        image_features = self.scratch.head1(path_1)
        if self.half_res:
            return image_features
        
        # (b, 512, h, w)
        image_features = self.scratch.output_conv(image_features)

        return image_features
    
    @torch.no_grad()
    def decode_feature(self, image_features, labelset=''):
        # # image_features:(b, 4096, h//16, w//16)
        # # split image_features into 4 parts
        # layer_1, layer_2, layer_3, layer_4 = torch.split(image_features, 1024, dim=1)
        
        # # DPT head
        # pretrained = self.pretrained
        # layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
        # layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
        # layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
        # layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)
        
        # # refinenet
        # layer_1_rn = self.scratch.layer1_rn(layer_1)
        # layer_2_rn = self.scratch.layer2_rn(layer_2)
        # layer_3_rn = self.scratch.layer3_rn(layer_3)
        # layer_4_rn = self.scratch.layer4_rn(layer_4)

        # path_4 = self.scratch.refinenet4(layer_4_rn)
        # path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        # path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        # path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        # image_features = self.scratch.head1(path_1)
        imshape = image_features.shape
        
        # encode text
        if labelset == '':
            text = self.text
        else:
            text = clip.tokenize(labelset)
        
        self.logit_scale = self.logit_scale.to(image_features.device)
        text = text.to(image_features.device)
        text_features = self.clip_pretrained.encode_text(text)
        image_features = image_features.permute(0,2,3,1).reshape(-1, self.out_c)
        
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits_per_image = self.logit_scale * image_features.half() @ text_features.t()
        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)

        if self.arch_option in [1, 2]:
            for _ in range(self.block_depth - 1):
                out = self.scratch.head_block(out)
            out = self.scratch.head_block(out, False)

        if self.half_res:
            out = self.scratch.output_conv(out)
            
        return out

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        print(f"Loading checkpoint from: {pretrained_model_name_or_path}")
        ckpt = torch.load(pretrained_model_name_or_path, map_location='cpu', weights_only=False)
        print(f"Checkpoint loaded. Keys in checkpoint: {ckpt.keys()}")
        
        print("Processing state dict...")
        new_state_dict = {k[len("net."):]: v for k, v in ckpt['state_dict'].items() if k.startswith("net.")}
        print(f"Processed state dict. Number of keys: {len(new_state_dict)}")
        
        print("Initializing model...")
        model = cls(*args, **kwargs)
        
        print("Loading state dict into model...")
        model.load_state_dict(new_state_dict, strict=True)
        print("State dict loaded successfully.")
        
        print("Cleaning up...")
        del ckpt
        del new_state_dict
        
        print("Model loading complete.")
        return model

def forward_layers(pretrained, x):
    b, c, h, w = x.shape
    
    # encoder
    glob = pretrained.model.forward_flex(x)
    
    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    unflatten = nn.Sequential(
        nn.Unflatten(
            2,
            torch.Size(
                [
                    h // pretrained.model.patch_size[1],
                    w // pretrained.model.patch_size[0],
                ]
            ),
        )
    )
    
    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)
    
    return layer_1, layer_2, layer_3, layer_4
