import torch
import torch.nn as nn
from einops import rearrange
from large_spatial_model.configs import LSMConfig

from large_spatial_model.ptv3 import PTV3
from large_spatial_model.dust3r_with_feature import Dust3RWithFeature
from large_spatial_model.gaussian_head import GaussianHead
from large_spatial_model.lseg import LSegFeatureExtractor
from large_spatial_model.utils.points_process import merge_points

class LSM_Dust3R(nn.Module):
    def __init__(self, config: LSMConfig):
        super().__init__()
        self.config = LSMConfig(**config)
        
        # Initialize components with typed configs
        self.dust3r = Dust3RWithFeature(**self.config.dust3r_config)
        self.point_transformer = PTV3(**self.config.point_transformer_config)
        self.gaussian_head = GaussianHead(**self.config.gaussian_head_config)
        self.lseg_feature_extractor = LSegFeatureExtractor.from_pretrained(**self.config.lseg_config)
        self.tokenizer = nn.AvgPool2d(kernel_size=8, stride=8)  # pool each 8*8 patch into a single value
        self.feature_expansion = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(64, 512, kernel_size=1, stride=1)
        ) # (b, 64, h, w) -> (b, 512, h//2, w//2)
        self.feature_reduction = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        ) # (b, 512, h//2, w//2) -> (b, d_features, h, w)
        # freeze parameters and set to eval mode
        if self.config.freeze_dust3r:
            print("Freezing dust3r")
            self.dust3r.eval()
            for param in self.dust3r.parameters():
                param.requires_grad = False
        if self.config.freeze_lseg:
            print("Freezing lseg")
            self.lseg_feature_extractor.eval()
            for param in self.lseg_feature_extractor.parameters():
                param.requires_grad = False
        
    def forward(self, view1, view2):
        # Dust3R forward pass
        dust3r_output, dust3r_feature = self.dust3r(view1, view2)

        # LSeg forward pass
        lseg_token_feature, lseg_res_feature = self.extract_lseg_features(view1, view2)
        multi_scale_feature = lseg_token_feature.clone()
        # merge points from two views
        data_dict = merge_points(dust3r_output, view1, view2)
        
        # PointTransformerV3 forward pass
        point_transformer_output = self.point_transformer(data_dict, dust3r_feature, lseg_token_feature, multi_scale_feature)
        
        # Gaussian head forward pass
        final_output = self.gaussian_head(point_transformer_output, lseg_res_feature)
        final_output[0].update(**dust3r_output[0])
        final_output[1].update(**dust3r_output[1])
        return final_output
    
    def extract_lseg_features(self, view1, view2):
        # concat view1 and view2
        img = torch.cat([view1['img'], view2['img']], dim=0) # (v*b, 3, h, w)
        # extract features
        lseg_features = self.lseg_feature_extractor.extract_features(img) # (v*b, 512, h//2, w//2)
        # average pooling
        lseg_token_feature = self.tokenizer(lseg_features)
        # reshape to (b, 2v, d)
        lseg_token_feature = rearrange(lseg_token_feature, '(v b) c h w -> b (v h w) c', v=2)
        # feature reduction
        lseg_res_feature = self.feature_reduction(lseg_features)
        return lseg_token_feature, lseg_res_feature

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, use_pretrained_lseg: bool = True, use_pretrained_dust3r: bool = True, device: str = 'cuda'):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False) # load checkpoint to cpu for saving memory
        args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
        print(f"instantiating {args}")
        model = eval(args)
        state_dict = ckpt['model']
        # if use_pretrained_lseg, remove lseg related keys
        if use_pretrained_lseg:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('lseg_feature_extractor')}
        # if use_pretrained_dust3r, remove dust3r related keys
        if use_pretrained_dust3r:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dust3r')}
        model.load_state_dict(state_dict, strict=False)
        del ckpt
        return model.to(device)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    from torch.utils.data import DataLoader
    from large_spatial_model.utils.path_manager import init_all_submodules
    init_all_submodules()

    # Add configuration file related parameters
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--overrides', type=str, nargs='+', default=[],
                       help='Override parameters in config file, format: key=value')
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config)
    
    # Process command line override parameters
    if args.overrides:
        overrides = OmegaConf.from_dotlist(args.overrides)
        config = OmegaConf.merge(config, overrides)
    
    # Convert to LSMConfig type
    config = LSMConfig(**config)
    
    # Initialize model
    model = LSM_Dust3R(config)
    model.to('cuda')
    # Load Data
    from large_spatial_model.datasets.scannet import Scannet
    dataset = Scannet(split='train', ROOT="data/scannet_processed", resolution=256)
    # Print dataset
    print(dataset)
    # Test model
    data_loader = DataLoader(dataset, batch_size=3, shuffle=True)
    data = next(iter(data_loader))
    # move data to cuda
    for view in data:
        view['img'] = view['img'].to('cuda')
        view['depthmap'] = view['depthmap'].to('cuda')
        view['camera_pose'] = view['camera_pose'].to('cuda')
        view['camera_intrinsics'] = view['camera_intrinsics'].to('cuda')
    # Forward pass
    output = model(*data[:2])
    
    # Loss
    from large_spatial_model.loss import GaussianLoss
    loss = GaussianLoss()
    loss_value = loss(*data, *output, model)
    print(loss_value)
