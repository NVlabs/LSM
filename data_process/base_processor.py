from typing import List, Dict
import torch
from abc import abstractmethod
import numpy as np
import os
import json
from torchvision.utils import save_image
from concurrent.futures import ProcessPoolExecutor
import tqdm
import multiprocessing as mp
import random
import cv2

class BaseSceneProcessorConfig:
    def __init__(self, root_dir: str, save_dir: str, device: torch.device, num_workers: int = 16):
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.device = device
        self.load_frame_parallel = False
        self.target_height = 480
        self.target_width = 640
        self.num_workers = num_workers

        # create dirs
        os.makedirs(self.save_dir, exist_ok=True)

class BaseSceneProcessor:
    def __init__(self, config: BaseSceneProcessorConfig):
        # Set multiprocessing start method to 'spawn'
        mp.set_start_method('spawn', force=True)
        # set random seed
        random.seed(0)
        np.random.seed(0)
        self.config = config
        self.scene_paths = self.get_all_scene_paths()
    
    def get_scene_image_save_path(self, scene_name: str) -> str:
        """
        Get scene image save path
        """
        return os.path.join(self.config.image_save_dir, scene_name)
    
    def get_image_save_path(self, scene_name: str, frame_idx: int) -> str:
        """
        Get image save path
        """
        return os.path.join(self.get_scene_image_save_path(scene_name), f'{frame_idx:06d}.png')
    
    def get_all_scene_paths(self) -> List[str]:
        """
        Get a list of scene names from the root directory
        """
        pass
    
    def load_and_process_scene(self, scene_path: str) -> dict:
        """
        Load and process scene data from scene_path
        Args:
            scene_path: str, path to the scene
        Returns:
            scene_data: dict, including processed frames, instance_id_to_class_map
        """
        # 1. Get frame paths and instance mapping
        frame_paths = self.get_all_frame_paths(scene_path)
        intrinsics = self.get_intrinsics(scene_path)
        
        # 2. Load all frames
        frames_data = {}
        for frame_idx, frame_path in frame_paths.items():
            frames_data[frame_idx] = self.load_single_frame(frame_path)

        # 3. Stack and process frames
        depth_data = torch.stack([frame_data['depth_data'] for frame_data in frames_data.values()], dim=0)
        color_data = torch.stack([frame_data['color_data'] for frame_data in frames_data.values()], dim=0)
        pose_data = torch.stack([frame_data['pose_data'] for frame_data in frames_data.values()], dim=0)

        # Filter out invalid frames
        valid_mask = self._get_valid_frame_mask(scene_path, depth_data, color_data, pose_data)
        if not valid_mask.any():
            print(f"No valid frames found in scene {scene_path}")
            return None

        depth_data = depth_data[valid_mask]
        color_data = color_data[valid_mask]
        pose_data = pose_data[valid_mask]

        # Validate intrinsics separately
        if not self._validate_intrinsics(scene_path, intrinsics):
            return None

        # 4. Process image size and resize
        batch_size, original_h, original_w = depth_data.shape
        h_ratio = self.config.target_height / original_h
        w_ratio = self.config.target_width / original_w
        ratio = max(h_ratio, w_ratio)
        new_h, new_w = int(original_h * ratio), int(original_w * ratio)
        
        # Calculate adjusted intrinsics first
        new_intrinsic = torch.clone(intrinsics)
        new_intrinsic[0, 0] *= ratio
        new_intrinsic[1, 1] *= ratio
        start_x = (new_w - self.config.target_width) // 2
        start_y = (new_h - self.config.target_height) // 2
        new_intrinsic[0, 2] = (intrinsics[0, 2] * ratio) - start_x
        new_intrinsic[1, 2] = (intrinsics[1, 2] * ratio) - start_y
        
        # Resize images
        depth_data = torch.nn.functional.interpolate(depth_data.unsqueeze(1), size=(new_h, new_w), mode='nearest')
        color_data = torch.nn.functional.interpolate(color_data.permute(0, 3, 1, 2), size=(new_h, new_w), mode='bilinear')
        
        # Crop images
        depth_data = depth_data[:, :, start_y:start_y + self.config.target_height, start_x:start_x + self.config.target_width]
        color_data = color_data[:, :, start_y:start_y + self.config.target_height, start_x:start_x + self.config.target_width]
        
        torch.cuda.empty_cache()
        
        return {
            "frames": {
                'color_data': color_data,
                'depth_data': depth_data,
                'pose_data': pose_data,
            },
            "intrinsics": new_intrinsic,
            "frame_num": batch_size,
            "scene_name": scene_path.split('/')[-1]
        }
    
    @abstractmethod
    def get_intrinsics(self, scene_path: str) -> dict:
        """
        Get intrinsics from scene_path
        """
        pass
    
    @abstractmethod
    def get_all_frame_paths(self, scene_path: str) -> Dict[str, List[str]]:
        """
        Get all frame paths from scene_path
        input:
            scene_path: str, path to the scene
        return:
            Dict: {
                "frame_idx": [depth_path, color_path, pose_path]
            }
        """
        pass
    
    @abstractmethod
    def load_single_frame(self, frame_path: str) -> dict:
        """
        Load frame data from frame_path
        Args:
            frame_path: str, path to the frame
        Returns:
            dict: {
                "frame_idx": [depth_data, color_data, pose_data]
            }
        """
        pass

    def process_single_scene(self, scene_path: str) -> dict:
        # Load and process scene data
        scene_data = self.load_and_process_scene(scene_path)
        if scene_data is None:
            return False
        # save scene data
        scene_name = scene_data['scene_name']
        scene_save_path = os.path.join(self.config.save_dir, scene_name)
        scene_color_save_path = os.path.join(scene_save_path, 'color')
        scene_depth_save_path = os.path.join(scene_save_path, 'depth')
        scene_pose_save_path = os.path.join(scene_save_path, 'pose')
        os.makedirs(scene_color_save_path, exist_ok=True)
        os.makedirs(scene_depth_save_path, exist_ok=True)
        os.makedirs(scene_pose_save_path, exist_ok=True)
        # save color data
        for frame_idx, color_data in enumerate(scene_data['frames']['color_data']):
            color_save_path = os.path.join(scene_color_save_path, f'{frame_idx:06d}.png')
            # Convert to uint8 range [0, 255] if needed
            color_data = color_data.cpu().numpy()
            color_data = color_data.transpose(1, 2, 0).astype(np.uint8)
            # Save as JPG for color images
            cv2.imwrite(color_save_path, color_data)

        # save depth data
        for frame_idx, depth_data in enumerate(scene_data['frames']['depth_data']):
            depth_save_path = os.path.join(scene_depth_save_path, f'{frame_idx:06d}.png')
            # ScanNet depth is in millimeters, stored as 16-bit PNG
            # Ensure depth is in uint16 format with proper scaling
            depth_data = depth_data.cpu().numpy()
            depth_data = depth_data.transpose(1, 2, 0).astype(np.uint16)
            # Save as 16-bit PNG for depth images
            cv2.imwrite(depth_save_path, depth_data)
        # save meta data
        for frame_idx, pose_data in enumerate(scene_data['frames']['pose_data']):
            pose_save_path = os.path.join(scene_pose_save_path, f'{frame_idx:06d}.npz')
            meta_data = {
                'camera_intrinsics': scene_data['intrinsics'].cpu().numpy(),
                'camera_pose': pose_data.cpu().numpy()
            }
            np.savez(pose_save_path, **meta_data)

        return True
    
    def _process_scene_with_gpu(self, scene_path: str, gpu_id: int) -> dict:
        """
        Process a single scene with specified GPU
        Args:
            scene_path: str, path to the scene
            gpu_id: int, GPU device ID
        Returns:
            dict: processed conversations
        """
        # Set CUDA device for this process
        torch.cuda.set_device(gpu_id)
        try:
            result = self.process_single_scene(scene_path)
            # Clean up GPU memory after processing
            torch.cuda.empty_cache()
            return result
        except Exception as e:
            print(f"Error processing scene {scene_path} on GPU {gpu_id}: {e}")
            # Clean up GPU memory in case of error
            torch.cuda.empty_cache()
            return None

    def process_all_scenes_serial(self):
        """
        Process all scenes in serial
        """
        for scene_path in tqdm.tqdm(self.scene_paths):
            self.process_single_scene(scene_path)
        
    def process_all_scenes_parallel(self):
        """
        Process all scenes in parallel with tasks distributed across available GPUs
        """
        # Filter out processed scenes
        unprocessed_scenes = []
        for scene_path in self.scene_paths:
            scene_name = os.path.basename(scene_path)
            scene_save_path = os.path.join(self.config.save_dir, scene_name)
            
            if os.path.exists(scene_save_path):
                if self.is_scene_processed(scene_save_path):
                    print(f"Skipping already processed scene: {scene_name}")
                    continue
            unprocessed_scenes.append(scene_path)

        if not unprocessed_scenes:
            print("All scenes have been processed!")
            return
        
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPU available for processing")
        
        # Calculate workers per GPU
        workers_per_gpu = max(1, self.config.num_workers // num_gpus)
        total_workers = workers_per_gpu * num_gpus
        
        print(f"Processing {len(unprocessed_scenes)} scenes with {total_workers} workers across {num_gpus} GPUs "
              f"({workers_per_gpu} workers per GPU)...")
        
        # Distribute scenes across GPUs
        with ProcessPoolExecutor(max_workers=total_workers) as executor:
            # Create tasks with GPU assignments
            futures = []
            for i, scene_path in enumerate(unprocessed_scenes):
                gpu_id = i % num_gpus  # Distribute scenes across GPUs in round-robin fashion
                futures.append(
                    executor.submit(self._process_scene_with_gpu, scene_path, gpu_id)
                )
            
            # Process results with progress bar
            for future in tqdm.tqdm(futures, total=len(unprocessed_scenes)):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in processing scene: {e}")
                    continue

    def is_scene_processed(self, scene_save_path):
        """Check if a scene has been fully processed"""
        # Check required directories and their contents
        required_dirs = {
            'color': '.png',
            'depth': '.png',
            'pose': '.npz'
        }
        
        for dir_name, file_ext in required_dirs.items():
            dir_path = os.path.join(scene_save_path, dir_name)
            # Check if directory exists
            if not os.path.isdir(dir_path):
                return False
            
            # Check if directory contains files with correct extension
            files = [f for f in os.listdir(dir_path) if f.endswith(file_ext)]
            if not files:
                return False
            
            # Check if all frame indices are continuous
            frame_indices = sorted([int(os.path.splitext(f)[0]) for f in files])
            if not frame_indices:
                return False
            
            # Check if the number of files matches across directories
            if dir_name == 'color':
                expected_count = len(files)
            elif len(files) != expected_count:
                return False

        return True

    def _get_valid_frame_mask(self, scene_path: str, depth_data: torch.Tensor, 
                            color_data: torch.Tensor, pose_data: torch.Tensor) -> torch.Tensor:
        """
        Get mask of valid frames
        Args:
            scene_path: path to the scene for error reporting
            depth_data: depth tensor data
            color_data: color tensor data
            pose_data: pose tensor data
        Returns:
            torch.Tensor: boolean mask indicating valid frames
        """
        frame_count = len(depth_data)
        valid_mask = torch.ones(frame_count, dtype=torch.bool)

        # Check each frame
        for i in range(frame_count):
            frame_valid = True
            tensors_to_check = {
                'depth': depth_data[i],
                'color': color_data[i],
                'pose': pose_data[i]
            }

            for name, tensor in tensors_to_check.items():
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"Invalid {name} data found in frame {i} of scene {scene_path}")
                    frame_valid = False
                    break

            valid_mask[i] = frame_valid

        return valid_mask

    def _validate_intrinsics(self, scene_path: str, intrinsics: torch.Tensor) -> bool:
        """
        Validate intrinsics data
        Args:
            scene_path: path to the scene for error reporting
            intrinsics: camera intrinsics tensor
        Returns:
            bool: True if intrinsics are valid, False otherwise
        """
        if torch.isnan(intrinsics).any() or torch.isinf(intrinsics).any():
            print(f"Invalid intrinsics found in scene {scene_path}")
            return False
        return True
