<h2 align="center"> <a href="https://arxiv.org/abs/2410.18956"> Large Spatial Model: End-to-end Unposed Images to Semantic 3D

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2403.20309-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.18956)
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://largespatialmodel.github.io/)

</h5>

<div align="center">
This repository is the official implementation of the Large Spatial Model.
LSM reconstructs explicit radiance fields from two unposed images in real-time, capturing geometry, appearance, and semantics.
</div>
<br>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Updates](#updates)
- [Feature and RGB Rendering](#feature-and-rgb-rendering)
  - [Feature Visualization](#feature-visualization)
  - [RGB Color Rendering](#rgb-color-rendering)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Testing / Evaluation](#testing--evaluation)
  - [Inference](#inference)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Updates

**[2025-10-28]** Updated LSM final checkpoint download instructions. The model is now hosted on Hugging Face: [Journey9ni/large_spatial_model](https://huggingface.co/Journey9ni/large_spatial_model).

**[2025-04-12]** Added test dataset download instructions and testing process description. See [data_process/data.md](data_process/data.md) for details.

**[2025-03-09]** Added ScanNet++ data preprocessing pipeline. For detailed instructions, please refer to [data_process/data.md](data_process/data.md).

**[2025-03-06]** Added ScanNet data preprocessing pipeline improvements. For detailed instructions, please refer to [data_process/data.md](data_process/data.md).

## Feature and RGB Rendering

### Feature Visualization
[https://github.com/user-attachments/assets/1c4f533f-af01-49b0-beeb-f9f8793c92b7](https://github.com/user-attachments/assets/f53990f9-96a4-4a2e-8378-da5b27260235)

### RGB Color Rendering
[https://github.com/user-attachments/assets/e984bc3c-d6f1-4b5c-8850-be00d501c683](https://github.com/user-attachments/assets/d0b09707-c74a-4959-8c58-75771e53995e
)

## Get Started

### Installation
1. **Download repo:**
   ````
   git clone --recurse-submodules https://github.com/NVlabs/LSM.git
   ````
2. **Create and activate conda environment:**
   ````bash
   conda create -n lsm python=3.10
   conda activate lsm
   ````

3. **Install PyTorch and related packages:**
   ````bash
   conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
   conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
   ````

4. **Install other Python dependencies:**
   ````bash
   pip install -r requirements.txt
   pip install flash-attn --no-build-isolation
   ````

5. **Install PointTransformerV3:**
   ````bash
   cd submodules/PointTransformerV3/Pointcept/libs/pointops
   python setup.py install
   cd ../../../../..
   ````

6. **Install 3D Gaussian Splatting modules:**
   ````bash
   pip install submodules/3d_gaussian_splatting/diff-gaussian-rasterization
   pip install submodules/3d_gaussian_splatting/simple-knn
   ````

7. **Install OpenAI CLIP:**
   ````bash
   pip install git+https://github.com/openai/CLIP.git
   ````

8. **Build croco model:**
   ````bash
   cd submodules/dust3r/croco/models/curope
   python setup.py build_ext --inplace
   cd ../../../../..
   ````

9. **Download pre-trained models:**

   The following three model weights need to be downloaded:

   ```bash
   # 1. Create directory for checkpoints
   mkdir -p checkpoints/pretrained_models

   # 2. DUSt3R model weights
   wget -P checkpoints/pretrained_models https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

   # 3. LSEG demo model weights
   gdown 1FTuHY1xPUkM-5gaDtMfgCl3D0gR89WV7 -O checkpoints/pretrained_models/demo_e200.ckpt

   # 4. LSM final checkpoint
   wget -O checkpoints/pretrained_models/checkpoint-final.pth https://huggingface.co/Journey9ni/large_spatial_model/resolve/main/checkpoint-final.pth
   ```

### Data Preparation
1. **For training**: The model can be trained on ScanNet and ScanNet++ datasets.
   - Both datasets require signing agreements to access
   - Detailed data preparation instructions are available in [data_process/data.md](data_process/data.md)

2. **For testing**: Refer to [data_process/data.md](data_process/data.md) for details on the test dataset.

### Training
After preparing the datasets, you can train the model using the following command:
```bash
bash scripts/train.sh
```

The training results will be saved to `SAVE_DIR`. By default, it is set to `checkpoints/output`.

Optional parameters in `scripts/train.sh`:
```bash
# Directory to save training outputs
--output_dir "checkpoints/output"
```

### Testing / Evaluation
Run the evaluation pipeline with:

```bash
bash scripts/test.sh
```

You can also run `test.py` directly:

```bash
python test.py \
  --pretrained "checkpoints/pretrained_models/checkpoint-final.pth" \
  --test_dataset "TestDataset(split='test', ROOT='data/scannet_test', resolution=(256, 256), seed=777)" \
  --test_criterion "TestLoss()" \
  --batch_size 1 \
  --test_results_dir "outputs/eval"
```

Evaluation outputs are written to `--test_results_dir`:
- `metrics_summary.json`: aggregated avg/med metrics per dataset.
- `{scene_name}/view_{k}_rgb_gt.png`: target RGB ground truth.
- `{scene_name}/view_{k}_rgb_ours.png`: rendered RGB prediction.
- `{scene_name}/view_{k}_sem_gt.png`: dataset semantic ground truth.
- `{scene_name}/view_{k}_sem_lseg.png`: LSeg semantic pseudo-labels.
- `{scene_name}/view_{k}_sem_ours.png`: rendered semantic prediction.

### Inference
1. Data preparation
   - Prepare any two images of indoor scenes (preferably indoor images, as the model is trained on indoor scene datasets).
   - Place your images in a directory of your choice.

   Example directory structure:
   ````bash
   demo_images/
   └── indoor/
       ├── scene1/
       │   ├── image1.jpg
       │   └── image2.jpg
       └── scene2/
           ├── room1.png
           └── room2.png
   ````

2. Commands
   ````bash
   # Reconstruct 3D scene and generate video using two images
   bash scripts/infer.sh
   ````

   Optional parameters in `scripts/infer.sh` (default settings recommended):
   ```bash
   # Path to your input images
   --file_list "demo_images/indoor/scene2/image1.jpg" "demo_images/indoor/scene2/image2.jpg"

   # Output directory for Gaussian points and rendered video
   --output_path "outputs/indoor/scene2"

   # Image resolution for processing
   --resolution "256"
   ```

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [DUSt3R](https://github.com/naver/dust3r)
- [Language-Driven Semantic Segmentation (LSeg)](https://github.com/isl-org/lang-seg)
- [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3)
- [pixelSplat](https://github.com/dcharatan/pixelsplat)
- [Feature 3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs)
- [ScanNet](https://github.com/ScanNet/ScanNet)
- [ScanNet++](https://github.com/scannetpp/scannetpp)

## Citation
If you find our work useful in your research, please consider giving a star :star: and citing the following paper :pencil:.

```bibTeX
@misc{fan2024largespatialmodelendtoend,
      title={Large Spatial Model: End-to-end Unposed Images to Semantic 3D},
      author={Zhiwen Fan and Jian Zhang and Wenyan Cong and Peihao Wang and Renjie Li and Kairun Wen and Shijie Zhou and Achuta Kadambi and Zhangyang Wang and Danfei Xu and Boris Ivanovic and Marco Pavone and Yue Wang},
      year={2024},
      eprint={2410.18956},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.18956},
}
```
