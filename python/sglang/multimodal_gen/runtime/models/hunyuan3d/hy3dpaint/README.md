# Hunyuan3D-Paint 2.1

Hunyuan3D-Paint 2.1 is a high quality PBR texture generation model for 3D meshes, powered by [RomanTex](https://github.com/oakshy/RomanTex) and [MaterialMVP](https://github.com/ZebinHe/MaterialMVP/).


## Quick Inference
You need to manually download the RealESRGAN weight to the `ckpt` folder using the following command:
```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ckpt
```

Given a 3D mesh `mesh.glb` and a reference image `image.png`, you can run inference using the following code. The result will be saved as `textured_mesh.glb`.

```bash
python3 demo.py
```
**Optional arguments in `demo.py`:**

- `max_num_view` : Maximum number of views, adaptively selected by the model (integer between 6 to 12)

- `resolution` : Resolution for generated PBR textures (512 or 768)

**Memory Recommendation:** For `max_num_view=6` and `resolution=512`, we recommend using a GPU with at least **21GB VRAM**. 

## Training

### Data Prepare
We provide a piece of data in `train_examples` for the overfitting training test. The data structure should be organized as follows:

```
train_examples/
├── examples.json
└── 001/
    ├── render_tex/                 # Rendered generated PBR images
    │   ├── 000.png                 # Rendered views (RGB images)
    │   ├── 000_albedo.png          # Albedo maps for each view
    │   ├── 000_mr.png              # Metallic-Roughness maps for each view, R and G channels
    │   ├── 000_normal.png          # Normal maps
    │   ├── 000_normal.png          # Normal maps
    │   ├── 000_pos.png             # Position maps
    │   ├── 000_pos.png             # Position maps
    │   ├── 001.png                 # Additional views...
    │   ├── 001_albedo.png
    │   ├── 001_mr.png
    │   ├── 001_normal.png
    │   ├── 001_pos.png
    │   └── ...                     # More views (002, 003, 004, 005, ...)
    └── render_cond/                # Rendered reference images (at least two light conditions should be rendered to facilitate consistency loss)
        ├── 000_light_AL.png        # Light condition 1 (Area Light)
        ├── 000_light_ENVMAP.png    # Light condition 2 (Environment map)
        ├── 000_light_PL.png        # Light condition 3 (Point lighting)
        ├── 001_light_AL.png        
        ├── 001_light_ENVMAP.png
        ├── 001_light_PL.png
        └── ...                      # More lighting conditions (002-005, ...)
```

Each training example contains:
- **render_tex/**: Multi-view renderings with PBR material properties
  - Main RGB images (`XXX.png`)
  - Albedo maps (`XXX_albedo.png`)
  - Metallic-Roughness maps (`XXX_mr.png`)
  - Normal maps (`XXX_normal.png/jpg`)
  - Position maps (`XXX_pos.png/jpg`)
  - Camera transforms (`transforms.json`)
- **render_cond/**: Lighting condition maps for each view
  - Ambient lighting (`XXX_light_AL.png`)
  - Environment map lighting (`XXX_light_ENVMAP.png`)
  - Point lighting (`XXX_light_PL.png`)

### Launch Training


```bash
python3 train.py --base 'cfgs/hunyuan-paint-pbr.yaml' --name overfit --logdir logs/
```

## BibTeX

If you found Hunyuan3D-Paint 2.1 helpful, please cite our papers:

```bibtex
@article{feng2025romantex,
  title={RomanTex: Decoupling 3D-aware Rotary Positional Embedded Multi-Attention Network for Texture Synthesis},
  author={Feng, Yifei and Yang, Mingxin and Yang, Shuhui and Zhang, Sheng and Yu, Jiaao and Zhao, Zibo and Liu, Yuhong and Jiang, Jie and Guo, Chunchao},
  journal={arXiv preprint arXiv:2503.19011},
  year={2025}
}

@article{he2025materialmvp,
  title={MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion},
  author={He, Zebin and Yang, Mingxin and Yang, Shuhui and Tang, Yixuan and Wang, Tao and Zhang, Kaihao and Chen, Guanying and Liu, Yuhong and Jiang, Jie and Guo, Chunchao and Luo, Wenhan},
  journal={arXiv preprint arXiv:2503.10289},
  year={2025}
}
```
