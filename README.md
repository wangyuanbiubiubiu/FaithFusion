<div align="center">   
  
# FaithFusion: Harmonizing Reconstruction and Generation via Pixel-wise Information Gain
</div>

A pixel-wise Expected Information Gain (EIG)-driven 3DGS-Diffusion fusion framework for faithful and 3D-consistent driving scene synthesis!
</p>

## [Project Page](https://shalfun.github.io/faithfusion/) | [Paper](none)

<div align="center">
  <img src="docs/FaithFusion/demo.gif" alt=""  width="1100" />
</div>

## Abstract 
In controllable driving-scene reconstruction and 3D scene generation, maintaining geometric fidelity while synthesizing visually plausible appearance under large viewpoint shifts is crucial. However, effective fusion of geometry-based 3DGS and appearance-driven diffusion models faces inherent challenges, as the absence of pixel-wise, 3D-consistent editing criteria often leads to over-restoration and geometric drift. To address these issues, we introduce **FaithFusion**, a 3DGS-diffusion fusion framework driven by pixel-wise Expected Information Gain (EIG). EIG acts as a unified policy for coherent spatio-temporal synthesis: it guides diffusion as a spatial prior to refine high-uncertainty regions, while its pixel-level weighting distills the edits back into 3DGS. The resulting plug-and-play system is free from extra prior conditions and structural modifications. Extensive experiments on the Waymo dataset demonstrate that our approach attains SOTA performance across NTA-IoU, NTL-IoU, and FID, maintaining an FID of 107.47 even at 6 meters lane shift. Our code will be released soon.

## 🔥 Update Log
- [2025/11/28] The construction pipeline for the Waymo cross-camera rendering restoration training dataset is now available!
- [2025/11/25] The calculation process for Expected Information Gain (EIG) has been released and is seamlessly integrated into the DriveStudio codebase.
- [2025/11/25] 📢 📢  Repository Initialization.

## TODO
- [x] Release Expected Information Gain (EIG) calculation process
- [x] Release cross-camera rendering dataset construction pipeline
- [ ] Release EIG-based weighted fusion framework
- [ ] Release [**VideoPainter**](https://github.com/TencentARC/VideoPainter)-based training code
- [ ] Release [**Difix3D+**](https://github.com/nv-tlabs/Difix3D)-based training code
- [ ] Release related model weights/checkpoints
- [ ] Release [**WAN2.1**](https://github.com/Wan-Video/Wan2.1)-based training code

## 🔨 Installation

- 3DGS for Driving Scene: please refer to the installation documentation provided by [**DriveStudio**](https://github.com/ziyc/drivestudio/blob/main/README.md).

## 📊 Prepare Data
The Waymo data is processed following the pipeline established by the **DriveStudio** project:

- Waymo: [Data Process Instruction](docs/Waymo.md)

## 📊 Prepare EIGent Restoration Data
The necessary pipeline to prepare the cross-camera training pairs for the EIGent restoration task is integrated into the DriveStudio project:
```shell
bash scripts/trans_camera_demo.sh
```
- This is a multi-GPU parallel execution version. The script first defines the set of scenes to be processed (`scene_list`), specifies the set of cameras used for training in `train_cam_ids`, and renders the results to the set of target cameras specified in `render_cam_ids`.

## 🚀 Running
### Rendering EIGg
```shell
bash scripts/render_EIG_demo.sh
```
- We have only adapted the one camera logic for the Waymo dataset and have disabled pedestrian rendering by default, prioritizing the synthesis of rigid vehicles.
- We provide additional novel view rendering trajectory configurations, specifically lane shift. More details can be found by inspecting the `render.render_novel` section within the configuration YAML files (See `configs/faithfusion/`).

### Training
coming soon
## Citation
If you find this codebase helpful, please kindly cite:
```
coming soon
```
## Acknowledgement 
Many thanks to the following open-source projects:
* [drivestudio](https://github.com/ziyc/drivestudio)
* [FisherRF](https://github.com/JiangWenPL/FisherRF)
* [VideoPainter](https://github.com/TencentARC/VideoPainter)
* [Difix3D+](https://github.com/nv-tlabs/Difix3D)
