# Pohang Quality-Aware SingleFusion

## Task

MQLA-BEV targets ship 3D detection under camera and LiDAR degradation. It is evaluated on a multiview subset of the Pohang Canal Dataset; vessel 3D boxes were annotated in LiDAR coordinates with SUSTechPOINTS and verified in the front, left, and right camera views.

The robustness protocol covers nine individual corruption types and five combined conditions, including rain and fog, illumination changes, camera-view loss, LiDAR point or beam dropout, scan-motion blur, and calibration shifts. Online corruption sampling during training uses a user-configured record count.

The model maps camera and LiDAR features to a shared BEV space, dynamically weights them according to local observation quality, and uses LiDAR-guided local attention to aggregate nearby valid camera features before ship 3D box prediction.

## Model structure

- Camera branch: per-view quality scaling and calibration-based DepthLSS projection.
- LiDAR branch: PointPillars feature extraction.
- Fusion and detection: spatial quality weighting, LiDAR-query attention over a 5 by 5 camera BEV neighborhood, one BEV fusion stage, and a CenterPoint detection head.
- The model does not include a learned offset predictor or a separate LiDAR safety detector.

## Files

- `singlefusion_attention.py`: model and training configuration.
- `singlefusion.py`: detector, quality-aware fusion module, dataset adapter, and hooks.
- `corruptions.py`: online corruption transforms and virtual-sample dataset.
- `weather_assets/`: rain/fog source code and required numerical assets.
- `requirements.txt`: Python and framework dependency constraints for this project.

## Environment and installation

Use Linux, Python 3.10, one NVIDIA RTX 4090, and a CUDA-enabled PyTorch build matching your CUDA toolkit. Install PyTorch and its matching torchvision build first. From an MMDetection3D 1.4.0 checkout, place `singlefusion.py`, `corruptions.py`, and `weather_assets/` beside the BEVFusion project package. Keep the configuration at a location you choose.

Set `PROJECT_ROOT` to this project directory, then install the dependencies from the MMDetection3D root:

```bash
export PROJECT_ROOT=/path/to/singlefusion-project
pip install -U openmim
mim install -r requirements/mminstall.txt
pip install -r requirements/runtime.txt
pip install -r "$PROJECT_ROOT/requirements.txt"
TORCH_CUDA_ARCH_LIST=8.9 pip install -v -e .
```

## Data and training

Set `train_ann_file` and `clean_count` in the configuration. `clean_count` is the number of source records and is user-defined. The train, validation, and test annotation fields are blank; validation and test loops are disabled in this recipe.

```bash
export POHANG_DATA_ROOT=/path/to/pohang
export CONFIG_FILE=/path/to/singlefusion_attention.py
export WORK_DIR=/path/to/work_dir
python tools/train.py "$CONFIG_FILE" --work-dir "$WORK_DIR"
```

The per-GPU batch size is 2. Install `PyMieScatt` through `requirements.txt` for the bundled rain model. No dataset, annotation PKL, or detector checkpoint is included.

The dataset will be made publicly available upon acceptance of the associated paper.
