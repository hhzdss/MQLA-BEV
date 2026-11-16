from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.registry import DATASETS, TRANSFORMS
from projects.BEVFusion.singlefusion import PohangDatasetKeepEmpty


RAIN_RATES_MM_H = {1: 1.0, 2: 10.0, 3: 50.0}
FOG_LEVELS = {
    1: dict(mor_m=500.0, alpha=0.007824, beta=0.000184),
    2: dict(mor_m=350.0, alpha=0.01117714, beta=0.00026286),
    3: dict(mor_m=250.0, alpha=0.015648, beta=0.000368),
}
POINT_KEEP_RATIOS = {1: 0.80, 2: 0.60, 3: 0.40}
BEAM_KEEP_RATIOS = {1: 0.75, 2: 0.50, 3: 0.25}
MOTION_STD_M = {1: 0.02, 2: 0.05, 3: 0.10}
SPATIAL_LIMITS = {
    1: (0.10, 0.02),
    2: (0.30, 0.05),
    3: (0.50, 0.10),
}
CAMERA_DROPOUT_PROBABILITIES = {1: 0.20, 2: 0.40, 3: 0.60}
IMAGE_BLUR_SIGMAS = {1: 0.06, 2: 0.10, 3: 0.13}
BRIGHTNESS_OFFSETS = {1: 0.50, 2: 0.60, 3: 0.70}
DARKNESS_CAPS = {1: 0.60, 2: 0.40, 3: 0.30}
DARKNESS_GAUSSIAN_STD = {1: 0.008, 2: 0.018, 3: 0.026}

_ALIASES = {
    'rain': 'paired_rain',
    'fog': 'paired_fog',
    'points_reducing': 'point_drop',
    'beams_reducing': 'beam_drop',
    'missing_camera': 'camera_dropout',
    'spatial_misalignment': 'spatial_misalignment',
    'temporal_misalignment': 'temporal_misalignment',
}
_SUPPORTED = frozenset({
    'paired_rain', 'paired_fog', 'point_drop', 'beam_drop',
    'motion_blur', 'brightness', 'darkness', 'camera_dropout',
    'temporal_misalignment', 'spatial_misalignment'
})


def _stable_seed(base_seed: int, sample_id: Any, corruption: str,
                 severity: int) -> int:
    """Return a process-independent seed for a fixed evaluation condition."""
    message = f'{base_seed}|{sample_id}|{corruption}|{severity}'.encode('utf-8')
    return int.from_bytes(hashlib.blake2b(message, digest_size=8).digest(),
                          byteorder='little') % (2**32 - 1)


def _rotation_xyz(angles_rad: np.ndarray) -> np.ndarray:
    rx, ry, rz = angles_rad
    cx, cy, cz = np.cos(angles_rad)
    sx, sy, sz = np.sin(angles_rad)
    rotation_x = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])
    rotation_y = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
    rotation_z = np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
    return rotation_z @ rotation_y @ rotation_x


def _as_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def _image_rain(image: np.ndarray, severity: int,
                rng: np.random.Generator) -> np.ndarray:
    """Camera rain paired with the LISA 1/10/50 mm/h severity labels."""
    profiles = {
        1: (120, 0.016, 1, 0.12, 0.99, 3, (-8., 8.)),
        2: (450, 0.026, 1, 0.23, 0.95, 3, (-10., 10.)),
        3: (1000, 0.038, 1, 0.32, 0.89, 5, (-12., 12.)),
    }
    density, length_ratio, width, opacity, brightness, blur_size, angles = \
        profiles[severity]
    image = _as_uint8(image)
    height, image_width = image.shape[:2]
    streak_count = max(1, int(round(density * height * image_width / 1_000_000)))
    base_length = max(5, int(round(height * length_ratio)))
    layer = np.zeros((height, image_width, 3), dtype=np.uint8)
    for _ in range(streak_count):
        length = max(3, int(round(base_length * rng.uniform(0.65, 1.35))))
        angle = math.radians(rng.uniform(*angles))
        dx = int(round(math.tan(angle) * length))
        x0 = int(rng.integers(-abs(dx), image_width + abs(dx) + 1))
        y0 = int(rng.integers(-length, height + 1))
        shade = int(rng.integers(185, 231))
        cv2.line(layer, (x0, y0), (x0 + dx, y0 + length),
                 (shade, shade, shade), width, lineType=cv2.LINE_AA)
    if blur_size > 1:
        layer = cv2.GaussianBlur(layer, (blur_size, blur_size), 0)
    rainy = image.astype(np.float32) * brightness + layer.astype(np.float32) * opacity
    return _as_uint8(rainy)


def _image_fog(image: np.ndarray, severity: int,
               rng: np.random.Generator) -> np.ndarray:
    density = {1: 0.20, 2: 0.40, 3: 0.70}[severity]
    image = _as_uint8(image).astype(np.float32) / 255.0
    height, width = image.shape[:2]
    vertical = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    horizon_profile = 0.25 + 0.75 * np.exp(
        -0.5 * ((vertical - 0.65) / 0.16) ** 2)
    coarse_h, coarse_w = max(2, height // 64), max(2, width // 64)
    texture = cv2.resize(
        rng.uniform(0.90, 1.10, size=(coarse_h, coarse_w)).astype(np.float32),
        (width, height), interpolation=cv2.INTER_CUBIC)
    fog_strength = np.clip(density * horizon_profile * texture, 0.0, 1.0)
    fog_color = np.array((205.0, 211.0, 215.0), dtype=np.float32) / 255.0
    foggy = image * (1.0 - fog_strength[..., None])
    foggy += fog_color[None, None, :] * fog_strength[..., None]
    return _as_uint8(foggy * 255.0)


def _image_motion_blur(image: np.ndarray, severity: int,
                       rng: np.random.Generator) -> np.ndarray:
    """A 15-pixel-radius linear exposure blur with an angle in [-45, 45]."""
    radius = 15
    sigma = IMAGE_BLUR_SIGMAS[severity]
    size = radius * 2 + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = radius
    angle = math.radians(rng.uniform(-45., 45.))
    dx = int(round(radius * math.cos(angle)))
    dy = int(round(radius * math.sin(angle)))
    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy),
             color=1., thickness=1, lineType=cv2.LINE_AA)
    # MultiCorrupt stores sigma as a normalized strength.  Blur the exposure
    # trace lightly before normalization, retaining its declared 15-pixel radius.
    blur_sigma = max(0.5, sigma * 100.0 / 8.0)
    kernel = cv2.GaussianBlur(kernel, (0, 0), blur_sigma)
    kernel /= max(float(kernel.sum()), np.finfo(np.float32).eps)
    return _as_uint8(cv2.filter2D(_as_uint8(image), -1, kernel,
                                  borderType=cv2.BORDER_REPLICATE))


def _image_brightness(image: np.ndarray, severity: int) -> np.ndarray:
    hsv = cv2.cvtColor(_as_uint8(image), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] / 255. + BRIGHTNESS_OFFSETS[severity], 0., 1.) * 255.
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _image_darkness(image: np.ndarray, severity: int,
                    rng: np.random.Generator) -> np.ndarray:
    image = _as_uint8(image).astype(np.float32) / 255.
    image_min, image_max = float(image.min()), float(image.max())
    if image_max > image_min:
        image = ((image - image_min) / (image_max - image_min)) ** 2
    else:
        image = np.zeros_like(image)
    image *= DARKNESS_CAPS[severity]
    poisson_scale = {1: 250., 2: 120., 3: 50.}[severity]
    image = rng.poisson(np.clip(image, 0., 1.) * poisson_scale) / poisson_scale
    image += rng.normal(0., DARKNESS_GAUSSIAN_STD[severity], size=image.shape)
    return _as_uint8(image * 255.)


class _WeatherSimPhysics:
    """Lazy adapter for the user's validated LISA and Hahner implementations."""

    def __init__(self, root: Optional[str], intensity_scale: float) -> None:
        bundled_root = Path(__file__).resolve().parent / 'weather_assets'
        self.intensity_scale = float(intensity_scale)
        if root:
            self.root = Path(root).expanduser().resolve()
            lidar_dir = self.root / 'lidar'
            rain_dir = lidar_dir / 'rain'
            fog_source = (lidar_dir / 'fog_model.py'
                          if (lidar_dir / 'fog_model.py').exists()
                          else lidar_dir / 'lidar.py')
            lookup_dir = lidar_dir / 'fog_lookup_tables'
        else:
            self.root = bundled_root
            rain_dir = bundled_root
            fog_source = bundled_root / 'fog_model.py'
            lookup_dir = bundled_root
        required = (fog_source, rain_dir / 'atmos_models.py',
                    rain_dir / 'mie_q.npz', lookup_dir)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                'weather_sim is incomplete; expected the maritime LiDAR physics '
                'files below:\n' + '\n'.join(missing))
        self._lidar = self._load_module('pohang_weather_sim_lidar', fog_source)
        atmos = self._load_module('pohang_weather_sim_atmos_models',
                                  rain_dir / 'atmos_models.py')
        self._lisa = atmos.LISA(atm_model='rain', rmax=500)

    @staticmethod
    def _load_module(name: str, source: Path):
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot import weather model: {source}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def rain(self, points: np.ndarray, rain_rate_mm_h: float,
             seed: int) -> np.ndarray:
        previous_state = np.random.get_state()
        np.random.seed(seed)
        try:
            lidar_input = points[:, :4].copy()
            lidar_input[:, 3] = np.clip(lidar_input[:, 3] / self.intensity_scale,
                                        0., 1.)
            augmented = self._lisa.augment(lidar_input, rain_rate_mm_h)
        finally:
            np.random.set_state(previous_state)
        result = points.copy()
        result[:, :4] = augmented[:, :4]
        result[:, 3] = np.clip(result[:, 3] * self.intensity_scale, 0.,
                               self.intensity_scale)
        return result[np.any(result[:, :3] != 0., axis=1)]

    def fog(self, points: np.ndarray, severity: int, seed: int) -> np.ndarray:
        self._lidar.RNG = np.random.default_rng(seed)
        foggy, _, _, _ = self._lidar.simulate_fog(severity, points.copy(), noise=10)
        positive_intensity = points[points[:, 3] > 0, 3]
        detection_floor = float(positive_intensity.min()) if len(positive_intensity) else 0.
        detectable = (points[:, 3] <= 0.) | (foggy[:, 3] >= detection_floor)
        return foggy[detectable]


@TRANSFORMS.register_module()
class MaritimeMultiModalCorruption(BaseTransform):
    """Apply one online maritime sensor corruption to a loaded sample.

    Training uses ``mode='train'`` and samples one corruption with probability
    ``prob``.  Evaluation uses ``mode='eval'`` with a fixed ``corruption`` and
    ``severity``.  Fixed evaluation seeds are derived from ``sample_idx`` so a
    condition is exactly repeatable across runs and worker counts.

    ``temporal_misalignment`` is deliberately optional.  It is disabled in the
    supplied training recipe because this dataset retains only every fifth raw
    frame; one stored-frame lag may already represent a large physical delay.
    """

    def __init__(self,
                 mode: str = 'train',
                 prob: float = 0.30,
                 corruption_types: Sequence[str] = ('paired_rain', 'paired_fog',
                                                     'camera_dropout', 'point_drop',
                                                     'beam_drop'),
                 corruption: Optional[str] = None,
                 severity: Optional[int] = None,
                 severity_probs: Sequence[float] = (0.50, 0.35, 0.15),
                 seed: int = 20260913,
                 physics_root: Optional[str] = None,
                 intensity_scale: float = 255.,
                 camera_dropout_mode: str = 'independent',
                 temporal_target: str = 'camera',
                 temporal_lag_frames: Optional[int] = None) -> None:
        if mode not in {'train', 'eval'}:
            raise ValueError("mode must be 'train' or 'eval'")
        if not 0. <= prob <= 1.:
            raise ValueError('prob must be in [0, 1]')
        if len(severity_probs) != 3 or sum(severity_probs) <= 0:
            raise ValueError('severity_probs must contain three positive weights')
        self.mode = mode
        self.prob = float(prob)
        self.corruption_types = tuple(self._normalise_corruption(item)
                                      for item in corruption_types)
        self.corruption = (self._normalise_corruption(corruption)
                           if corruption is not None else None)
        self.severity = severity
        if mode == 'eval' and (self.corruption is None or severity not in (1, 2, 3)):
            raise ValueError("eval mode requires corruption='<type>' and severity=1, 2 or 3")
        if not self.corruption_types and mode == 'train':
            raise ValueError('corruption_types cannot be empty in train mode')
        self.severity_probs = np.asarray(severity_probs, dtype=np.float64)
        self.severity_probs /= self.severity_probs.sum()
        self.seed = int(seed)
        self.physics_root = physics_root
        self.intensity_scale = float(intensity_scale)
        self.camera_dropout_mode = camera_dropout_mode
        self.temporal_target = temporal_target
        self.temporal_lag_frames = temporal_lag_frames
        self._physics: Optional[_WeatherSimPhysics] = None

    @staticmethod
    def _normalise_corruption(corruption: str) -> str:
        corruption = _ALIASES.get(corruption, corruption)
        if corruption not in _SUPPORTED:
            raise ValueError(f'unsupported maritime corruption: {corruption}')
        return corruption

    def _sample_condition(self, results: Dict[str, Any]) -> Tuple[Optional[str], Optional[int], np.random.Generator]:
        sample_id = results.get('sample_idx', results.get('lidar_path', 'unknown'))
        if self.mode == 'eval':
            assert self.corruption is not None and self.severity is not None
            rng = np.random.default_rng(_stable_seed(self.seed, sample_id,
                                                      self.corruption, self.severity))
            return self.corruption, self.severity, rng
        rng = np.random.default_rng(np.random.randint(0, 2**32 - 1, dtype=np.uint32))
        if rng.random() > self.prob:
            return None, None, rng
        corruption = str(rng.choice(self.corruption_types))
        severity = int(rng.choice(np.array([1, 2, 3]), p=self.severity_probs))
        return corruption, severity, rng

    @staticmethod
    def _valid_points(results: Dict[str, Any]) -> Tuple[Any, np.ndarray]:
        points = results['points']
        array = points.tensor.detach().cpu().numpy().copy()
        if array.ndim != 2 or array.shape[1] < 5:
            raise ValueError('maritime LiDAR corruptions require N x 5 points (xyz, intensity, ring)')
        # Ouster recordings include zero-valued padding.  Never corrupt those
        # placeholders: otherwise motion noise turns them into fabricated returns.
        return points, array[np.any(array[:, :3] != 0., axis=1)]

    @staticmethod
    def _store_points(results: Dict[str, Any], template: Any,
                      points: np.ndarray) -> None:
        results['points'] = template.new_point(np.asarray(points, dtype=np.float32))

    def _physics_backend(self) -> _WeatherSimPhysics:
        if self._physics is None:
            self._physics = _WeatherSimPhysics(
                self.physics_root, self.intensity_scale)
        return self._physics

    @staticmethod
    def _replace_images(results: Dict[str, Any], function) -> None:
        results['img'] = [function(image) for image in results['img']]

    def _camera_dropout(self, results: Dict[str, Any], severity: int,
                        rng: np.random.Generator) -> np.ndarray:
        count = len(results['img'])
        valid = np.ones(count, dtype=np.bool_)
        if self.camera_dropout_mode == 'independent':
            valid = rng.random(count) >= CAMERA_DROPOUT_PROBABILITIES[severity]
        elif self.camera_dropout_mode == 'front':
            valid[0] = False
        elif self.camera_dropout_mode == 'side':
            valid[int(rng.integers(1, count))] = False
        elif self.camera_dropout_mode == 'two_views':
            valid[:] = False
            valid[int(rng.integers(0, count))] = True
        elif self.camera_dropout_mode == 'all':
            valid[:] = False
        else:
            raise ValueError('camera_dropout_mode must be independent, front, side, two_views or all')
        if 'camera_valid_mask' in results:
            previous = np.asarray(results['camera_valid_mask'], dtype=np.bool_)
            if previous.shape != valid.shape:
                raise ValueError('existing camera_valid_mask has the wrong shape')
            valid &= previous
        results['img'] = [image if enabled else np.zeros_like(image)
                          for image, enabled in zip(results['img'], valid)]
        results['camera_valid_mask'] = valid
        return valid

    @staticmethod
    def _point_drop(points: np.ndarray, severity: int,
                    rng: np.random.Generator) -> np.ndarray:
        count = max(1, int(round(len(points) * POINT_KEEP_RATIOS[severity])))
        return points[rng.permutation(len(points))[:count]]

    @staticmethod
    def _beam_drop(points: np.ndarray, severity: int) -> np.ndarray:
        rings = np.unique(points[:, 4])
        if not len(rings):
            return points
        indices = np.arange(len(rings))
        if severity == 1:
            selected = indices[indices % 4 != 3]       # 64 -> 48
        elif severity == 2:
            selected = indices[indices % 2 == 0]        # 64 -> 32
        else:
            selected = indices[indices % 4 == 0]        # 64 -> 16
        return points[np.isin(points[:, 4], rings[selected])]

    @staticmethod
    def _lidar_motion(points: np.ndarray, severity: int,
                      rng: np.random.Generator) -> np.ndarray:
        std = MOTION_STD_M[severity]
        moved = points.copy()
        moved[:, :3] += rng.normal(0., std, size=3).astype(np.float32)
        point_std = np.array([std * .15, std * .20, std * .12], dtype=np.float32)
        jitter = rng.normal(0., point_std, size=(len(moved), 3))
        jitter = np.clip(jitter, -3. * point_std, 3. * point_std)
        moved[:, :3] += jitter.astype(np.float32)
        return moved

    @staticmethod
    def _previous_path(path: str, lag: int) -> Optional[Path]:
        current = Path(path)
        try:
            frame_id = int(current.stem)
        except ValueError:
            return None
        if frame_id < lag:
            return None
        return current.with_name(f'{frame_id - lag:0{len(current.stem)}d}{current.suffix}')

    def _temporal_misalignment(self, results: Dict[str, Any], severity: int) -> bool:
        lag = self.temporal_lag_frames or severity
        target = self.temporal_target
        if target not in {'camera', 'lidar', 'both'}:
            raise ValueError("temporal_target must be 'camera', 'lidar' or 'both'")
        if target in {'camera', 'both'}:
            paths = results.get('img_path', results.get('filename'))
            if not paths:
                return False
            previous_paths = [self._previous_path(path, lag) for path in paths]
            if any(path is None or not path.is_file() for path in previous_paths):
                return False
            images = []
            for path in previous_paths:
                assert path is not None
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    return False
                images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32))
            results['img'] = images
        if target in {'lidar', 'both'}:
            lidar_path = results.get('lidar_path')
            previous_path = self._previous_path(lidar_path, lag) if lidar_path else None
            if previous_path is None or not previous_path.is_file() or previous_path.suffix.lower() != '.bin':
                return False
            template, _ = self._valid_points(results)
            values = np.fromfile(previous_path, dtype=np.float32)
            if values.size % template.points_dim:
                raise ValueError(f'invalid temporal point cloud: {previous_path}')
            previous = values.reshape(-1, template.points_dim)
            previous = previous[np.any(previous[:, :3] != 0., axis=1)]
            self._store_points(results, template, previous)
        results['temporal_lag_frames'] = lag
        return True

    @staticmethod
    def _spatial_misalignment(results: Dict[str, Any], severity: int,
                               rng: np.random.Generator) -> Dict[str, np.ndarray]:
        rotation_limit, translation_limit = SPATIAL_LIMITS[severity]
        angles_deg = rng.uniform(-rotation_limit, rotation_limit, size=3)
        translation = rng.uniform(-translation_limit, translation_limit, size=3)
        delta = np.eye(4, dtype=np.float32)
        delta[:3, :3] = _rotation_xyz(np.deg2rad(angles_deg)).astype(np.float32)
        delta[:3, 3] = translation.astype(np.float32)
        lidar2cam = np.asarray(results['lidar2cam'], dtype=np.float32)
        cam2img = np.asarray(results['cam2img'], dtype=np.float32)
        corrupted_lidar2cam = np.einsum('ij,njk->nik', delta, lidar2cam)
        results['lidar2cam'] = corrupted_lidar2cam
        results['cam2lidar'] = np.linalg.inv(corrupted_lidar2cam).astype(np.float32)
        results['lidar2img'] = np.matmul(cam2img, corrupted_lidar2cam).astype(np.float32)
        return dict(rotation_deg=angles_deg.astype(np.float32),
                    translation_m=translation.astype(np.float32))

    def _apply_corruption(self, results: Dict[str, Any], corruption: str,
                          severity: int,
                          rng: np.random.Generator) -> Dict[str, Any]:
        """Apply one selected event and return its metadata."""
        metadata: Dict[str, Any] = dict(type=corruption, severity=severity)
        sample_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))

        if corruption == 'paired_rain':
            self._replace_images(results, lambda image: _image_rain(image, severity, rng))
            template, points = self._valid_points(results)
            self._store_points(results, template,
                               self._physics_backend().rain(points, RAIN_RATES_MM_H[severity], sample_seed))
            metadata['rain_rate_mm_h'] = RAIN_RATES_MM_H[severity]
        elif corruption == 'paired_fog':
            self._replace_images(results, lambda image: _image_fog(image, severity, rng))
            template, points = self._valid_points(results)
            self._store_points(results, template,
                               self._physics_backend().fog(points, severity, sample_seed))
            metadata.update(FOG_LEVELS[severity])
        elif corruption == 'point_drop':
            template, points = self._valid_points(results)
            self._store_points(results, template, self._point_drop(points, severity, rng))
            metadata['point_keep_ratio'] = POINT_KEEP_RATIOS[severity]
        elif corruption == 'beam_drop':
            template, points = self._valid_points(results)
            reduced = self._beam_drop(points, severity)
            self._store_points(results, template, reduced)
            metadata['kept_beams'] = len(np.unique(reduced[:, 4]))
        elif corruption == 'motion_blur':
            self._replace_images(results, lambda image: _image_motion_blur(image, severity, rng))
            template, points = self._valid_points(results)
            self._store_points(results, template, self._lidar_motion(points, severity, rng))
            metadata['lidar_motion_std_m'] = MOTION_STD_M[severity]
        elif corruption == 'brightness':
            self._replace_images(results, lambda image: _image_brightness(image, severity))
            metadata['brightness_v_offset'] = BRIGHTNESS_OFFSETS[severity]
        elif corruption == 'darkness':
            self._replace_images(results, lambda image: _image_darkness(image, severity, rng))
            metadata['brightness_cap'] = DARKNESS_CAPS[severity]
        elif corruption == 'camera_dropout':
            metadata['camera_valid_mask'] = self._camera_dropout(results, severity, rng)
        elif corruption == 'temporal_misalignment':
            metadata['applied'] = self._temporal_misalignment(results, severity)
        elif corruption == 'spatial_misalignment':
            metadata.update(self._spatial_misalignment(results, severity, rng))

        if 'camera_valid_mask' not in results:
            results['camera_valid_mask'] = np.ones(len(results['img']), dtype=np.bool_)
        return metadata

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        corruption, severity, rng = self._sample_condition(results)
        if corruption is None:
            results['maritime_corruption'] = dict(type='clean', severity=0)
            results['camera_valid_mask'] = np.ones(len(results['img']), dtype=np.bool_)
            return results
        assert severity is not None
        results['maritime_corruption'] = self._apply_corruption(
            results, corruption, severity, rng)
        return results


@TRANSFORMS.register_module()
class MaritimeRobustnessCorruption(MaritimeMultiModalCorruption):
    """Sample clean, single, compound and full-camera-failure regimes.

    Defaults allocate 67.5% clean, 22.5% single corruption, 5% compound
    corruption and 5% complete camera dropout.  Temporal misalignment is not
    accepted here because Pohang stores only every fifth source frame.
    """

    def __init__(
            self,
            clean_prob: float = .675,
            single_prob: float = .225,
            compound_prob: float = .05,
            full_camera_dropout_prob: float = .05,
            single_corruptions: Sequence[str] = (
                'paired_rain', 'paired_fog', 'point_drop', 'beam_drop',
                'motion_blur', 'brightness', 'darkness', 'camera_dropout',
                'spatial_misalignment'),
            single_weights: Sequence[float] = (
                .16, .16, .12, .12, .10, .07, .09, .10, .08),
            compound_corruptions: Sequence[Sequence[str]] = (
                ('paired_fog', 'beam_drop'),
                ('paired_fog', 'camera_dropout'),
                ('paired_rain', 'camera_dropout'),
                ('paired_rain', 'beam_drop'),
                ('darkness', 'point_drop')),
            severity_probs: Sequence[float] = (.55, .35, .10),
            spatial_severity_probs: Sequence[float] = (.70, .30),
            seed: int = 20260913,
            physics_root: Optional[str] = None,
            intensity_scale: float = 255.) -> None:
        super().__init__(
            mode='train', prob=1., corruption_types=single_corruptions,
            severity_probs=severity_probs, seed=seed,
            physics_root=physics_root, intensity_scale=intensity_scale,
            camera_dropout_mode='independent')
        regime_probs = np.asarray((clean_prob, single_prob, compound_prob,
                                   full_camera_dropout_prob), dtype=np.float64)
        if np.any(regime_probs < 0) or not np.isclose(regime_probs.sum(), 1.):
            raise ValueError('robustness regime probabilities must sum to 1')
        self.regime_probs = regime_probs

        self.single_corruptions = tuple(
            self._normalise_corruption(item) for item in single_corruptions)
        weights = np.asarray(single_weights, dtype=np.float64)
        if (len(weights) != len(self.single_corruptions)
                or np.any(weights < 0) or weights.sum() <= 0):
            raise ValueError(
                'single_weights must be nonnegative and match single_corruptions')
        self.single_weights = weights / weights.sum()

        compounds = []
        for events in compound_corruptions:
            normalized = tuple(self._normalise_corruption(item)
                               for item in events)
            if len(normalized) < 2:
                raise ValueError('each compound corruption needs at least two events')
            if 'temporal_misalignment' in normalized:
                raise ValueError('temporal_misalignment is disabled for robustness training')
            compounds.append(normalized)
        if not compounds:
            raise ValueError('compound_corruptions cannot be empty')
        self.compound_corruptions = tuple(compounds)

        spatial_probs = np.asarray(spatial_severity_probs, dtype=np.float64)
        if len(spatial_probs) != 2 or np.any(spatial_probs < 0) or spatial_probs.sum() <= 0:
            raise ValueError('spatial_severity_probs must contain L1/L2 weights')
        self.spatial_severity_probs = spatial_probs / spatial_probs.sum()
        if 'temporal_misalignment' in self.single_corruptions:
            raise ValueError('temporal_misalignment is disabled for robustness training')

    def _training_rng(self) -> np.random.Generator:
        return np.random.default_rng(
            np.random.randint(0, 2**32 - 1, dtype=np.uint32))

    def _draw_severity(self, corruption: str,
                       rng: np.random.Generator) -> int:
        if corruption == 'spatial_misalignment':
            return int(rng.choice(np.array([1, 2]),
                                  p=self.spatial_severity_probs))
        return int(rng.choice(np.array([1, 2, 3]), p=self.severity_probs))

    @staticmethod
    def _full_camera_dropout(results: Dict[str, Any]) -> Dict[str, Any]:
        count = len(results['img'])
        results['img'] = [np.zeros_like(image) for image in results['img']]
        results['camera_valid_mask'] = np.zeros(count, dtype=np.bool_)
        return dict(type='camera_dropout', severity='all',
                    camera_valid_mask=results['camera_valid_mask'].copy())

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        rng = self._training_rng()
        regime = str(rng.choice(
            ('clean', 'single', 'compound', 'full_camera_dropout'),
            p=self.regime_probs))
        results['camera_valid_mask'] = np.ones(
            len(results['img']), dtype=np.bool_)

        if regime == 'clean':
            events = []
        elif regime == 'full_camera_dropout':
            events = [self._full_camera_dropout(results)]
        elif regime == 'single':
            corruption = str(rng.choice(self.single_corruptions,
                                        p=self.single_weights))
            severity = self._draw_severity(corruption, rng)
            events = [self._apply_corruption(
                results, corruption, severity, rng)]
        else:
            selected = self.compound_corruptions[int(
                rng.integers(0, len(self.compound_corruptions)))]
            events = []
            for corruption in selected:
                severity = self._draw_severity(corruption, rng)
                events.append(self._apply_corruption(
                    results, corruption, severity, rng))

        results['maritime_corruption'] = dict(
            type=regime, events=events)
        results['robustness_regime'] = regime
        return results


MULTICORRUPT32_CAMERA_ORDER = ('front', 'left', 'right')
MULTICORRUPT32_CALIBRATION_LEVELS = {
    1: (1.0, 0.25),
    2: (3.0, 0.75),
    3: (5.0, 1.50),
}
MULTICORRUPT32_SCAN_MOTION_LEVELS = {
    1: (0.5, 0.20),
    2: (1.5, 0.50),
    3: (3.0, 1.00),
}




CAMERA_ORDER = ('front', 'left', 'right')

BRIGHTNESS_EXPOSURE_EV = {1: 1.5, 2: 3.0, 3: 4.5}
DARKNESS_PROFILES = {
    1: (-1.0, 250.0, 0.010),
    2: (-1.5, 150.0, 0.015),
    3: (-3.0, 50.0, 0.035),
}
CALIBRATION_PROFILES = {
    1: (4.0, 0.40),
    2: (8.0, 0.80),
    3: (12.0, 1.20),
}
CALIBRATION_VIEW_PATTERNS = {
    'front': dict(yaw_sign=1.0, translation_direction=(0.0, 1.0, 0.0)),
    'left': dict(yaw_sign=-1.0, translation_direction=(1.0, 0.0, 0.0)),
    'right': dict(yaw_sign=1.0, translation_direction=(-1.0, 0.0, 0.0)),
}
BENCHMARK_CONDITION_NAMES = {
    'brightness': {
        1: 'brightness_v2_l2',
        2: 'brightness_v2_l3',
        3: 'brightness_v3_l3',
    },
    'darkness': {
        1: 'darkness_v3_l1',
        2: 'darkness_v2_l2',
        3: 'darkness_v2_l3',
    },
    'spatial_misalignment': {
        1: 'calib_error_v2_l2',
        2: 'calib_error_v2_l3',
        3: 'calib_error_v3_l3',
    },
}


@TRANSFORMS.register_module()
class MaritimeRobustnessCorruptionFinalThree(MaritimeRobustnessCorruption):
    """Use the final three-level exposure and fixed-calibration definitions."""

    def __init__(
            self,
            camera_order: Sequence[str] = CAMERA_ORDER,
            calibration_severity_probs: Sequence[float] = (.55, .35, .10),
            **kwargs) -> None:
        self.camera_order = tuple(camera_order)
        if self.camera_order != CAMERA_ORDER:
            raise ValueError(
                f'camera_order must be exactly {CAMERA_ORDER}, got '
                f'{self.camera_order}')

        calibration_probs = np.asarray(
            calibration_severity_probs, dtype=np.float64)
        if (len(calibration_probs) != 3
                or np.any(calibration_probs < 0)
                or calibration_probs.sum() <= 0):
            raise ValueError(
                'calibration_severity_probs must contain nonnegative L1/L2/L3 '
                'weights')
        self.calibration_severity_probs = (
            calibration_probs / calibration_probs.sum())

        # The parent historically accepts only L1/L2 spatial probabilities.
        # This subclass overrides the spatial severity draw with all three
        # final calibration levels, so the parent value is only a placeholder.
        kwargs.pop('spatial_severity_probs', None)
        super().__init__(spatial_severity_probs=(1.0, 0.0), **kwargs)

    def _draw_severity(self, corruption: str,
                       rng: np.random.Generator) -> int:
        if corruption == 'spatial_misalignment':
            return int(rng.choice(
                np.array([1, 2, 3]), p=self.calibration_severity_probs))
        return super()._draw_severity(corruption, rng)

    @staticmethod
    def _srgb_to_linear(image: np.ndarray) -> np.ndarray:
        image = np.clip(image, 0.0, 1.0)
        return np.where(
            image <= 0.04045,
            image / 12.92,
            ((image + 0.055) / 1.055) ** 2.4)

    @staticmethod
    def _linear_to_srgb(image: np.ndarray) -> np.ndarray:
        image = np.clip(image, 0.0, 1.0)
        return np.where(
            image <= 0.0031308,
            image * 12.92,
            1.055 * np.power(image, 1.0 / 2.4) - 0.055)

    def _brightness_exposure(self, image: np.ndarray,
                             severity: int) -> np.ndarray:
        exposure_ev = BRIGHTNESS_EXPOSURE_EV[severity]
        srgb = _as_uint8(image).astype(np.float32) / 255.0
        linear = self._srgb_to_linear(srgb)
        exposed = np.clip(linear * (2.0 ** exposure_ev), 0.0, 1.0)
        return _as_uint8(self._linear_to_srgb(exposed) * 255.0)

    def _darkness_exposure(self, image: np.ndarray, severity: int,
                           rng: np.random.Generator) -> np.ndarray:
        exposure_ev, poisson_scale, gaussian_std = DARKNESS_PROFILES[severity]
        srgb = _as_uint8(image).astype(np.float32) / 255.0
        linear = self._srgb_to_linear(srgb)
        exposed = np.clip(linear * (2.0 ** exposure_ev), 0.0, 1.0)
        noisy = rng.poisson(exposed * poisson_scale) / poisson_scale
        noisy += rng.normal(0.0, gaussian_std, size=noisy.shape)
        return _as_uint8(self._linear_to_srgb(noisy) * 255.0)

    def _fixed_calibration(self, results: Dict[str, Any],
                           severity: int) -> Dict[str, Any]:
        rotation_deg, translation_m = CALIBRATION_PROFILES[severity]
        lidar2cam = np.asarray(results['lidar2cam'], dtype=np.float32)
        if len(lidar2cam) != len(self.camera_order):
            raise ValueError(
                f'Expected {len(self.camera_order)} calibration matrices in '
                f'order {self.camera_order}, got {len(lidar2cam)}')

        corrupted = lidar2cam.copy()
        rotations_by_view = {}
        translations_by_view = {}
        for index, view_name in enumerate(self.camera_order):
            pattern = CALIBRATION_VIEW_PATTERNS[view_name]
            angles_deg = np.array(
                [0.0, 0.0, pattern['yaw_sign'] * rotation_deg],
                dtype=np.float32)
            translation = (
                np.asarray(pattern['translation_direction'], dtype=np.float32)
                * translation_m)
            delta_lidar = np.eye(4, dtype=np.float32)
            delta_lidar[:3, :3] = _rotation_xyz(
                np.deg2rad(angles_deg)).astype(np.float32)
            delta_lidar[:3, 3] = translation
            corrupted[index] = np.matmul(lidar2cam[index], delta_lidar)
            rotations_by_view[view_name] = angles_deg.copy()
            translations_by_view[view_name] = translation.copy()

        cam2img = np.asarray(results['cam2img'], dtype=np.float32)
        results['lidar2cam'] = corrupted
        results['cam2lidar'] = np.linalg.inv(corrupted).astype(np.float32)
        results['lidar2img'] = np.matmul(cam2img, corrupted).astype(np.float32)
        return dict(
            type='spatial_misalignment',
            severity=severity,
            profile='per_camera_fixed_calibration_final',
            benchmark_condition=BENCHMARK_CONDITION_NAMES[
                'spatial_misalignment'][severity],
            fixed_view_pattern=True,
            camera_order=self.camera_order,
            rotation_deg_by_view=rotations_by_view,
            translation_m_by_view=translations_by_view,
            delta_coordinate_frame='lidar',
            composition='lidar2cam @ delta_lidar',
            points_and_annotations_unchanged=True)

    def _apply_corruption(self, results: Dict[str, Any], corruption: str,
                          severity: int,
                          rng: np.random.Generator) -> Dict[str, Any]:
        if corruption == 'brightness':
            exposure_ev = BRIGHTNESS_EXPOSURE_EV[severity]
            self._replace_images(
                results,
                lambda image: self._brightness_exposure(image, severity))
            metadata = dict(
                type='brightness',
                severity=severity,
                profile='brightness_exposure_final',
                benchmark_condition=BENCHMARK_CONDITION_NAMES[
                    corruption][severity],
                exposure_ev=exposure_ev,
                exposure_gain=2.0 ** exposure_ev,
                color_space='linear_srgb')
        elif corruption == 'darkness':
            exposure_ev, poisson_scale, gaussian_std = (
                DARKNESS_PROFILES[severity])
            self._replace_images(
                results,
                lambda image: self._darkness_exposure(image, severity, rng))
            metadata = dict(
                type='darkness',
                severity=severity,
                profile='darkness_exposure_final',
                benchmark_condition=BENCHMARK_CONDITION_NAMES[
                    corruption][severity],
                exposure_ev=exposure_ev,
                exposure_gain=2.0 ** exposure_ev,
                poisson_scale=poisson_scale,
                gaussian_std=gaussian_std,
                color_space='linear_srgb')
        elif corruption == 'spatial_misalignment':
            metadata = self._fixed_calibration(results, severity)
        else:
            return super()._apply_corruption(
                results, corruption, severity, rng)

        if 'camera_valid_mask' not in results:
            results['camera_valid_mask'] = np.ones(
                len(results['img']), dtype=np.bool_)
        return metadata


@DATASETS.register_module()
class PohangBalancedOnlineCorruptionDataset(PohangDatasetKeepEmpty):
    GROUP_RATIOS = (
        ('full_camera_dropout', 0.27266666666666667),
        ('partial_camera_dropout', 0.09133333333333334),
        ('single_other', 0.36333333333333335),
        ('compound', 0.09133333333333334),
    )

    def __init__(
            self,
            clean_count: Optional[int] = None,
            full_camera_count: Optional[int] = None,
            partial_camera_count: Optional[int] = None,
            other_single_count: Optional[int] = None,
            compound_count: Optional[int] = None,
            virtual_index_seed: int = 20260928,
            **kwargs) -> None:
        if clean_count is None or int(clean_count) <= 0:
            raise ValueError('Set clean_count to the number of source frames')
        self.clean_count = int(clean_count)
        configured_counts = (
            full_camera_count,
            partial_camera_count,
            other_single_count,
            compound_count,
        )
        self.group_counts = tuple(
            (name, int(count) if count is not None else
             int(round(self.clean_count * ratio)))
            for (name, ratio), count in zip(
                self.GROUP_RATIOS, configured_counts))
        self.virtual_index_seed = int(virtual_index_seed)
        self.virtual_index_stride = 1103
        while math.gcd(self.virtual_index_stride, self.clean_count) != 1:
            self.virtual_index_stride += 2

        super().__init__(**kwargs)
        source_count = (len(self.data_address) if self.serialize_data
                        else len(self.data_list))
        if self.clean_count != source_count:
            raise ValueError(
                f'Expected {self.clean_count} source frames, found '
                f'{source_count}')
        if any(count < 0 for _, count in self.group_counts):
            raise ValueError('Corruption counts must be nonnegative')

    def __len__(self) -> int:
        if not getattr(self, '_fully_initialized', False):
            self.full_init()
        return self.clean_count + sum(count for _, count in self.group_counts)

    def get_data_info(self, idx: int) -> Dict:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        if idx < self.clean_count:
            source_idx = idx
            group = 'clean'
        else:
            virtual_idx = idx - self.clean_count
            offset = self.virtual_index_seed % self.clean_count
            source_idx = (
                virtual_idx * self.virtual_index_stride + offset
            ) % self.clean_count
            group = ''
            cursor = 0
            for name, count in self.group_counts:
                if virtual_idx < cursor + count:
                    group = name
                    break
                cursor += count
            if not group:
                raise RuntimeError(f'No corruption group assigned to index {idx}')

        info = dict(super().get_data_info(source_idx))
        info['online_corruption_group'] = group
        return info

@TRANSFORMS.register_module()
class PohangBalancedOnlineCorruption(MaritimeRobustnessCorruptionFinalThree):
    """Apply the corruption family assigned by the virtual dataset index."""

    def transform(self, results: Dict) -> Dict:
        group = results.pop('online_corruption_group', None)
        if group is None:
            raise KeyError(
                'PohangBalancedOnlineCorruption requires '
                'online_corruption_group from its dataset')

        rng = self._training_rng()
        results['camera_valid_mask'] = np.ones(
            len(results['img']), dtype=np.bool_)

        if group == 'clean':
            events = []
            regime = 'clean'
        elif group == 'full_camera_dropout':
            events = [self._full_camera_dropout(results)]
            regime = 'full_camera_dropout'
        elif group == 'partial_camera_dropout':
            # Always preserve at least one valid camera. Draw one or two
            # missing views so this block cannot silently become all-camera
            # failure, which has its own explicitly counted block.
            view_count = len(results['img'])
            if view_count < 2:
                raise ValueError('Partial camera dropout needs >= 2 views')
            drop_count = int(rng.choice((1, min(2, view_count - 1))))
            dropped = np.sort(rng.choice(
                view_count, size=drop_count, replace=False))
            valid = np.ones(view_count, dtype=np.bool_)
            valid[dropped] = False
            results['img'] = [
                image if keep else np.zeros_like(image)
                for image, keep in zip(results['img'], valid)]
            results['camera_valid_mask'] = valid
            events = [dict(
                type='camera_dropout',
                severity=drop_count,
                profile='balanced_exact_partial_view_loss',
                dropped_camera_indices=dropped.tolist(),
                camera_valid_mask=valid.copy())]
            regime = 'partial_camera_dropout'
        elif group == 'single_other':
            candidates = [
                (name, weight)
                for name, weight in zip(
                    self.single_corruptions, self.single_weights)
                if name != 'camera_dropout']
            names = [name for name, _ in candidates]
            weights = np.asarray([weight for _, weight in candidates],
                                 dtype=np.float64)
            weights /= weights.sum()
            corruption = str(rng.choice(names, p=weights))
            severity = self._draw_severity(corruption, rng)
            events = [self._apply_corruption(
                results, corruption, severity, rng)]
            regime = 'single_other'
        elif group == 'compound':
            selected = self.compound_corruptions[int(
                rng.integers(0, len(self.compound_corruptions)))]
            events = []
            for corruption in selected:
                severity = self._draw_severity(corruption, rng)
                events.append(self._apply_corruption(
                    results, corruption, severity, rng))
            regime = 'compound'
        else:
            raise ValueError(f'Unknown online corruption group: {group}')

        results['maritime_corruption'] = dict(type=regime, events=events)
        results['robustness_regime'] = regime
        return results



