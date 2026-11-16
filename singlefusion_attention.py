import os

custom_imports = dict(
    imports=[
        'projects.BEVFusion.bevfusion',
        'projects.BEVFusion.singlefusion',
        'projects.BEVFusion.corruptions',
    ],
    allow_failed_imports=False)

class_names = ['Vessel']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=True)
data_root = os.environ.get('POHANG_DATA_ROOT', '')
train_ann_file = ''
val_ann_file = ''
test_ann_file = ''
clean_count = None
full_camera_count = None
partial_camera_count = None
other_single_count = None
compound_count = None
weather_sim_root = os.environ.get('WEATHER_SIM_ROOT', '')
point_cloud_range = [0.0, -60.8, -5.0, 153.6, 60.8, 10.0]
pillar_voxel_size = [0.2, 0.2, 15.0]
pillar_grid_size = [768, 608, 1]
image_size = [256, 704]
feature_size = [32, 88]
max_epochs = 12

model = dict(
    type='PohangQualityAwareSingleFusionNoSafety',
    point_cloud_range=point_cloud_range,
    expected_lidar_beams=64,
    density_cap=20,
    view_availability_loss_weight=0.05,
    view_quality_encoder=dict(
        type='PohangViewQualityEncoder',
        hidden_channels=16,
        enabled=True),
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=20,
            point_cloud_range=point_cloud_range,
            voxel_size=pillar_voxel_size,
            max_voxels=[30000, 40000],
            voxelize_reduce=False)),
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=(
                'https://download.openmmlab.com/mmdetection3d/'
                'v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth'))),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=image_size,
        feature_size=feature_size,
        xbound=[0.0, 153.601, 0.2],
        ybound=[-60.8, 60.801, 0.2],
        zbound=[-5.0, 10.0, 15.0],
        dbound=[1.0, 166.0, 1.0],
        downsample=1),
    fusion_layer=dict(
        type='PohangQualityAwareAlignConvFuser',
        in_channels=[80, 64],
        out_channels=64,
        quality_channels=16,
        pool_factor=4,
        lidar_prior=0.05,
        use_camera_quality=True,
        use_lidar_quality=True,
        use_disagreement=True,
        use_lidar_density=True,
        use_ring_coverage=True,
        use_camera_availability=True,
        hard_camera_fallback=True,
        fallback_mode='convfuser_lidar',
        use_masked_reconstruction=True,
        reconstruction_mask_ratio=0.15,
        use_modality_aux_heads=True,
        use_local_cross_attention=True,
        local_attention_radius=2,
        quality_loss_weight=0.10,
        reconstruction_loss_weight=0.05,
        modality_aux_loss_weight=0.10,
        freeze_loaded_bn_stats=True),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=pillar_voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False),
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[pillar_grid_size[1], pillar_grid_size[0]]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    bbox_head=dict(
        type='CenterHead',
        in_channels=384,
        tasks=[dict(num_class=1, class_names=class_names)],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-10.0, -70.0, -10.0, 160.0, 70.0, 15.0],
            max_num=100,
            score_threshold=0.01,
            out_size_factor=4,
            voxel_size=pillar_voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=dict(
            grid_size=pillar_grid_size,
            voxel_size=pillar_voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=100,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 0.0, 0.0]),
        test_cfg=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-10.0, -70.0, -10.0, 160.0, 70.0, 15.0],
            max_per_img=100,
            max_pool_nms=False,
            min_radius=[4],
            score_threshold=0.01,
            out_size_factor=4,
            voxel_size=pillar_voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))

robustness_corruption = dict(
    type='PohangBalancedOnlineCorruption',
    single_corruptions=(
        'paired_rain', 'paired_fog', 'point_drop', 'beam_drop',
        'motion_blur', 'brightness', 'darkness', 'camera_dropout',
        'spatial_misalignment'),
    single_weights=(0.16, 0.16, 0.12, 0.12, 0.10, 0.07, 0.09, 0.10, 0.08),
    compound_corruptions=(
        ('paired_fog', 'beam_drop'),
        ('paired_fog', 'camera_dropout'),
        ('paired_rain', 'camera_dropout'),
        ('paired_rain', 'beam_drop'),
        ('darkness', 'point_drop')),
    severity_probs=(0.55, 0.35, 0.10),
    calibration_severity_probs=(0.55, 0.35, 0.10),
    camera_order=('front', 'left', 'right'),
    seed=20260913,
    physics_root=weather_sim_root,
    intensity_scale=255.0)

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        num_views=3,
        camera_names=('front', 'left', 'right'),
        backend_args=None),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=None),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    robustness_corruption,
    dict(
        type='ImageAug3D',
        final_dim=image_size,
        resize_lim=[0.35, 0.40],
        bot_pct_lim=[0.30, 0.40],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        rot_range=[-0.0873, 0.0873],
        scale_ratio_range=[0.98, 1.02],
        translation_std=[0.0, 0.0, 0.0]),
    dict(
        type='PohangFusionRandomFlip3D',
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=(
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img',
            'cam2lidar', 'img_aug_matrix', 'lidar_aug_matrix',
            'box_type_3d', 'sample_idx', 'lidar_path', 'img_path',
            'num_pts_feats', 'num_views', 'camera_valid_mask',
            'maritime_corruption', 'robustness_regime')),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='PohangBalancedOnlineCorruptionDataset',
        clean_count=clean_count,
        full_camera_count=full_camera_count,
        partial_camera_count=partial_camera_count,
        other_single_count=other_single_count,
        compound_count=compound_count,
        virtual_index_seed=20260928,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(pts='', img=''),
        filter_empty_gt=False,
        metainfo=metainfo,
        modality=input_modality,
        box_type_3d='LiDAR',
        pipeline=train_pipeline))

val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=999)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.05),
        'img_neck': dict(lr_mult=0.10),
        'view_transform': dict(lr_mult=0.10),
        'pts_voxel_encoder': dict(lr_mult=0.10),
        'pts_middle_encoder': dict(lr_mult=0.10),
        'pts_backbone': dict(lr_mult=0.10),
        'pts_neck': dict(lr_mult=0.10),
        'bbox_head': dict(lr_mult=0.25),
        'view_quality_encoder': dict(lr_mult=1.0),
        'fusion_layer': dict(lr_mult=1.0),
    }),
    clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False,
         begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, end=max_epochs,
         T_max=max_epochs, by_epoch=True, eta_min_ratio=0.05,
         convert_to_iter_based=True),
]
custom_hooks = [
    dict(type='PohangFusionWarmupHook', warmup_epochs=1,
         module_names=(
             'pts_voxel_encoder', 'pts_middle_encoder', 'pts_backbone',
             'pts_neck', 'bbox_head')),
    dict(type='PohangFreezeBNStatsHook',
         module_names=('img_neck', 'view_transform')),
]
find_unused_parameters = True
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=25),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=12),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
randomness = dict(seed=42, deterministic=False)
load_from = None
resume = False
work_dir = None
auto_scale_lr = dict(enable=False, base_batch_size=2)
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
launcher = 'none'
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
    name='visualizer')
