_base_ = [
    './grounding_dino_swin-t_finetune_16xb2_1x_coco.py',
]

load_from = './weights/gdino-swin-b/zerowaste_f_finetuned_best_coco_bbox_mAP.pth'

data_root = './data/'
class_name = ('rigid_plastic', 'cardboard', 'metal', 'soft_plastic')
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[
        (0, 128, 255),  # Blue for rigid_plastic
        (255, 165, 0),  # Orange for cardboard
        (220, 20, 60),  # Red for metal
        (34, 139, 34)   # Green for soft_plastic
    ])

model = dict(
    type='GroundingDINO',
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.3,
        patch_norm=True
    ),
    neck=dict(in_channels=[256, 512, 1024]),
    bbox_head=dict(
        type='SemiSupGroundingDINOHead',
        num_classes=num_classes)
)

train_dataloader = dict(
   _delete_=True,
   batch_sampler=dict(type='AspectRatioBatchSampler'),
   #batch_sampler=dict(type='RatioBalancedBatchSampler', ratio_per_dataset=[0.25, 0.75], drop_last=False),
   batch_size=4,
   num_workers=2,
   persistent_workers=True,
   sampler=dict(shuffle=True, type='DefaultSampler'),
   dataset=dict(
       type='ConcatDataset',
       datasets=[
           dict(
               type='CocoDataset',
               ann_file='./data/zerowaste-f/train/labels.json',
               data_prefix=dict(img='./data/zerowaste-f/train/data'),
               metainfo=metainfo,
               backend_args=None,
               filter_cfg=dict(filter_empty_gt=False, min_size=32),
               pipeline=[
                   dict(backend_args=None, type='LoadImageFromFile'),
                   #dict(type='LoadAnnotations', with_bbox=True),
                   dict(
                        type='LoadAnnotationsWithConfidence',
                        with_bbox=True,
                        with_confidence=True,
                        confidence_key='score'
                    ),
                   dict(prob=0.5, type='RandomFlip'),
                   dict(
                       transforms=[
                           [
                               dict(
                                   keep_ratio=True,
                                   scales=[
                                       (480, 1333),
                                       (512, 1333),
                                       (544, 1333),
                                       (576, 1333),
                                       (608, 1333),
                                       (640, 1333),
                                       (672, 1333),
                                       (704, 1333),
                                       (736, 1333),
                                       (768, 1333),
                                       (800, 1333),
                                   ],
                                   type='RandomChoiceResize'),
                           ],
                           [
                               dict(
                                   keep_ratio=True,
                                   scales=[
                                       (400, 4200),
                                       (500, 4200),
                                       (600, 4200),
                                   ],
                                   type='RandomChoiceResize'),
                               dict(
                                   allow_negative_crop=True,
                                   crop_size=(384, 600),
                                   crop_type='absolute_range',
                                   type='RandomCrop'),
                               dict(
                                   keep_ratio=True,
                                   scales=[
                                       (480, 1333),
                                       (512, 1333),
                                       (544, 1333),
                                       (576, 1333),
                                       (608, 1333),
                                       (640, 1333),
                                       (672, 1333),
                                       (704, 1333),
                                       (736, 1333),
                                       (768, 1333),
                                       (800, 1333),
                                   ],
                                   type='RandomChoiceResize'),
                           ],
                       ],
                       type='RandomChoice'),
                   dict(type='SetPseudoFlag', flag_value=False),
                   dict(
                       meta_keys=(
                           'img_id',
                           'img_path',
                           'ori_shape',
                           'img_shape',
                           'scale_factor',
                           'flip',
                           'flip_direction',
                           'text',
                           'custom_entities',
                           'is_pseudo',
                           'gt_confidences',
                       ),
                       type='PackDetInputs'),
               ],
               return_classes=True
           ),
           dict(
               type='CocoDatasetWithScores',
               ann_file='./data/pseudo_labels/zerowaste-s_ensemble_consensus_pseudo_annotations_offset.json',
               data_prefix=dict(img='./data/zerowaste-s/data'),
               metainfo=metainfo,
               backend_args=None,
               filter_cfg=dict(filter_empty_gt=False, min_size=32),
               pipeline=[
                   dict(backend_args=None, type='LoadImageFromFile'),
                   #dict(type='LoadAnnotations', with_bbox=True),
                   dict(
                        type='LoadAnnotationsWithConfidence',
                        with_bbox=True,
                        with_confidence=True,
                        confidence_key='score'
                    ),
                   dict(prob=0.5, type='RandomFlip'),
                   dict(
                       transforms=[
                           [
                               dict(
                                   keep_ratio=True,
                                   scales=[
                                       (480, 1333),
                                       (512, 1333),
                                       (544, 1333),
                                       (576, 1333),
                                       (608, 1333),
                                       (640, 1333),
                                       (672, 1333),
                                       (704, 1333),
                                       (736, 1333),
                                       (768, 1333),
                                       (800, 1333),
                                   ],
                                   type='RandomChoiceResize'),
                           ],
                           [
                               dict(
                                   keep_ratio=True,
                                   scales=[
                                       (400, 4200),
                                       (500, 4200),
                                       (600, 4200),
                                   ],
                                   type='RandomChoiceResize'),
                               dict(
                                   allow_negative_crop=True,
                                   crop_size=(384, 600),
                                   crop_type='absolute_range',
                                   type='RandomCrop'),
                               dict(
                                   keep_ratio=True,
                                   scales=[
                                       (480, 1333),
                                       (512, 1333),
                                       (544, 1333),
                                       (576, 1333),
                                       (608, 1333),
                                       (640, 1333),
                                       (672, 1333),
                                       (704, 1333),
                                       (736, 1333),
                                       (768, 1333),
                                       (800, 1333),
                                   ],
                                   type='RandomChoiceResize'),
                           ],
                       ],
                       type='RandomChoice'),
                   dict(type='SetPseudoFlag', flag_value=True),
                   dict(
                       meta_keys=(
                           'img_id',
                           'img_path',
                           'ori_shape',
                           'img_shape',
                           'scale_factor',
                           'flip',
                           'flip_direction',
                           'text',
                           'custom_entities',
                           'is_pseudo',
                           'gt_confidences',
                       ),
                       type='PackDetInputs'),
               ],
               return_classes=True
           )
       ]
   )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='zerowaste-f/test/labels.json',
        data_prefix=dict(img='zerowaste-f/test/data')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'zerowaste-f/test/labels.json')
test_evaluator = val_evaluator

max_epoch = 12

default_hooks = dict(
    checkpoint=dict(
        interval=1, 
        max_keep_ckpts=4, 
        save_best='coco/bbox_mAP',
        save_optimizer=False,
        save_param_scheduler=False
    ),
    logger=dict(type='LoggerHook', interval=50))

train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=200),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[10],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.05),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=4)

work_dir = 'experiments/swin-b_zerowaste_semi-supervised'