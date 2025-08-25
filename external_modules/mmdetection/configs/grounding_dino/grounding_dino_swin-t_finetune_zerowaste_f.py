_base_ = 'grounding_dino_swin-t_finetune_16xb2_1x_coco.py'

data_root = './data/zerowaste-f'
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

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/labels.json',
        data_prefix=dict(img='train/data')))

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test/labels.json',
        data_prefix=dict(img='test/data')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + '/test/labels.json')
test_evaluator = val_evaluator

max_epoch = 15

default_hooks = dict(
    checkpoint=dict(
        interval=1, 
        max_keep_ckpts=1, 
        save_best='coco/bbox_mAP',
        save_optimizer=False,
        save_param_scheduler=False
    ),
    logger=dict(type='LoggerHook', interval=50))

train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    #dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=8)

work_dir = 'experiments/swin-t_zerowaste-f_finetune'