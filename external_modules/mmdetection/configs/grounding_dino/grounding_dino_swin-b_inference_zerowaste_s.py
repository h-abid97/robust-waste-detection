_base_ = 'grounding_dino_swin-t_finetune_16xb2_1x_coco.py'

# Metadata for the dataset
data_root = './data/zerowaste-s'
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

# Model configuration
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
    bbox_head=dict(num_classes=num_classes)
)

# Test dataloader for the unlabeled dataset
test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='labels.json',
        data_prefix=dict(img='data'),
        test_mode=True  # Enables inference mode
    )
)

test_evaluator = dict(ann_file=data_root + '/labels.json')

# Default hooks for logging
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10)  # Log every 10 batches
)

work_dir = 'experiments/swin-b_fine-tuned_zerowaste-s_evaluation'