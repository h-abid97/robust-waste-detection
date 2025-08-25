from mmdet.models.dense_heads import GroundingDINOHead
#from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import InstanceList, reduce_mean
#from mmdet.models.losses import QualityFocalLoss
from mmengine.structures import InstanceData
from mmdet.registry import MODELS

from typing import List, Tuple
import numpy as np
from torch import Tensor
import torch



@MODELS.register_module()
class SemiSupGroundingDINOHead(GroundingDINOHead):
    """GroundingDINO head adapted for semi-supervised learning with confidence-weighted losses."""
    
    def __init__(self, *args, enable_logging=True, confidence_power=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_logging = enable_logging
        self.confidence_power = confidence_power
        self.batch_counter = 0
        print(f"\nInitialized SemiSupGroundingDINOHead:")
        print(f"- Background class weight: {self.bg_cls_weight}")
        print(f"- Logging enabled: {enable_logging}")
        print(f"- Confidence power: {confidence_power}")
    
    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                        gt_instances: InstanceData, img_meta: dict) -> tuple:
        """Modified target generation to apply confidence weighting."""
        # Get targets from parent class
        targets = super()._get_targets_single(cls_score, bbox_pred, gt_instances, img_meta)
        labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds = targets
        
        # Only apply confidence weighting for pseudo-labeled data
        is_pseudo = img_meta.get('is_pseudo', False)
        
        if is_pseudo and 'gt_confidences' in img_meta:
            # Convert numpy array to tensor
            confidences = torch.tensor(
                img_meta['gt_confidences'], 
                dtype=torch.float32, 
                device=cls_score.device
            )
            
            # Clamp confidence scores to [0, 1] range
            confidences = torch.clamp(confidences, min=0.0, max=1.0)
            
            # Get matched gt indices for positive predictions
            assign_result = self.assigner.assign(
                pred_instances=InstanceData(scores=cls_score, bboxes=bbox_pred),
                gt_instances=gt_instances,
                img_meta=img_meta)
            
            pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
            
            # Apply confidence scores to both label and bbox weights
            for i, gt_idx in enumerate(pos_assigned_gt_inds):
                if gt_idx >= 0 and gt_idx < len(confidences):
                    # Apply power function to confidence
                    confidence = confidences[gt_idx]
                    confidence_weighted = confidence ** self.confidence_power
                    
                    # Apply to classification weight
                    label_pos = pos_inds[i]
                    if 0 <= label_pos < len(label_weights):
                        label_weights[label_pos] *= confidence_weighted
                    
                    # Apply to bbox weight
                    bbox_pos = pos_inds[i]
                    if 0 <= bbox_pos < len(bbox_weights):
                        bbox_weights[bbox_pos] *= confidence_weighted
        
        return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds
    
    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss computation with logging of confidence statistics."""
        # Calculate losses using the parent method (which will use our modified _get_targets_single)
        loss_cls, loss_bbox, loss_iou = super().loss_by_feat_single(
            cls_scores, bbox_preds, batch_gt_instances, batch_img_metas)
        
        # Collect statistics for logging
        if self.enable_logging and self.batch_counter % 10 == 0:
            pseudo_count = 0
            total_count = len(batch_gt_instances)
            avg_confidence = 0.0
            confidence_count = 0
            
            for img_meta in batch_img_metas:
                is_pseudo = img_meta.get('is_pseudo', False)
                if is_pseudo:
                    pseudo_count += 1
                    if 'gt_confidences' in img_meta:
                        # Get confidence scores directly from img_meta
                        confs = img_meta['gt_confidences']
                        # Convert to tensor if needed for calculation
                        if not isinstance(confs, torch.Tensor):
                            confs = torch.tensor(confs, dtype=torch.float32)
                        # Add to running average
                        avg_confidence += confs.mean().item()
                        confidence_count += 1
            
            if confidence_count > 0:
                avg_confidence /= confidence_count
                #print(f"Batch {self.batch_counter}: Pseudo ratio={pseudo_count/total_count:.2f}, "
                #    f"Avg confidence={avg_confidence:.4f}")
        
        self.batch_counter += 1
        
        return loss_cls, loss_bbox, loss_iou






    # def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
    #                         gt_instances: InstanceData, img_meta: dict) -> tuple:
    #     """Modified target generation to apply confidence weighting with detailed verification."""
    #     # Get targets from parent class
    #     targets = super()._get_targets_single(cls_score, bbox_pred, gt_instances, img_meta)
    #     labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds = targets
        
    #     # Only apply confidence weighting for pseudo-labeled data
    #     is_pseudo = img_meta.get('is_pseudo', False)
        
    #     # Print basic info for every sample
    #     print(f"\n=== Sample Processing ===")
    #     print(f"Is pseudo: {is_pseudo}")
    #     print(f"img_id: {img_meta.get('img_id', 'unknown')}")
    #     print(f"Shape of cls_score: {cls_score.shape}")
    #     print(f"Shape of bbox_pred: {bbox_pred.shape}")
    #     print(f"Shape of labels: {labels.shape}")
    #     print(f"Shape of label_weights: {label_weights.shape}")
    #     print(f"Shape of bbox_targets: {bbox_targets.shape}")
    #     print(f"Shape of bbox_weights: {bbox_weights.shape}")
    #     print(f"Number of pos_inds: {len(pos_inds)}")
    #     print(f"Number of neg_inds: {len(neg_inds)}")
        
    #     if is_pseudo and 'gt_confidences' in img_meta:
    #         print(f"\n=== Processing Pseudo-Labeled Sample ===")
            
    #         # Convert numpy array to tensor
    #         confidences = torch.tensor(
    #             img_meta['gt_confidences'], 
    #             dtype=torch.float32, 
    #             device=cls_score.device
    #         )
            
    #         # Print confidence score distribution
    #         print(f"Confidence scores: {confidences}")
    #         print(f"Mean confidence: {confidences.mean().item():.4f}")
    #         print(f"Min confidence: {confidences.min().item():.4f}")
    #         print(f"Max confidence: {confidences.max().item():.4f}")
            
    #         # Clamp confidence scores to [0, 1] range
    #         confidences = torch.clamp(confidences, min=0.0, max=1.0)
            
    #         # Get matched gt indices for positive predictions
    #         assign_result = self.assigner.assign(
    #             pred_instances=InstanceData(scores=cls_score, bboxes=bbox_pred),
    #             gt_instances=gt_instances,
    #             img_meta=img_meta)
            
    #         pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
            
    #         # Print assignment information
    #         print(f"pos_assigned_gt_inds: {pos_assigned_gt_inds}")
            
    #         # Record original weights for verification
    #         original_label_weights = label_weights.clone()
    #         original_bbox_weights = bbox_weights.clone()
            
    #         # Track sum of weights for calculating averages
    #         sum_original_label_weights = 0.0
    #         sum_new_label_weights = 0.0
    #         sum_original_bbox_weights = 0.0
    #         sum_new_bbox_weights = 0.0
    #         num_modified = 0
            
    #         # Apply confidence scores to both label and bbox weights
    #         for i, gt_idx in enumerate(pos_assigned_gt_inds):
    #             if gt_idx >= 0 and gt_idx < len(confidences):
    #                 confidence = confidences[gt_idx]
    #                 print(f"\nProcessing match {i}:")
    #                 print(f"  - gt_idx: {gt_idx}")
    #                 print(f"  - confidence: {confidence:.4f}")
                    
    #                 # Verify and apply to classification weight
    #                 label_pos = pos_inds[i]
    #                 if 0 <= label_pos < len(label_weights):
    #                     old_label_weight = label_weights[label_pos].item()
    #                     label_weights[label_pos] *= confidence
    #                     new_label_weight = label_weights[label_pos].item()
    #                     print(f"  - Label weight at pos {label_pos}: {old_label_weight:.4f} -> {new_label_weight:.4f}")
                        
    #                     # Track for average calculation
    #                     sum_original_label_weights += old_label_weight
    #                     sum_new_label_weights += new_label_weight
    #                     num_modified += 1
    #                 else:
    #                     print(f"  - WARNING: Invalid label position {label_pos} (out of bounds)")
                    
    #                 # Verify and apply to bbox weight
    #                 bbox_pos = pos_inds[i]
    #                 if 0 <= bbox_pos < len(bbox_weights):
    #                     old_bbox_weight = bbox_weights[bbox_pos][0].item()  # Just show first element for brevity
    #                     bbox_weights[bbox_pos] *= confidence
    #                     new_bbox_weight = bbox_weights[bbox_pos][0].item()
    #                     print(f"  - BBox weight at pos {bbox_pos}: {old_bbox_weight:.4f} -> {new_bbox_weight:.4f}")
                        
    #                     # Track for average calculation
    #                     sum_original_bbox_weights += old_bbox_weight
    #                     sum_new_bbox_weights += new_bbox_weight
    #                 else:
    #                     print(f"  - WARNING: Invalid bbox position {bbox_pos} (out of bounds)")
            
    #         # Verification summary
    #         print("\n=== Verification Summary ===")
    #         print(f"Total positive matches: {len(pos_inds)}")
    #         print(f"Weights modified: {num_modified}")
            
    #         # Calculate and show average weights
    #         if num_modified > 0:
    #             avg_orig_label = sum_original_label_weights / num_modified
    #             avg_new_label = sum_new_label_weights / num_modified
    #             avg_orig_bbox = sum_original_bbox_weights / num_modified
    #             avg_new_bbox = sum_new_bbox_weights / num_modified
                
    #             print(f"Average label weight: {avg_orig_label:.4f} -> {avg_new_label:.4f}")
    #             print(f"Average bbox weight: {avg_orig_bbox:.4f} -> {avg_new_bbox:.4f}")
        
    #     return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds
















# @MODELS.register_module()
# class SemiSupGroundingDINOHead(GroundingDINOHead):
#     def __init__(self, *args, enable_logging=True, verification_mode=True, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.enable_logging = enable_logging
#         self.verification_mode = verification_mode
#         self.batch_counter = 0
#         print(f"\nInitialized SemiSupGroundingDINOHead:")
#         print(f"- Background class weight: {self.bg_cls_weight}")
#         print(f"- Verification mode: {verification_mode}")
#         print(f"- Logging enabled: {enable_logging}")

#     def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
#                          gt_instances: InstanceData, img_meta: dict) -> tuple:
#         """Modified target generation to handle pseudo-labeled data differently."""
#         # Get targets from parent class
#         targets = super()._get_targets_single(cls_score, bbox_pred, gt_instances, img_meta)
#         labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds = targets
        
#         # Zero out negative sample weights for pseudo-labeled data
#         if img_meta.get('is_pseudo', False):
#             label_weights[neg_inds] = 0
        
#         return targets

#     def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
#                         batch_gt_instances: InstanceList,
#                         batch_img_metas: List[dict]) -> Tuple[Tensor]:
#         """Loss computation."""
#         num_imgs = cls_scores.size(0)
#         cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
#         bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        
#         # Get targets (this will use our modified _get_targets_single)
#         with torch.no_grad():
#             cls_reg_targets = self.get_targets(cls_scores_list,
#                                             bbox_preds_list,
#                                             batch_gt_instances,
#                                             batch_img_metas)
        
#         (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
#          num_total_pos, num_total_neg) = cls_reg_targets

#         # Stack and concatenate targets
#         labels = torch.stack(labels_list, 0)
#         label_weights = torch.stack(label_weights_list, 0)
#         bbox_targets = torch.cat(bbox_targets_list, 0)
#         bbox_weights = torch.cat(bbox_weights_list, 0)

#         # Text mask handling
#         assert (self.text_masks.dim() == 2)
#         text_masks = self.text_masks.new_zeros((self.text_masks.size(0), self.max_text_len))
#         text_masks[:, :self.text_masks.size(1)] = self.text_masks
#         text_mask = (text_masks > 0).unsqueeze(1)
#         text_mask = text_mask.repeat(1, cls_scores.size(1), 1)

#         # Apply masks
#         cls_scores = torch.masked_select(cls_scores, text_mask).contiguous()
#         labels = torch.masked_select(labels, text_mask)
#         label_weights = label_weights[..., None].repeat(1, 1, text_mask.size(-1))
#         label_weights = torch.masked_select(label_weights, text_mask)

#         # Classification loss calculation
#         cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
#         if self.sync_cls_avg_factor:
#             cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
#         cls_avg_factor = max(cls_avg_factor, 1)

#         # Calculate classification loss
#         if isinstance(self.loss_cls, QualityFocalLoss):
#             raise NotImplementedError('QualityFocalLoss for GroundingDINOHead is not supported yet.')
#         else:
#             loss_cls = self.loss_cls(
#                 cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

#         # Compute average number of gt boxes
#         num_total_pos = loss_cls.new_tensor([num_total_pos])
#         num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

#         # Construct factors for bbox rescaling
#         factors = []
#         for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
#             img_h, img_w = img_meta['img_shape']
#             factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(
#                 bbox_pred.size(0), 1)
#             factors.append(factor)
#         factors = torch.cat(factors, 0)

#         # DETR bbox regression
#         bbox_preds = bbox_preds.reshape(-1, 4)
#         bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
#         bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

#         # Calculate IoU and bbox losses
#         loss_iou = self.loss_iou(
#             bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)
#         loss_bbox = self.loss_bbox(
#             bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

#         return loss_cls, loss_bbox, loss_iou

