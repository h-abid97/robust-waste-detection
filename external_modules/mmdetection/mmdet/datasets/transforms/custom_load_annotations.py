from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms.loading import LoadAnnotations
import numpy as np

@TRANSFORMS.register_module()
class LoadAnnotationsWithConfidence(LoadAnnotations):
    """Load annotations with confidence scores.
    
    Extracts 'score' values from annotations if present, otherwise defaults to 1.0.
    This approach works for both ground truth and pseudo-label annotations.
    
    Args:
        with_confidence (bool): Whether to parse and load confidence scores.
            Defaults to True.
        confidence_key (str): The key in annotations where confidence scores 
            are stored. Defaults to 'score'.
        **kwargs: Arguments passed to parent LoadAnnotations class.
    """
    
    def __init__(self,
                 with_confidence=True,
                 confidence_key='score',
                 **kwargs):
        super().__init__(**kwargs)
        self.with_confidence = with_confidence
        self.confidence_key = confidence_key
    
    # def _load_confidence(self, results):
    #     """Load confidence scores from preserved values."""
    #     if 'original_confidence_scores' in results:
    #         results['gt_confidences'] = np.array(results['original_confidence_scores'], dtype=np.float32)
    #     else:
    #         # Default to 1.0 for all bboxes
    #         num_bboxes = len(results.get('gt_bboxes', []))
    #         results['gt_confidences'] = np.ones(num_bboxes, dtype=np.float32)
    #     return results
    
    def _load_confidence(self, results):
        """Private function to load confidence scores."""
        gt_confidences = []
        img_path = results.get('img_path', 'unknown')
        
        # First try to access gt_instances
        gt_instances = results.get('gt_instances', [])
        if gt_instances:
            for instance in gt_instances:
                # Check if instance has the score field
                if self.confidence_key in instance:
                    confidence = instance[self.confidence_key]
                else:
                    # Default to 1.0 if no score field
                    confidence = 1.0
                gt_confidences.append(confidence)
        else:
            # Fall back to looking for instances if gt_instances not found
            instances = results.get('instances', [])
            if instances:
                for instance in instances:
                    if self.confidence_key in instance:
                        confidence = instance[self.confidence_key]
                    else:
                        confidence = 1.0
                    gt_confidences.append(confidence)
            else:
                # If no instances or gt_instances, try to match with gt_bboxes count
                gt_bboxes = results.get('gt_bboxes', [])
                if len(gt_bboxes) > 0:
                    #print(f"WARNING: No instances or gt_instances found, using gt_bboxes count for {img_path}")
                    gt_confidences = [1.0] * len(gt_bboxes)
                else:
                    # No annotations found at all
                    #print(f"ERROR: No annotations found for {img_path}")
                    gt_confidences = []
        
        # Store the confidence values
        results['gt_confidences'] = np.array(gt_confidences, dtype=np.float32)
        return results
    
    def transform(self, results):
        """Function to load annotations with confidence scores.
        
        Args:
            results (dict): Result dict from dataset.
            
        Returns:
            dict: The dict with loaded annotations and confidence scores.
        """
        # First load standard annotations
        results = super().transform(results)
        
        # Then load confidence scores if required
        if self.with_confidence:
            results = self._load_confidence(results)
        
        return results
    
    def __repr__(self):
        repr_str = super().__repr__()[:-1]  # Remove the closing parenthesis
        repr_str += f', with_confidence={self.with_confidence}, '
        repr_str += f"confidence_key='{self.confidence_key}')"
        return repr_str