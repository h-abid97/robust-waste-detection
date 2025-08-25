from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class PreserveScores(BaseTransform):
    """Preserve confidence scores before standard annotation loading."""
    
    def __init__(self, confidence_key='score'):
        self.confidence_key = confidence_key
        
    def transform(self, results):
        """Extract scores from raw annotations and store them."""
        # Detailed debug output
        print(f"\nDEBUG PreserveScores: Processing {results.get('img_path', 'unknown')}")
        print(f"Results keys: {list(results.keys())}")
        
        if 'ann_info' in results:
            print(f"Ann_info keys: {list(results['ann_info'].keys())}")
            
            if 'anns' in results['ann_info']:
                anns = results['ann_info']['anns']
                print(f"Found {len(anns)} annotations")
                
                if anns:
                    print(f"First annotation keys: {list(anns[0].keys())}")
                    if self.confidence_key in anns[0]:
                        print(f"Score value: {anns[0][self.confidence_key]}")
                    else:
                        print(f"WARNING: '{self.confidence_key}' not found in annotation")
                
                # Extract scores from original annotations
                scores = []
                for ann in anns:
                    score = ann.get(self.confidence_key, 1.0)
                    scores.append(score)
                
                # Store in results for later use
                results['original_confidence_scores'] = scores
                print(f"Stored {len(scores)} scores: {scores[:5]}")
            else:
                print("No 'anns' field in ann_info")
                results['original_confidence_scores'] = []
        else:
            print("No 'ann_info' field in results")
            results['original_confidence_scores'] = []
            
        return results