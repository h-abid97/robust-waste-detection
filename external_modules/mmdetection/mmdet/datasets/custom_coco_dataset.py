from mmdet.registry import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module()
class CocoDatasetWithScores(CocoDataset):
    """Extended COCO dataset that preserves confidence scores."""
    
    def parse_data_info(self, raw_data_info):
        """Parse raw annotation to target format with score preservation."""
        # First call the parent method to do standard parsing
        data_info = super().parse_data_info(raw_data_info)
        
        # Get access to raw annotations
        ann_info = raw_data_info['raw_ann_info']
        
        # Debug print
        #print(f"CocoDatasetWithScores: Processing image {data_info.get('img_id')}")
        #print(f"Found {len(ann_info)} raw annotations")
        #if ann_info:
            #print(f"Sample scores: {[ann.get('score', 'N/A') for ann in ann_info[:3]]}")
        
        # Add scores to instances if they exist
        count_with_score = 0
        for i, (instance, ann) in enumerate(zip(data_info['instances'], ann_info)):
            if 'score' in ann:
                instance['score'] = ann['score']
                count_with_score += 1
        
        #print(f"Added scores to {count_with_score}/{len(data_info['instances'])} instances")
        
        return data_info