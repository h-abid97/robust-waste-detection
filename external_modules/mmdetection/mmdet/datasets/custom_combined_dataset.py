from mmdet.registry import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmengine.dataset import force_full_init


class CombinedDataset(CocoDataset):
    """A dataset that combines labeled and pseudo-labeled data."""

    def __init__(self,
                 labeled_ann_file,
                 pseudo_ann_file,
                 data_root=None,
                 data_prefix=dict(img=''),
                 pipeline=None,
                 **kwargs):
        print(f"Initializing CombinedDataset with:")
        print(f"data_root: {data_root}")
        print(f"data_prefix: {data_prefix}")
        print(f"labeled_ann_file: {labeled_ann_file}")
        print(f"pseudo_ann_file: {pseudo_ann_file}")

        # Use the labeled annotation file for the parent init
        kwargs['ann_file'] = labeled_ann_file
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            **kwargs
        )

        print("After parent initialization:")
        print(f"Parent data_list length: "
              f"{len(super().data_list) if hasattr(super(), 'data_list') else 'No data_list'}")

        # Load labeled data
        print("Loading labeled data...")
        self.labeled_data = self.load_data_list()
        self.labeled_size = len(self.labeled_data)
        print(f"Loaded {self.labeled_size} labeled samples")

        # Load pseudo-labeled data
        print("Loading pseudo-labeled data...")
        original_ann_file = self.ann_file
        self.ann_file = pseudo_ann_file
        self.pseudo_data = self.load_data_list()
        self.ann_file = original_ann_file
        print(f"Loaded {len(self.pseudo_data)} pseudo-labeled samples")

        # Combine the labeled and pseudo-labeled data
        print("Before combining datasets:")
        print(f"Current data_list length: "
              f"{len(self.data_list) if hasattr(self, 'data_list') else 'No data_list'}")

        self.data_list = self.labeled_data + self.pseudo_data

        print("After combining datasets:")
        print(f"Updated data_list length: {len(self.data_list)}")

        # Mark as fully initialized so that the framework doesn't attempt a re-init
        self._fully_initialized = True

    def __len__(self):
        """Get dataset length."""
        if not hasattr(self, 'data_list'):
            print("WARNING: data_list not found in __len__")
            return 0
        length = len(self.data_list)
        print(f"Dataset length called: {length}")
        return length

    def prepare_data(self, idx):
        """Prepare data for training."""
        if not hasattr(self, 'data_list'):
            print("WARNING: data_list not found in prepare_data")
            raise RuntimeError("Dataset not properly initialized")
            
        data = self.data_list[idx].copy()
        
        # Add source information
        data['is_pseudo'] = 1 if idx >= self.labeled_size else 0
        
        # Ensure required fields exist with default values
        data.setdefault('img_path', '')
        data.setdefault('img_id', -1)
        data.setdefault('instances', [])
        
        # Apply pipeline transformation
        if self.pipeline is not None:
            data = self.pipeline(data)
            
        return data

    def __getitem__(self, idx):
        """Get dataset item."""
        if idx >= len(self.data_list):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data_list)}")
        
        return self.prepare_data(idx)


@DATASETS.register_module()
class CombinedDatasetWrapper(CombinedDataset):
    """Wrapper to register CombinedDataset with MMDetection."""
    pass