import numpy as np
from typing import Sequence, List
from torch.utils.data import BatchSampler, Sampler
from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class RatioBalancedBatchSampler(BatchSampler):
    """BatchSampler that ensures a fixed ratio between labeled and pseudo-labeled datasets.
    Oversamples smaller datasets to maintain the ratio throughout the entire epoch.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        ratio_per_dataset (List[float]): Ratio of samples from each dataset. Must sum to 1.
        drop_last (bool): If ``True``, drop incomplete batches.
    """

    def __init__(self,
                 sampler: Sampler,
                 batch_size: int,
                 ratio_per_dataset: List[float] = [0.5, 0.5],
                 drop_last: bool = False) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                          f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                           f'but got batch_size={batch_size}')
        
        super().__init__(sampler, batch_size, drop_last)
        
        # Validate dataset count matches ratios
        num_datasets = len(getattr(self.sampler.dataset, 'cumulative_sizes', [1]))
        if len(ratio_per_dataset) != num_datasets:
            raise ValueError(f'Expected {num_datasets} ratios, got {len(ratio_per_dataset)}')
        
        # Validate ratios
        if not np.isclose(sum(ratio_per_dataset), 1.0):
            raise ValueError(f"Ratios must sum to 1, got {sum(ratio_per_dataset)}")
            
        # Calculate samples per dataset
        self.samples_per_dataset = [int(batch_size * r) for r in ratio_per_dataset]
        # Handle rounding error
        self.samples_per_dataset[0] += batch_size - sum(self.samples_per_dataset)
        
        print(f"\nInitialized RatioBalancedBatchSampler:")
        print(f"Batch size: {batch_size}")
        print(f"Ratios: {ratio_per_dataset}")
        print(f"Samples per dataset: {self.samples_per_dataset}")
        
        # Initialize buckets for each dataset
        self._dataset_buckets = [[] for _ in range(len(ratio_per_dataset))]

    def __iter__(self) -> Sequence[int]:
        import random
        
        # Get all indices first and sort them by dataset
        dataset_indices = [[] for _ in range(len(self.samples_per_dataset))]
        
        for idx in self.sampler:
            dataset_idx = self._get_dataset_index(idx)
            dataset_indices[dataset_idx].append(idx)
        
        # Store the original indices of each dataset for oversampling
        original_indices = [indices.copy() for indices in dataset_indices]
        
        print(f"\nDataset sizes: {[len(indices) for indices in dataset_indices]}")
        
        # Calculate expected number of batches based on the larger dataset
        largest_dataset_idx = max(range(len(dataset_indices)), key=lambda i: len(dataset_indices[i]) / self.samples_per_dataset[i] if self.samples_per_dataset[i] > 0 else 0)
        largest_dataset_size = len(dataset_indices[largest_dataset_idx])
        samples_needed_per_batch = self.samples_per_dataset[largest_dataset_idx]
        
        if samples_needed_per_batch > 0:
            total_batches = largest_dataset_size // samples_needed_per_batch
            if not self.drop_last and largest_dataset_size % samples_needed_per_batch > 0:
                total_batches += 1
        else:
            total_batches = 0
            
        print(f"Expected to generate {total_batches} batches")
        
        batch_count = 0
        while batch_count < total_batches:
            batch = []
            
            # Check if any dataset needs oversampling for this batch
            for dataset_idx in range(len(dataset_indices)):
                samples_needed = self.samples_per_dataset[dataset_idx]
                if samples_needed > 0 and len(dataset_indices[dataset_idx]) < samples_needed:
                    # Need to oversample this dataset for the current batch
                    remaining = samples_needed - len(dataset_indices[dataset_idx])
                    
                    # If we've used all original samples, reshuffle and use them again
                    if len(original_indices[dataset_idx]) > 0:
                        # Shuffle to avoid repeating the same order
                        additional_indices = original_indices[dataset_idx].copy()
                        random.shuffle(additional_indices)
                        # Take only what we need for this batch
                        dataset_indices[dataset_idx].extend(additional_indices[:remaining])
                    
                    # If we still need more after adding all available, cycle through again
                    while len(dataset_indices[dataset_idx]) < samples_needed:
                        additional_needed = samples_needed - len(dataset_indices[dataset_idx])
                        if len(original_indices[dataset_idx]) <= additional_needed:
                            # Add all and continue if we need more
                            dataset_indices[dataset_idx].extend(original_indices[dataset_idx])
                        else:
                            # Add just what we need after shuffling
                            shuffle_indices = original_indices[dataset_idx].copy()
                            random.shuffle(shuffle_indices)
                            dataset_indices[dataset_idx].extend(shuffle_indices[:additional_needed])
                            break
            
            # Form batch from all datasets according to ratio
            for dataset_idx in range(len(dataset_indices)):
                samples_needed = self.samples_per_dataset[dataset_idx]
                if samples_needed > 0:
                    batch.extend(dataset_indices[dataset_idx][:samples_needed])
                    dataset_indices[dataset_idx] = dataset_indices[dataset_idx][samples_needed:]
            
            if len(batch) == self.batch_size:
                yield batch
            elif not self.drop_last and len(batch) > 0:
                if batch_count == total_batches - 1 or len(batch) >= self.batch_size // 2:
                    yield batch
            
            batch_count += 1

    def _get_dataset_index(self, idx: int) -> int:
        """Determine which dataset an index belongs to."""
        dataset_idx = 0
        for cum_size in self.sampler.dataset.cumulative_sizes:
            if idx < cum_size:
                break
            dataset_idx += 1
        return dataset_idx

    def __len__(self) -> int:
        """Calculate the number of batches based on the largest dataset and its ratio."""
        max_batches = 0
        for dataset_idx, samples_needed in enumerate(self.samples_per_dataset):
            if samples_needed == 0:
                continue
            dataset_size = self._get_dataset_size(dataset_idx)
            possible_batches = dataset_size // samples_needed
            if not self.drop_last and dataset_size % samples_needed > 0:
                possible_batches += 1
            max_batches = max(max_batches, possible_batches)
        return max_batches

    def _get_dataset_size(self, dataset_idx: int) -> int:
        """Get the size of a specific dataset."""
        if dataset_idx == 0:
            return self.sampler.dataset.cumulative_sizes[0]
        return (self.sampler.dataset.cumulative_sizes[dataset_idx] - 
                self.sampler.dataset.cumulative_sizes[dataset_idx - 1])





# @DATA_SAMPLERS.register_module()
# class RatioBalancedBatchSampler(BatchSampler):
#     """BatchSampler that ensures a fixed ratio between labeled and pseudo-labeled datasets.

#     Args:
#         sampler (Sampler): Base sampler.
#         batch_size (int): Size of mini-batch.
#         ratio_per_dataset (List[float]): Ratio of samples from each dataset. Must sum to 1.
#         drop_last (bool): If ``True``, drop incomplete batches.
#     """

#     def __init__(self,
#                  sampler: Sampler,
#                  batch_size: int,
#                  ratio_per_dataset: List[float] = [0.5, 0.5],
#                  drop_last: bool = False) -> None:
#         if not isinstance(sampler, Sampler):
#             raise TypeError('sampler should be an instance of ``Sampler``, '
#                           f'but got {sampler}')
#         if not isinstance(batch_size, int) or batch_size <= 0:
#             raise ValueError('batch_size should be a positive integer value, '
#                            f'but got batch_size={batch_size}')
        
#         super().__init__(sampler, batch_size, drop_last)
        
#         # Validate dataset count matches ratios
#         num_datasets = len(getattr(self.sampler.dataset, 'cumulative_sizes', [1]))
#         if len(ratio_per_dataset) != num_datasets:
#             raise ValueError(f'Expected {num_datasets} ratios, got {len(ratio_per_dataset)}')
        
#         # Validate ratios
#         if not np.isclose(sum(ratio_per_dataset), 1.0):
#             raise ValueError(f"Ratios must sum to 1, got {sum(ratio_per_dataset)}")
            
#         # Calculate samples per dataset
#         self.samples_per_dataset = [int(batch_size * r) for r in ratio_per_dataset]
#         # Handle rounding error
#         self.samples_per_dataset[0] += batch_size - sum(self.samples_per_dataset)
        
#         print(f"\nInitialized RatioBalancedBatchSampler:")
#         print(f"Batch size: {batch_size}")
#         print(f"Ratios: {ratio_per_dataset}")
#         print(f"Samples per dataset: {self.samples_per_dataset}")
        
#         # Initialize buckets for each dataset
#         self._dataset_buckets = [[] for _ in range(len(ratio_per_dataset))]

#     def __iter__(self) -> Sequence[int]:
#         # Get all indices first and sort them by dataset
#         dataset_indices = [[] for _ in range(len(self.samples_per_dataset))]
        
#         for idx in self.sampler:
#             dataset_idx = self._get_dataset_index(idx)
#             dataset_indices[dataset_idx].append(idx)
        
#         #print(f"\nDataset sizes: {[len(indices) for indices in dataset_indices]}")
        
#         # Keep track of original target sizes
#         original_targets = self.samples_per_dataset.copy()
        
#         while any(len(indices) > 0 for indices in dataset_indices):
#             batch = []
#             available_datasets = [i for i, indices in enumerate(dataset_indices) if len(indices) > 0]
            
#             if not available_datasets:
#                 break
                
#             if len(available_datasets) == len(dataset_indices):
#                 # All datasets available - use original ratios
#                 target_sizes = original_targets
#             else:
#                 # Some datasets exhausted - redistribute among available ones
#                 samples_per_available = self.batch_size // len(available_datasets)
#                 target_sizes = [samples_per_available if i in available_datasets else 0 
#                             for i in range(len(dataset_indices))]
            
#             # Form batch from available datasets
#             for dataset_idx in range(len(dataset_indices)):
#                 if dataset_idx not in available_datasets:
#                     continue
                    
#                 available = len(dataset_indices[dataset_idx])
#                 target = min(target_sizes[dataset_idx], available)
                
#                 if target > 0:
#                     batch.extend(dataset_indices[dataset_idx][:target])
#                     dataset_indices[dataset_idx] = dataset_indices[dataset_idx][target:]
            
#             if len(batch) == 0:
#                 break
                
#             # Only yield batch if it's complete or if it's large enough
#             if len(batch) == self.batch_size:
#                 #print(f"Yielding batch of size {len(batch)}")
#                 yield batch
#             elif not self.drop_last and len(batch) >= self.batch_size // 2:
#                 # Only yield incomplete batches if they're at least half the target size
#                 #print(f"Yielding final batch of size {len(batch)}")
#                 yield batch

#     def _get_dataset_index(self, idx: int) -> int:
#         """Determine which dataset an index belongs to."""
#         dataset_idx = 0
#         for cum_size in self.sampler.dataset.cumulative_sizes:
#             if idx < cum_size:
#                 break
#             dataset_idx += 1
#         return dataset_idx

#     def __len__(self) -> int:
#         # Calculate based on the dataset that will exhaust first
#         if self.drop_last:
#             min_batches = float('inf')
#             for dataset_idx, samples_needed in enumerate(self.samples_per_dataset):
#                 if samples_needed == 0:
#                     continue
#                 dataset_size = self._get_dataset_size(dataset_idx)
#                 possible_batches = dataset_size // samples_needed
#                 min_batches = min(min_batches, possible_batches)
#             return min_batches
#         else:
#             return (len(self.sampler) + self.batch_size - 1) // self.batch_size

#     def _get_dataset_size(self, dataset_idx: int) -> int:
#         """Get the size of a specific dataset."""
#         if dataset_idx == 0:
#             return self.sampler.dataset.cumulative_sizes[0]
#         return (self.sampler.dataset.cumulative_sizes[dataset_idx] - 
#                 self.sampler.dataset.cumulative_sizes[dataset_idx - 1])