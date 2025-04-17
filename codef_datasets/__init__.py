from .distributed_weighted_sampler import DistributedWeightedSampler
from .video_dataset import VideoDataset,VideoDatasets

dataset_dict = {'video': VideoDataset,'videos':VideoDatasets}

custom_sampler_dict = {'weighted': DistributedWeightedSampler}