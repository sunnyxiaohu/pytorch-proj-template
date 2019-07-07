import os
import logging
import shutil
import torch
from datetime import datetime
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import RandomSampler, BatchSampler
import torch.distributed as dist
import math
import numpy as np

class GivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, last_iter=-1):

        world_size = 1
        rank = 0
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter*self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter+1)*self.batch_size:])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):

        np.random.seed(0)

        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size-1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)
        beg = self.total_size * self.rank
        indices = indices[beg:beg+self.total_size]

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        return self.total_size

class DistributedGivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=-1):
        if world_size is None:
            world_size = dist.get_world_size()
        else:
            world_size = 1
        if rank is None:
            rank = dist.get_rank()
        else:
            rank = 0
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter*self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter+1)*self.batch_size:])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):

        # each process shuffle all list with same seed, and pick one piece according to rank
        np.random.seed(0)

        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size-1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)
        beg = self.total_size * self.rank
        indices = indices[beg:beg+self.total_size]

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        #return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size


class DistributedSampler(Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset.

    Note:
        Dataset is assumed to be of constant size.

    Arguments:
        dataset (Dataset): dataset used for sampling.
        num_replicas (int): number of processes participating in distributed training, optional.
        rank (int): rank of the current process within num_replicas, optional.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(torch.randperm(len(self.dataset), generator=g))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class TestDistributedSampler(Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset.

    Note:
        Do not align the total size to be divisible by world_size.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = len(range(rank, len(self.dataset), num_replicas))
        self.total_size = len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        # g = torch.Generator()
        # g.manual_seed(self.epoch)
        # indices = list(torch.randperm(len(self.dataset), generator=g))
        indices = torch.arange(len(self.dataset))

        # subsample
        indices = indices[self.rank::self.num_replicas]
        # offset = self.num_samples * self.rank
        # indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.

    Arguments:
        sampler (Sampler): base sampler.
        group_ids (list): group id of each image.
        batch_size (int): size of mini-batch.
        training (bool): if True, training mode.
    """

    def __init__(self, sampler, group_ids, batch_size, training=True):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids)
        assert self.group_ids.dim() == 1
        self.batch_size = batch_size
        self.training = training

        self.groups = torch.unique(self.group_ids).sort(0)[0]
        self._can_reuse_batches = False

    def _prepare_batches(self):
        sampled_ids = torch.as_tensor(list(self.sampler))
        sampled_group_ids = self.group_ids[sampled_ids]
        clusters = [sampled_ids[sampled_group_ids == i] for i in self.groups]
        target_batch_num = int(np.ceil(len(sampled_ids) / float(self.batch_size)))

        splits = [c.split(self.batch_size) for c in clusters]
        merged = tuple(itertools.chain.from_iterable(splits))

        # re-permuate the batches by the order that
        # the first element of each batch occurs in the original sampled_ids
        first_element_of_batch = [t[0].item() for t in merged]
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch])
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        batches = [merged[i].tolist() for i in permutation_order]

        if self.training:
            # ensure number of batches in different gpus are the same
            if len(batches) > target_batch_num:
                batches = batches[:target_batch_num]
            assert len(batches) == target_batch_num, "Error, uncorrect target_batch_num!"
        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        return iter(batches)

    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)


class IterationBasedBatchSampler(BatchSampler):
    """Wraps a BatchSampler, resampling a specified number of iterations"""

    def __init__(self, batch_sampler, num_iterations=None, start_iter=None):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        assert self.start_iter is not None, "must set start iter before training"
        assert self.num_iterations is not None, "must set total number of iterations before training"

        iteration = self.start_iter
        len_dataset = len(self.batch_sampler)
        while iteration < self.num_iterations:
            if hasattr(self.batch_sampler.sampler, 'set_epoch'):
                self.batch_sampler.sampler.set_epoch(iteration // len_dataset)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations

    def set_start_iter(self, start_iter):
        self.start_iter = start_iter

    def set_num_iterations(self, num_iterations):
        self.num_iterations = num_iterations

