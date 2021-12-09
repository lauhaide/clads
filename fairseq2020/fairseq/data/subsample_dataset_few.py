# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np

from . import BaseWrapperDataset


logger = logging.getLogger(__name__)


class FewSubsampleDataset(BaseWrapperDataset):
    """Subsamples a given dataset by a specified ratio. Subsampling is done on the number of examples

            Args:
                dataset (~torch.utils.data.Dataset): dataset to subsample
                size_ratio(float): the ratio to subsample to. must be between 0 and 1 (exclusive)
            """

    def __init__(self, dataset, indices, size_ratio, few_inst):
        super().__init__(dataset)

        self.actual_size = len(indices)
        self.indices = indices
        logger.info(
            "subsampled dataset from {} to {} (ratio={})".format(
                len(self.dataset), self.actual_size, size_ratio
            )
        )

        self.few_instance = self.indices[:min(few_inst, len(self.indices))] if few_inst else []
        if few_inst:
            print(len(self.few_instance), few_inst)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def isFewInstance(self, index):
        return (self.few_instance == []) or self.indices[index] in self.few_instance

    def __len__(self):
        return self.actual_size

    def collater(self, samples):
        return self.dataset.collater(samples)

    @property
    def sizes(self):
        return self.dataset.sizes[self.indices]

    @property
    def name(self):
        return self.dataset.name

    def num_tokens(self, index):
        return self.dataset.num_tokens(self.indices[index])

    def size(self, index):
        return self.dataset.size(self.indices[index])

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    def prefetch(self, indices):
        self.dataset.prefetch(self.indices[indices])
