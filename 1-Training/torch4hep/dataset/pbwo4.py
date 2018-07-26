from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from torch4hep.dataset.base import BaseTreeDataset

import ROOT

import numpy as np


class PbWO4Dataset(BaseTreeDataset):
    def __init__(self, path, tree_name="crystal", transform=None):
        super(PbWO4Dataset, self).__init__(path, tree_name, transform)

    def __getitem__(self, idx):
        self._tree.GetEntry(idx)

        energy_deposit = np.array(self._tree.energy_deposit, dtype=np.float32)
        energy_deposit = energy_deposit.reshape(1, 9, 9)

        total_energy = np.float32(self._tree.total_energy)

        example = dict()
        example["energy_deposit"] = energy_deposit
        example["total_energy"] = total_energy
        return example

