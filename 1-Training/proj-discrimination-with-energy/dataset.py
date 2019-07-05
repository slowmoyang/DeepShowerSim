from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
sys.path.append("..")

from torch4hep.datasets.base import BaseTreeDataset

import ROOT

import numpy as np


class PbWO4Dataset(BaseTreeDataset):
    def __init__(self, path, tree_name="crystal"):
        super(PbWO4Dataset, self).__init__(path, tree_name)

    def __getitem__(self, idx):
        self._tree.GetEntry(idx)

        energy_deposit = np.array(self._tree.energy_deposit, dtype=np.float32)
        energy_deposit = energy_deposit.reshape(1, 9, 9)

        total_energy = np.float32(self._tree.total_energy)

        example = dict()
        example["energy_deposit"] = energy_deposit
        example["total_energy"] = total_energy
        return example


if __name__ == "__main__":
    dset = PbWO4Dataset("/data/slowmoyang/DeepShowerSim/PbWO4_positron_uniform-1-100.root")
    example = dset[0]
    for key, value in example.iteritems():
        print(key, value.shape, value.dtype)
