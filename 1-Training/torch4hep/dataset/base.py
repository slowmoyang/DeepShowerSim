from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from torch.utils.data import Dataset
from torchvision import transforms

import ROOT


class BaseTreeDataset(Dataset):
    def __init__(self, path, tree_name, transform=None):
        self._root_file = ROOT.TFile.Open(path, "READ")
        self._tree = self._root_file.Get(tree_name)
        # TODO check if everything is fine


        self.transform = transform
        
        self._path = path
        self._tree_name = tree_name

    def __len__(self):
        return int(self._tree.GetEntries())

    def __getitem__(self, idx):
        raise NotImplementedError
