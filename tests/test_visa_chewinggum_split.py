import sys
import types
import unittest


def install_torch_stub():
    if 'torch' in sys.modules:
        return
    torch = types.ModuleType('torch')
    utils = types.ModuleType('torch.utils')
    data = types.ModuleType('torch.utils.data')

    class Dataset:
        pass

    class ConcatDataset:
        def __init__(self, datasets):
            self.datasets = datasets

    data.Dataset = Dataset
    data.ConcatDataset = ConcatDataset
    utils.data = data
    torch.utils = utils

    sys.modules['torch'] = torch
    sys.modules['torch.utils'] = utils
    sys.modules['torch.utils.data'] = data


install_torch_stub()

if 'cv2' not in sys.modules:
    sys.modules['cv2'] = types.ModuleType('cv2')

from dataset import dataset_dict


class VisaChewinggumSplitTests(unittest.TestCase):
    def test_visa_chewinggum_split_uses_official_train_test_split(self):
        cls_names, dataset_class, _ = dataset_dict['visa_chewinggum_split']

        train_ds = dataset_class(transform=None, target_transform=None, clsnames=cls_names, training=True)
        test_ds = dataset_class(transform=None, target_transform=None, clsnames=cls_names, training=False)

        self.assertEqual(cls_names, ['chewinggum'])
        self.assertEqual(len(train_ds), 453)
        self.assertEqual(len(test_ds), 150)

        train_paths = {item['img_path'] for item in train_ds.data_all}
        test_paths = {item['img_path'] for item in test_ds.data_all}
        self.assertTrue(train_paths.isdisjoint(test_paths))


if __name__ == '__main__':
    unittest.main()

