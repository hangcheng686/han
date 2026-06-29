import importlib.util
import os
import sys
import unittest


class OptionalCv2ImportTests(unittest.TestCase):
    def test_tools_visualization_imports_without_cv2(self):
        sys.modules.pop('cv2', None)

        repo_root = os.path.dirname(os.path.dirname(__file__))
        module_path = os.path.join(repo_root, 'tools', 'visualization.py')
        spec = importlib.util.spec_from_file_location('visualization_under_test', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertTrue(hasattr(module, 'plot_sample_cv2'))
        self.assertTrue(callable(module.plot_sample_cv2))


if __name__ == '__main__':
    unittest.main()
