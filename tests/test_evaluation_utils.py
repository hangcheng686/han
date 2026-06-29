import json
import os
import tempfile
import unittest

from evaluation_utils import build_dataset_result_paths, save_metrics_summary


class EvaluationUtilsTests(unittest.TestCase):
    def test_build_dataset_result_paths_groups_outputs_by_checkpoint_and_dataset(self):
        paths = build_dataset_result_paths(
            save_root='workspaces',
            checkpoint_path=r'workspaces\models\demo_best.pth',
            dataset_name='visa_chewinggum_split',
        )

        self.assertEqual(paths['run_name'], 'demo_best-visa_chewinggum_split')
        self.assertTrue(paths['csv_path'].endswith(os.path.join('eval', 'demo_best-visa_chewinggum_split', 'metrics.csv')))
        self.assertTrue(paths['log_path'].endswith(os.path.join('eval', 'demo_best-visa_chewinggum_split', 'log.txt')))
        self.assertTrue(paths['image_dir'].endswith(os.path.join('eval', 'demo_best-visa_chewinggum_split', 'images')))
        self.assertTrue(paths['summary_path'].endswith(os.path.join('eval', 'demo_best-visa_chewinggum_split', 'summary.json')))

    def test_save_metrics_summary_writes_json_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = os.path.join(tmp_dir, 'summary.json')
            metric_dict = {
                'Average': {
                    'auroc_im': 71.3,
                    'f1_im': 83.12,
                    'ap_im': 82.64,
                    'auroc_px': 49.64,
                    'f1_px': 1.12,
                    'ap_px': 0.56,
                }
            }

            save_metrics_summary(
                summary_path=summary_path,
                checkpoint_path=r'workspaces\models\demo_best.pth',
                dataset_name='visa_chewinggum_split',
                metric_dict=metric_dict,
            )

            with open(summary_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)

        self.assertEqual(payload['checkpoint_path'], r'workspaces\models\demo_best.pth')
        self.assertEqual(payload['dataset_name'], 'visa_chewinggum_split')
        self.assertEqual(payload['average']['f1_px'], 1.12)


if __name__ == '__main__':
    unittest.main()
