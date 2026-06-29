import json
import os


def checkpoint_stem(checkpoint_path):
    return os.path.splitext(os.path.basename(checkpoint_path))[0]


def build_dataset_result_paths(save_root, checkpoint_path, dataset_name):
    run_name = f'{checkpoint_stem(checkpoint_path)}-{dataset_name}'
    run_root = os.path.join(save_root, 'eval', run_name)

    return {
        'run_name': run_name,
        'run_root': run_root,
        'csv_path': os.path.join(run_root, 'metrics.csv'),
        'log_path': os.path.join(run_root, 'log.txt'),
        'image_dir': os.path.join(run_root, 'images'),
        'summary_path': os.path.join(run_root, 'summary.json'),
    }


def ensure_parent_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def ensure_result_directories(paths):
    os.makedirs(paths['run_root'], exist_ok=True)
    os.makedirs(paths['image_dir'], exist_ok=True)


def _to_builtin(value):
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if hasattr(value, 'item'):
        return value.item()
    return value


def save_metrics_summary(summary_path, checkpoint_path, dataset_name, metric_dict):
    ensure_parent_dir(summary_path)
    payload = {
        'checkpoint_path': checkpoint_path,
        'dataset_name': dataset_name,
        'average': _to_builtin(metric_dict.get('Average', {})),
        'metrics': _to_builtin(metric_dict),
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
