import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import json

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from dataset import get_data, dataset_dict
from evaluation_utils import build_dataset_result_paths, ensure_result_directories, save_metrics_summary
from method import AdaCLIP_Trainer
from tools import Logger, setup_seed, write2csv

setup_seed(111)


def build_model(args, device):
    config_path = os.path.join('./model_configs', f'{args.model}.json')
    with open(config_path, 'r') as f:
        model_configs = json.load(f)

    n_layers = model_configs['vision_cfg']['layers']
    substage = n_layers // 4
    features_list = [substage, substage * 2, substage * 3, substage * 4]

    model = AdaCLIP_Trainer(
        backbone=args.model,
        feat_list=features_list,
        input_dim=model_configs['vision_cfg']['width'],
        output_dim=model_configs['embed_dim'],
        learning_rate=0.,
        device=device,
        image_size=args.image_size,
        prompting_depth=args.prompting_depth,
        prompting_length=args.prompting_length,
        prompting_branch=args.prompting_branch,
        prompting_type=args.prompting_type,
        use_hsf=args.use_hsf,
        k_clusters=args.k_clusters,
    ).to(device)
    model.load(args.ckt_path)
    return model


def log_args(logger, args):
    for key, value in sorted(vars(args).items()):
        logger.info(f'{key} = {value}')


def evaluate_dataset(args, model):
    assert args.testing_data in dataset_dict.keys(), (
        f"You entered {args.testing_data}, but we only support {dataset_dict.keys()}"
    )

    result_paths = build_dataset_result_paths(args.save_path, args.ckt_path, args.testing_data)
    ensure_result_directories(result_paths)

    logger = Logger(result_paths['log_path'])
    log_args(logger, args)
    logger.info(f"run_name = {result_paths['run_name']}")
    logger.info(f"checkpoint = {args.ckt_path}")

    test_data_cls_names, test_data, test_data_root = get_data(
        dataset_type_list=args.testing_data,
        transform=model.preprocess,
        target_transform=model.transform,
        training=False,
    )
    logger.info(f"test_data_root = {test_data_root}")

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    metric_dict = model.evaluation(
        test_dataloader,
        test_data_cls_names,
        args.save_fig,
        result_paths['image_dir'],
    )

    for tag, data in metric_dict.items():
        logger.info(
            '{:>15} \t\tI-Auroc:{:.2f} \tI-F1:{:.2f} \tI-AP:{:.2f} \tP-Auroc:{:.2f} \tP-F1:{:.2f} \tP-AP:{:.2f}'.format(
                tag,
                data['auroc_im'],
                data['f1_im'],
                data['ap_im'],
                data['auroc_px'],
                data['f1_px'],
                data['ap_px'],
            )
        )

    for tag in metric_dict.keys():
        write2csv(metric_dict[tag], test_data_cls_names, tag, result_paths['csv_path'])

    save_metrics_summary(
        summary_path=result_paths['summary_path'],
        checkpoint_path=args.ckt_path,
        dataset_name=args.testing_data,
        metric_dict=metric_dict,
    )

    logger.info(f"csv_path = {result_paths['csv_path']}")
    logger.info(f"summary_path = {result_paths['summary_path']}")
    if args.save_fig:
        logger.info(f"image_dir = {result_paths['image_dir']}")


def evaluate_image(args, model):
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV (cv2) is required for single-image visualization. Install opencv-python first."
        ) from exc

    assert os.path.isfile(args.image_path), f"Please verify the input image path: {args.image_path}"

    ori_image = cv2.resize(cv2.imread(args.image_path), (args.image_size, args.image_size))
    pil_img = Image.open(args.image_path).convert('RGB')

    img_input = model.preprocess(pil_img).unsqueeze(0).to(model.device)

    with torch.no_grad():
        anomaly_map, anomaly_score = model.clip_model(img_input, [args.class_name], aggregation=True)

    anomaly_map = anomaly_map[0, :, :].cpu().numpy()
    anomaly_score = anomaly_score[0].cpu().numpy()

    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
    anomaly_map = (anomaly_map * 255).astype(np.uint8)

    heat_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    vis_map = cv2.addWeighted(heat_map, 0.5, ori_image, 0.5, 0)
    vis_map = cv2.hconcat([ori_image, vis_map])

    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, args.save_name)
    print(f"Anomaly detection results are saved in {save_path}, with an anomaly of {anomaly_score:.3f}")
    cv2.imwrite(save_path, vis_map)


def evaluate(args):
    assert os.path.isfile(args.ckt_path), (
        f"Please check the path of pre-trained model, {args.ckt_path} is not valid."
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(args, device)

    if args.testing_model == 'dataset':
        evaluate_dataset(args, model)
    elif args.testing_model == 'image':
        evaluate_image(args, model)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AdaCLIP", add_help=True)

    parser.add_argument(
        "--ckt_path",
        type=str,
        default='weights/pretrained_mvtec_colondb.pth',
        help="Path to the pre-trained model (default: weights/pretrained_mvtec_colondb.pth)",
    )
    parser.add_argument(
        "--testing_model",
        type=str,
        default="dataset",
        choices=["dataset", "image"],
        help="Model for testing (default: 'dataset')",
    )
    parser.add_argument("--testing_data", type=str, default="visa", help="Dataset for testing (default: 'visa')")
    parser.add_argument("--image_path", type=str, default="asset/img.png", help="Testing image path")
    parser.add_argument("--class_name", type=str, default="candle", help="Class name for single-image testing")
    parser.add_argument("--save_name", type=str, default="test.png", help="Output file name for image mode")
    parser.add_argument("--save_path", type=str, default='./workspaces', help="Directory to save results")
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-L-14-336",
        choices=["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336"],
        help="The CLIP model to be used (default: 'ViT-L-14-336')",
    )
    parser.add_argument("--save_fig", type=str2bool, default=False, help="Save figures for visualizations")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--image_size", type=int, default=518, help="Size of the input images (default: 518)")
    parser.add_argument("--prompting_depth", type=int, default=4, help="Depth of prompting (default: 4)")
    parser.add_argument("--prompting_length", type=int, default=5, help="Length of prompting (default: 5)")
    parser.add_argument(
        "--prompting_type",
        type=str,
        default='SD',
        choices=['', 'S', 'D', 'SD'],
        help="Type of prompting. 'S' for Static, 'D' for Dynamic, 'SD' for both (default: 'SD')",
    )
    parser.add_argument(
        "--prompting_branch",
        type=str,
        default='VL',
        choices=['', 'V', 'L', 'VL'],
        help="Branch of prompting. 'V' for Visual, 'L' for Language, 'VL' for both (default: 'VL')",
    )
    parser.add_argument("--use_hsf", type=str2bool, default=True, help="Use HSF for aggregation")
    parser.add_argument("--k_clusters", type=int, default=20, help="Number of clusters (default: 20)")

    args = parser.parse_args()

    if args.batch_size != 1:
        raise NotImplementedError(
            "Currently, only batch size of 1 is supported due to unresolved bugs. Please set --batch_size to 1."
        )

    evaluate(args)
