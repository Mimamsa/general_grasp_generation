import os, sys
import time
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
import imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask
import yaml
from MatchAnything.SAM6D.SAM6D.Instance_Segmentation_Model.utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from MatchAnything.SAM6D.SAM6D.Instance_Segmentation_Model.utils.bbox_utils import ResizePad, xyxy_to_xywh
from MatchAnything.SAM6D.SAM6D.Instance_Segmentation_Model.model.utils import Detections, convert_npz_to_json
from MatchAnything.SAM6D.SAM6D.Instance_Segmentation_Model.model.loss import Similarity
from MatchAnything.SAM6D.SAM6D.Instance_Segmentation_Model.utils.inout import load_json, save_json_bop23

inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )


def visualize(rgb, detections, save_path):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    best_score = 0.
    for mask_idx, det in enumerate(detections):
        if best_score < det['score']:
            best_score = det['score']
            best_det = detections[mask_idx]

    mask = rle_to_mask(best_det["segmentation"])
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    obj_id = best_det["category_id"]
    temp_id = obj_id - 1

    r = int(255*colors[temp_id][0])
    g = int(255*colors[temp_id][1])
    b = int(255*colors[temp_id][2])
    img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
    img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
    img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
    img[edge, :] = 255
    
    # ===== 把原圖中物體區域抓出來 =====
    rgb_np = np.array(rgb).copy()
    object_only = rgb_np.copy()
    object_only[~mask] = 0 
    #  取得 mask 的 bounding box
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # +1 是為了包含該點   
    object_only_cropped = object_only[y0:y1, x0:x1]
    object_only_img = Image.fromarray(object_only_cropped)
    object_only_img.save(save_path+"/object_only.png")

    fat_mask = binary_dilation(mask, np.ones((args.fat_mask, args.fat_mask)))
    mask_binary = (fat_mask > 0).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_binary).convert("1")  # ➜ mode "1" = 1-bit
    mask_img.save(save_path + "/object_mask.png")

    img = Image.fromarray(np.uint8(img))
    img.save(save_path+"/vis_ism.png")
    prediction = Image.open(save_path+"/vis_ism.png")
    
    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

def batch_input_data(depth_path, cam_path, device):
    batch = {}
    cam_info = load_json(cam_path)
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam_info['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

def run_inference(segmentor_model, output_dir, template_img, rgb_path, stability_score_thresh, exp_name):
    start_time = time.time()
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='run_inference.yaml')

    if segmentor_model == "sam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')
        cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    elif segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')
    else:
        raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))

    logging.info("Initializing model")
    model = instantiate(cfg.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")
    init_complet = time.time()
    print(f"完成init model time: {init_complet - start_time:.6f}秒")
    #改完輸入多張圖片，沒額外mask來crop，只做resize and padding
    logging.info("Initializing template")

    # transform 應該根據圖片再多加設計
    gallery_transform = T.Compose([
    T.CenterCrop((512, 512)),     
    ])

    extensions = ['png', 'jpg', 'jpeg']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(template_img, f"*.{ext}")))
    image_paths.sort()  
    templates = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image = gallery_transform(image)
        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        templates.append(image)
    templates = torch.stack(templates)  # shape: [N, H, W, 3]
    templates = templates.permute(0, 3, 1, 2)  # shape: [N, 3, H, W]
    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    proposal_processor = ResizePad(processing_config.image_size)
    templates = proposal_processor(images=templates).to(device)

    model.ref_data = {}
    model.ref_data["descriptors"] = model.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
    model.ref_data["appe_descriptors"] = model.descriptor_model.compute_patch_feature(
                    templates).unsqueeze(0).data

    
    # run inference
    rgb = Image.open(rgb_path).convert("RGB")
    detections = model.segmentor_model.generate_masks(np.array(rgb))
    detections = Detections(detections)
    # descriptor_model = DINOv2
    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(np.array(rgb), detections)

    # matching descriptors
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = model.compute_semantic_score(query_decriptors)

    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]
    # print("best_template = ", best_template.shape)        # 維度(n), proposal 數量
    # print("pred_idx_objects = ", pred_idx_objects.shape)  # 維度(n), proposal 數量
    # print("query_appe_descriptors = ", query_appe_descriptors.shape)  # 維度(n, 256, 1024), proposal 數量, patches 數量, patches 維度

    # compute the appearance score
    appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

    final_score = (semantic_score + appe_scores) / (2)
    match_complet = time.time()
    print(f"完成matching: {match_complet - init_complet:.6f}秒")

    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))   
         
    detections.to_numpy()
    save_path = f"{output_dir}/{exp_name}/detection_ism"
    detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
    detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
    save_json_bop23(save_path+".json", detections)
    vis_img = visualize(rgb, detections, f"{output_dir}/{exp_name}")
    vis_img.save(f"{output_dir}/{exp_name}/vis_match_anything.png")
    vis_complet = time.time()
    print(f"完成視覺化: {vis_complet - match_complet:.6f}秒")
    print(f"完成全部流程: {vis_complet - start_time:.6f}秒")

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='MA_config.yaml', help='Path to config file')
    cli_args = parser.parse_args()

    # 讀取 YAML
    with open(cli_args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 可以把 dict 轉成 Namespace，如果你想用 args.xxx 風格
    return argparse.Namespace(**config)

class ObjectQueryHelper():
    def __init__(
        self,
        config,
        device=None
    ):
        self.config = config
        self.segmentor_model = config.segmentor_model
        self.gallery_path = config.template_img
        self.stability_score_thresh = config.stability_score_thresh
        self.gallery_transform = T.Compose([
            T.CenterCrop((512, 512)),     
        ])
        self.matching_thresh = config.matching_thresh
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._load_model()
        self._register_gallery()

    def _load_model(self):
        with initialize(version_base=None, config_path="configs"):
            self.cfg = compose(config_name='run_inference.yaml')

        if self.segmentor_model == "sam":
            with initialize(version_base=None, config_path="configs/model"):
                self.cfg.model = compose(config_name='ISM_sam.yaml')
            self.cfg.model.segmentor_model.stability_score_thresh = self.stability_score_thresh
        elif self.segmentor_model == "fastsam":
            with initialize(version_base=None, config_path="configs/model"):
                self.cfg.model = compose(config_name='ISM_fastsam.yaml')
        else:
            raise ValueError("The segmentor_model {} is not supported now!".format(self.segmentor_model))
        
        logging.info("Initializing model")
        print(self.cfg.model)
        self.model = instantiate(self.cfg.model)
        
        self.model.descriptor_model.model = self.model.descriptor_model.model.to(self.device)
        self.model.descriptor_model.model.device = self.device
        # if there is predictor in the model, move it to device
        if hasattr(self.model.segmentor_model, "predictor"):
            self.model.segmentor_model.predictor.model = (
                self.model.segmentor_model.predictor.model.to(self.device)
            )
        else:
            self.model.segmentor_model.model.setup_model(device=self.device, verbose=True)

    def _register_gallery(self):
        extensions = ['png', 'jpg', 'jpeg']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(self.gallery_path, f"*.{ext}")))
        image_paths.sort()  
        templates = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            image = self.gallery_transform(image)
            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            templates.append(image)

        templates = torch.stack(templates)  # shape: [N, H, W, 3]
        templates = templates.permute(0, 3, 1, 2)  # shape: [N, 3, H, W]
        processing_config = OmegaConf.create(
            {
                "image_size": 224,
            }
        )
        proposal_processor = ResizePad(processing_config.image_size)
        templates = proposal_processor(images=templates).to(self.device)

        self.model.ref_data = {}
        self.model.ref_data["descriptors"] = self.model.descriptor_model.compute_features(
                        templates, token_name="x_norm_clstoken"
                    ).unsqueeze(0).data
        self.model.ref_data["appe_descriptors"] = self.model.descriptor_model.compute_patch_feature(
                        templates).unsqueeze(0).data
        
    def reset_gallery(self, gallery_path:str):
        self.gallery_path = gallery_path
        self._register_gallery()

    def query(self, image:np.ndarray):
        detections = self.model.segmentor_model.generate_masks(image)
        detections = Detections(detections)
        # descriptor_model = DINOv2
        query_decriptors, query_appe_descriptors = self.model.descriptor_model.forward(image, detections)

        # matching descriptors
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = self.model.compute_semantic_score(query_decriptors)

        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]
        appe_scores, ref_aux_descriptor= self.model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

        final_score = (semantic_score + appe_scores) / (2)
        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", torch.zeros_like(final_score))
        detections.to_numpy()

        scores = detections.scores
        indices = np.where(scores >= self.matching_thresh)[0]

        object_ids = detections.object_ids[indices]
        scores = scores[indices]
        masks = detections.masks[indices]
        unique_ids = np.unique(object_ids)
        # 挑出個category中分數最高分的query

        results = []
        for u_id in unique_ids:
            object_indices = np.where(object_ids==u_id)[0]
            object_scores = scores[object_indices]
            object_masks = masks[object_indices]
            
            
            best_score_id = np.argmax(object_scores)
            best_score = object_scores[best_score_id]
            best_mask = object_masks[best_score_id]
            
            # 擴張mask, Thinking: 如果物件很靠近, 有可能切到其他物件
            best_mask = binary_dilation(best_mask[0], np.ones((self.config.fat_mask, self.config.fat_mask)))
            best_cand = {
                "category": u_id,
                "scores": best_score,
                "mask": best_mask
            }
            results.append(best_cand)

        return results
        
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--segmentor_model", default='fastsam', help="The segmentor model in ISM")
    # parser.add_argument("--output_dir", default='/workspace/SAM-6D/SAM-6D/Instance_Segmentation_Model/Example/outputs', nargs="?", help="Path to root directory of the output")
    # parser.add_argument("--template_img", default='/workspace/SAM-6D/SAM-6D/Instance_Segmentation_Model/Example/inputs/template/ele', nargs="?", help="Path to root directory of the template image")
    # parser.add_argument("--rgb_path", default='/workspace/SAM-6D/SAM-6D/Instance_Segmentation_Model/Example/inputs/target_img/ele_rgbd_camera.png', nargs="?", help="Path to RGB image")
    # parser.add_argument("--exp_name", default="rgbd_component")
    # parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
    # args = parser.parse_args()
    args = load_config()
    # os.makedirs(f"{args.output_dir}/{args.exp_name}", exist_ok=True)
    # run_inference(
    #     args.segmentor_model, args.output_dir, args.template_img, args.rgb_path, 
    #     stability_score_thresh=args.stability_score_thresh,exp_name=args.exp_name,
    # )
    rgb_path = "Example/inputs/target_img/ele_on_table.jpg"
    rgb = Image.open(rgb_path).convert("RGB")
    rgb = np.array(rgb)
    object_query_helper = ObjectQueryHelper(args)
    object_query_helper.query(rgb)
    