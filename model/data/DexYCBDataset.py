import os
import cv2
import yaml
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

YCB_CLASSES = {
    1 : "002_master_chef_can",
    2 : "003_cracker_box",
    3 : "004_suagr_box",
    4 : "005_tomato_soup_can",
    5 : "006_mustard_bottle",
    6 : "007_tuna_fish_can",
    7 : "008_pudding_box",
    8 : "009_gelatin_box",
    9 : "010_potted_mest_can",
    10 : "011_banana",
    11 : "019_pitcher_base",
    12 : "021_bleach_cleanser", 
    13 : "024_bowl",
    14 : "025_mug",
    15 : "035_power_drill",
    16 : "036_wood_block",
    17 : "037_scissors",
    18 : "040_large_marker",
    19 : "051_large_large_clamp",
    20 : "052_extra_large_clamp",
    21 : "061_foam_brick"
}

class DexYCBDataset(Dataset):
    def __init__(self, split = 'train', transform=None):
        super().__init__()

        assert 'DEX_YCB_DIR' in os.environ, "environment variable 'DEX_YCB_DIR' is not set"
        self.root_dir = os.environ['DEX_YCB_DIR']
        self.transform = transform

        # 预加载内参字典
        self.intrinsics_cache = self._preload_intrinsics()

        # 扫描并构建样本列表
        self.samples = self._build_index(split)
        print(f"[{split}] Dataset initialized with {len(self.samples)} samples.")

    def _preload_intrinsics(self):
        """
        根据 calibration/intrinsics/ 下的文件名加载所有内参
        文件名格式: 836212060125_640x480.yml
        """
        cache = {}
        calib_path = os.path.join(self.root_dir, "calibration", "intrinsics")
        if not os.path.exists(calib_path):
            print("Warning: Calibration path not found, using identity matrix.")
            return cache

        for fname in os.listdir(calib_path):
            if fname.endswith(".yml") or fname.endswith(".yaml"):
                serial = fname.split('_')[0] 
                
                with open(os.path.join(calib_path, fname), 'r') as f:
                    data = yaml.load(f, Loader=yaml.FullLoader)
                    if 'color' in data:
                        cam_data = data['color']
                        fx = cam_data['fx']
                        fy = cam_data['fy']
                        cx = cam_data['ppx'] # ppx 对应 cx
                        cy = cam_data['ppy'] # ppy 对应 cy
                        
                        # 构建 3x3 内参矩阵 K
                        k = np.array([
                            [fx, 0,  cx],
                            [0,  fy, cy],
                            [0,  0,  1 ]
                        ], dtype=np.float32)
                        
                        cache[serial] = k
                    else:
                        print(f"Warning: 'color' field not found in {fname}")
                        
        return cache

    def _build_index(self, split):
        samples = []
        all_subjects = sorted([d for d in os.listdir(self.root_dir) if "subject" in d])
        if split == 'train':
            target_subjects = all_subjects[:-2]
        else:
            target_subjects = all_subjects[-2:]
        
        for subj in target_subjects:
            subj_dir = os.path.join(self.root_dir, subj)
            if not os.path.isdir(subj_dir): continue

            for seq in os.listdir(subj_dir):
                seq_dir = os.path.join(subj_dir, seq)
                if not os.path.isdir(seq_dir): continue
                
                # --- 读取 meta.yml ---
                meta_path = os.path.join(seq_dir, "meta.yml")
                if not os.path.exists(meta_path): continue
                with open(meta_path, 'r') as f:
                    meta = yaml.safe_load(f)
                
                # 获取该序列中包含的物体 ID 列表
                seq_ycb_ids = np.array(meta.get('ycb_ids', []))
                
                # 遍历该序列下的相机
                for cam_serial in os.listdir(seq_dir):
                    cam_dir = os.path.join(seq_dir, cam_serial)
                    if not os.path.isdir(cam_dir): continue
                    
                    label_files = sorted(glob.glob(os.path.join(cam_dir, "labels_*.npz")))
                    for label_path in label_files:
                        # 提取 frame_id: labels_000000.npz -> 000000
                        frame_id = os.path.basename(label_path).split('_')[1].split('.')[0]
                        
                        samples.append({
                            'img_path': os.path.join(cam_dir, f"color_{frame_id}.jpg"),
                            'depth_path': os.path.join(cam_dir, f"aligned_depth_to_color_{frame_id}.png"),
                            'label_path': label_path,
                            'cam_serial': cam_serial,
                            'ycb_ids': seq_ycb_ids,
                            'sample_name': f"{subj}_{seq}_{cam_serial}_{frame_id}"
                        })
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        info = self.samples[idx]

        img = cv2.imread(info['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        depth = cv2.imread(info['depth_path'], cv2.IMREAD_ANYDEPTH)
        if depth is None:
            depth = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        else:
            depth = depth.astype(np.float32) / 1000.0

        data = np.load(info['label_path'])
        seg = data['seg']
        pose_y = data['pose_y']
        ycb_ids = info['ycb_ids']
        obj_rots = pose_y[:, :3, :3]
        obj_trans = pose_y[:, :3, 3]
        pose_m = data['pose_m'].flatten() # [51]
        has_hand = not np.allclose(pose_m, 0)
        k3d = data['joint_3d'][0]
        k2d = data['joint_2d'][0]
        K = self.intrinsics_cache.get(info['cam_serial'], np.eye(3))

        # 转换为 Tensor
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)
        
        target = {
            'seg': torch.from_numpy(seg).long(),
            
            # 物体信息
            'ycb_ids': torch.from_numpy(ycb_ids).long(),     # [N_obj] -> [1, 5, 6, 15]
            'obj_rot': torch.from_numpy(obj_rots).float(),   # [N_obj, 3, 3]
            'obj_trans': torch.from_numpy(obj_trans).float(),# [N_obj, 3]
            
            # 手部信息
            'mano_pca': torch.from_numpy(pose_m[:48]).float(),
            'mano_trans': torch.from_numpy(pose_m[48:51]).float(),
            'has_hand': torch.tensor(1.0 if has_hand else 0.0),

            'joint_3d': torch.from_numpy(k3d),
            'joint_2d': torch.from_numpy(k2d),
            
            'K': torch.from_numpy(K).float(),
            'id': info['sample_name']
        }

        if self.transform:
            # apply transforms
            pass
            
        return img_tensor, depth_tensor, target

if __name__ == '__main__':
    data = DexYCBDataset()
    print(len(data))
    sample = data[0]
    img_tensor, depth_tensor, target = sample

    print(f"1. 图像 Tensor 形状: {img_tensor.shape}")  # 期望: [3, 480, 640]
    print(f"2. 深度 Tensor 形状: {depth_tensor.shape}") # 期望: [1, 480, 640]
    print(f"3. 包含的元数据 Keys: {list(target.keys())}")

    # 检查具体数值
    print(f"4. 当前帧物体 ID (ycb_ids): {target['ycb_ids']}")
    print(f"5. 是否有手 (has_hand): {target['has_hand'].item()}")
    print(f"6. 相机内参 (K):\n{target['K'].numpy()}\n")

    print(f"7. k3d形状: {target['joint_3d'].shape}")
    print(f"8. k2d形状: {target['joint_2d'].shape}")
