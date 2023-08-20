import json
import math
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms as T

from psdet.datasets.base import BaseDataset
from psdet.datasets.registry import DATASETS
from psdet.utils.precision_recall import calc_average_precision, calc_precision_recall

from .process_data import boundary_check, overlap_check, rotate_centralized_marks, rotate_image, generalize_marks
from .utils import match_marking_points, match_slots 

@DATASETS.register
class ParkingSlotDataset(BaseDataset):
    def __init__(self, cfg, logger=None):
        super(ParkingSlotDataset, self).__init__(cfg=cfg, logger=logger)

        assert self.root_path.exists()

        # if cfg.mode == 'train':
        #     data_dir = self.root_path / 'annotations' / 'train'
        # elif cfg.mode == 'val':
        #     data_dir = self.root_path / 'annotations' / 'test'
        if cfg.mode == "train":
            data_dir = self.root_path / "json"
        elif cfg.mode == 'val':
            data_dir = self.root_path / "json"
        assert data_dir.exists()

        self.json_files = [p for p in data_dir.glob("*.json")]
        self.json_files.sort()

        if cfg.mode == "train":
            # data augmentation
            self.image_transform = T.Compose(
                [
                    T.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                    ),
                    T.ToTensor(),
                ]
            )
        else:
            self.image_transform = T.Compose([T.ToTensor()])

        if self.logger:
            self.logger.info(
                "Loading PSV {} dataset with {} samples".format(
                    cfg.mode, len(self.json_files)
                )
            )

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_file = Path(self.json_files[idx])
        # load label
        with open(str(json_file), "r") as f:
            data = json.load(f)

        labels = np.array(data["slot"])
        marks = []
        slots = []
        mark_id = 1
        for label in labels:
            category = int(label["category"])
            point1 = label["points"][0]
            point2 = label["points"][1]
            angle1 = int(label["angle1"] / np.pi * 180)
            angle2 = int(label["angle2"] / np.pi * 180)
            ignore = label["ignore"]
            vacant = label["vacant"]
            marks.append(point1 + [angle1])
            marks.append(point2 + [angle2])
            slots.append([mark_id, mark_id + 1, category, angle1])
            mark_id += 2
        # marks[i] = (x, y, angle)
        # slots[i] = (id_p1, id_p2, category, angle)
        marks = np.array(marks)
        slots = np.array(slots)
        # print(f"marks.shape: {marks.shape}   slots.shape: {slots.shape}")

        assert slots.size > 0
        if len(marks.shape) < 2:
            marks = np.expand_dims(marks, axis=0)
        if len(slots.shape) < 2:
            slots = np.expand_dims(slots, axis=0)

        num_points = marks.shape[0]
        max_points = self.cfg.max_points
        # assert max_points >= num_points
        if max_points < num_points:  # 只要前max_points个点
            # 从slots配对表中删去带有max_points外的点的配对
            slots = slots[slots[:, 0] <= max_points]
            slots = slots[slots[:, 1] <= max_points]
            # 从marks点表中删去max_points外的点
            marks = marks[:max_points, :]
            num_points = marks.shape[0]
            print("max_points: ", max_points)
            print("num_points: ", num_points)

        # centralize (image size = 512 x 512)
        marks[:, 0:2] -= 256.5

        img_file = (
            str(self.json_files[idx]).replace(".json", ".jpg").replace("json", "imgs")
        )
        image = Image.open(img_file)
        image = image.resize((512, 512), Image.BILINEAR)

        marks = generalize_marks(marks)  # 点坐标归一化到 [0, 1]
        image = self.image_transform(image)

        # make sample with the max num points
        # 为了输入对齐，这里需要做填充
        marks_full = np.full((max_points, marks.shape[1]), 0.0, dtype=np.float32)
        marks_full[:num_points] = marks
        # match_targets 是给后面点配对做监督数据用的
        # match_targets[ver1][0]=ver2
        # match_targets[ver1][1]=angle
        match_targets = np.full((max_points, 2), -1, dtype=np.int32)

        for slot in slots:
            match_targets[slot[0] - 1, 0] = slot[1] - 1
            match_targets[slot[0] - 1, 1] = slot[3]  # 之前是0

        input_dict = {
            "marks": marks_full,  # 一张图下的所有点坐标 (max_points, 3) -> (x, y, angle)
            "match_targets": match_targets,  # 一张图下的所有点配对 (max_points, 2) -> (ver2, angle)
            "npoints": num_points,  # max_points下的有效点个数
            "frame_id": idx,  # item索引，没有
            "image": image,  # 输入图像
        }

        return input_dict

    def generate_prediction_dicts(self, batch_dict, pred_dicts):
        pred_list = []
        pred_slots = pred_dicts['pred_slots']
        for i, slots in enumerate(pred_slots):
            single_pred_dict = {}
            single_pred_dict['frame_id'] = batch_dict['frame_id'][i]
            single_pred_dict['slots'] = slots
            pred_list.append(single_pred_dict)
        return pred_list
     
    def evaluate_point_detection(self, predictions_list, ground_truths_list):
        precisions, recalls = calc_precision_recall(
            ground_truths_list, predictions_list, match_marking_points)
        average_precision = calc_average_precision(precisions, recalls)
        self.logger.info('precesions:')
        self.logger.info(precisions[-5:])
        self.logger.info('recalls:')
        self.logger.info(recalls[-5:])
        self.logger.info('Point detection: average_precision {}'.format(average_precision))

    def evaluate_slot_detection(self, predictions_list, ground_truths_list):
                
        precisions, recalls = calc_precision_recall(
            ground_truths_list, predictions_list, match_slots)
        average_precision = calc_average_precision(precisions, recalls)

        self.logger.info('precesions:')
        self.logger.info(precisions[-5:])
        self.logger.info('recalls:')
        self.logger.info(recalls[-5:])
        self.logger.info('Slot detection: average_precision {}'.format(average_precision))
