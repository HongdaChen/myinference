import os
import torch
import cv2
import numpy as np
import json
from tqdm import tqdm
from psdet.utils.config import get_config
from psdet.models.builder import build_model


def convert2real(x1, y1, x2, y2, angle1, angle2):
    x1 *= 512
    y1 *= 512
    x2 *= 512
    y2 *= 512
    angle1 = (angle1 * 360) / 180 * np.pi
    angle2 = (angle2 * 360) / 180 * np.pi
    return x1, y1, x2, y2, angle1, angle2

def main():
    cfg = get_config()
    model = build_model(cfg.model)
    model.load_params_from_file(filename=cfg.ckpt, to_cpu=False)
    model.cuda()
    model.eval()
    if 1==1:
        with torch.no_grad():
            img_dir = '/work/data/visual-parking-space-line-recognition-test-set/'
            json_dir = '/work/output/'
            # img_dir = r'E:\workspace\comp\xunfei\car\datasets\train\testing'
            # json_dir = r'.\output'
            img_list = os.listdir(img_dir)
            none_det = []
            for img_name in tqdm(img_list, desc="my-inference-model"):
                img_path = os.path.join(img_dir, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (512, 512))
                img = img / 255.
                
                data_dict = {}
                data_dict['image'] = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).cuda()
                pred_dicts, ret_dict = model(data_dict)  # points_angle_pred_batch, slots_pred
                
                slots_pred = pred_dicts['slots_pred'][0]
                slots_list = []
                for slot in slots_pred:
                    scores, (x1, y1, x2, y2, angle1, angle2) = slot
                    x1, y1, x2, y2, angle1, angle2 = convert2real(x1, y1, x2, y2, angle1, angle2)
                    slot_dict = {
                        "points": [[x1, y1], [x2, y2]],
                        "angle1": angle1,
                        "angle2": angle2,
                        "scores": scores.item(),
                    }
                    slots_list.append(slot_dict)
                slots_dict = {}
                slots_dict["slot"] = slots_list
                if len(slots_list) == 0:
                    none_det.append(img_name)
                    print(f"{img_name} detect nothing!")
                name = img_name.split('.')[0]
                json_name = name + '.json'
                json_path = os.path.join(json_dir, json_name)
                with open(json_path, 'w') as f:
                    json.dump(slots_dict, f)
        print(f"nothing detected in these img: {none_det}, in total: {len(none_det)}")
    else:
        with torch.no_grad():
            json_dir = '/work/output/'
            img_list = ['image20160722192751_3720.jpg', 'p2_img25_0528.jpg', 'img7_1557.jpg', 'image20160722192751_3612.jpg', 'image20160722192751_3604.jpg', 'img8_1740.jpg', 'img5_6867.jpg', 'img5_6213.jpg', 'p2_img10_0126.jpg', 'p2_img49_3156.jpg', 'image20160722192751_3748.jpg', 'image20160722192751_3316.jpg', 'p2_img29_1788.jpg', 'p2_img83_3870.jpg', 'img5_7023.jpg', 'p2_img53_0054.jpg', 'image20160722193621_1316.jpg', 'p2_img104_0150.jpg', 'img5_7521.jpg', 'p2_img11_0468.jpg', 'p2_img51_3618.jpg', 'p2_img34_1938.jpg', 'image20160722192751_3728.jpg', 'image20160722192751_2960.jpg', 'p2_img118_0312.jpg', 'p2_img21_2292.jpg', 'image20160722192751_3192.jpg', 'p2_img27_3312.jpg', 'image20160722192751_3248.jpg', '20161111-03-362.jpg', 'img9_4872.jpg', 'p2_img8_0294.jpg', 'img9_1290.jpg', 'image20160722192751_2376.jpg', 'img5_7032.jpg', 'p2_img118_0288.jpg', 'image20160722192751_1076.jpg', 'p2_img26_0384.jpg', 'image20160725142318_1116.jpg', 'img9_6435.jpg', 'image20160725152215_252.jpg', 'img5_9111.jpg', 'img5_6990.jpg', '20161019-2-653.jpg', 'image20160722192751_1776.jpg', 'p2_img116_0114.jpg', 'image20160725142318_1156.jpg', 'p2_img21_2214.jpg', 'p2_img15_0732.jpg', 'p2_img118_0420.jpg', 'p2_img119_0348.jpg', 'image20160722192751_3732.jpg', 'image20160722192751_3736.jpg', 'image20160722192751_3608.jpg', 'image20160725142318_1112.jpg', 'image20160722192751_3752.jpg', 'p2_img21_1452.jpg', 'image20160722192751_3488.jpg', 'img5_6996.jpg', 'p2_img15_1686.jpg', 'p2_img25_0552.jpg', 'img5_6957.jpg', 'p2_img19_0852.jpg', 'p2_img118_0480.jpg', 'img5_6963.jpg', 'image20160722192751_3480.jpg', 'image20160725151308_308.jpg', 'p2_img52_1878.jpg', 'image20160722192751_2920.jpg', 'p2_img34_0546.jpg', 'img5_6987.jpg', 'p2_img31_0090.jpg', 'p2_img28_4140.jpg', 'p2_img119_0306.jpg', 'p2_img23_0582.jpg', 'image20160722192751_3496.jpg', 'p2_img28_0330.jpg', 'p2_img12_0234.jpg', 'image20160722192751_3600.jpg', 'img5_6999.jpg', 'p2_img86_3972.jpg', 'img9_1158.jpg', 'p2_img52_2004.jpg', 'p2_img25_0516.jpg', 'p2_img118_0078.jpg', 'img5_6900.jpg', 'img5_7005.jpg', 'p2_img12_0702.jpg', 'img5_6972.jpg', 'image20160722192751_4752.jpg', 'image20160725143627_560.jpg', 'p2_img26_0372.jpg', 'image20160722192751_1780.jpg', 'p2_img23_0648.jpg', 'image20160722192751_1740.jpg', 'p2_img117_0516.jpg', 'p2_img27_3330.jpg', 'image20160722192751_3208.jpg', 'p2_img18_0294.jpg', 'image20160722192751_4500.jpg', 'image20160725142318_1152.jpg', 'image20160722193621_860.jpg', 'p2_img23_0654.jpg', 'p2_img23_1638.jpg', 'p2_img118_0042.jpg', 'p2_img15_0552.jpg', 'image20160722192751_3320.jpg', 'p2_img26_2748.jpg', 'p2_img10_0114.jpg', 'image20160722192751_3484.jpg', 'p2_img117_0594.jpg', 'p2_img13_0444.jpg', 'p2_img118_0258.jpg', 'p2_img21_2712.jpg', 'image20160722192751_3740.jpg', 'img7_0387.jpg', 'p2_img15_0534.jpg', 'img5_7011.jpg', 'p2_img25_0510.jpg', 'p2_img23_0600.jpg', 'img9_1254.jpg', 'image20160722193621_972.jpg', 'img3_5574.jpg']
            img_dir = r"/workspace/ParkingSlotDetection/data_comp/train/imgs"
            none_det = []
            for img_name in tqdm(img_list, desc="my-inference-model"):
                img_path = os.path.join(img_dir, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (512, 512))
                img = img / 255.
                
                data_dict = {}
                data_dict['image'] = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).cuda()
                pred_dicts, ret_dict = model(data_dict)  # points_angle_pred_batch, slots_pred
                
                slots_pred = pred_dicts['slots_pred'][0]
                slots_list = []
                for slot in slots_pred:
                    scores, (x1, y1, x2, y2, angle1, angle2) = slot
                    x1, y1, x2, y2, angle1, angle2 = convert2real(x1, y1, x2, y2, angle1, angle2)
                    slot_dict = {
                        "points": [[x1, y1], [x2, y2]],
                        "angle1": angle1,
                        "angle2": angle2,
                        "scores": scores.item(),
                    }
                    slots_list.append(slot_dict)
                slots_dict = {}
                slots_dict["slot"] = slots_list
                if len(slots_list) == 0:
                    none_det.append(img_name)
                    print(f"{img_name} detect nothing!")
                name = img_name.split('.')[0]
                json_name = name + '.json'
                json_path = os.path.join(json_dir, json_name)
                with open(json_path, 'w') as f:
                    json.dump(slots_dict, f)
        print(f"nothing detected in these img: {none_det}, in total: {len(none_det)}")
       
        
if __name__ == '__main__':
    main()