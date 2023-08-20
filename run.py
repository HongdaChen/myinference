import os
import torch
import cv2
import numpy as np
import json
import tqdm

from psdet.utils.config import get_config
from psdet.models.builder import build_model

def convert2real(x1, y1, x2, y2, angle1, angle2):
    x1 *= 512
    y1 *= 512
    x2 *= 512
    y2 *= 512
    angle1 = angle1 * np.pi
    angle2 = angle2 * np.pi
    return x1, y1, x2, y2, angle1, angle2

def main():
    cfg = get_config()
    model = build_model(cfg.model)
    model.load_params_from_file(filename=cfg.ckpt, to_cpu=False)
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        img_dir = '/work/data/visual-parking-space-line-recognition-test-set/'
        json_dir = '/work/output/'
        # img_dir = r'E:\workspace\comp\xunfei\ps2.0\ps2\testing\imgs'
        # json_dir = r'.\output'
        
        img_list = os.listdir(img_dir)
        pbar = tqdm.tqdm(total=len(img_list), desc='Total process')
        for img_name in img_list:
            pbar.update()
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
            
            name = img_name.split('.')[0]
            json_name = name + '.json'
            json_path = os.path.join(json_dir, json_name)
            with open(json_path, 'w') as f:
                json.dump(slots_dict, f)
        
        
if __name__ == '__main__':
    main()