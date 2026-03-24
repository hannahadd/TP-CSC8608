import numpy as np
import cv2
from pathlib import Path

from sam_utils import load_sam_predictor, predict_mask_from_box

# prendre une image jpg/png
imgs = list(Path("TP1/data/images").glob("*.jpg")) + list(Path("TP1/data/images").glob("*.png"))
img_path = imgs[0]

bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

ckpt = "TP1/models/sam_vit_b_01ec64.pth"
pred = load_sam_predictor(ckpt, model_type="vit_b")

box = np.array([50, 50, 250, 250], dtype=np.int32)

mask, score = predict_mask_from_box(pred, rgb, box, multimask=True)
print("img", rgb.shape, "mask", mask.shape, "score", score, "mask_sum", int(mask.sum()))
