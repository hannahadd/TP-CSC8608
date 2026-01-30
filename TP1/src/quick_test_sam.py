import numpy as np
import cv2
from pathlib import Path

from sam_utils import load_sam_predictor, predict_mask_from_box
from geom_utils import mask_area, mask_bbox, mask_perimeter
from viz_utils import render_overlay


def main():
    # Choisir une image (jpg ou png)
    imgs = list(Path("TP1/data/images").glob("*.jpg")) + list(Path("TP1/data/images").glob("*.png"))
    if len(imgs) == 0:
        raise FileNotFoundError("Aucune image trouvée dans TP1/data/images (jpg/png).")
    img_path = imgs[0]

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Image illisible: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # SAM (local: vit_b recommandé)
    ckpt = "TP1/models/sam_vit_b_01ec64.pth"
    pred = load_sam_predictor(ckpt, model_type="vit_b")

    # bbox à la main (à ajuster si besoin)
    box = np.array([50, 50, 250, 250], dtype=np.int32)

    mask, score = predict_mask_from_box(pred, rgb, box, multimask=True)

    m_area = mask_area(mask)
    m_bbox = mask_bbox(mask)
    m_per = mask_perimeter(mask)

    overlay = render_overlay(rgb, mask, box, alpha=0.5)

    out_dir = Path("TP1/outputs/overlays")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"overlay_{img_path.stem}.png"

    # Sauvegarde via OpenCV (BGR)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("image", img_path.name)
    print("score", score, "area", m_area, "bbox", m_bbox, "perimeter", m_per)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
