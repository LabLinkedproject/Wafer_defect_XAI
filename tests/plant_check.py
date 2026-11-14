"""
tests/plant_check.py
실제 스캔 맵에 인공 결함(검은 원)을 심어 검출 감도를 잰다. 학습·평가가 아니라 전처리 조정용 점검 도구다.
실행: OMP_NUM_THREADS=4 taskset -c 4-7 ./venv/bin/python -m tests.plant_check <스캔 폴더> [밝기 배율 0.35]
"""
import sys

import cv2
import numpy as np

import wafer_detect as WD
from wafer_model import die_layout

PLAN = [(10, (8, 8), 5), (11, (-8, 8), 5), (12, (8, -8), 5), (13, (-8, -8), 5), (17, (8, 8), 4), (20, (-8, 8), 4),
        (25, (8, -8), 5), (28, (-8, -8), 5), (33, (8, 8), 5), (34, (-8, -8), 4.5)]


def main():
    cv2.setNumThreads(4)
    d = sys.argv[1].rstrip("/") + "/"
    level = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
    meta = WD.load_meta(d)
    bgr = cv2.imread(d + meta["map"])
    seen = cv2.imread(d + meta["mask"], 0)
    s = meta["px_per_gantry_mm"]
    cx, cy = meta["wafer"]["center_px"]
    dies = {x["id"]: x for x in die_layout()}
    img = bgr.copy()
    for i, (dx, dy), dia in PLAN:
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(m, (int(cx + (dies[i]["x_mm"] + dx) * s), int(cy - (dies[i]["y_mm"] + dy) * s)), int(dia / 2 * s), 1, -1)
        img[m > 0] = (img[m > 0] * level).astype(np.uint8)
    for name, src in (("원본", bgr), ("인공 결함", img)):
        gray, prof = WD.detection_channel(src, meta)
        r = WD.detect_array(gray, meta, seen=seen, min_area_mm2=prof["min_area_mm2"], sharp_z=prof["sharp_z"],
                            dark_only=prof["dark_only"])
        hit = sorted({x["die"] for x in r["defects"]})
        planted = [p[0] for p in PLAN if p[0] in r["inspected"]]
        print(f"{name}: 검사 {len(r['inspected'])}, 검출 다이 {hit}, 심은 다이 중 검사됨 {planted}, "
              f"잡음 {r['noise']['sharp_sigma']:.3f}")


if __name__ == "__main__":
    main()
