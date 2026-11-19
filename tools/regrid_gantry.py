"""이미 합성한 맵의 격자 정보를 갠트리 기준(wafer_stitch.gantry_grid)으로 다시 쓴다. 영상은 그대로 둔다.
11-18 실측 정리: 격자 측정 실패(분석 불가) 네 장과 피치가 3 % 어긋난 두 장을 맵을 다시 합성하지 않고 고친다.
웨이퍼 원이 사진에서 잡힌 맵(from_gantry 아님)은 원 자체가 틀렸을 수 있어 건너뛴다 — 다시 합성해야 한다.
실행: ./venv/bin/python tools/regrid_gantry.py [스캔 ...]   (인자 없으면 전부)"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wafer_model import WAFER_MODEL
import wafer_stitch as W


def regrid(path: str) -> str:
    with open(path) as f:
        m = json.load(f)
    w = m.get("wafer") or {}
    if not w.get("from_gantry"):
        return "건너뜀(사진에서 잡은 원)"
    old = m.get("grid")
    s = w["radius_px"] / w["radius_gantry_mm"]                  # 맵 px / 갠트리 mm
    g = W.gantry_grid({"r": w["radius_px"], "from_gantry": True},
                      {"angle_deg": old["angle_deg"], "phase": old.get("phase", {}), "lines": old.get("lines")} if old else None)
    m["grid"] = {"angle_deg": g["angle_deg"], "pitch_px": round(g["pitch_px"], 1), "phase": g["phase"], "lines": g["lines"],
                 "pitch_gantry_mm": round(g["pitch_px"] / s, 2), "pitch_mm": WAFER_MODEL["pitch_mm"], "from_gantry": True}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return f"피치 {old['pitch_px'] if old else '없음'} → {m['grid']['pitch_px']} px, 기울기 {m['grid']['angle_deg']}"


if __name__ == "__main__":
    ids = sys.argv[1:] or [os.path.basename(os.path.dirname(p)) for p in glob.glob("captures/scans/*_scan/wafer_map.json")]
    for sid in sorted(ids):
        print(sid, regrid(os.path.join("captures", "scans", sid, "wafer_map.json")))
