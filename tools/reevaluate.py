"""모든 스캔을 지금 코드로 다시 검출하고(캐시 갱신), 정답 기록이 있는 스캔은 AI 판정(judged)과 평가를 다시 남긴다.
정답 기록의 사용자 입력(다이·패턴·메모)은 바꾸지 않는다. 대시보드와 같은 판정 경로(adc.session)를 쓴다.
실행: OMP_NUM_THREADS=1 ./venv/bin/python tools/reevaluate.py [스캔 ...]"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adc import scan_input, session, truth


def judge(sid: str) -> list[str]:
    s = session.Session.__new__(session.Session)          # 최신 스캔을 자동으로 여는 생성자는 건너뛴다
    s._lock, s.scenario, s.target, s._cache = threading.RLock(), "donut", None, None
    s.load_scan(sid)
    return s.state()["labels"]


if __name__ == "__main__":
    ids = sys.argv[1:] or [x["id"] for x in reversed(scan_input.scans())]
    for sid in ids:
        try:
            labels = judge(sid)
        except Exception as e:                            # 분석 실패도 그대로 알린다
            print(sid, "분석 실패:", e, flush=True)
            continue
        t = truth.load(sid)
        if t is None:
            print(sid, "판정", labels, "(정답 기록 없음)", flush=True)
            continue
        t["judged"] = labels
        truth.save(sid, t)
        ev = truth.record(sid)
        print(sid, "판정", labels, "정답", t["patterns"], "평가", {k: ev[k] for k in ("tp", "fn", "fp", "tn", "pattern_ok")}, flush=True)
