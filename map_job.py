"""
map_job.py
웨이퍼 맵 합성과 결함 검출을 별도 프로세스로 실행한다. 웹 서버는 이 프로세스를 띄우고 진행 줄을 읽는다.

합성은 한 번에 수백 MB 를 쓰고, 같은 프로세스에서 되풀이하면 메모리를 다 돌려주지 않는다.
보드(3.8 GB)에서 웹 서버가 합성 세 번 만에 1.5 GB 를 넘겨 강제 종료된 뒤(2026-11-03) 따로 돌리게 했다.

출력 줄: 'phase <stitch|detect>', 'log <메시지>', 실패하면 'error <메시지>' 와 종료 코드 1.
실행: ./venv/bin/python map_job.py <스캔 폴더> [--flip-y] [--flip-x] [--scale-hint=<px/갠트리 mm>] [--turn-hint=<도>]
"""
import os
import sys

# 맵 합성 CPU 사용은 설정 map_cpu 로 고른다 (scan_config.py 설명 참고). 전원이 약하면 계산이 몰리는 순간
# 전압이 떨어져 보드가 멈추거나 파일시스템이 읽기 전용으로 잠긴다(2026-11-14 노트북 USB 전원에서 세 번).
MODES = {"all": (None, None), "little4": ({0, 1, 2, 3}, 4), "little1": ({0}, 1)}


def _mode():
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import scan_config
        return MODES.get(scan_config.load().get("map_cpu"), MODES["all"])
    except Exception:
        return MODES["little4"]


CORES, THREADS = _mode()
if THREADS:
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_k, str(THREADS))


def say(kind: str, msg: str = ""):
    print(f"{kind} {msg}", flush=True)


def main() -> int:
    scan_dir = sys.argv[1]
    if CORES:
        try:
            os.sched_setaffinity(0, CORES & os.sched_getaffinity(0) or os.sched_getaffinity(0))
        except (AttributeError, OSError):
            pass
    try:
        import cv2
        if THREADS:
            cv2.setNumThreads(THREADS)
        import wafer_stitch
        from adc import scan_input
        say("phase", "stitch")
        hint = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--scale-hint=")), None)
        turn = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--turn-hint=")), None)
        wafer_stitch.build(scan_dir, log=lambda m: say("log", m), flip_y="--flip-y" in sys.argv,
                           flip_x="--flip-x" in sys.argv, scale_hint=hint,
                           turn_hint=turn)
        say("phase", "detect")
        say("log", "결함 검출")
        try:
            sid = os.path.basename(scan_dir.rstrip("/"))
            scan_input.detect(sid)                                      # 분석 화면이 바로 쓰도록 미리 계산
            from adc import truth
            truth.record(sid)                                           # 정답이 있으면 혼동행렬을 데이터로 남긴다
            from adc import overlay
            if overlay.build(sid):                                      # 템플릿 위에 검출 결함을 붙인 결과 화면
                say("log", "결과 화면(템플릿 + 검출) 생성")
        except Exception as e:                                          # 맵은 이미 저장됐다. 분석 화면이 다시 시도한다
            say("log", f"결함 검출 실패 ({e})")
    except Exception as e:
        say("error", f"{type(e).__name__}: {e}")
        return 1
    finally:
        os.sync()                     # 루트가 commit=600 이라 갑자기 꺼지면 결과 파일이 깨진다(2026-11-14)
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
