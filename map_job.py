"""
map_job.py
웨이퍼 맵 합성과 결함 검출을 별도 프로세스로 실행한다. 웹 서버는 이 프로세스를 띄우고 진행 줄을 읽는다.

합성은 한 번에 수백 MB 를 쓰고, 같은 프로세스에서 되풀이하면 메모리를 다 돌려주지 않는다.
보드(3.8 GB)에서 웹 서버가 합성 세 번 만에 1.5 GB 를 넘겨 강제 종료된 뒤(2026-11-03) 따로 돌리게 했다.

출력 줄: 'phase <stitch|detect>', 'log <메시지>', 실패하면 'error <메시지>' 와 종료 코드 1.
실행: ./venv/bin/python map_job.py <스캔 폴더> [--flip-y] [--scale-hint=<px/갠트리 mm>] [--turn-hint=<도>]
"""
import os
import sys

# 작은 코어(A55) 4개로 제한한다. 큰 코어(A76)로 계산을 시작하는 순간마다 보드가 멈췄다
# (2026-11-14 16:28·16:39·17:03, 큰 코어 4개로 줄여도 같음, 온도·메모리는 정상 → 순간 전력 부족 의심).
# 작은 코어는 전력이 훨씬 적다. 합성이 2~3배 느려지는 대신 멈추지 않게 한다
CORES = {0, 1, 2, 3}
THREADS = 4
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, str(THREADS))


def say(kind: str, msg: str = ""):
    print(f"{kind} {msg}", flush=True)


def main() -> int:
    scan_dir = sys.argv[1]
    try:
        os.sched_setaffinity(0, CORES & os.sched_getaffinity(0) or os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    try:
        import cv2
        cv2.setNumThreads(THREADS)
        import wafer_stitch
        from adc import scan_input
        say("phase", "stitch")
        hint = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--scale-hint=")), None)
        turn = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--turn-hint=")), None)
        wafer_stitch.build(scan_dir, log=lambda m: say("log", m), flip_y="--flip-y" in sys.argv, scale_hint=hint,
                           turn_hint=turn)
        say("phase", "detect")
        say("log", "결함 검출")
        try:
            scan_input.detect(os.path.basename(scan_dir.rstrip("/")))   # 분석 화면이 바로 쓰도록 미리 계산
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
