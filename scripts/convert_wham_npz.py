#!/usr/bin/env python3
"""
WHAM 결과(pkl)에서 pose·trans 만 뽑아 npz 로 저장합니다.

apply_wham_pose.py 가 읽는 형식으로 맞춰 주는 다리 역할입니다. 그동안 수동으로
하던 변환을 스크립트로 고정했습니다.

  python convert_wham_npz.py \
      --pkl ~/data/04_wham/zombie_sample1/wham_output.pkl \
      --out ~/data/06_rig/zombie_sample1/wham_pose.npz

출력 npz 키:
  pose   (T, 72)  프레임별 24관절 axis-angle
  trans  (T, 3)   루트 이동

pkl 에 트랙이 여러 개면 첫 번째를 쓰고 경고합니다 (--track 으로 지정 가능).
트랙이 2개 이상이라는 건 WHAM 이 다른 물체를 사람으로 잡았다는 뜻이라,
그대로 진행하기 전에 overlay.mp4 를 확인하는 편이 좋습니다.
"""
import argparse
import os
import sys

from pathlib import Path

import numpy as np

try:
    import joblib
except ImportError:
    sys.exit("joblib 이 없습니다. 설치: pip3 install joblib")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="WHAM 출력 wham_output.pkl")
    ap.add_argument("--out", required=True, help="출력 wham_pose.npz")
    ap.add_argument("--track", default=None,
                    help="트랙 키 지정 (생략 시 첫 번째)")
    return ap.parse_args()


def main():
    a = parse_args()
    data = joblib.load(a.pkl)
    if not data:
        raise RuntimeError(f"트랙이 비어 있습니다: {a.pkl}")

    keys = list(data.keys())
    if a.track is not None:
        key = a.track if a.track in data else type(keys[0])(a.track)
        if key not in data:
            raise RuntimeError(f"트랙 {a.track} 없음. 있는 트랙: {keys}")
    else:
        key = keys[0]
        if len(keys) > 1:
            print(f"[!] 트랙이 {len(keys)}개입니다 {keys} — {key} 를 사용합니다. "
                  "overlay.mp4 로 사람을 제대로 잡았는지 확인하세요.")
    track = data[key]

    for k in ("pose", "trans"):
        if k not in track:
            raise RuntimeError(f"pkl 에 '{k}' 가 없습니다. 있는 키: {list(track)}")
    pose = np.asarray(track["pose"])
    trans = np.asarray(track["trans"])
    if pose.shape[0] != trans.shape[0]:
        raise RuntimeError(f"프레임 수 불일치: pose {pose.shape} vs trans {trans.shape}")
    if pose.shape[-1] != 72:
        raise RuntimeError(f"pose 형상이 (T,72) 가 아닙니다: {pose.shape}")

    out_path = Path(os.path.expanduser(a.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)   # 규약 경로가 아직 없을 수 있다
    np.savez(out_path, pose=pose, trans=trans)
    print(f"저장: {a.out}")
    print(f"  track={key}  pose {pose.shape}  trans {trans.shape}  ({pose.shape[0]}프레임)")


if __name__ == "__main__":
    main()
