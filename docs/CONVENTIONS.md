# 경로·CLI 규약

파이프라인 스크립트가 공유하는 입출력 규약. 개별 스크립트의 사용법은 각 파일의
독스트링과 `--help`, 전체 실행 절차는 `docs/RUNBOOK.md` 를 본다.

## 1. `data/` 번호표

볼륨 루트는 `/workspace/data/`, 맥북 미러는 `~/data/` 다.

| 번호 | 폴더 | 단계 | 내용 |
|---|---|---|---|
| 00 | `00_raw` | G0 입력 | 원본 영상 |
| 01 | `01_pre` | S0 · S1 | fps/해상도/색공간 정규화 클립, 컷 분할 클립 |
| 02 | `02_sam2` | S2 | 마스크, 마스킹 클립, 키프레임 후보, `candidates.json` |
| 03 | `03_trellis` | S3 | GLB, `textures/`, `params.json`, 턴테이블 프리뷰 |
| 04 | `04_wham` | S5 | `wham_output.pkl`, `overlay.mp4` |
| 05 | `05_smpl_mesh` | S4 입력 | `smpl_frame<N>.obj`, `smpl_tpose.obj`, `joints_frame<N>.json`, `smpl_faces.npy` |
| 06 | `06_rig` | S4 · S6 · S7 중간물 | `aligned/rigged/transferred/animated.blend`, `aligned_params.json`, `wham_pose.npz` |
| 07 | `07_unity` | S7 최종 | `zombie_wham.fbx`, Unity 반입용 텍스처 PNG |

**폴더 번호와 S번호는 독립된 식별자다.** 폴더 번호는 산출물이 생기는 순서(의존
순서)를 나타내고, S번호는 파이프라인 설계도(README v4)의 단계 라벨이다. 둘은 1:1이
아니다 — 예를 들어 `05_smpl_mesh` 는 S5(WHAM) 결과에서 만들어지지만 S4(리깅)의
**입력**이고, `06_rig` 하나에 S4·S6·S7 산출물이 함께 들어간다. 폴더 번호를 보고
S번호를 추정하지 말 것. **S번호는 이 규약과 무관하게 바뀌지 않는다.**

`04_wham` 은 WHAM 의 원출력이고 `06_rig` 는 그것을 가공한 결과다. 이 성격 구분에 따라
`wham_pose.npz`(pose·trans 추출본, S6)는 `06_rig` 에 둔다.

### 구번호 대응표

2026-09-03 이전에 작성된 기록·데이터의 번호는 아래로 읽는다.

| 구번호 | 신번호 |
|---|---|
| `05_wham` | `04_wham` |
| `06_smpl_mesh` | `05_smpl_mesh` |

`00_raw` · `02_sam2` · `03_trellis` 는 번호가 바뀌지 않았다.

`docs/PROJECT_STATUS.md` 의 **날짜가 박힌 일지 항목은 소급 수정하지 않는다.** 그 시점의
사실을 그대로 두고, 읽을 때 이 대응표로 해석한다.

### 맥북 미러

맥북 작업 산출물도 같은 번호 체계를 쓴다. 볼륨 구조를 그대로 미러링해
`~/data/<번호_이름>/<샘플명>/` 에 둔다.

```
~/data/06_rig/zombie_sample1/aligned.blend
~/data/06_rig/zombie_sample1/rigged.blend
~/data/05_smpl_mesh/zombie_sample1/smpl_frame3.obj
```

`~/Desktop` 에 파일을 흩어 두지 않는다. 회수(`scp`)도 같은 경로로 받는다.

## 2. CLI 공통 규약

파이프라인 단계 스크립트(유틸 제외)는 다음을 지킨다.

- **argparse 필수.** `sys.argv` 직접 파싱이나 위치 인자 전용은 쓰지 않는다.
- **입력**: 단계마다 의미가 명확한 이름을 쓴다 — 영상은 `--video`, 이미지는
  `--image`, WHAM 결과는 `--pkl`, Blender 파일은 `--blend`. 같은 이름을 다른
  의미로 쓰지 않는다.
- **출력**: 폴더든 파일이든 **`--out`** 하나로 통일한다. `--outdir` 같은 변형을
  새로 만들지 않는다.
- **이름**: 샘플 단위 산출물은 `--name` 으로 받고, 생략하면 `--out` 폴더 이름을 쓴다.
- **기본값**: 규약 경로를 기본값으로 둔다. 단, **특정 샘플명을 기본값에 박지
  않는다** (`zombie_sample1` 등). 샘플명이 필요한 인자는 필수로 만든다.
- 같은 개념은 같은 인자 이름을 쓴다. 키프레임 번호는 스크립트를 가리지 않고 `--frame`.

## 3. `params.json` 규약

파이프라인 단계 스크립트는 **산출물 폴더에 실행 기록을 남긴다.** 형식은
`run_trellis.py` 가 남기는 `params.json` 을 표준으로 삼는다.

```json
{
  "name": "zombie_sample1",
  "inputs": ["<절대 경로 …>"],
  "<단계 파라미터>": "…",
  "<산출물 통계>": "…",
  "time_s": {"generate": 5.7, "glb": 20.2, "total": 111.1},
  "date": "2026-09-02 06:58:17"
}
```

필수 키는 `name` · `inputs`(절대 경로) · `time_s` · `date` 이고, 단계별 파라미터와
산출물 통계(정점 수, 프레임 수 등)를 함께 남긴다. GPU 작업은 `gpu` · `torch` ·
`peak_vram_gb` 를, 외부 레포에 의존하는 단계는 그 커밋 해시를 추가한다.

목적은 재현과 대조다 — 같은 입력을 다시 돌렸을 때 무엇이 달랐는지 이 파일로 가린다.
`align_smpl_to_trellis.py` 의 `aligned_params.json` 도 같은 역할이다.

## 4. 로그 규약

5분 이상 걸리는 작업은 백그라운드로 돌리고 로그를 남긴다.

```
/workspace/logs/<작업>_<MMDD_HHMM>.log
```

예: `trellis_setup_0902_1216.log`, `trellis_smoke_0902_1554.log`.

> 예외 기록: `run_wham.sh` 는 `wham_<YYYYmmdd_HHMMSS>.log` 를 쓴다. 규약과 다르지만
> 동작에 영향이 없어 두었다. 손댈 일이 생기면 그때 맞춘다.

## 5. `config.yaml` 의 역할 경계

레포 루트의 `config.yaml` 은 **환경 상수만** 담는다.

**담는 것** — 볼륨 경로 루트와 `data/` 번호표, 환경 이름(`wham` / `sam2` / `trellis`),
micromamba 경로, fps 표준.

**담지 않는 것** — 단계별 생성 파라미터(샘플링 스텝, cfg, simplify, 텍스처 크기,
max-dist, 좌표 보정 각도 등). 이것들은 **각 스크립트의 인자 소관**이다. 실행마다
바뀌는 값을 설정 파일에 넣으면 `params.json` 기록과 실제 실행이 어긋난다.

현재 이 파일을 읽는 스크립트는 없다. 규약의 기계 판독 가능한 사본으로 먼저 두고,
연결은 게이트·오케스트레이터 작업(W4) 때 한다. 그 시점에 이 경계도 재검토한다 —
오케스트레이터가 단계별 파라미터를 주입해야 한다면 그때 스키마를 넓힌다.

## 6. fps 표준

**24 fps.** W1에서 확보한 좀비 영상 4종이 실측 24fps인 데 따른다.

> 미정리: `quick_vis.py` 와 `smpl_pkl_to_fbx.py` 의 `--fps` 기본값은 30이다.
> 검증된 동작이라 이번에 건드리지 않았다. 통일은 추후 과제.

## 7. 미정리 목록

- **v2 잔재 2건** — `wham_to_smplfbx.py`, `smpl_pkl_to_fbx.py` 는 폐기된
  SMPL→Mixamo 리타게팅(v2) 경로의 도구다. `scripts/deprecated/` 이동 후보로
  기록해 둔다. 아직 옮기지 않았다.
- `pipeline/qa/` 는 비어 있다(`.gitkeep` 뿐). 게이트·오케스트레이터 구현은 W4 과제.
- `glb_to_fbx.py` 독스트링 예시가 `~/Downloads`·`~/Desktop` 를 쓴다 — 파이프라인 사슬 밖의
  유틸이라 이번에 건드리지 않았다. 다음 수정 기회에 미러 규약으로 맞춘다.
- `99_reports` — 옛 번호 체계의 빈 폴더로 볼륨에 남아 있던 것을 2026-09-04 에 지웠다.
  게이트 판정 리포트 자리가 필요해지면 W4 게이트 작업 때 규약에 정식 배정한 뒤
  다시 만든다.
- `transfer_weights.py` 의 2차 전파(BFS)가 UV 심에서 끊긴다. glTF 임포트는 심에서 정점을
  분리하므로 표면상 이어진 부위도 별개 조각으로 보이고, 수신 정점이 없는 조각은 직선거리
  폴백으로 떨어진다 (zombie1 은 160/5307 = 3.0% 로 무해했다). 전파 전 위치 병합(1e-4)
  그래프에서 BFS 를 도는 것이 옳다. 검증된 스크립트라 다음 수정 기회에 손댄다.
