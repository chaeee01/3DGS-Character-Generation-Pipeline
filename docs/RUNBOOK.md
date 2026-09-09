# RUNBOOK — 영상 1편 → Unity 에셋 (수동 파이프라인)

좀비 영상 한 편을 넣어 Unity에서 재생되는 애니메이션 에셋을 얻기까지의 실행 절차서.
자동화(M2) 전까지 사람이 직접 밟는 순서를 그대로 적었다. 설계 배경은
[PROJECT_STATUS.md](PROJECT_STATUS.md), 일정은 [ROADMAP.md](ROADMAP.md) 참조.

**소요 시간 기준** — 69프레임 테스트 영상 기준이며, 실전 영상은 길이에 비례해 늘어난다:
Pod 준비 ~5분 · SAM2 ~2분(모델 로딩 포함) · TRELLIS Space 2~10분(대기열에 따라 변동) ·
WHAM 수 분 · 리깅 4단계 ~10분 · Unity ~15분.

**전제**: 맥북(Blender 4.5 LTS, Unity), RunPod 계정, Network Volume `pipeline-vol`(100GB,
EU-RO-1), SMPL/SMPLify 계정 인증 완료.

---

## 0. 입력 조건 (G0/G1 수동 적용)

영상을 고르는 단계. 여기서 거른 만큼 뒤 단계가 편해진다.

| 항목 | 조건 | 근거 |
|---|---|---|
| 길이 | 90프레임(3초) 이상 ~ 600프레임 이하 | WHAM 추적 안정성 / 처리 시간 |
| fps | 24fps | 동작 재생 기준 (`docs/CONVENTIONS.md` 6절 표준, 소싱 4종 실측치) |
| 해상도 | 720p 이상 | 마스크·메쉬 품질 |
| 인물 | 1명, 컷 전환 없음 | SAM2 단일 객체 추적 전제 |
| 크기 | 인물 bbox 높이 256px 이상, 발끝까지 프레임 안 | WHAM 관절 추정 |
| 가림 | 30% 미만 | 〃 |
| 카메라 | 고정 (DPVO 미설치, `--estimate_local_only`) | 카메라 이동 시 추가 설치 필요 |

**키프레임 조건 (TRELLIS용)** — 클립 안에 아래를 만족하는 프레임이 1장 이상 있어야 한다:
전신, 모션블러 없음, **팔이 몸통에서 떨어진 자세**, 정면~3/4 측면.

> 팔 벌린 자세는 취향이 아니라 요건이다. 팔이 몸통에 붙어 있으면 3단계 웨이트 전이에서
> 팔-몸통 근접부 웨이트가 번진다.

### 확인 포인트
- `ffprobe <video>`로 fps·해상도·길이를 **실측**한다. 표기와 다를 수 있다.
- 방위각 45° 이상 차이 나는 양질 프레임이 2장 이상이면 TRELLIS 다중 뷰 경로를 쓸 수 있다.

### 흔한 실패
- **fps를 표기만 믿음** — 테스트 클립 `zombie_sample1.mp4`는 30fps인 줄 알았으나 실측 24fps였다.
  뒤 단계의 프레임 번호 계산이 전부 어긋난다.
- **69프레임짜리로 진행** — 현재 테스트 클립은 90프레임 기준 미달이다. 검증용으로는 쓰되
  실제 에셋 제작에는 5~10초 클립을 새로 구한다.

---

## 1. Pod 준비

1. RunPod → **Storage** → `pipeline-vol` → **Deploy**. 볼륨에서 배포해야 `/workspace`가 붙는다.
   (Pods 메뉴에서 새로 만들면 볼륨이 연결되지 않는다.)
2. GPU 선택: **RTX 4090** ($0.69/hr) 기본. 리전은 볼륨과 같은 **EU-RO-1** 고정 — 다른 리전은
   애초에 이 볼륨을 붙일 수 없다. VRAM 24GB면 SAM2·WHAM 모두 충분하고, 더 큰 카드는 낭비다.
3. SSH(Direct TCP)로 접속하고 VSCode Remote-SSH를 붙인다.
4. 환경 활성화 — 단계마다 **다른 환경**을 쓴다:

```bash
export MAMBA_ROOT_PREFIX=/workspace/micromamba
eval "$(/workspace/micromamba/bin/micromamba shell hook -s bash)"

micromamba activate sam2      # 2단계 (python 3.10, torch cu121)
micromamba activate trellis   # 3단계 (python 3.10, torch 2.4.0+cu121, CUDA 툴체인 내장)
micromamba activate wham      # 4·5단계 (python 3.9, torch 2.0.0+cu118)
```

레포 의존성이 충돌(torch 1.11~2.5, CUDA 11.3~12.4)해서 단일 환경으로 합칠 수 없다.
환경 분리는 우회가 아니라 확정된 원칙이다.

### 확인 포인트
- `ls /workspace/repos` → `WHAM`, `sam2`, `TRELLIS`가 보이면 볼륨이 제대로 붙은 것이다.
- `python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"` → `True`.
- 볼륨은 Pod을 지워도 유지된다(검증 완료). 환경을 다시 만들 필요 없다.

### 흔한 실패
- **포트 혼동** — Jupyter 포트(8888)는 **브라우저 프록시 전용**이다. `ssh`·`scp`는 RunPod 콘솔의
  **"SSH over exposed TCP"** 에 표시된 포트를 써야 한다. 8888로 `scp`하면 `Connection refused`가 난다.
  포트와 IP는 Pod을 재배포할 때마다 바뀌므로 접속 전 콘솔에서 다시 확인한다.
- **환경 혼동** — `wham` 환경에서 `run_sam2.py`를 돌리면 torch 버전이 맞지 않아 죽는다.
  프롬프트의 환경 이름을 확인하는 습관을 들인다.
- **레포를 볼륨 밖에 clone** — `cd /workspace/repos` 없이 clone하면 `/root` 아래로 들어간다.
  `/workspace` 밖은 Pod을 Terminate하면 **사라진다**. 레포는 항상
  `/workspace/repos/Video2UnityAvatar-Pipeline`에 둔다.
- **Stop으로 두기** — Stop은 스토리지가 2배로 과금된다. 작업이 끝나면 반드시 **Terminate**.

---

## 2. SAM2 — 객체 분리 (S2)

먼저 **첫 프레임을 뽑아 좀비 좌표를 확인한다.** `--point` 는 추적 시작점이라
빗나가면 전 프레임이 어긋난다. 중앙 기본값에 기대지 말고 눈으로 확인한다.

```bash
ffmpeg -v error -y -i /workspace/data/00_raw/<샘플>.mp4 \
    -vf "select=eq(n\,0)" -vframes 1 /workspace/data/02_sam2/<샘플>/frame0.png
scp runpod:/workspace/data/02_sam2/<샘플>/frame0.png ~/data/02_sam2/<샘플>/
```

맥북에서 열어 좀비 몸통 한가운데 픽셀 좌표를 읽고, 그 값을 `--point` 에 넣는다.

```bash
micromamba activate sam2
python /workspace/repos/Video2UnityAvatar-Pipeline/scripts/run_sam2.py \
    --video /workspace/data/00_raw/zombie_sample1.mp4 \
    --out   /workspace/data/02_sam2/zombie_sample1 \
    --point 640,360
```

`--point`는 **첫 프레임에서 좀비가 있는 픽셀 좌표**다. 생략하면 프레임 중앙을 쓴다.
좌표를 모르겠으면 첫 프레임을 열어 좀비 몸통 한가운데를 찍어 그 좌표를 넣는다.

**산출물** (`--out` 아래):

| 경로 | 용도 |
|---|---|
| `masks/00000.png …` | 프레임별 마스크 (흑백) |
| `<영상명>_masked.mp4` | 배경 검게 지운 클립 → **4단계 WHAM 입력** |
| `keyframes/key1_f00007.png …` | 알파 키프레임 후보 → **3단계 TRELLIS 입력** |
| `keyframes/candidates.json` | 후보 선정 근거(프레임·점수·bbox) |

### 확인 포인트
- **`masks/` 장수 = 원본 프레임 수.** 중간에 비면 추적이 끊긴 것이다. 이것이 가장 확실한 지표다.
  (진행바의 it/s 는 `nohup` 으로 파일에 리다이렉트하면 버퍼링돼 로그에 남지 않는다.
  참고값: 4090에서 240프레임 약 9분.)
- 마스크 몇 장을 원본 프레임에 겹쳐 **전신을 잡았는지** 눈으로 본다 — 손끝·발끝·옷자락 포함,
  배경 혼입 없음.
- 마스킹 클립을 재생해 **좀비만 남고 배경이 검은지**, 중간에 마스크가 튀지 않는지 본다.
- 키프레임 후보를 열어 **팔이 벌어진 프레임인지** 확인한다. 자동 점수(마스크 폭/높이 × 채움률)가
  항상 최선을 고르지는 않는다 — `candidates.json`을 보고 직접 골라도 된다.

### 흔한 실패
- **config-체크포인트 짝 어긋남** — `--model-cfg`와 `--checkpoint`는 **같은 세대**여야 한다.
  현재 기본값은 `configs/sam2.1/sam2.1_hiera_l.yaml` + `sam2.1_hiera_large.pt` 조합이다.
  구버전 `sam2_hiera_t.yaml`에 2.1 체크포인트를 물리면 로드 단계에서 죽는다.
- **`--point`가 배경을 찍음** — 첫 프레임에서 좀비가 화면 중앙에 없으면 기본값이 배경을 잡아
  엉뚱한 것을 추적한다. 마스크가 이상하면 제일 먼저 이걸 의심한다.
- **재생 안 되는 클립** — ffmpeg가 없으면 스크립트가 **에러로 멈춘다**(의도된 동작). 예전엔 조용히
  mp4v로 떨어졌는데 그 결과물이 브라우저·Jupyter에서 재생되지 않아 한참 헤맸다.

---

## 3. TRELLIS.2 — 외형 복원 (S3)

```bash
micromamba activate trellis2
python /workspace/repos/Video2UnityAvatar-Pipeline/scripts/run_trellis2.py \
    --image /workspace/data/02_sam2/<샘플>/keyframes/<키프레임>.png \
    --out   /workspace/data/03_trellis/gen2/<샘플> \
    --pipeline-type 512
```

`--pipeline-type` 이 해상도 다이얼이다 — `512` / `1024` / `1024_cascade` / `1536_cascade`.
`--image` 는 이미지 **1장만** 받는다 (upstream `run()` 이 그렇다). `--video` 는 HDRI 를
요구하는데 `opencv-python-headless` 에 OpenEXR 코덱이 없어 실패한다 — 통제 비교는
`scripts/render_glb_compare.py` 로 한다.

**산출물** (`--out` 아래):

| 경로 | 용도 |
|---|---|
| `<이름>.glb` | PBR 메쉬 → **5-2 정렬 입력** |
| `textures/<이름>_tex_0.png` | baseColor → **Unity Base Map** |
| `textures/<이름>_tex_1.png` | metallicRoughness → 6절 채널 규약 참조 |
| `input_0.png` | 모델이 실제로 본 전처리 입력 |
| `params.json` | 파라미터·시간·VRAM·메쉬 통계 + `texture_slots` 매핑 |

**소요 시간** (RTX 4090, 2026-09-08 실측):

| 구간 | 512 | 1024 |
|---|---|---|
| 모델 로드 | ~155s | ~158s |
| 생성 | ~22s | ~53s |
| GLB 변환 | ~20s | ~51s |
| **총계** | **~196s** | **~262s** |
| **peak VRAM** | **2.6GB** | **3.1GB** |

모델 로드 155초는 15.12GB 가중치를 볼륨에서 읽는 시간이라 매 실행 발생한다.
VRAM 은 해상도의 병목이 아니다(`low_vram=True` 가 모델을 CPU 에 두고 필요할 때만 올린다) —
비용은 시간에 붙는다. 24GB GPU 로 충분하다.

### 확인 포인트
- `params.json` 의 `input_has_alpha` 가 `true` 인지. `false` 면 RMBG 가 배경을 뗀 것이다.
- `peak_vram_gb` (512 기준 2.6GB), `vertices` (실물 GLB 기준으로 기록된다).
- `texture_slots` 에 baseColor·metallicRoughness 매핑이 있는지 — 6절에서 쓴다.
- GLB 를 Blender 에 임포트해 **뒷면**을 본다. 2세대는 뒤통수가 뭉개지지 않고 찢어진 옷이
  구멍으로 표현되는 것이 정상이다.

### 흔한 실패
- **환경 혼동** — `trellis2` 환경이다. 1세대는 `trellis` 로 따로 있다.
- **게이트 리포 접근** — DINOv3(`facebook/dinov3-vitl16-*`)와 RMBG(`briaai/RMBG-2.0`)가
  게이트다. HF 접근 승인 + `HF_TOKEN` 환경변수가 필요하다. 없으면 파이프라인 생성 자체가
  실패한다. RMBG 는 알파 있는 입력에서는 호출되지 않지만 로드는 무조건 된다.
- **transformers 버전** — 4.57.3 으로 고정돼 있다. 5.x 는 DINOv3ViTModel 구조가 달라져
  `AttributeError: 'DINOv3ViTModel' object has no attribute 'layer'` 가 난다.

---

## 4. WHAM — 동작 복원 (S5)

```bash
micromamba activate wham
bash /workspace/repos/Video2UnityAvatar-Pipeline/scripts/run_wham.sh \
    /workspace/data/02_sam2/zombie_sample1/zombie_sample1_masked.mp4
```

출력 폴더는 `--out`으로 바꿀 수 있고, 기본값은 규약 경로 `/workspace/data/04_wham`이다.
산출물은 `/workspace/data/04_wham/<영상명>/`에 떨어진다:
`wham_output.pkl`(pose·trans·betas·verts), `tracking_results.pth`, `slam_results.pth`.

**재투영 오버레이는 따로 만든다.** WHAM 의 `--visualize` 는 pytorch3d 를 요구하는데
py39+cu118 빌드가 torch 2.1.2 이상만 있어 torchvision 0.15.1(torch 2.0.0 고정)과 공존할 수
없다. `run_wham.sh` 에서 `--visualize` 를 뺐고, 대신 이 스크립트로 `overlay.mp4` 를 만든다:

```bash
python scripts/overlay_vis.py \
    --pkl   /workspace/data/04_wham/<영상명>/wham_output.pkl \
    --video /workspace/data/02_sam2/<샘플>/<샘플>_masked.mp4
```

pkl 이 있는 폴더에 `overlay.mp4` 가 생긴다. WHAM 내장 렌더러(3D 메쉬)와 달리
**2D 관절 스켈레톤 재투영**이라 그림이 다르지만, 관절 정합 확인이라는 목적은 같다.
`skeleton_preview.mp4` 등 `--visualize` 의 다른 부산물은 더 이상 생기지 않는다.

### 확인 포인트
- pkl 형상: `pose (T,72)`, `trans (T,3)`, `betas (T,10)`, `verts (T,6890,3)`.
- **트랙 1개**여야 한다. 2개 이상이면 다른 객체를 사람으로 잡은 것이다.
- 포즈 표준편차가 0.4 이상이면 동작이 확실히 잡힌 것 (테스트 클립 0.44).
- `overlay.mp4`를 재생해 **관절이 좀비 위에 얹혀 있는지** 눈으로 본다. 이게 가장 확실한 검증이다.
  판정 기준은 **몸통·대관절은 엄격, 말단은 관대** — 손목·손끝이 가끔 이탈하는 것은 WHAM 말단
  관절의 알려진 특성이라 허용한다. 골반·척추·무릎·어깨가 어긋나면 불합격이다.

### 흔한 실패
- **원본 영상을 그대로 넣음** — 배경이 남아 있으면 추적이 흔들린다. 2단계의 마스킹 클립을 넣는다.
- **카메라가 움직이는 영상** — `--estimate_local_only`는 카메라 고정 전제다. 흔들리는 영상은
  DPVO를 따로 설치해야 한다.

---

## 5. SMPL 리깅 4단계 (S4 + S7)

v4의 핵심 구간. 골격을 SMPL로 통일해 **리타게팅 없이** 동작을 재생한다.
5-1은 Pod(wham 환경), 5-2~5-5는 **맥북 Blender 4.5 LTS**에서 돈다.

### 5-1. SMPL 메쉬 생성 (Pod)

```bash
python scripts/generate_smpl_mesh.py \
    --pkl /workspace/data/04_wham/zombie_sample1_masked/wham_output.pkl \
    --frame 3
```

WHAM 출력 폴더는 마스킹 클립 이름(`<샘플>_masked`)을 따르므로 `--pkl` 경로에 `_masked` 가 붙는다.
산출물 폴더의 샘플명은 거기서 접미사를 떼어 유추하며, 틀리면 `--name` 으로 지정한다.

`--frame`은 **TRELLIS에 넣은 키프레임과 같은 번호**여야 한다. 애매하면 `overlay.mp4`에서 그 장면을 찾는다.
출력(`/workspace/data/05_smpl_mesh/<샘플명>/`): `smpl_frame3.obj`, `smpl_tpose.obj`, `joints_frame3.json`, `smpl_faces.npy`.
샘플별 폴더에 넣는 이유는 `smpl_tpose.obj`·`smpl_faces.npy` 에 샘플 구분이 없어 평평하게 쓰면
다음 샘플이 이전 것을 덮어쓰기 때문이다.

**확인 포인트** — `smpl_frame3.obj`를 열어 TRELLIS 좀비와 **같은 자세**인지 본다. 이게 맞아야 다음 정렬이 쉬워진다.

### 5-2. 정렬 (맥북)

```bash
/Applications/Blender4.5.app/Contents/MacOS/Blender --background \
    --python scripts/align_smpl_to_trellis.py -- \
    --trellis ~/data/03_trellis/zombie_sample1/zombie_sample1.glb \
    --smpl ~/data/05_smpl_mesh/zombie_sample1/smpl_frame3.obj \
    --out ~/data/06_rig/zombie_sample1/aligned.blend
```

**확인 포인트**
- 출력되는 **bbox IoU ≥ 0.5** (zombie_sample1 0.717, zombie1 0.797).
- **스케일 0.55~0.65** 밖이면 상류 단계 오류 신호다 — SMPL 표준 신장과 TRELLIS 정규화 크기의
  비율에서 나오는 구조 상수라 캐릭터가 달라도 이 범위를 벗어나지 않는다
  (zombie_sample1 0.588, zombie1 0.5906).
- **수치만으로는 자세가 틀어졌는지 알 수 없다.** 캡처를 떠서 눈으로 본다:

  ```bash
  /Applications/Blender4.5.app/Contents/MacOS/Blender --background \
      --python scripts/render_align_check.py -- \
      --blend ~/data/06_rig/<샘플>/aligned.blend \
      --out   ~/data/06_rig/<샘플>
  ```

  `aligned_front.png`·`aligned_side.png` 가 나온다. 초록 반투명이 SMPL이다. 팔·다리 위상이
  대응하면 통과. 몸통에서 SMPL이 안 보이는 것은 옷 두께만큼 TRELLIS가 큰 정상 상태다.

`aligned_params.json`에 scale·offset이 기록되니 이전 실행과 대조할 수 있다.

### 5-3. 아마추어 생성 + 바인딩 (맥북)

```bash
/Applications/Blender4.5.app/Contents/MacOS/Blender --background \
    --python scripts/create_smpl_armature.py -- \
    --blend ~/data/06_rig/zombie_sample1/aligned.blend \
    --joints ~/data/05_smpl_mesh/zombie_sample1/joints_frame3.json \
    --out ~/data/06_rig/zombie_sample1/rigged.blend
```

**확인 포인트**
- 후보 회전 4개 중 선택된 것의 **평균 관절-메쉬 거리 ≤ 0.2** (경고 임계값). zombie_sample1 0.028,
  zombie1 0.0293이었고 차순위와 각각 100배·90배 차이였다. 차이가 작으면 좌표계 판정이 애매하다는
  뜻이니 결과를 눈으로 확인한다.
- 포즈 시험은 **5-4 이후에 한다.** 이 단계에서는 SMPL 몸체에만 웨이트가 있어 정작 확인하고 싶은
  TRELLIS 좀비가 움직이지 않는다.

### 5-4. 웨이트 전이 (맥북)

```bash
/Applications/Blender4.5.app/Contents/MacOS/Blender --background \
    --python scripts/transfer_weights.py -- \
    --blend ~/data/06_rig/zombie_sample1/rigged.blend \
    --out ~/data/06_rig/zombie_sample1/transferred.blend
```

SMPL 몸체의 웨이트를 TRELLIS 메쉬로 옮긴다. 1차는 0.08m 안쪽만 정밀 전이, 2차는 메쉬 엣지를
따라(BFS) 전파한다. 공간상 가깝지만 천으로는 떨어진 부위로 웨이트가 건너뛰는 것을 막기 위해서다.

**확인 포인트**
- **웨이트 무배정 0%**. 남은 정점이 있으면 로그에 뜬다.
- **포즈 시험** (5-3에서 옮겨온 항목) — Blender에서 열어 Pose Mode → `R_Knee`·`L_Shoulder`·`Spine2`를
  돌려 본다. **관절 경계에서 메쉬가 자연스럽게 갈라지고**, 허리 자락이 팔을 따라가지 않으면 통과.
- 로그의 "엣지로 도달 불가(고립 조각) N개"는 물리적 고립 섬이 아닐 수 있다. glTF 임포트는 UV 심에서
  정점을 분리하므로 표면상 이어진 곳도 별개 조각으로 보인다 (zombie1: 연결 요소 148개였지만 거리
  병합 시 1개). 비율이 크면 그때 실제 고립 여부를 따진다.

### 5-5. WHAM 동작 베이킹 (맥북)

```bash
/Applications/Blender4.5.app/Contents/MacOS/Blender --background \
    --python scripts/apply_wham_pose.py -- \
    --blend ~/data/06_rig/zombie_sample1/transferred.blend \
    --npz ~/data/06_rig/zombie_sample1/wham_pose.npz \
    --out ~/data/06_rig/zombie_sample1/animated.blend \
    --frame 3
```

`--frame`은 **5-1의 `--frame`과 같은 번호**여야 한다. 리그의 rest 자세가 그 프레임이라,
각 프레임 회전을 그 기준의 상대 회전(delta)으로 바꿔 적용하기 때문이다.

`wham_pose.npz`는 WHAM 결과에서 `pose`·`trans`만 뽑은 파일이다. 먼저 변환한다:

```bash
python3 scripts/convert_wham_npz.py \
    --pkl ~/data/04_wham/zombie_sample1/wham_output.pkl \
    --out ~/data/06_rig/zombie_sample1/wham_pose.npz
```

출력되는 프레임 수가 WHAM 클립 길이와 같은지 확인한다. 트랙이 2개 이상이면 경고가 뜨는데,
사람이 아닌 것을 잡았다는 신호이므로 `overlay.mp4`부터 다시 본다.

**확인 포인트** — Blender에서 재생해 **원본 영상과 같은 보행 패턴**이 나오는지 본다.

### 흔한 실패 (5단계 공통)
- **Blender 5.2 사용** — 5.2의 FBX 임포터에 조명 객체 파싱 버그가 있다. **4.5 LTS**로 돈다.
- **5-1과 5-5의 `--frame` 불일치** — 자세가 어긋난 채로 베이킹되어 동작이 뭉개진다.
- **찢어진 옷자락이 팔을 따라감** — TRELLIS가 본체와 분리해 생성한 고립 섬(테스트 좀비 기준
  5.4만 정점)은 엣지로 도달할 수 없어 직선거리 폴백이 걸리고, 그 결과 팔에 오배정된다.
  **구조적 예외이며 미해결**이다 (EH-157). 자락이 크게 흔들리면 이 문제다.

---

## 6. Unity 반입 (S7)

```bash
/Applications/Blender4.5.app/Contents/MacOS/Blender --background \
    --python scripts/export_unity_fbx.py -- \
    --blend ~/data/06_rig/zombie_sample1/animated.blend \
    --out ~/data/07_unity/zombie_sample1/zombie_wham.fbx
```

메쉬 + SMPL 골격 + 웨이트 + 베이킹된 애니메이션이 함께 나간다(보조 메쉬 `SMPL_body`는 제외).
텍스처는 FBX에 임베드되지 않으므로 3단계의 `textures/<샘플>_tex_0.png` 를 따로 연결한다.

Unity에서:
1. FBX를 임포트하고 Rig → **Animation Type: Generic**으로 설정한다.

   > **Generic vs Humanoid — 의도된 트레이드오프.** 현재 리그는 rest가 키프레임 자세라
   > **Generic**으로 반입한다(WHAM 동작 무손실). **Humanoid**(제3자 애니메이션 호환)가 필요하면
   > `smpl_tpose.obj` 기반으로 **T포즈 rest 리그를 재생성**해야 한다 — 별도 과제 EH-211.
   > Humanoid로 억지 설정하면 근육 변환 과정에서 동작이 왜곡된다.

2. Material 생성 → URP **Base Map**에 3단계의 baseColor 텍스처를 연결한다.
   어느 파일이 어느 슬롯인지는 `params.json` 의 `texture_slots` 를 본다.

   > **PBR 텍스처 채널 규약 (2세대).** TRELLIS.2 는 텍스처를 2장 낸다. glTF 는
   > metallicRoughness 를 한 장에 묶는데 **G = roughness, B = metallic** 이고,
   > Unity URP 는 **Metallic Map 의 R = metallic, A = smoothness(= 1 − roughness)** 로
   > 다르게 읽는다. 그대로 물리면 금속·거칠기가 뒤바뀌므로 채널을 재배치하거나,
   > 값이 균일하면 텍스처 대신 슬라이더로 대체한다.
   >
   > 최종 metallic = `metallicFactor` × 텍스처 B 다. GLB 의 `metallicFactor` 가 1.0 이면
   > (2세대 기본값. 1세대는 키 자체가 없어 glTF 기본 1.0) Unity 가 Metallic 슬라이더를
   > 1.0 으로 잡을 수 있다. **좀비가 금속처럼 번쩍이면 이것부터 확인한다.**
   > zombie1 512 실측은 텍스처 B 평균 1.0/255 로 사실상 비금속이라 Metallic 0 으로 두면 된다.
   > 1024 는 B 평균 254.7 로 전신 금속이라 별도 규명 대상이다.
3. Animator Controller를 만들어 FBX 안의 클립을 물리고 재생한다.

### 확인 포인트
- 재생했을 때 **원본 영상과 같은 보행 패턴**이 나오면 통과.
- 메쉬 관통, 발이 바닥을 뚫거나 뜨는지 본다 (G4 항목).

### 흔한 실패
- **Humanoid로 설정** — 위 트레이드오프 상자를 보라. 동작이 미끄러지거나 팔이 접히면 이걸 의심한다.
- **금속처럼 번쩍임** — `metallicFactor` 가 1.0 이거나 채널 규약이 어긋난 것이다. 위 상자를 보라.
- **텍스처가 회색으로 나옴** — Material 연결을 빠뜨린 것이다. FBX에 텍스처가 임베드돼 있지 않으면
  3단계에서 추출한 PNG를 직접 연결한다.

---

## 7. 회수 · 정리

Terminate 전에 **반드시** 결과를 맥북으로 내린다. Pod을 지우면 `/workspace` 밖은 사라진다.

```bash
mkdir -p ~/data/02_sam2 ~/data/03_trellis ~/data/04_wham ~/data/05_smpl_mesh

scp -r runpod:/workspace/data/02_sam2/<이름>      ~/data/02_sam2/
scp -r runpod:/workspace/data/03_trellis/<이름>   ~/data/03_trellis/
scp -r runpod:/workspace/data/04_wham/<이름>      ~/data/04_wham/
scp -r runpod:/workspace/data/05_smpl_mesh/<이름> ~/data/05_smpl_mesh/
```

IP·포트를 명령에 적지 않는다. 접속 정보는 `~/.ssh/config`의 `Host runpod` 한 곳에서만
관리하고, Pod을 새로 띄우면 그 항목의 `HostName`·`Port`만 갱신한다.

회수 목록 체크:

- [ ] SAM2 키프레임 후보 + 마스킹 클립
- [ ] `wham_output.pkl` + `overlay.mp4`
- [ ] `smpl_frame<N>.obj`, `smpl_tpose.obj`, `joints_frame<N>.json`
- [ ] TRELLIS GLB + `textures/` + `params.json`
- [ ] 최종 `zombie_wham.fbx` + 텍스처 PNG
- [ ] **Pod Terminate** (Stop 아님 — 스토리지 2배 과금)

`/workspace` 안에 둔 것은 볼륨에 남으므로 다시 받을 필요는 없다. 애매하면 내려두는 편이 싸다.

### 흔한 실패
- **맥북 쪽 권한(TCC)** — macOS가 터미널의 `~/Downloads`·`~/Desktop`·`~/Documents` 접근을 막으면
  `scp`와 스크립트가 `Operation not permitted`로 실패한다. 파일이 보이는데 열리지 않으면 이것이다.
  시스템 설정 → 개인정보 보호 및 보안 → **전체 디스크 접근 권한**에서 터미널을 허용하고 재시작한다.
  규약 경로 `~/data`는 TCC 보호 대상이 아니라 이 문제를 피한다 — 예전 산출물이 `~/Desktop`에
  남아 있을 때만 해당된다.
- **Terminate 잊음** — 초당 과금이다. 작업이 끝났으면 바로 지운다.

---

## 부록: 레거시 — TRELLIS 1세대 (S3)

2026-09-09 에 TRELLIS.2 로 전환했다. 1세대 절차는 대조군·기준선 재현용으로 남긴다.
산출물은 `03_trellis/gen1/` 에 둔다.

### 3. TRELLIS 1세대 — 외형 복원 (S3)

```bash
micromamba activate trellis
python /workspace/repos/Video2UnityAvatar-Pipeline/scripts/run_trellis.py \
    --image /workspace/data/02_sam2/zombie_sample1/keyframes/key3_f00007.png \
    --out   /workspace/data/03_trellis/zombie_sample1 \
    --video
```

`--image`는 2단계 `keyframes/`에서 고른 RGBA PNG다. 좋은 프레임이 2장 이상이면 여러 개를
넘긴다(방위각 45° 이상 차이). `--video`는 턴테이블 mp4를 함께 렌더한다(+1~2분, 육안 확인용).

**산출물** (`--out` 아래):

| 경로 | 용도 |
|---|---|
| `<이름>.glb` | 메쉬 + 베이크 텍스처 → **5-2 정렬 입력** |
| `textures/<이름>_tex_0.png` | 2048² 텍스처 → **Unity Material** |
| `input_0.png` | 모델이 실제로 본 전처리 입력(알파 crop → 518²). G1a 디버깅용 |
| `params.json` | 입력·파라미터·시간·VRAM·메쉬 통계 |
| `preview_gs.mp4` / `preview_mesh.mp4` | `--video` 지정 시 턴테이블 |

**소요 시간** (4090, 키프레임 1장, 2026-09-02 실측):

| 구간 | 첫 실행 | 이후 |
|---|---|---|
| 모델 다운로드 + 로드 | ~80s | **~56s** (다운로드만 생략, 볼륨→GPU 로드는 매번 발생) |
| 생성 | ~6s | ~6s |
| GLB 변환 (단순화 + 2048² 베이킹) | ~20s | ~20s |
| **총계** | **~111s** | **~85s** |

### 확인 포인트
- `params.json`의 `input_has_alpha`가 `true`인지 본다. `false`면 rembg가 배경을 뗀 것이라
  마스크 품질을 의심해야 한다 (로그에 `[경고] 알파 없음`이 찍힌다).
- `peak_vram_gb` **~9.7GB** (24GB 중 40%). 크게 벗어나면 파라미터가 달라진 것이다.
- 정점·면 수도 `params.json`에 남는다 (실측 4,770 / 6,552). 이전 실행과 대조한다.
- GLB를 Blender에 임포트해 **뒷면**을 본다. 앞면만 그럴듯한 경우가 있다.
- 얼굴 디테일은 기대하지 않는다 — 좀비 컨셉에서는 허용 범위로 판단했다.

### 흔한 실패
- **환경 혼동** — `trellis` 환경에서 돌려야 한다. `wham`/`sam2`에는 TRELLIS가 없다.
- **알파 없는 입력** — SAM2 키프레임은 RGBA라 정상이지만, 다른 경로로 만든 PNG를 넣으면
  rembg(u2net)를 타서 마스크가 달라진다. 경고를 흘려보내지 않는다.
- **transformers 배너** — 실행 첫머리의 `[transformers] Disabling PyTorch …` 두 줄은
  무해하다. 이미지 경로는 transformers를 타지 않는다.
- **텍스처 소실** — GLB를 FBX로 변환하거나 Mixamo를 경유하면 텍스처가 떨어져 나간다.
  FBX에서 되살리려 하지 말고 **원본 GLB에서 직접 추출**한다:

  ```bash
  python scripts/glb_tex.py char1.glb ~/data/03_trellis/char1/textures
  ```

  캐릭터가 여러 개면 출력 폴더를 반드시 나눈다(파일명이 겹친다).


---

## 부록: 단계별 산출물 한눈에

| 단계 | 입력 | 출력 | 위치 |
|---|---|---|---|
| 2. SAM2 | 원본 mp4 | 마스크, 마스킹 클립, 키프레임 후보 | Pod `/workspace/data/02_sam2/` |
| 3. TRELLIS | 키프레임 PNG | GLB, 텍스처 PNG, `params.json` | Pod `/workspace/data/03_trellis/` |
| 4. WHAM | 마스킹 클립 | `wham_output.pkl`, `overlay.mp4` | Pod `/workspace/data/04_wham/` |
| 5-1. SMPL 메쉬 | pkl | obj, joints json | Pod `/workspace/data/05_smpl_mesh/<샘플명>/` |
| 5-2. 정렬 | GLB + obj | `aligned.blend` | 맥북 |
| 5-3. 아마추어 | `aligned.blend` + joints | `rigged.blend` | 맥북 |
| 5-4. 웨이트 | `rigged.blend` | `transferred.blend` | 맥북 |
| 5-5. 베이킹 | `transferred.blend` + npz | `animated.blend` | 맥북 |
| 6. Unity | `animated.blend` | `zombie_wham.fbx` | 맥북 |
