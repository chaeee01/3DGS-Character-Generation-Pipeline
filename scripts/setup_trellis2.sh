#!/usr/bin/env bash
# TRELLIS.2(microsoft/TRELLIS.2-4B) 로컬 설치 — RunPod Network Volume 기준.
#
#   bash scripts/setup_trellis2.sh            # 전체 설치 (멱등: 다시 돌리면 빠진 것만)
#   bash scripts/setup_trellis2.sh --check    # 설치 상태만 점검
#
# 설계:
#   - setup_trellis.sh(1세대)와 같은 방식. 볼륨(/workspace)의 micromamba 환경 `trellis2` 에
#     설치하고, CUDA 툴체인·컴파일러까지 환경 안에 넣는다. 1세대 환경 `trellis` 는 건드리지
#     않는다 — 통제 실험의 대조군이라 살려 둬야 한다.
#   - 버전·URL 은 전부 upstream setup.sh 원문에서 가져왔다 (추측 없음):
#       python 3.10 / torch 2.6.0 + torchvision 0.21.0 (cu124) / CUDA 12.4 /
#       flash-attn 2.7.3 / nvdiffrast v0.4.0 / nvdiffrec(renderutils 브랜치) /
#       CuMesh / FlexGEMM / o-voxel(레포 동봉) / utils3d @9a4eb15e (1세대와 같은 핀)
#   - upstream 과 일부러 다르게 한 것 (이유는 각 단계 주석):
#       * 확장 소스를 /tmp 가 아니라 /workspace/build 에 둔다 (Pod 삭제 시 소실 방지)
#       * gradio 제외 (app.py 웹데모 전용, 파이프라인에 불필요)
#       * pillow-simd 제외 (sudo apt libjpeg-dev 필요 + Pillow 와 충돌)
#   - GPU: upstream 이 "at least 24GB, verified on A100/H100" 이라 명시. A100 기준으로 띄운다.
#   - 캐시는 1세대와 같은 /workspace/.cache 를 공유한다 (HF 모델은 별개 레포라 충돌 없음).
#
# -u 는 쓰지 않는다: conda 컴파일러 활성화 스크립트가 미정의 변수를 참조해 -u 에서 죽는다.
set -eo pipefail

VOL=/workspace
MAMBA_DIR=$VOL/micromamba
REPO=$VOL/repos/TRELLIS.2
EXT_DIR=$VOL/build/trellis2_ext
ENV_NAME=trellis2
PY_VER=3.10
CUDA_VER=12.4
LOG_DIR=$VOL/logs
mkdir -p $VOL/repos $VOL/build $LOG_DIR $VOL/.cache/pip $VOL/.cache/huggingface \
         $VOL/.cache/torch $VOL/.cache/torch_extensions $VOL/data/03_trellis2

export PIP_CACHE_DIR=$VOL/.cache/pip
export HF_HOME=$VOL/.cache/huggingface
export TORCH_HOME=$VOL/.cache/torch
export TORCH_EXTENSIONS_DIR=$VOL/.cache/torch_extensions
# 빌드 대상 아키텍처. A100=8.0, 4090=8.9, H100=9.0. 지금은 A100 이지만 추후 4090 하향
# 가능성이 있어 둘 다 넣는다. 늘리면 빌드 시간이 비례해 늘어난다.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.9}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

CHECK_ONLY=0
ATTN=flash-attn          # upstream 기본값. 폴백은 --attn xformers 로만 (자동 전환 없음)
while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK_ONLY=1; shift ;;
        --attn)  ATTN="${2:?--attn 뒤에 flash-attn 또는 xformers}"; shift 2 ;;
        *) echo "알 수 없는 인자: $1"; exit 1 ;;
    esac
done
case "$ATTN" in flash-attn|xformers) ;; *) echo "--attn 은 flash-attn 또는 xformers"; exit 1 ;; esac

log() { echo ""; echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- 1. micromamba
if [ ! -f "$MAMBA_DIR/bin/micromamba" ]; then
    [ $CHECK_ONLY = 1 ] && { echo "micromamba 없음"; exit 1; }
    log "[1/8] micromamba 설치"
    mkdir -p $MAMBA_DIR/bin
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $MAMBA_DIR bin/micromamba
else
    log "[1/8] micromamba 재사용"
fi
export MAMBA_ROOT_PREFIX=$MAMBA_DIR
eval "$($MAMBA_DIR/bin/micromamba shell hook -s bash)"

# ---------------------------------------------------------------- 2. 환경
if ! micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    [ $CHECK_ONLY = 1 ] && { echo "환경 '$ENV_NAME' 없음"; exit 1; }
    log "[2/8] python $PY_VER + CUDA $CUDA_VER 툴체인 환경 생성 (conda-forge, ~2GB)"
    micromamba create -y -n $ENV_NAME -c conda-forge \
        python=$PY_VER pip \
        cuda-version=$CUDA_VER cuda-compiler=$CUDA_VER cuda-libraries-dev=$CUDA_VER \
        gxx_linux-64=11 ninja git
else
    log "[2/8] 환경 '$ENV_NAME' 재사용"
fi
micromamba activate $ENV_NAME

# 확장 빌드가 환경 안의 CUDA 를 보도록 (1세대에서 확인된 필수 설정)
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}
export PATH=$CONDA_PREFIX/bin:$PATH

if ! command -v nvcc >/dev/null || ! nvcc --version | grep -q "release $CUDA_VER"; then
    echo "nvcc $CUDA_VER 를 찾지 못함 (PATH=$PATH)"; nvcc --version || true; exit 1
fi
log "nvcc: $(nvcc --version | grep release)  /  CXX: ${CXX:-c++}"

# ---------------------------------------------------------------- 3. torch
if ! python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.6.0') else 1)" 2>/dev/null; then
    [ $CHECK_ONLY = 1 ] && { echo "torch 2.6.0 없음"; exit 1; }
    log "[3/8] torch 2.6.0 + torchvision 0.21.0 (cu124) 설치 (~2.5GB)"
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
else
    log "[3/8] torch 재사용: $(python -c 'import torch; print(torch.__version__)')"
fi

# ---------------------------------------------------------------- 4. 기본 의존성
# upstream --basic 목록에서 gradio(웹데모 전용), pillow-simd(sudo apt 필요 + Pillow 충돌) 제외.
if [ $CHECK_ONLY = 0 ]; then
    log "[4/8] 기본 패키지"
    # transformers 는 upstream 대로 핀 없이 둔다 — 5.x 가 요구하는 torch>=2.5 를 여기선
    # 만족한다(2.6.0). 1세대의 "transformers<5" 핀은 torch 2.4 때문이었다.
    pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
        trimesh transformers tensorboard pandas lpips zstandard kornia timm
    # utils3d 는 upstream 이 커밋을 고정해 둔 것 (1세대와 같은 핀).
    python -c "import utils3d" 2>/dev/null || \
        pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
fi

# ---------------------------------------------------------------- 5. flash-attn
# upstream 기본 백엔드. A100 은 지원한다. prebuilt 휠이 없으면 소스 빌드로 넘어가 오래 걸린다.
# 실패하면 xformers 로 폴백하고 run_trellis2.py 가 ATTN_BACKEND 를 맞춘다.
if [ $CHECK_ONLY = 0 ]; then
    if [ "$ATTN" = "xformers" ]; then
        python -c "import xformers" 2>/dev/null && log "[5/8] xformers 재사용 (--attn xformers)" || {
            log "[5/8] xformers 설치 (--attn xformers 로 명시 지정됨)"
            pip install xformers --index-url https://download.pytorch.org/whl/cu124
        }
    elif python -c "import flash_attn" 2>/dev/null; then
        log "[5/8] flash-attn 재사용"
    else
        log "[5/8] flash-attn 2.7.3 설치 (prebuilt 휠 직접)"
        pip install packaging wheel >/dev/null
        # prebuilt 휠을 URL 로 직접 설치한다. `pip install flash-attn==2.7.3` 은 설치기가
        # 휠을 받아 pip 캐시로 os.rename 하다 죽는다 — 받은 위치(/tmp, 컨테이너 로컬)와
        # PIP_CACHE_DIR(/workspace, MooseFS)가 다른 파일시스템이라 EXDEV(Errno 18).
        # flash-attn setup.py 가 shutil.move 대신 os.rename 을 쓰는 탓이다.
        #
        # URL 에 torch2.6 / cp310 / cxx11abiFALSE 가 박혀 있다 — [3/8] 의 torch 버전이나
        # [2/8] 의 python 버전을 바꾸면 이 URL 도 함께 갱신해야 한다. ABI 는
        # torch._C._GLIBCXX_USE_CXX11_ABI 값과 일치해야 한다 (현재 False).
        FA_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
        # 자동 폴백하지 않는다. attention 백엔드는 실험 조건이라 조용히 바뀌면
        # 1세대와의 대조가 무너진다. 실패하면 멈추고, 우회는 사람이 고른다.
        if ! pip install "$FA_WHL" 2>&1 | tee $LOG_DIR/build_flash_attn.log | tail -5; then
            echo ""
            echo "  ! flash-attn prebuilt 휠 설치 실패 — 로그: $LOG_DIR/build_flash_attn.log"
            echo "    자동 진행하지 않는다. 아래 중 하나를 사람이 고른다:"
            echo "      1) 버전 독립 우회 (pip 캐시를 로컬 디스크로 두어 EXDEV 회피):"
            echo "         PIP_CACHE_DIR=/root/.cache/pip pip install flash-attn==2.7.3 --no-build-isolation"
            echo "      2) 백엔드 교체 (실험 조건이 바뀌므로 기록 필요):"
            echo "         bash scripts/setup_trellis2.sh --attn xformers"
            echo "    URL 이 404 면 torch/python/ABI 조합이 바뀐 것이다 — FA_WHL 을 갱신하라."
            exit 1
        fi
    fi
fi

# ---------------------------------------------------------------- 6. TRELLIS.2 레포
# o-voxel 이 레포 안에 동봉돼 있어 확장 설치보다 먼저 클론해야 한다.
if [ ! -d "$REPO" ]; then
    [ $CHECK_ONLY = 1 ] && { echo "TRELLIS.2 레포 없음"; exit 1; }
    log "[6/8] TRELLIS.2 레포 클론"
    git clone -b main --recursive https://github.com/microsoft/TRELLIS.2.git $REPO
else
    log "[6/8] TRELLIS.2 레포 재사용 ($(git -C $REPO rev-parse --short HEAD))"
fi

# ---------------------------------------------------------------- 7. CUDA 확장 5종
build_ext() {   # 이름 URL 클론옵션 서브경로  (URL 이 "-" 면 레포 동봉 소스)
    local mod=$1 url=$2 opt=$3 sub=$4 dir=$EXT_DIR/$1
    if python -c "import $mod" 2>/dev/null; then echo "  - $mod: 있음"; return; fi
    [ $CHECK_ONLY = 1 ] && { echo "  - $mod: 없음"; return; }
    if [ "$url" = "-" ]; then
        rm -rf "$dir"; cp -r "$REPO/$sub" "$dir"; sub="."
    else
        [ -d "$dir" ] || git clone $opt $url $dir
    fi
    echo "  - $mod: 빌드 ($dir/$sub)"
    pip install --no-build-isolation -v "$dir/$sub" 2>&1 | tee $LOG_DIR/build_${mod}.log \
        | grep -E "error|Error|Successfully|warning: unsupported" || true
    python -c "import $mod" || { echo "  ! $mod 빌드 실패 — $LOG_DIR/build_${mod}.log 확인"; exit 1; }
}
log "[7/8] CUDA 확장 (nvdiffrast / nvdiffrec / cumesh / flexgemm / o_voxel)"
mkdir -p $EXT_DIR
[ $CHECK_ONLY = 1 ] || pip install setuptools wheel ninja >/dev/null
build_ext nvdiffrast https://github.com/NVlabs/nvdiffrast.git        "-b v0.4.0"     "."
build_ext nvdiffrec  https://github.com/JeffreyXiang/nvdiffrec.git   "-b renderutils" "."
build_ext cumesh     https://github.com/JeffreyXiang/CuMesh.git      "--recursive"   "."
build_ext flexgemm   https://github.com/JeffreyXiang/FlexGEMM.git    "--recursive"   "."
build_ext o_voxel    -                                               ""              "o-voxel"

# ---------------------------------------------------------------- 8. 검증
log "[8/8] import 검증"
cd $REPO
python - <<'EOF'
import importlib, torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  VRAM {p.total_memory/2**30:.1f}GB  sm_{p.major}{p.minor}")
import numpy; print("numpy", numpy.__version__)
mods = ["torchvision", "utils3d", "trimesh", "kornia", "timm", "transformers",
        "nvdiffrast", "nvdiffrec", "cumesh", "flexgemm", "o_voxel"]
try:
    import flash_attn; backend = "flash_attn"
except ImportError:
    import xformers; backend = "xformers"
mods.append(backend)
for m in mods:
    mod = importlib.import_module(m)
    # utils3d 처럼 모듈 수준 __getattr__ 로 lazy import 하는 패키지는 없는 속성에
    # ModuleNotFoundError 를 던진다. getattr 기본값은 AttributeError 만 잡는다 (1세대 교훈).
    try:
        v = mod.__version__
    except Exception:
        v = "ok"
    print(f"  {m:28s} {v}")
# 백엔드는 실험 조건이라 검증 출력에 반드시 남긴다 — 1세대 대조 시 조건 일치 확인용.
print(f"  ATTN_BACKEND (설치 기준)      {backend}")
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel.postprocess
print("trellis2 import ok")
EOF

echo ""
echo "설치 완료. 스모크 테스트:"
echo "  micromamba activate trellis2"
echo "  python /workspace/repos/Video2UnityAvatar-Pipeline/scripts/run_trellis2.py \\"
echo "      --image /workspace/data/02_sam2/zombie1/keyframes/key1_f00228.png \\"
echo "      --out   /workspace/data/03_trellis2/zombie1"
echo "첫 실행은 4B 모델 가중치를 받으므로 오래 걸린다 (수 GB, /workspace/.cache 에 캐시)."
