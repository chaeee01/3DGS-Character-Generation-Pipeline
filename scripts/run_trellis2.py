"""
TRELLIS.2 외형 복원 (S3, 2세대): SAM2 키프레임 PNG 1장 → PBR GLB.

run_trellis.py(1세대)와 입출력 규약이 같다 — --image / --out / --name / params.json /
[1/4]~[4/4] 출력. 1세대 산출물과 대조하기 위해 실행 조건을 params.json 에 남긴다.

  micromamba activate trellis2
  python run_trellis2.py \
      --image /workspace/data/02_sam2/zombie1/keyframes/key1_f00228.png \
      --out   /workspace/data/03_trellis2/zombie1

  # 해상도 다이얼 (비용 곡선 측정용)
  python run_trellis2.py --image ... --out ... --pipeline-type 1024

출력 (--out 아래):
  <name>.glb              PBR 메쉬 (baseColor / roughness / metallic / opacity)
  textures/<name>_tex_N.png   GLB 에서 뽑은 텍스처 (glb_tex.py, 1세대와 같은 도구)
  input_0.png             모델이 실제로 본 전처리 입력
  params.json             입력·파라미터·시간·VRAM·메쉬 통계 + 텍스처 슬롯 매핑
  preview.mp4             --video 지정 시 PBR 턴테이블 (HDRI 환경광 필요)

1세대와 달라지는 점:
  - 해상도가 pipeline_type 으로 갈린다: '512' / '1024' / '1024_cascade' / '1536_cascade'.
    1세대의 ss-steps/slat-steps 같은 단계별 스텝 인자는 노출하지 않는다 (upstream 이
    sampler params 를 dict 로 받고 기본값을 config 에 둔다).
  - 텍스처가 여러 장이다 (PBR). 어느 파일이 어느 슬롯인지 params.json 의
    "texture_slots" 에 기록한다 — Unity Material 연결(RUNBOOK 6절)에서 추측을 없애려는 것.
  - GLB 는 OPAQUE 모드로 나간다. 알파는 텍스처에 보존되지만 비활성이라, 투명이 필요하면
    3D 소프트웨어에서 알파 채널을 opacity 에 수동 연결해야 한다 (upstream README 주의).
  - attention 백엔드는 실험 조건이라 자동 전환하지 않는다. --attn 으로 명시하고
    params.json 에 실제 사용 백엔드를 남긴다.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VOL = "/workspace"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", nargs="+", required=True,
                    help="입력 키프레임 PNG (TRELLIS.2 run() 은 1장만 받는다)")
    ap.add_argument("--out", required=True, help="출력 폴더 (예: /workspace/data/03_trellis2/zombie1)")
    ap.add_argument("--name", default=None, help="출력 파일 이름. 기본값: --out 폴더 이름")
    ap.add_argument("--trellis2-root", default=f"{VOL}/repos/TRELLIS.2")
    ap.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    ap.add_argument("--seed", type=int, default=0,
                    help="기본 0 — 1세대 run_trellis.py 와 맞춘 값 (upstream 기본은 42)")
    ap.add_argument("--pipeline-type", default="512",
                    choices=["512", "1024", "1024_cascade", "1536_cascade"],
                    help="해상도 다이얼. 비용 곡선 측정용으로 512/1024 를 각각 재 본다")
    ap.add_argument("--max-num-tokens", type=int, default=49152, help="upstream 기본값")
    ap.add_argument("--decimation-target", type=int, default=1000000,
                    help="to_glb 데시메이션 목표 정점 수 (upstream 예제 1000000). "
                         "1세대 simplify 와 성격이 달라 통제 실험에서 조정한다")
    ap.add_argument("--texture-size", type=int, default=2048,
                    help="기본 2048 — 기존 에셋 관례이자 1세대와 대조 조건 통일 "
                         "(upstream 예제는 4096)")
    ap.add_argument("--simplify-limit", type=int, default=16777216,
                    help="mesh.simplify 상한 (upstream 예제: nvdiffrast limit)")
    ap.add_argument("--attn", default=None, choices=["flash_attn", "xformers"],
                    help="attention 백엔드. 생략 시 설치된 것을 자동 감지하되 전환은 하지 않는다")
    ap.add_argument("--video", action="store_true", help="PBR 턴테이블 mp4 렌더 (HDRI 필요)")
    ap.add_argument("--hdri", default=None,
                    help="--video 용 HDRI. 기본: <trellis2-root>/assets/hdri/forest.exr")
    ap.add_argument("--no-tex", action="store_true", help="텍스처 추출 생략")
    return ap.parse_args()


def setup_env(args):
    """trellis2 를 import 하기 전에 잡아야 하는 것들."""
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"          # EnvMap 이 .exr 를 읽는다
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # upstream 권장
    if args.attn:
        os.environ["ATTN_BACKEND"] = args.attn
    os.environ.setdefault("HF_HOME", f"{VOL}/.cache/huggingface")
    os.environ.setdefault("TORCH_HOME", f"{VOL}/.cache/torch")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"{VOL}/.cache/torch_extensions")
    # nvdiffrast JIT 빌드가 환경 안의 nvcc 와 헤더를 쓰도록 (1세대 교훈).
    if "CONDA_PREFIX" in os.environ:
        prefix = os.environ["CONDA_PREFIX"]
        targets = f"{prefix}/targets/x86_64-linux"
        os.environ.setdefault("CUDA_HOME", prefix)
        for var, paths in (("CPATH", [f"{targets}/include"]),
                           ("LIBRARY_PATH", [f"{targets}/lib", f"{targets}/lib/stubs"])):
            cur = os.environ.get(var, "")
            os.environ[var] = ":".join(paths + ([cur] if cur else []))
    sys.path.insert(0, args.trellis2_root)


def detect_backend():
    """실제로 쓰이는 백엔드를 확인만 한다. 없는 것을 설치하거나 바꾸지 않는다."""
    env = os.environ.get("ATTN_BACKEND")
    if env:
        return env
    try:
        import flash_attn  # noqa: F401
        return "flash_attn"
    except ImportError:
        import xformers    # noqa: F401
        return "xformers"


def texture_slots(glb_path):
    """GLB 의 material 슬롯 → 이미지 인덱스 매핑을 읽는다.

    glb_tex.py 는 images 배열 순서대로 <name>_tex_<i>.png 를 만든다. 그 i 가 어느
    슬롯인지 여기서 알아내 params.json 에 남긴다 — Unity Material 연결에서
    baseColor 를 눈으로 찾지 않아도 되게 하려는 것.
    webp 확장(EXT_texture_webp)으로 나가면 source 가 extensions 안에 들어간다.
    """
    with open(glb_path, "rb") as f:
        data = f.read()
    off, js = 12, None
    while off < len(data):
        ln, typ = struct.unpack_from("<II", data, off)
        if typ == 0x4E4F534A:
            js = json.loads(data[off + 8: off + 8 + ln].decode("utf-8"))
            break
        off += 8 + ln + ((4 - ln % 4) % 4)
    if not js:
        return {}

    def src(tex_index):
        t = js.get("textures", [])[tex_index]
        if "source" in t:
            return t["source"]
        for ext in (t.get("extensions") or {}).values():
            if isinstance(ext, dict) and "source" in ext:
                return ext["source"]
        return None

    slots = {}
    for mat in js.get("materials", []):
        pbr = mat.get("pbrMetallicRoughness", {})
        for key, node in (("baseColor", pbr.get("baseColorTexture")),
                          ("metallicRoughness", pbr.get("metallicRoughnessTexture")),
                          ("normal", mat.get("normalTexture")),
                          ("emissive", mat.get("emissiveTexture")),
                          ("occlusion", mat.get("occlusionTexture"))):
            if node and "index" in node:
                i = src(node["index"])
                if i is not None:
                    slots[key] = i
    return slots


def main():
    args = parse_args()
    setup_env(args)

    import cv2
    import imageio
    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.utils import render_utils
    from trellis2.renderers import EnvMap
    import o_voxel.postprocess

    if len(args.image) > 1:
        raise SystemExit(
            f"TRELLIS.2 의 run() 은 이미지 1장만 받는다 (받은 개수: {len(args.image)}). "
            "다중 뷰가 필요하면 upstream API 를 먼저 확인할 것.")

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    name = args.name or os.path.basename(os.path.normpath(out))
    backend = detect_backend()

    img_path = os.path.expanduser(args.image[0])
    image = Image.open(img_path)
    has_alpha = image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255)
    if not has_alpha:
        print(f"[경고] 알파 없음 → 전처리가 배경을 추정한다: {img_path}")
    image.save(os.path.join(out, "input_0.png"))
    print(f"[1/4] 입력 1장  seed={args.seed}  attn={backend}  pipeline_type={args.pipeline_type}")

    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    t_load = time.time() - t0
    print(f"[2/4] 모델 로드 {t_load:.0f}s  ({args.model})")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    mesh = pipeline.run(image, seed=args.seed, pipeline_type=args.pipeline_type,
                        max_num_tokens=args.max_num_tokens)[0]
    mesh.simplify(args.simplify_limit)
    t_gen = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"[3/4] 생성 {t_gen:.0f}s  peak VRAM {peak:.1f}GB")

    t0 = time.time()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target, texture_size=args.texture_size,
        remesh=True, remesh_band=1, remesh_project=0, verbose=True,
    )
    glb_path = os.path.join(out, f"{name}.glb")
    glb.export(glb_path, extension_webp=True)
    t_glb = time.time() - t0
    n_v, n_f = len(mesh.vertices), len(mesh.faces)
    mb = os.path.getsize(glb_path) / 2 ** 20
    print(f"[4/4] GLB {t_glb:.0f}s  정점 {n_v:,}  면 {n_f:,}  {mb:.1f}MB → {glb_path}")

    # 텍스처·params.json 을 먼저 남긴다. 미리보기(--video)는 부가물인데 그것이 죽으면
    # 필수 기록까지 날아가는 순서였다 (2026-09-08 스모크에서 실제로 겪음).
    tex_dir, slots = None, {}
    if not args.no_tex:
        tex_dir = os.path.join(out, "textures")
        subprocess.run([sys.executable, os.path.join(HERE, "glb_tex.py"), glb_path, tex_dir], check=True)
        slots = {k: f"{name}_tex_{i}.png" for k, i in texture_slots(glb_path).items()}
        if slots:
            print("      텍스처 슬롯:", ", ".join(f"{k}={v}" for k, v in slots.items()))

    try:
        commit = subprocess.check_output(["git", "-C", args.trellis2_root, "rev-parse", "--short", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = None
    params = {
        "name": name,
        "generation": 2,
        "inputs": [os.path.abspath(img_path)],
        "input_has_alpha": [has_alpha],
        "model": args.model,
        "trellis2_commit": commit,
        "seed": args.seed,
        "pipeline_type": args.pipeline_type,
        "max_num_tokens": args.max_num_tokens,
        "decimation_target": args.decimation_target,
        "texture_size": args.texture_size,
        "simplify_limit": args.simplify_limit,
        "attn_backend": backend,
        "glb": glb_path,
        "vertices": n_v,
        "faces": n_f,
        "glb_mb": round(mb, 2),
        "textures": tex_dir,
        "texture_slots": slots,
        "time_s": {"load": round(t_load, 1), "generate": round(t_gen, 1),
                   "glb": round(t_glb, 1), "total": round(t_load + t_gen + t_glb, 1)},
        "peak_vram_gb": round(peak, 2),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # --video 는 마지막에, 실패해도 진행. opencv-python-headless 는 OpenEXR 코덱을
    # 포함하지 않아(build 정보 OpenEXR: NO) HDRI(.exr) 를 못 읽는다. 1·2세대 통제
    # 비교는 scripts/render_glb_compare.py 로 하므로 미리보기는 필수가 아니다.
    video_status = "skipped"
    if args.video:
        try:
            hdri = args.hdri or os.path.join(args.trellis2_root, "assets/hdri/forest.exr")
            raw = cv2.imread(hdri, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise RuntimeError(f"HDRI 를 읽지 못했다: {hdri} "
                                   f"(opencv {cv2.__version__} 에 OpenEXR 코덱이 없을 수 있다)")
            envmap = EnvMap(torch.tensor(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB),
                                         dtype=torch.float32, device="cuda"))
            video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
            imageio.mimsave(os.path.join(out, "preview.mp4"), video, fps=15)
            video_status = "ok"
            print("      턴테이블 저장: preview.mp4")
        except Exception as e:
            video_status = f"failed({type(e).__name__}: {e})"
            print(f"[경고] 턴테이블 렌더 실패 — 산출물에는 영향 없음: {video_status}")
    params["video"] = video_status

    with open(os.path.join(out, "params.json"), "w") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"완료 {params['time_s']['total']:.0f}s → {out}")


if __name__ == "__main__":
    main()
