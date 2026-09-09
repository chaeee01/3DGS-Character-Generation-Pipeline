"""
TRELLIS 외형 복원 (S3): SAM2 키프레임 PNG 1장(또는 여러 장) → GLB.

HF Space(JeffreyXiang/TRELLIS)에서 손으로 하던 일을 Pod 에서 돌리는 것. 기본 파라미터는
Space 의 기본값과 같다 (seed 0 / sparse 12 step·cfg 7.5 / slat 12 step·cfg 3.0 / simplify 0.95).
텍스처 크기만 2048 로 두었다 — 지금까지 확보한 좀비 에셋이 2048 기준이라서.

  micromamba activate trellis
  python run_trellis.py \
      --image /workspace/data/02_sam2/zombie_sample1/keyframes/key3_f00007.png \
      --out   /workspace/data/03_trellis/zombie_sample1

  # 다중 뷰 (방위각 45° 이상 차이 나는 양질 프레임 2장 이상)
  python run_trellis.py --image a.png b.png --out ... --multi-mode stochastic

출력 (--out 아래):
  <name>.glb              메쉬 + 베이크 텍스처 (→ 5-2 정렬 입력)
  textures/<name>_tex_N.png  GLB 에서 뽑은 텍스처 (glb_tex.py, → Unity Material)
  input_0.png …           모델이 실제로 본 전처리 입력 (알파 crop → 518²). G1a 디버깅용
  params.json             입력·파라미터·시간·VRAM·메쉬 통계. 같은 입력을 다시 돌릴 때 대조용
  preview_gs.mp4 / preview_mesh.mp4   --video 지정 시 턴테이블 (Space 미리보기와 같은 것)

주의:
  - 입력이 RGBA 면 알파를 마스크로 쓰고, 알파가 없으면 rembg(u2net)로 배경을 뗀다. SAM2 키프레임은
    RGBA 라 rembg 를 타지 않는다. rembg 를 탔다면 경고가 찍히니 마스크 품질을 의심할 것.
  - 첫 실행은 모델(~3GB)과 DINOv2 를 받고 nvdiffrast 를 JIT 빌드한다. 캐시는 /workspace/.cache.
  - seed 를 고정해도 결과가 미세하게 달라질 수 있다(비결정적 CUDA 커널). 형상 수준 재현은 된다.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VOL = "/workspace"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", nargs="+", required=True,
                    help="키프레임 PNG (RGBA 권장). 2장 이상이면 다중 뷰")
    ap.add_argument("--out", required=True, help="출력 폴더 (예: /workspace/data/03_trellis/zombie_sample1)")
    ap.add_argument("--name", default=None, help="출력 파일 이름. 기본값: --out 폴더 이름")
    ap.add_argument("--trellis-root", default=f"{VOL}/repos/TRELLIS")
    ap.add_argument("--model", default="microsoft/TRELLIS-image-large")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ss-steps", type=int, default=12, help="sparse structure 샘플링 스텝")
    ap.add_argument("--ss-cfg", type=float, default=7.5, help="sparse structure guidance")
    ap.add_argument("--slat-steps", type=int, default=12, help="structured latent 샘플링 스텝")
    ap.add_argument("--slat-cfg", type=float, default=3.0, help="structured latent guidance")
    ap.add_argument("--simplify", type=float, default=0.95,
                    help="제거할 면의 비율. 0=단순화 없음, 0.95=95%% 제거 (Space 기본 0.9~0.98). "
                         "1.0 은 전부 제거라 빈 메쉬가 되어 GLB 변환이 죽는다")
    ap.add_argument("--texture-size", type=int, default=2048, choices=[512, 1024, 1536, 2048])
    ap.add_argument("--multi-mode", default="stochastic", choices=["stochastic", "multidiffusion"],
                    help="다중 뷰 결합 방식 (Space 기본 stochastic)")
    ap.add_argument("--attn", default="xformers", choices=["xformers", "flash_attn"],
                    help="attention 백엔드. setup_trellis.sh 는 xformers 만 설치한다")
    ap.add_argument("--video", action="store_true", help="턴테이블 mp4 도 렌더 (Space 미리보기 상당, +1~2분)")
    ap.add_argument("--ply", action="store_true", help="가우시안 PLY 도 저장")
    ap.add_argument("--no-tex", action="store_true", help="텍스처 추출 생략")
    return ap.parse_args()


def setup_env(args):
    """trellis 를 import 하기 전에 잡아야 하는 것들."""
    os.environ["ATTN_BACKEND"] = args.attn
    os.environ["SPCONV_ALGO"] = "native"      # auto 는 첫 실행마다 벤치마크를 돌려 느리고, 가끔 멈춘다
    # 캐시를 볼륨에. 기본 ~/.cache 는 Pod 과 함께 사라져 매번 다시 받는다 (setup_trellis.sh 와 동일).
    os.environ.setdefault("HF_HOME", f"{VOL}/.cache/huggingface")
    os.environ.setdefault("TORCH_HOME", f"{VOL}/.cache/torch")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"{VOL}/.cache/torch_extensions")
    os.environ.setdefault("U2NET_HOME", f"{VOL}/.cache/u2net")
    # nvdiffrast JIT 빌드가 환경 안의 nvcc 와 헤더를 쓰도록 (setup_trellis.sh 의 export 와 같은 3줄).
    # conda-forge 는 헤더·라이브러리를 targets/ 아래에 두고 include/ 로 링크를 걸어 두는데,
    # 링크가 없는 버전도 있어 경로를 직접 얹는다. 기존 값이 있으면 그 앞에 붙인다.
    if "CONDA_PREFIX" in os.environ:
        prefix = os.environ["CONDA_PREFIX"]
        targets = f"{prefix}/targets/x86_64-linux"
        os.environ.setdefault("CUDA_HOME", prefix)
        for var, paths in (("CPATH", [f"{targets}/include"]),
                           ("LIBRARY_PATH", [f"{targets}/lib", f"{targets}/lib/stubs"])):
            cur = os.environ.get(var, "")
            os.environ[var] = ":".join(paths + ([cur] if cur else []))
    sys.path.insert(0, args.trellis_root)


def has_alpha(im):
    if im.mode != "RGBA":
        return False
    lo, hi = im.getchannel("A").getextrema()
    return not (lo == 255 and hi == 255)


def main():
    args = parse_args()
    setup_env(args)

    import torch
    import numpy as np
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    out = args.out
    name = args.name or os.path.basename(os.path.normpath(out))
    os.makedirs(out, exist_ok=True)
    t_start = time.time()

    # ---- 입력
    images = []
    for p in args.image:
        im = Image.open(p)
        if not has_alpha(im):
            print(f"[경고] 알파 없음 → rembg 로 배경 제거함: {p}")
        images.append(im)
    print(f"[1/4] 입력 {len(images)}장  seed={args.seed}  attn={args.attn}")

    # ---- 모델
    t0 = time.time()
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    print(f"[2/4] 모델 로드 {time.time()-t0:.0f}s  ({args.model})")

    # Space 처럼 전처리를 따로 하고 run 에는 preprocess_image=False. 모델이 실제로 본 입력을 남기기 위해.
    processed = []
    for i, im in enumerate(images):
        pi = pipeline.preprocess_image(im)
        pi.save(os.path.join(out, f"input_{i}.png"))
        processed.append(pi)

    # ---- 생성
    ss_params = {"steps": args.ss_steps, "cfg_strength": args.ss_cfg}
    slat_params = {"steps": args.slat_steps, "cfg_strength": args.slat_cfg}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    if len(processed) == 1:
        outputs = pipeline.run(
            processed[0], seed=args.seed, formats=["gaussian", "mesh"], preprocess_image=False,
            sparse_structure_sampler_params=ss_params, slat_sampler_params=slat_params)
    else:
        outputs = pipeline.run_multi_image(
            processed, seed=args.seed, formats=["gaussian", "mesh"], preprocess_image=False,
            sparse_structure_sampler_params=ss_params, slat_sampler_params=slat_params,
            mode=args.multi_mode)
    t_gen = time.time() - t0
    print(f"[3/4] 생성 {t_gen:.0f}s  peak VRAM {torch.cuda.max_memory_allocated()/2**30:.1f}GB")

    # ---- GLB
    t0 = time.time()
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0], outputs["mesh"][0],
        simplify=args.simplify, texture_size=args.texture_size, verbose=False)
    glb_path = os.path.join(out, f"{name}.glb")
    glb.export(glb_path)
    t_glb = time.time() - t0
    n_v, n_f = int(glb.vertices.shape[0]), int(glb.faces.shape[0])
    print(f"[4/4] GLB {t_glb:.0f}s  정점 {n_v:,}  면 {n_f:,}  "
          f"{os.path.getsize(glb_path)/2**20:.1f}MB → {glb_path}")

    if args.ply:
        outputs["gaussian"][0].save_ply(os.path.join(out, f"{name}_gs.ply"))

    if args.video:
        import imageio
        from trellis.utils import render_utils
        v = render_utils.render_video(outputs["gaussian"][0], num_frames=120)["color"]
        imageio.mimsave(os.path.join(out, "preview_gs.mp4"), v, fps=30)
        v = render_utils.render_video(outputs["mesh"][0], num_frames=120)["normal"]
        imageio.mimsave(os.path.join(out, "preview_mesh.mp4"), v, fps=30)
        print("      턴테이블 저장: preview_gs.mp4, preview_mesh.mp4")

    # ---- 텍스처 (기존 glb_tex.py 그대로 사용 — 맥북에서 하던 것을 여기서 미리)
    tex_dir = None
    if not args.no_tex:
        tex_dir = os.path.join(out, "textures")
        subprocess.run([sys.executable, os.path.join(HERE, "glb_tex.py"), glb_path, tex_dir], check=True)

    # ---- 기록
    try:
        commit = subprocess.check_output(["git", "-C", args.trellis_root, "rev-parse", "--short", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = None
    params = {
        "name": name,
        "inputs": [os.path.abspath(p) for p in args.image],
        "input_has_alpha": [has_alpha(im) for im in images],
        "model": args.model, "trellis_commit": commit,
        "seed": args.seed, "sparse_structure": ss_params, "slat": slat_params,
        "multi_mode": args.multi_mode if len(images) > 1 else None,
        "simplify": args.simplify, "texture_size": args.texture_size, "attn_backend": args.attn,
        "glb": glb_path, "vertices": n_v, "faces": n_f,
        "glb_mb": round(os.path.getsize(glb_path) / 2**20, 2),
        "textures": tex_dir,
        "time_s": {"generate": round(t_gen, 1), "glb": round(t_glb, 1),
                   "total": round(time.time() - t_start, 1)},
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out, "params.json"), "w") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"완료 {params['time_s']['total']:.0f}s → {out}")


if __name__ == "__main__":
    main()
