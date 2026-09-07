"""
GLB 통제 렌더: 서로 다른 생성 결과를 같은 카메라·조명으로 찍어 비교한다.

모델마다 크기·위치가 달라 그냥 렌더하면 비교가 안 된다. 바운딩박스로 정규화해
프레임 안에서 같은 크기로 맞추고, 카메라·조명 설정을 고정한다.
(씬 구성은 render_align_check.py 와 같은 방식)

  /Applications/Blender4.5.app/Contents/MacOS/Blender --background \
      --python render_glb_compare.py -- \
      --glb ~/data/03_trellis/zombie1/zombie1.glb \
      --out /tmp/cmp --tag trellis1 --angles 0 135

출력 (--out 아래): <tag>_a<각도>.png
"""
import argparse
import math
import os
import sys

import bpy
import mathutils


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True, help="입력 GLB")
    ap.add_argument("--out", required=True, help="출력 폴더")
    ap.add_argument("--tag", required=True, help="파일명 접두 (모델 구분)")
    ap.add_argument("--angles", type=float, nargs="+", default=[0, 135],
                    help="Z축 회전 각도 목록 (0=정면, 135=후측면)")
    ap.add_argument("--res", type=int, nargs=2, default=[800, 1000])
    ap.add_argument("--light", type=float, nargs=2, default=[3.0, 1.2],
                    help="key/fill 태양광 세기. 모델마다 baseColor 밝기가 달라 조정이 필요할 수 있다")
    return ap.parse_args(argv)


def main():
    a = parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=os.path.expanduser(a.glb))
    sc = bpy.context.scene

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("GLB 에 메쉬가 없습니다")
    mn = mathutils.Vector((1e9,) * 3)
    mx = mathutils.Vector((-1e9,) * 3)
    for o in meshes:
        for c in o.bound_box:
            w = o.matrix_world @ mathutils.Vector(c)
            mn = mathutils.Vector(min(mn[i], w[i]) for i in range(3))
            mx = mathutils.Vector(max(mx[i], w[i]) for i in range(3))
    ctr, size = (mn + mx) / 2, max(mx - mn)
    print(f"  {a.tag}: 메쉬 {len(meshes)}개, 정점 {sum(len(m.data.vertices) for m in meshes)}, 크기 {size:.3f}")

    cam = bpy.data.objects.new("cmp_cam", bpy.data.cameras.new("cmp_cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = size * 1.15          # 두 모델을 같은 비율로 채운다

    key = bpy.data.objects.new("key", bpy.data.lights.new("key", type="SUN"))
    key.data.energy = a.light[0]
    sc.collection.objects.link(key)
    key.rotation_euler = (math.radians(55), 0, math.radians(35))
    fill = bpy.data.objects.new("fill", bpy.data.lights.new("fill", type="SUN"))
    fill.data.energy = a.light[1]
    sc.collection.objects.link(fill)
    fill.rotation_euler = (math.radians(70), 0, math.radians(215))

    sc.world = bpy.data.worlds.new("w")
    sc.world.use_nodes = True
    sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.18, 0.18, 0.19, 1)

    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        try:
            sc.render.engine = eng
            break
        except TypeError:
            continue
    print(f"  엔진: {sc.render.engine}")
    sc.render.resolution_x, sc.render.resolution_y = a.res
    sc.render.image_settings.file_format = "PNG"

    for deg in a.angles:
        r = math.radians(deg)
        cam.location = (ctr.x + math.sin(r) * size * 2,
                        ctr.y - math.cos(r) * size * 2,
                        ctr.z)
        cam.rotation_euler = (math.radians(90), 0, r)
        sc.render.filepath = os.path.join(out, f"{a.tag}_a{int(deg)}.png")
        bpy.ops.render.render(write_still=True)
        print(f"캡처: {sc.render.filepath}")


if __name__ == "__main__":
    main()
