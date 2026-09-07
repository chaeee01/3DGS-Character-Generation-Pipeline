"""
정렬 확인 캡처 (5-2 보조): aligned.blend 를 정면·측면으로 렌더한다.

bbox IoU 수치만으로는 자세가 틀어졌는지 알 수 없어 눈으로 볼 그림이 필요하다.
SMPL 메쉬를 반투명 초록으로 칠해 TRELLIS 좀비와 겹쳐 보이게 한다.

  /Applications/Blender4.5.app/Contents/MacOS/Blender --background \
      --python render_align_check.py -- \
      --blend ~/data/06_rig/zombie1/aligned.blend \
      --out   ~/data/06_rig/zombie1

출력 (--out 아래): aligned_front.png, aligned_side.png
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
    ap.add_argument("--blend", required=True, help="aligned.blend (5-2 출력)")
    ap.add_argument("--out", required=True, help="캡처 출력 폴더")
    ap.add_argument("--smpl-name", default="SMPL_body",
                    help="반투명으로 칠할 SMPL 메쉬 이름 (부분 일치)")
    ap.add_argument("--res", type=int, nargs=2, default=[900, 1200], help="해상도 W H")
    return ap.parse_args(argv)


def scene_bounds(objs):
    mn = mathutils.Vector((1e9,) * 3)
    mx = mathutils.Vector((-1e9,) * 3)
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ mathutils.Vector(c)
            mn = mathutils.Vector(min(mn[i], w[i]) for i in range(3))
            mx = mathutils.Vector(max(mx[i], w[i]) for i in range(3))
    return mn, mx


def main():
    a = parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=os.path.expanduser(a.blend))
    sc = bpy.context.scene

    for ob in bpy.data.objects:
        if ob.type == "MESH" and a.smpl_name.lower() in ob.name.lower():
            m = bpy.data.materials.new("align_check_smpl")
            m.use_nodes = True
            b = m.node_tree.nodes["Principled BSDF"]
            b.inputs["Base Color"].default_value = (0.1, 0.9, 0.2, 1)
            b.inputs["Alpha"].default_value = 0.35
            m.blend_method = "BLEND"
            ob.data.materials.clear()
            ob.data.materials.append(m)

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("메쉬가 없습니다")
    mn, mx = scene_bounds(meshes)
    ctr, size = (mn + mx) / 2, max(mx - mn)

    cam = bpy.data.objects.new("align_cam", bpy.data.cameras.new("align_cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = size * 1.4

    light = bpy.data.objects.new("align_sun", bpy.data.lights.new("align_sun", type="SUN"))
    sc.collection.objects.link(light)
    light.rotation_euler = (math.radians(50), 0, math.radians(40))

    sc.render.resolution_x, sc.render.resolution_y = a.res
    sc.render.image_settings.file_format = "PNG"

    for tag, deg in (("front", 0), ("side", 90)):
        r = math.radians(deg)
        cam.location = (ctr.x + math.sin(r) * size * 2, ctr.y - math.cos(r) * size * 2, ctr.z)
        cam.rotation_euler = (math.radians(90), 0, r)
        sc.render.filepath = os.path.join(out, f"aligned_{tag}.png")
        bpy.ops.render.render(write_still=True)
        print(f"캡처: {sc.render.filepath}")


if __name__ == "__main__":
    main()
