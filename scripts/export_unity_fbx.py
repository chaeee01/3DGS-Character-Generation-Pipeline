"""
SMPL 리깅 4단계 마무리: animated.blend -> Unity 반입용 FBX.

메쉬 + SMPL 골격 + 웨이트 + 베이킹된 WHAM 애니메이션을 FBX 로 내보냅니다.
SMPL_body(보조 메쉬)는 내보내기에서 제외합니다 — Unity 에 필요한 건
TRELLIS 좀비뿐입니다.

  /Applications/Blender4.5.app/Contents/MacOS/Blender --background \
      --python export_unity_fbx.py -- \
      --blend ~/data/06_rig/zombie_sample1/animated.blend \
      --out ~/data/07_unity/zombie_sample1/zombie_wham.fbx
"""
import argparse
import sys

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rig-name", default="SMPL_rig")
    ap.add_argument("--smpl-name", default="SMPL_body",
                    help="내보내기에서 제외할 보조 메쉬")
    ap.add_argument("--keep-smpl", action="store_true",
                    help="SMPL_body 도 포함하고 싶으면 지정")
    return ap.parse_args(argv)


def main():
    a = parse_args()
    bpy.ops.wm.open_mainfile(filepath=a.blend)

    rig = bpy.data.objects.get(a.rig_name)
    if rig is None:
        raise RuntimeError(f"{a.rig_name} 없음")

    # SMPL 보조 메쉬 제거 (역할 종료)
    if not a.keep_smpl:
        smpl = bpy.data.objects.get(a.smpl_name)
        if smpl:
            bpy.data.objects.remove(smpl, do_unlink=True)
            print(f"제외: {a.smpl_name}")

    # 아마추어 + 그 자식 메쉬만 선택
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    for child in rig.children_recursive:
        child.select_set(True)
    bpy.context.view_layer.objects.active = rig
    n_sel = len(bpy.context.selected_objects)
    print(f"내보낼 오브젝트 {n_sel}개")

    bpy.ops.export_scene.fbx(
        filepath=a.out,
        use_selection=True,
        # --- 골격 ---
        add_leaf_bones=False,          # Unity 에 불필요한 말단 본 방지
        armature_nodetype="NULL",
        use_armature_deform_only=True,
        # --- 애니메이션 ---
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        # --- 좌표/스케일 (Blender Z-up -> Unity Y-up) ---
        apply_scale_options="FBX_SCALE_ALL",
        axis_forward="-Z",
        axis_up="Y",
        # --- 기타 ---
        path_mode="COPY",
        embed_textures=True,
        mesh_smooth_type="FACE",
    )
    print(f"저장: {a.out}")
    print("Unity: 임포트 -> Rig 탭 Generic -> Animation 탭에서 클립 확인 -> 씬 재생")
    print("      Humanoid 로 두면 근육 변환에서 동작이 왜곡된다 (rest 가 T포즈가 아니라서).")
    # T포즈 rest 리그 재생성은 별도 과제 EH-211.


if __name__ == "__main__":
    main()
