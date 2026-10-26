# 一条命令：她上传的 lean EXR → assets/
#   python3 ingest.py <dir_with_exr> [--no-assets]
# dir 里放 land_cave.exr、land_city.exr、port_cave.exr、port_city.exr（缺哪张就跳过哪个版本/层）。
# 1) exr2npy 转成 npy（Combined + Position；full 版的 Normal/DiffCol/灯光组也照转）
# 2) 相机 json：优先用 dir 里已有的；否则用 pc_cam/（她 PC 上出图时打印的 LAN_INFO，含光柱位置、weights_baked/denoised 标记）
# 3) grade.py land / port → 写 ../assets/{v}_cave.webp、{v}_city.webp、{v}_depth.png、{v}_city_depth.png、{v}.json，review/ 出显示图
import sys, os, json, glob, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
d = os.path.abspath(sys.argv[1]); no_assets = '--no-assets' in sys.argv
py = sys.executable
exrs = sorted(glob.glob(os.path.join(d, '*.exr')))
if not exrs: sys.exit('no .exr in ' + d)
tags = [os.path.basename(p)[:-4] for p in exrs]
print('EXR:', ', '.join(f'{t} ({os.path.getsize(os.path.join(d, t + ".exr")) / 1e6:.1f} MB)' for t in tags))
subprocess.run([py, os.path.join(HERE, 'exr2npy.py'), d] + tags, check=True)
for v in ('land', 'port', 'low'):
    if f'{v}_cave' not in tags: print('skip', v, '(no cave EXR)'); continue
    cj = os.path.join(d, f'{v}_cam.json')
    if not os.path.exists(cj):
        src = os.path.join(HERE, 'pc_cam', f'{v}_cam.json')
        if not os.path.exists(src): sys.exit(f'missing {src}')
        shutil.copy(src, cj)
    args = [py, os.path.join(HERE, 'grade.py'), v, d]
    if no_assets: args += ['--preview', 'ingest']
    subprocess.run(args, check=True)
print('done →', 'review/gpu/ingest_*' if no_assets else os.path.normpath(os.path.join(HERE, '..', 'assets')))
