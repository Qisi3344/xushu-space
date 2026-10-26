# 虚数空间 · 鬼洞之雾 — Blender 背景板生成脚本 v5.1（可复现，BLENDER_MCP_SAFE_MODE 兼容）
#
# 只 import bpy / bmesh / mathutils / math；无 lambda / class / globals / eval / exec / open / dunder。发送前跑 `python3 safecheck.py scene.py`。
# 不碰她自己的东西：只建/删名字以 'Lan_X_' 开头的数据块、名为 'Lan_Xushu' 的场景和集合。不切她的场景、不保存 .blend。
#
# ── 配置：包装脚本把下面这一行文本替换成具体字典（safe mode 禁 globals()，所以是文本替换）──
CFG = {}
MODE    = CFG.get('mode', 'all')          # 'build'：只搭场景 | 'render'：用已搭好的场景出图 | 'all'：都做 | 'cleanup'：删掉本脚本的一切
VARIANT = CFG.get('variant', 'land')      # 'land'：2560x1280，竖直 FOV 48° | 'port'：1080x1920，FOV 46°
LAYER   = CFG.get('layer', 'cave')        # 'cave' 洞窟本体 | 'city' 恶罗海城（洞窟 holdout、透明底）
SCALE   = CFG.get('scale', 100)           # 分辨率百分比
RES     = CFG.get('res', None)            # [w, h] 直接指定（预览用，覆盖 SCALE）
SAMPLES = CFG.get('samples', 128)
NOISE_T = CFG.get('noise_threshold', 0.01)
DEVICE  = CFG.get('device', 'CPU')        # 她的 PC：'GPU'（偏好设置里已选 OptiX）
DENOISE = CFG.get('denoise', False)       # True：Cycles 降噪（OPTIX → 失败退 OIDN），lean 输出就靠它
DENOISER = CFG.get('denoiser', 'OPTIX')
THREADS = CFG.get('threads', 0)
SUBSURF = CFG.get('subsurf', True)        # False：不加细分（沙箱算 Position 预览时省时间）
LIGHTGROUPS = CFG.get('lightgroups', False)   # True：出灯光组通道（旧流程）；False：把下面的权重直接乘进灯的强度
WEIGHTS = CFG.get('weights', {'key': 6.0, 'cloud': 10.0, 'bounce': 1.8, 'fill': 1.8, 'back': 0.6, 'rim': 1.0})
PASSES  = CFG.get('passes', 'depth')      # 'depth'：Combined + Z | 'lean'：Combined + Position | 'full'：再加 Normal、DiffCol
FORMAT  = CFG.get('format', 'multilayer') # 'multilayer'（终版）| 'single'（预览：只 Combined 的单层 EXR）
HALF    = CFG.get('half', True)           # 半精度 EXR
CODEC   = CFG.get('codec', 'ZIP')
OUT_DIR = CFG.get('out_dir', '')          # 空 = bpy.app.tempdir
OUT_NAME = CFG.get('out_name', '')        # 空 = {variant}_{layer}
RENDER  = CFG.get('render', True)
REPORT_SIZE = CFG.get('report_size', True)  # 渲完把文件当图片载入并 pack 一下，打印字节数（safe mode 没有 os.path.getsize）
#
# 坐标约定：Blender (X 右, Y 向里, Z 上) ↔ three.js (x, y, z) = (X, Z, -Y)。
# 相机和 index.html 一致：camBase=(0,0.3,d) 看向 (0,-1.25,0)，横屏 d=12.391、竖屏(390x844) d=16.68。
# 云团在原点，水面 y=G=-2.596，光柱 (-3.4,7.2,-1.2)→(0.9,G,0.9)（three 坐标）。云团区域（画面中上）保持干净、暗。
import bpy, bmesh, math
from mathutils import Vector, Matrix, noise
from mathutils.bvhtree import BVHTree

PFX = 'Lan_X_'
SCN = 'Lan_Xushu'
G = -2.596
def B(x, y, z): return Vector((x, -z, y))     # three → blender
def toThree(v): return [v.x, v.z, -v.y]

CAMS = {
    'land': {'pos': B(0, 0.3, 12.391), 'res': (2560, 1280), 'vfov': 48.0},
    'port': {'pos': B(0, 0.3, 16.68), 'res': (1080, 1920), 'vfov': 46.0},
    # v8 低机位仰拍：眼球右前方、贴着前景石头，水平偏右 30°，仰角 19°，22mm（2:1 画幅竖直 FOV 44°）
    'low':  {'pos': B(3.2, -2.2, 5.54), 'target': B(0.328, -0.296, -3.848), 'res': (2560, 1280), 'vfov': 44.0},
}
if CFG.get('cam_pos'): CAMS['low']['pos'] = B(*CFG['cam_pos'])
if CFG.get('cam_target'): CAMS['low']['target'] = B(*CFG['cam_target'])
if CFG.get('cam_vfov'): CAMS['low']['vfov'] = CFG['cam_vfov']
TARGET = B(0, -1.25, 0)
SCREEN = {'land': (1.6, 40.0), 'port': (0.462, 40.0), 'low': (2.0, 44.0)}
SH_TOP = B(-3.4, 7.2, -1.2); SH_BOT = B(0.9, G, 0.9); SH_DIR = (SH_BOT - SH_TOP).normalized()
SH2_BOT = B(1.9, G, -5.0)

_rng_state = [0]
def _rng_seed(s): _rng_state[0] = (int(s) * 2654435761) & 0xFFFFFFFF
def _rng_random():
    """mulberry32，确定性；不依赖 random 模块（两台机器布局一致）。"""
    _rng_state[0] = (_rng_state[0] + 0x6D2B79F5) & 0xFFFFFFFF
    t = _rng_state[0]
    t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
    t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
def rr(a, b): return a + (b - a) * _rng_random()

def cam_basis(pos, tgt=None):
    if tgt is None:
        tgt = TARGET
        for c_ in CAMS.values():
            if (c_['pos'] - pos).length < 1e-6 and 'target' in c_: tgt = c_['target']
    f = (tgt - pos).normalized(); r = f.cross(Vector((0, 0, 1))).normalized(); u = r.cross(f)
    return f, r, u
def aim(ob, target):
    ob.rotation_euler = (target - ob.location).to_track_quat('-Z', 'Y').to_euler()

# ───────────────────────── 清理（只动本脚本的东西） ─────────────────────────
def cleanup():
    sc = bpy.data.scenes.get(SCN)
    if sc is not None:
        for ob in [o for o in sc.objects if o.name.startswith(PFX)]:
            bpy.data.objects.remove(ob, do_unlink=True)
    for ob in [o for o in bpy.data.objects if o.name.startswith(PFX)]:
        bpy.data.objects.remove(ob, do_unlink=True)
    if sc is not None and len(bpy.data.scenes) > 1:
        bpy.data.scenes.remove(sc)
    for c in [c for c in bpy.data.collections if c.name.startswith(SCN) or c.name.startswith(PFX)]:
        bpy.data.collections.remove(c)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.worlds, bpy.data.images):
        for d in [d for d in coll if d.name.startswith(PFX)]:
            coll.remove(d)

# ───────────────────────── 节点小工具 ─────────────────────────
def _nodes_on(idb):
    try: idb.use_nodes = True        # Blender 5 里总是节点；4.x 需要打开
    except Exception: pass
def nn(N, t, ins=None, props=None):
    n = N.new(t)
    if props:      # safe mode：setattr 的属性名必须是字面量，所以逐个写
        if 'operation' in props: n.operation = props['operation']
        if 'feature' in props: n.feature = props['feature']
        if 'wave_type' in props: n.wave_type = props['wave_type']
        if 'bands_direction' in props: n.bands_direction = props['bands_direction']
        if 'rings_direction' in props: n.rings_direction = props['rings_direction']
        if 'wave_profile' in props: n.wave_profile = props['wave_profile']
    if ins:
        for k, v in ins.items(): n.inputs[k].default_value = v
    return n
def plug(L, sock, val):
    """val 是 socket 就连线，否则设默认值。"""
    if isinstance(val, (int, float, tuple, list)): sock.default_value = val
    else: L.new(val, sock)
def mth(N, L, op, a, b=0.0, c=None, clamp=False):
    n = N.new('ShaderNodeMath'); n.operation = op; n.use_clamp = clamp
    plug(L, n.inputs[0], a); plug(L, n.inputs[1], b)
    if c is not None: plug(L, n.inputs[2], c)
    return n.outputs[0]
def mrange(N, L, v, f0, f1, t0, t1, smooth=False):
    n = N.new('ShaderNodeMapRange')
    if smooth: n.interpolation_type = 'SMOOTHSTEP'
    plug(L, n.inputs['Value'], v)
    n.inputs['From Min'].default_value = f0; n.inputs['From Max'].default_value = f1
    n.inputs['To Min'].default_value = t0; n.inputs['To Max'].default_value = t1
    return n.outputs['Result']
def new_mat(name):
    m = bpy.data.materials.new(PFX + name); _nodes_on(m)
    N = m.node_tree.nodes
    bsdf = None; out = None
    for n in N:
        if n.type == 'BSDF_PRINCIPLED': bsdf = n
        if n.type == 'OUTPUT_MATERIAL': out = n
    if bsdf is None: bsdf = N.new('ShaderNodeBsdfPrincipled')
    if out is None: out = N.new('ShaderNodeOutputMaterial')
    if not out.inputs['Surface'].is_linked: m.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return m, N, m.node_tree.links, bsdf, out
def set_disp_method(m, meth):
    try: m.displacement_method = meth
    except Exception:
        try: m.cycles.displacement_method = meth
        except Exception: pass

# ───────────────────────── 材质 ─────────────────────────
def mat_rock(name, base=(0.040, 0.045, 0.053), wet=1.0, fs=1.0, disp=0.15, bump=0.35):
    """湿冷岩石：域扭曲 → 层理(波纹带) + 两级裂隙(Voronoi 到边距离) + 中频块面 → 真位移；细颗粒走 bump。
    颜色：冷灰蓝，层理带交替深浅，裂隙/凹处压暗；湿：竖向水痕 + 裂隙 + 近水面 → 低粗糙度 + 清漆层。"""
    m, N, L, bsdf, out = new_mat(name)
    geo = nn(N, 'ShaderNodeNewGeometry'); P = geo.outputs['Position']
    # 域扭曲
    wn = nn(N, 'ShaderNodeTexNoise', {'Scale': 0.32 * fs, 'Detail': 2.0, 'Roughness': 0.5}); L.new(P, wn.inputs['Vector'])
    wsub = nn(N, 'ShaderNodeVectorMath', props={'operation': 'SUBTRACT'}); L.new(wn.outputs['Color'], wsub.inputs[0]); wsub.inputs[1].default_value = (0.5, 0.5, 0.5)
    wsc = nn(N, 'ShaderNodeVectorMath', props={'operation': 'SCALE'}); L.new(wsub.outputs['Vector'], wsc.inputs[0]); wsc.inputs['Scale'].default_value = 1.1 / fs
    wadd = nn(N, 'ShaderNodeVectorMath', props={'operation': 'ADD'}); L.new(P, wadd.inputs[0]); L.new(wsc.outputs['Vector'], wadd.inputs[1])
    Pw = wadd.outputs['Vector']
    # 层理：沿 Z 的波纹带，被扭曲后成倾斜、断续的岩层
    wave = nn(N, 'ShaderNodeTexWave', {'Scale': 0.55 * fs, 'Distortion': 3.0, 'Detail': 3.0, 'Detail Roughness': 0.6},
              {'wave_type': 'BANDS', 'bands_direction': 'Z', 'wave_profile': 'SAW'})
    L.new(Pw, wave.inputs['Vector'])
    # 裂隙：大、小两级
    v1 = nn(N, 'ShaderNodeTexVoronoi', {'Scale': 0.85 * fs}, {'feature': 'DISTANCE_TO_EDGE'}); L.new(Pw, v1.inputs['Vector'])
    v2 = nn(N, 'ShaderNodeTexVoronoi', {'Scale': 3.4 * fs}, {'feature': 'DISTANCE_TO_EDGE'}); L.new(Pw, v2.inputs['Vector'])
    c1 = mrange(N, L, v1.outputs['Distance'], 0.0, 0.07, 0.0, 1.0, smooth=True)
    c2 = mrange(N, L, v2.outputs['Distance'], 0.0, 0.05, 0.25, 1.0, smooth=True)
    crack = mth(N, L, 'MINIMUM', c1, c2)
    # 块面：中频噪声
    mid = nn(N, 'ShaderNodeTexNoise', {'Scale': 1.9 * fs, 'Detail': 7.0, 'Roughness': 0.62}); L.new(Pw, mid.inputs['Vector'])
    # 位移高度 = 0.35*层理 + 0.75*裂隙 + 0.55*块面
    h = mth(N, L, 'MULTIPLY_ADD', wave.outputs['Fac'], 0.35, mth(N, L, 'MULTIPLY', crack, 0.75))
    h = mth(N, L, 'MULTIPLY_ADD', mid.outputs['Fac'], 0.55, h)
    dn = nn(N, 'ShaderNodeDisplacement', {'Midlevel': 1.0, 'Scale': disp}); L.new(h, dn.inputs['Height'])
    L.new(dn.outputs['Displacement'], out.inputs['Displacement'])
    set_disp_method(m, 'BOTH')
    # 细颗粒 bump
    fine = nn(N, 'ShaderNodeTexNoise', {'Scale': 16.0 * fs, 'Detail': 8.0, 'Roughness': 0.7}); L.new(P, fine.inputs['Vector'])
    bh = mth(N, L, 'MULTIPLY_ADD', fine.outputs['Fac'], 0.6, mth(N, L, 'MULTIPLY', c2, 0.5))
    bp = nn(N, 'ShaderNodeBump', {'Strength': bump, 'Distance': 0.02}); L.new(bh, bp.inputs['Height'])
    L.new(bp.outputs['Normal'], bsdf.inputs['Normal'])
    # 颜色
    ramp = nn(N, 'ShaderNodeValToRGB')
    e0, e1 = ramp.color_ramp.elements
    e0.position, e1.position = 0.25, 0.8
    e0.color = (base[0] * 0.55, base[1] * 0.55, base[2] * 0.58, 1); e1.color = (base[0] * 1.45, base[1] * 1.45, base[2] * 1.45, 1)
    tone = mth(N, L, 'MULTIPLY_ADD', wave.outputs['Fac'], 0.45, mth(N, L, 'MULTIPLY', mid.outputs['Fac'], 0.65))
    L.new(tone, ramp.inputs['Fac'])
    cav = mrange(N, L, crack, 0.0, 1.0, 0.3, 1.0)
    spk = nn(N, 'ShaderNodeTexNoise', {'Scale': 9.0 * fs, 'Detail': 2.0}); L.new(P, spk.inputs['Vector'])
    spk_m = mrange(N, L, spk.outputs['Fac'], 0.3, 0.75, 0.75, 1.2)
    cm = mth(N, L, 'MULTIPLY', cav, spk_m)
    col = nn(N, 'ShaderNodeVectorMath', props={'operation': 'SCALE'}); L.new(ramp.outputs['Color'], col.inputs[0]); L.new(cm, col.inputs['Scale'])
    L.new(col.outputs['Vector'], bsdf.inputs['Base Color'])
    # 湿：竖向拉长的水痕
    mp = nn(N, 'ShaderNodeMapping', {'Scale': (1.6 * fs, 1.6 * fs, 0.10 * fs)}); L.new(P, mp.inputs['Vector'])
    st = nn(N, 'ShaderNodeTexNoise', {'Scale': 1.7, 'Detail': 5.0, 'Roughness': 0.6}); L.new(mp.outputs['Vector'], st.inputs['Vector'])
    streak = mrange(N, L, st.outputs['Fac'], 0.5, 0.64, 0.0, 1.0, smooth=True)
    sep = nn(N, 'ShaderNodeSeparateXYZ'); L.new(P, sep.inputs['Vector'])
    nearw = mrange(N, L, sep.outputs['Z'], G + 1.6, G + 0.05, 0.0, 1.0, smooth=True)
    cwet = mrange(N, L, crack, 0.0, 0.6, 0.7, 0.0)
    wetm = mth(N, L, 'MAXIMUM', mth(N, L, 'MAXIMUM', streak, nearw), cwet)
    wetm = mth(N, L, 'MULTIPLY', wetm, wet, clamp=True)
    rough = mrange(N, L, wetm, 0.0, 1.0, 0.7, 0.17)
    rv = mrange(N, L, fine.outputs['Fac'], 0.3, 0.7, 0.88, 1.12)
    L.new(mth(N, L, 'MULTIPLY', rough, rv), bsdf.inputs['Roughness'])
    L.new(mrange(N, L, wetm, 0.0, 1.0, 0.0, 0.75), bsdf.inputs['Coat Weight'])
    bsdf.inputs['Coat Roughness'].default_value = 0.07
    bsdf.inputs['Coat IOR'].default_value = 1.33
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
    return m

def mat_water():
    """暗、静的镜面水：几乎全反射，粗糙度极低；只在光柱落点附近有一圈圈很淡的涟漪，远处有极缓的起伏。"""
    m, N, L, bsdf, out = new_mat('water')
    bsdf.inputs['Base Color'].default_value = (0.0008, 0.001, 0.0014, 1)
    bsdf.inputs['IOR'].default_value = 1.33
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
    geo = nn(N, 'ShaderNodeNewGeometry'); P = geo.outputs['Position']
    rn = nn(N, 'ShaderNodeTexNoise', {'Scale': 0.12, 'Detail': 2.0}); L.new(P, rn.inputs['Vector'])
    L.new(mrange(N, L, rn.outputs['Fac'], 0.35, 0.7, 0.006, 0.03), bsdf.inputs['Roughness'])
    # 涟漪：以光柱落点为圆心的同心环，随距离衰减
    mpr = nn(N, 'ShaderNodeMapping', {'Location': (-SH_BOT.x, -SH_BOT.y, 0.0)}); L.new(P, mpr.inputs['Vector'])
    rings = nn(N, 'ShaderNodeTexWave', {'Scale': 0.9, 'Distortion': 0.6, 'Detail': 1.0},
               {'wave_type': 'RINGS', 'rings_direction': 'SPHERICAL', 'wave_profile': 'SIN'})
    L.new(mpr.outputs['Vector'], rings.inputs['Vector'])
    dd = nn(N, 'ShaderNodeVectorMath', props={'operation': 'LENGTH'}); L.new(mpr.outputs['Vector'], dd.inputs[0])
    fade = mrange(N, L, dd.outputs['Value'], 0.5, 7.0, 1.0, 0.0, smooth=True)
    rh = mth(N, L, 'MULTIPLY', rings.outputs['Fac'], fade)
    # 远处极缓的起伏（让倒影有一点"水"的摆动）
    mps = nn(N, 'ShaderNodeMapping', {'Scale': (1.0, 2.6, 1.0)}); L.new(P, mps.inputs['Vector'])
    sw = nn(N, 'ShaderNodeTexNoise', {'Scale': 1.1, 'Detail': 3.0, 'Roughness': 0.5}); L.new(mps.outputs['Vector'], sw.inputs['Vector'])
    hh = mth(N, L, 'MULTIPLY_ADD', sw.outputs['Fac'], 0.35, rh)
    bp = nn(N, 'ShaderNodeBump', {'Strength': 0.06, 'Distance': 0.02}); L.new(hh, bp.inputs['Height'])
    L.new(bp.outputs['Normal'], bsdf.inputs['Normal'])
    return m

def mat_city():
    m, N, L, bsdf, out = new_mat('cityStone')
    geo = nn(N, 'ShaderNodeNewGeometry')
    n = nn(N, 'ShaderNodeTexNoise', {'Scale': 0.9, 'Detail': 4.0}); L.new(geo.outputs['Position'], n.inputs['Vector'])
    ramp = nn(N, 'ShaderNodeValToRGB'); e0, e1 = ramp.color_ramp.elements
    e0.color = (0.035, 0.040, 0.050, 1); e1.color = (0.065, 0.072, 0.086, 1)
    L.new(n.outputs['Fac'], ramp.inputs['Fac']); L.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.7
    return m

# ───────────────────────── 网格工具 ─────────────────────────
COLL = {}
ROOT = [None]
def coll(name):
    if name not in COLL:
        c = bpy.data.collections.new(PFX + name); ROOT[0].children.link(c); COLL[name] = c
    return COLL[name]

def obj_from_bm(bm, name, mat, cname, smooth=True, subsurf=0, bevel=0.0):
    me = bpy.data.meshes.new(PFX + name); bm.to_mesh(me); bm.free()
    if smooth:
        for p in me.polygons: p.use_smooth = True
    me.materials.append(mat)
    ob = bpy.data.objects.new(PFX + name, me); coll(cname).objects.link(ob)
    if bevel > 0:
        md = ob.modifiers.new('bevel', 'BEVEL'); md.width = bevel; md.segments = 2; md.limit_method = 'ANGLE'
    if subsurf and SUBSURF:
        md = ob.modifiers.new('sub', 'SUBSURF'); md.levels = 0; md.render_levels = subsurf
    return ob

def fbm(p, octv=5): return noise.fractal(p, 0.8, 2.1, octv, noise_basis='PERLIN_NEW')
def ridged(p, octv=5): return noise.ridged_multi_fractal(p, 0.9, 2.1, octv, 1.0, 2.0, noise_basis='PERLIN_NEW')

def blob(bm, center, radii, subdiv=5, amp=0.25, freq=0.6, rotz=0.0, seed=0.0, zmin=None, ymax=None, chips=8, chip_depth=(0.62, 0.86)):
    """一块岩体：icosphere → 噪声位移 → 被若干随机平面"削"出断裂面（真实岩石的块面感，而不是揉皱的泥团）。"""
    res = bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
    rmin = min(radii); off = Vector((seed * 13.1, seed * 7.7, seed * 3.3))
    cr, sr = math.cos(rotz), math.sin(rotz)
    planes = []
    for k in range(chips):
        n = noise.noise_vector(Vector((seed * 1.37 + k * 3.11, k * 0.71 - seed, seed * 0.53 + 9.0)))
        if n.length < 1e-4: continue
        n = n.normalized()
        if k % 3 == 0: n = Vector((n.x * 0.35, n.y * 0.35, n.z if abs(n.z) > 0.3 else 0.8)).normalized()   # 一部分是近水平的层面
        sup = math.sqrt((radii[0] * n.x) ** 2 + (radii[1] * n.y) ** 2 + (radii[2] * n.z) ** 2)
        u = 0.5 + 0.5 * noise.noise(Vector((seed * 2.3 + k, k * 1.9, 4.2)))
        planes.append((n, sup * (chip_depth[0] + (chip_depth[1] - chip_depth[0]) * u)))
    for v in res['verts']:
        d = v.co.normalized()
        p = Vector((d.x * radii[0], d.y * radii[1], d.z * radii[2]))
        w = p * (freq / rmin * 2.0) + off
        disp = fbm(w, 6) * 0.55 + ridged(w * 0.8, 5) * 0.3 - 0.15
        disp += 0.05 * math.sin((p.z + fbm(w * 0.5, 2) * 1.2) * 6.0)          # 岩层纹
        p = p + d * (disp * amp * rmin)
        for (n, dpl) in planes:
            s = p.dot(n) - dpl
            if s > 0: p = p - n * (s * 0.9)
        p = Vector((p.x * cr - p.y * sr, p.x * sr + p.y * cr, p.z)) + center
        if zmin is not None and p.z < zmin: p.z = zmin - 0.05 * (zmin - p.z)
        if ymax is not None and p.y > ymax: p.y = ymax
        v.co = p

def prof_r(prof, t):
    """车削体的半径轮廓（代替 lambda）：prof = (种类, 参数...)。"""
    kind = prof[0]
    if kind == 'taper':                       # ('taper', r, pow, add)
        return prof[1] * (1 - t) ** prof[2] + prof[3]
    if kind == 'drip':                        # ('drip', r, pow, add, phase)：钟乳石，带一圈圈生长环和尖端的小水滴
        base = prof[1] * (1 - t) ** prof[2] + prof[3]
        ring = 1 + 0.07 * math.sin(t * 46 + prof[4]) + 0.04 * math.sin(t * 17 + prof[4] * 2)
        bulb = 1 + 0.6 * math.exp(-((t - 0.965) / 0.025) ** 2)
        return base * ring * bulb
    if kind == 'column':                      # ('column', r, phase)
        r, ph = prof[1], prof[2]
        return r * (0.55 + 1.1 * (t ** 4 + (1 - t) ** 5) + 0.18 * math.sin(t * 9 + ph) + 0.1 * math.sin(t * 23 + ph * 2)
                    + 0.035 * math.sin(t * 61 + ph * 3))
    if kind == 'spire':                       # ('spire', r)
        return prof[1] * (1 - t) ** 1.2 + 0.03
    return 0.1

def tube(bm, cx, cy, z0, z1, prof, segs=12, rings=18, namp=0.25, freq=1.2, seed=0.0, lean=(0, 0)):
    """竖直方向的车削体（钟乳石、石笋、石柱、尖塔）；z0→z1 可以向下。"""
    off = Vector((seed * 5.3, seed * 2.9, seed * 8.1)); rows = []
    for j in range(rings + 1):
        t = j / rings; z = z0 + (z1 - z0) * t; r = prof_r(prof, t); row = []
        lx, ly = lean[0] * t * t, lean[1] * t * t
        for i in range(segs):
            a = 2 * math.pi * i / segs
            p = Vector((cx + lx + math.cos(a) * r, cy + ly + math.sin(a) * r, z))
            k = 1 + namp * fbm(p * freq + off, 4) if namp else 1.0
            row.append(bm.verts.new((cx + lx + math.cos(a) * r * k, cy + ly + math.sin(a) * r * k, z)))
        rows.append(row)
    for j in range(rings):
        for i in range(segs):
            a, b2 = rows[j][i], rows[j][(i + 1) % segs]; c, d = rows[j + 1][i], rows[j + 1][(i + 1) % segs]
            bm.faces.new((a, b2, d, c))
    for row, z in ((rows[0], z0), (rows[-1], z1)):
        cxy = Vector((0, 0, 0))
        for v in row: cxy = cxy + v.co
        cxy = cxy / len(row)
        tip = bm.verts.new((cxy.x, cxy.y, z))
        for i in range(segs): bm.faces.new((row[i], row[(i + 1) % segs], tip))

def box(bm, cx, cy, z0, w, d, h, rot=0.0):
    m = Matrix.Translation((cx, cy, z0 + h / 2)) @ Matrix.Rotation(rot, 4, 'Z') @ Matrix.Diagonal((w, d, h, 1))
    bmesh.ops.create_cube(bm, size=1.0, matrix=m)
def pyramid(bm, cx, cy, z0, w, d, h, segs=4):
    m = Matrix.Translation((cx, cy, z0 + h / 2)) @ Matrix.Diagonal((w * 0.7071 * 2, d * 0.7071 * 2, 1, 1)) @ Matrix.Rotation(math.pi / 4 if segs == 4 else 0, 4, 'Z')
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs, radius1=0.5, radius2=0.0, depth=h, matrix=m)

# ═════════════════════════ 搭场景 ═════════════════════════
def build():
    cleanup()
    scene = bpy.data.scenes.new(SCN)
    ROOT[0] = bpy.data.collections.new(SCN); scene.collection.children.link(ROOT[0])
    COLL.clear()
    _rng_seed(20261026)

    M_ROCK = mat_rock('rock', base=(0.040, 0.045, 0.053), wet=0.9, fs=0.55, disp=0.45, bump=0.3)
    M_FORM = mat_rock('formation', base=(0.046, 0.050, 0.058), wet=1.1, fs=1.1, disp=0.10, bump=0.3)
    M_NEAR = mat_rock('rockNear', base=(0.034, 0.038, 0.046), wet=1.15, fs=1.0, disp=0.16, bump=0.4)
    M_FLOOR = mat_rock('floor', base=(0.036, 0.040, 0.047), wet=1.6, fs=1.4, disp=0.04, bump=0.3)
    M_WATER = mat_water(); M_CITY = mat_city()

    # ── 大溶洞外壳 ──
    SHELL_C, SHELL_R = Vector((0, 36, 3.0)), Vector((44, 64, 23))
    bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=384, v_segments=192, radius=1.0)
    for v in bm.verts:
        d = v.co.normalized()
        p = Vector((d.x * SHELL_R.x, d.y * SHELL_R.y, d.z * SHELL_R.z)) + SHELL_C
        w = p * 0.06
        disp = fbm(w, 6) * 7.0 + ridged(p * 0.11, 5) * 2.4 + 0.6 * math.sin(p.z * 0.9 + fbm(w * 2, 2) * 3)
        if d.z > 0.25: disp += max(0.0, ridged(Vector((p.x * 0.35, p.y * 0.35, 0.0)), 4) - 0.6) * 6.0
        v.co = p - d * disp
    bmesh.ops.reverse_faces(bm, faces=bm.faces)
    shell_bm = bm.copy()
    obj_from_bm(bm, 'shell', M_ROCK, 'cave', subsurf=1)
    shell_bvh = BVHTree.FromBMesh(shell_bm); shell_bm.free()

    def ceiling_at(x, y):
        hit = shell_bvh.ray_cast(Vector((x, y, G + 0.5)), Vector((0, 0, 1)), 200)
        return hit[0].z if hit[0] is not None else None

    # ── 水面（暗镜） ──
    bm = bmesh.new(); bmesh.ops.create_grid(bm, x_segments=2, y_segments=2, size=160, matrix=Matrix.Translation((0, 40, G)))
    obj_from_bm(bm, 'water', M_WATER, 'water', smooth=False)
    # 水下岩床：大部分没在水里，只有零星湿亮的石头露出（v5 草稿里大片露出的地面在光柱下成了一团怪亮斑）
    bm = bmesh.new(); NXf, NYf = 300, 260
    X0, X1, Y0, Y1 = -26.0, 26.0, -18.0, 42.0
    vv = []
    for j in range(NYf + 1):
        row = []
        for i in range(NXf + 1):
            x = X0 + (X1 - X0) * i / NXf; y = Y0 + (Y1 - Y0) * j / NYf
            p = Vector((x, y, 0))
            h = fbm(p * 0.35, 5) * 0.16 + fbm(p * 1.6, 3) * 0.035 - 0.36
            h -= max(0.0, (y - 18) / 24) * 0.35
            row.append(bm.verts.new((x, y, G + h)))
        vv.append(row)
    for j in range(NYf):
        for i in range(NXf):
            bm.faces.new((vv[j][i], vv[j][i + 1], vv[j + 1][i + 1], vv[j + 1][i]))
    obj_from_bm(bm, 'floor', M_FLOOR, 'cave')
    # 水里零星几块半沉的湿石（避开画面正中——云团的倒影在那儿）
    bm = bmesh.new()
    for k, (x3, z3, r) in enumerate([(-3.4, 2.6, 0.55), (4.1, -1.2, 0.7), (-6.8, -5.5, 1.1), (7.4, -8.5, 1.3), (-2.9, -2.4, 0.35), (3.0, 3.4, 0.4)]):
        c = B(x3, G, z3); c.z = G + r * 0.12
        blob(bm, c, (r * 1.3, r, r * 0.6), subdiv=5, amp=0.3, freq=0.7, rotz=k * 0.9, seed=70 + k, zmin=G - 0.6, chips=7)
    obj_from_bm(bm, 'pondRocks', M_NEAR, 'cave', subsurf=1)

    # ── 前景：洞口岩框（按各自相机的屏幕位置摆） ──
    def place(variant, sx, sy, dist):
        pos = CAMS[variant]['pos']; f, r, u = cam_basis(pos); asp, vf = SCREEN[variant]
        hh = math.tan(math.radians(vf / 2)) * dist
        return pos + f * dist + r * (sx * hh * asp) + u * (sy * hh), hh
    FG = {
        'land': [
            (-1.22, 0.10, 7.0, 0.80, 0.55, 1.70), (-1.05, -0.88, 5.0, 0.62, 0.40, 0.34),
            (1.28, 0.30, 8.2, 0.85, 0.55, 1.70), (1.02, -0.92, 4.6, 0.62, 0.40, 0.36),
            (-0.62, 1.22, 6.0, 1.05, 0.50, 0.36), (0.66, 1.26, 7.4, 1.10, 0.50, 0.38),
            (0.02, 1.36, 6.6, 0.70, 0.45, 0.30), (-0.42, -1.08, 3.8, 0.30, 0.25, 0.14),
            (0.48, -1.12, 4.2, 0.26, 0.25, 0.13),
        ],
        'port': [
            (0.15, 1.22, 3.6, 1.70, 0.25, 0.30), (-1.36, 0.55, 3.2, 0.19, 0.22, 0.95),
            (1.36, -0.25, 3.0, 0.19, 0.22, 0.90), (-0.35, -1.12, 3.1, 0.55, 0.22, 0.20),
            (0.75, -1.15, 3.4, 0.40, 0.22, 0.16),
        ],
        # v8 低机位：右侧石壁贴着镜头压住画面右缘，右下一块近石，左下一点碎石
        'low': [
            (1.12, 0.15, 3.0, 0.95, 0.75, 2.6), (0.92, -0.85, 2.4, 0.55, 0.45, 0.45),
            (0.38, -1.02, 2.8, 0.42, 0.35, 0.16), (-0.55, -1.05, 4.2, 0.30, 0.30, 0.12),
        ],
    }
    FG_BM = {'land': bmesh.new(), 'port': bmesh.new(), 'low': bmesh.new()}
    for vname in ('land', 'port', 'low'):
        bm = FG_BM[vname]
        for k, spec in enumerate(FG[vname]):
            sx, sy, dist, rx, ry, rz = spec
            c, hh = place(vname, sx, sy, dist)
            rad = (rx * hh, ry * hh, rz * hh)
            if sy < -0.7: c.z = G + rad[2] * 0.25
            ymax = CAMS['land']['pos'].y - 0.35 if vname == 'port' else None
            blob(bm, c, rad, subdiv=6, amp=0.26, freq=0.55, rotz=rr(-0.3, 0.3), seed=k * 3.7 + (11 if vname == 'port' else 0),
                 zmin=G - 1.0, ymax=ymax, chips=9)
    bm = FG_BM['land']
    for k in range(30):                                     # 顶岩檐下挂的小钟乳石
        sx = rr(-1.1, 1.1); c, hh = place('land', sx, 1.0, rr(5.5, 7.5))
        top = c.z + 0.6; h = rr(0.25, 0.9)
        tube(bm, c.x, c.y, top, top - h - 0.6, ('drip', rr(0.05, 0.14), 1.4, 0.004, rr(0, 6.28)), segs=10, rings=16, namp=0.3, seed=k)
    fg_objs = {}
    for v_ in ('land', 'port', 'low'):
        fg_objs[v_] = obj_from_bm(FG_BM[v_], 'foreground_' + v_, M_NEAR, 'cave', subsurf=1)

    # ── 中景 / 远景：石柱、石笋、垂幕钟乳 ──
    def near_shaft(x, y, z0, z1, clear=3.2):
        for z in (z0, (z0 + z1) / 2, z1):
            p = Vector((x, y, z)); s_ = (p - SH_BOT).dot(SH_DIR)
            if s_ > 1.0: continue
            if (p - (SH_BOT + SH_DIR * s_)).length < clear + 0.12 * (-s_): return True
        return False
    bm = bmesh.new()
    _rng_seed(29)
    COLS = [(-8.5, 14, 1.8), (9.5, 19, 2.0), (-11.5, 29, 1.6), (13.5, 37, 1.9), (-17, 46, 2.6), (19, 52, 2.8), (-21, 64, 2.2), (23, 70, 2.0)]
    for k, (x, y, r) in enumerate(COLS):
        if k in CFG.get('skip_cols', []): continue
        top = ceiling_at(x, y) or 25
        ph = rr(0, 6.28)
        tube(bm, x, y, G - 1.0, top + 1.0, ('column', r, ph), segs=28, rings=72, namp=0.42, freq=0.45, seed=k + 20,
             lean=(rr(-2.5, 2.5), rr(-1.5, 1.5)))
    for k in range(70):                                     # 石笋（避开画面中央通道）
        y = rr(-2, 70); x = rr(-30, 30)
        if abs(x) < 2.5 + 0.22 * (y + 12): continue
        h = rr(1.0, 5.5) * (1 + y / 60)
        if near_shaft(x, y, G, G + h): continue
        tube(bm, x, y, G - 0.3, G + h, ('drip', rr(0.25, 0.9), 1.3, 0.01, rr(0, 6.28)), segs=12, rings=16, namp=0.3, seed=k + 50)
    for k in range(220):                                    # 顶上的钟乳石 / 长垂幕
        y = rr(-6, 85); x = rr(-34, 34)
        top = ceiling_at(x, y)
        if top is None: continue
        long_ = _rng_random() < 0.35
        h = rr(6, 15) if long_ else rr(1.5, 5)
        if abs(x) < 3 + 0.2 * (y + 12) and top - h < 4 + y * 0.18: h = max(1.0, top - (4 + y * 0.18))   # 别垂到云团正后面
        if near_shaft(x, y, top - h, top): continue
        r0 = rr(0.3, 1.2) * (1.4 if long_ else 1)
        tube(bm, x, y, top + 0.8, top - h, ('drip', r0, 1.6, 0.01, rr(0, 6.28)), segs=12, rings=28 if long_ else 18, namp=0.3,
             seed=k + 200, lean=(rr(-0.3, 0.3), rr(-0.3, 0.3)))
    obj_from_bm(bm, 'formations', M_FORM, 'cave', subsurf=1)

    # ── 恶罗海城（只在 city 层出图）：退台岩基 + 错落屋顶（坡顶/穹顶/叠檐）+ 塔群，倒角避免"积木块" ──
    _rng_seed(47)
    bm = bmesh.new()
    for i in range(6):
        y0 = 52 + i * 5.0; ztop = G + 1.2 + i * 2.3; w = 46 - i * 6
        box(bm, 0, y0 + 2.5, G - 1, w, 5.2, ztop - G + 1)
        for s in (-1, 1):
            for q in range(4): box(bm, s * w * 0.18, y0 - 0.4 - q * 0.35, G - 1 + (ztop - G + 1) * q / 4, 1.6, 0.4, (ztop - G + 1) * (q + 1) / 4)
        n = int(w * 0.8)
        for q in range(n):
            x = -w / 2 + 0.8 + (w - 1.6) * (q + rr(-0.35, 0.35)) / n
            if _rng_random() < 0.22: continue
            bw, bd = rr(0.8, 2.2), rr(0.9, 2.6)
            bh = rr(0.5, 2.2) * (1.35 if abs(x) < 6 else 1)
            yy = y0 + rr(0.6, 3.6)
            box(bm, x, yy, ztop, bw, bd, bh, rot=rr(-0.06, 0.06))
            rk = _rng_random()
            if rk < 0.28: pyramid(bm, x, yy, ztop + bh, bw * 1.05, bd * 1.05, rr(0.5, 1.2))              # 坡顶
            elif rk < 0.42: box(bm, x, yy, ztop + bh, bw * 0.7, bd * 0.7, bh * 0.4); pyramid(bm, x, yy, ztop + bh * 1.4, bw * 0.75, bd * 0.75, 0.5)
            elif rk < 0.55: blob(bm, Vector((x, yy, ztop + bh)), (bw * 0.45, bd * 0.45, bw * 0.5), subdiv=2, amp=0.0, zmin=ztop + bh, chips=0)
            if _rng_random() < 0.12:
                th = rr(2.0, 5.0)
                box(bm, x, yy, ztop + bh, bw * 0.45, bw * 0.45, th)
                tube(bm, x, yy, ztop + bh + th, ztop + bh + th + rr(2, 6), ('spire', bw * 0.3), segs=8, rings=4, namp=0.0)
        for q in range(int(w)):
            if q % 2 == 0: box(bm, -w / 2 + q + 0.5, y0 - 0.1, ztop, 0.55, 0.5, 0.45)
    zb = G + 1.2 + 4 * 2.3
    for q in range(6):
        s = 9.0 - q * 1.35; box(bm, 0, 74, zb + q * 1.25, s, s, 1.25)
    box(bm, 0, 74, zb + 7.5, 1.8, 1.8, 6.0)
    tube(bm, 0, 74, zb + 13.5, zb + 22, ('taper', 0.9, 1.2, 0.03), segs=8, rings=6, namp=0.0)
    for s in (-1, 1):
        box(bm, s * 3.2, 74, zb + 7.5, 1.0, 1.0, 4.0)
        tube(bm, s * 3.2, 74, zb + 11.5, zb + 17, ('taper', 0.5, 1.0, 0.02), segs=8, rings=4, namp=0.0)
        for q in range(3):
            box(bm, s * (5.5 + q * 2.2), 72 + q, zb + 3.0, 0.9, 0.9, 3.0 + q)
            tube(bm, s * (5.5 + q * 2.2), 72 + q, zb + 6.0 + q, zb + 9.0 + q * 1.5, ('spire', 0.45), segs=6, rings=3, namp=0.0)
    for k in range(14):
        s = -1 if k % 2 else 1; x = s * rr(14, 28); y = rr(58, 84); z0 = G + rr(2, 10)
        th_ = rr(2.5, 6)
        box(bm, x, y, z0, rr(1, 2), rr(1, 2), th_)
        if _rng_random() < 0.6:
            tube(bm, x, y, z0 + th_, z0 + th_ + rr(2, 5), ('taper', 0.6, 1.0, 0.02), segs=6, rings=3, namp=0.0)
    for k in range(60):
        s = -1 if k % 2 else 1; y = rr(48, 82); z = G + rr(3, 18)
        x = s * (28 + (z - G) * 0.15 + rr(-2, 2))
        box(bm, x, y, z, rr(1.0, 2.6), rr(0.8, 1.6), rr(0.8, 2.0))
    blob(bm, Vector((0, 86, G + 4)), (34, 10, 10), subdiv=4, amp=0.2, freq=0.4, seed=91, chips=6)
    obj_from_bm(bm, 'city', M_CITY, 'city', smooth=False, bevel=0.05)
    # 每段台地前面的岩崖（让城坐在山岩上，而不是摆在平板上）
    bm = bmesh.new()
    for i in range(6):
        y0 = 52 + i * 5.0; ztop = G + 1.2 + i * 2.3; w = 46 - i * 6
        blob(bm, Vector((0, y0 + 0.6, (G + ztop) / 2 - 0.2)), (w * 0.52, 1.1, (ztop - G) / 2 + 0.3), subdiv=4, amp=0.3, freq=0.9, seed=60 + i, chips=5)
    obj_from_bm(bm, 'cityCliff', M_ROCK, 'city')

    # ── 灯光：基准强度存在 lan_base，出图时按 WEIGHTS 乘进去 ──
    def light(name, kind, loc, energy, color, cname, group='key'):
        ld = bpy.data.lights.new(PFX + name, kind); ld.energy = energy; ld.color = color
        ld['lan_base'] = float(energy); ld['lan_group'] = group
        ob = bpy.data.objects.new(PFX + name, ld); ob.location = loc; coll(cname).objects.link(ob)
        return ob
    def below_ceiling(bot, d, margin=1.5):
        for zz in range(24, 6, -1):
            p = bot - d * ((zz - G) / -d.z); c = ceiling_at(p.x, p.y)
            if c is not None and p.z < c - margin: return p
        return bot - d * 12
    def wall_y(x, z):
        hit = shell_bvh.ray_cast(Vector((x, 40, z)), Vector((0, 1, 0)), 200)
        return hit[0].y if hit[0] is not None else 95
    sp = light('shaft', 'SPOT', below_ceiling(SH_BOT, SH_DIR), 26000, (0.70, 0.82, 1.0), 'lights', 'key')
    sp.data.spot_size = math.radians(15); sp.data.spot_blend = 0.9; sp.data.shadow_soft_size = 1.1
    aim(sp, SH_BOT)
    p2 = B(3.6, 8.5, -7.5) + (B(3.6, 8.5, -7.5) - SH2_BOT).normalized() * 12
    sp2 = light('shaft2', 'SPOT', p2, 5200, (0.66, 0.78, 1.0), 'lights', 'key')
    sp2.data.spot_size = math.radians(9); sp2.data.spot_blend = 0.9; sp2.data.shadow_soft_size = 0.7
    aim(sp2, SH2_BOT)
    cg = light('cloudGlow', 'POINT', Vector((0, 0, 0.1)), 300, (0.62, 0.74, 0.95), 'lights', 'cloud')
    cg.data.shadow_soft_size = 1.6; cg.visible_glossy = False
    vn = light('vanish', 'POINT', Vector((0, wall_y(0, 1.5) - 6, 1.5)), 700, (0.7, 0.82, 1.0), 'lights', 'back')
    vn.data.shadow_soft_size = 4; vn.visible_glossy = False
    bo = light('bounce', 'AREA', SH_BOT + Vector((0, 0, 0.25)), 900, (0.66, 0.78, 1.0), 'lights', 'bounce')
    bo.data.size = 5.0; bo.data.shape = 'DISK'; bo.rotation_euler = (math.pi, 0, 0)
    bo.visible_glossy = False; bo.visible_camera = False
    bo2 = light('bounceFwd', 'AREA', SH_BOT + Vector((0, -1.5, 1.2)), 500, (0.66, 0.78, 1.0), 'lights', 'bounce')
    bo2.data.size = 7.0; aim(bo2, CAMS['land']['pos'] + Vector((0, 0, -1.0))); bo2.visible_glossy = False; bo2.visible_camera = False
    fl = light('fill', 'AREA', Vector((0, 22, 15)), 9000, (0.62, 0.74, 1.0), 'lights', 'fill')
    fl.data.shape = 'RECTANGLE'; fl.data.size = 46; fl.data.size_y = 40; fl.visible_glossy = False; fl.visible_camera = False
    cr_ = light('cityRim', 'AREA', Vector((0, wall_y(0, 9) - 3, 9)), 60000, (0.66, 0.78, 1.0), 'citylights', 'rim')
    cr_.data.shape = 'RECTANGLE'; cr_.data.size = 40; cr_.data.size_y = 18
    aim(cr_, Vector((0, 50, 3)))

    # ── 世界：几乎全黑的冷色 ──
    world = bpy.data.worlds.new(PFX + 'World'); scene.world = world; _nodes_on(world)
    wn = world.node_tree.nodes; wl = world.node_tree.links
    bg = None; wo = None
    for n in wn:
        if n.type == 'BACKGROUND': bg = n
        if n.type == 'OUTPUT_WORLD': wo = n
    if bg is None: bg = wn.new('ShaderNodeBackground')
    if wo is None: wo = wn.new('ShaderNodeOutputWorld')
    if not wo.inputs['Surface'].is_linked: wl.new(bg.outputs['Background'], wo.inputs['Surface'])
    bg.inputs['Color'].default_value = (0.0015, 0.002, 0.003, 1); bg.inputs['Strength'].default_value = 1.0

    # ── 相机 ──
    cd = bpy.data.cameras.new(PFX + 'Cam'); cd.sensor_fit = 'VERTICAL'; cd.sensor_height = 24.0
    cd.clip_start = 0.05; cd.clip_end = 400
    cam = bpy.data.objects.new(PFX + 'Cam', cd); ROOT[0].objects.link(cam); scene.camera = cam
    scene['lan_shaft_pos'] = toThree(sp.location); scene['lan_shaft_dir'] = toThree(SH_DIR)
    scene['lan_shaft2_pos'] = toThree(sp2.location); scene['lan_shaft2_dir'] = toThree((SH2_BOT - sp2.location).normalized())
    print('[scene] built; shaft spot at', tuple(round(c, 2) for c in sp.location), flush=True)
    return scene

# ═════════════════════════ 出图设置 ═════════════════════════
def configure(scene):
    VL = scene.view_layers[0]
    C = CAMS[VARIANT]
    cam = bpy.data.objects[PFX + 'Cam']
    cam.data.lens = 12.0 / math.tan(math.radians(C['vfov'] / 2))
    cam.location = C['pos']; aim(cam, C.get('target', TARGET)); scene.camera = cam
    for v_ in ('land', 'port', 'low'):
        ob = bpy.data.objects.get(PFX + 'foreground_' + v_)
        if ob is not None: ob.hide_render = (v_ != VARIANT)    # 每个版本只出自己的岩框
    # 层切换用物体级开关（同一次执行里新建的场景 layer_collection 可能还没同步）
    is_cave = LAYER == 'cave'
    def objs(cname):
        c = bpy.data.collections.get(PFX + cname)
        return list(c.objects) if c is not None else []
    for ob in objs('city') + objs('citylights'): ob.hide_render = is_cave
    co = CFG.get('city_off', [0, 0, 0]) if VARIANT == 'low' else [0, 0, 0]   # three 坐标的平移（x, y, z）
    for ob in objs('city') + objs('citylights'):
        if 'lan_base' not in ob: ob['lan_base'] = list(ob.location)
        b_ = ob['lan_base']; ob.location = (b_[0] + co[0], b_[1] - co[2], b_[2] + co[1])
    for ob in objs('lights'): ob.hide_render = not is_cave
    for ob in objs('cave') + objs('water'):
        ob.is_holdout = not is_cave
        if ob.name.startswith(PFX + 'foreground_'): ob.hide_render = (ob.name != PFX + 'foreground_' + VARIANT)
    scene.render.film_transparent = not is_cave
    # 灯光：权重乘进强度（或者出灯光组）
    for ld in bpy.data.lights:
        if not ld.name.startswith(PFX): continue
        g = ld.get('lan_group', 'key')
        ld.energy = ld['lan_base'] * (1.0 if LIGHTGROUPS else float(WEIGHTS.get(g, 1.0)))
    if LIGHTGROUPS:
        for g in ('key', 'cloud', 'bounce', 'fill', 'back', 'rim'):
            if g not in VL.lightgroups: VL.lightgroups.add(name=g)
        for ob in scene.objects:
            if ob.type == 'LIGHT' and ob.name.startswith(PFX): ob.lightgroup = ob.data.get('lan_group', 'key')
    R = scene.render
    R.engine = 'CYCLES'
    if RES: R.resolution_x, R.resolution_y = RES[0], RES[1]; R.resolution_percentage = 100
    else: R.resolution_x, R.resolution_y = C['res']; R.resolution_percentage = SCALE
    if THREADS: R.threads_mode = 'FIXED'; R.threads = THREADS
    cy = scene.cycles
    cy.device = DEVICE; cy.samples = SAMPLES; cy.use_adaptive_sampling = True; cy.adaptive_threshold = NOISE_T
    cy.use_denoising = bool(DENOISE)
    if DENOISE:
        try: cy.denoiser = DENOISER
        except Exception:
            cy.denoiser = 'OPENIMAGEDENOISE'
        try: cy.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
        except Exception: pass
        try: cy.denoising_prefilter = 'ACCURATE'
        except Exception: pass
        try: cy.denoising_use_gpu = True
        except Exception: pass
    cy.max_bounces = 6; cy.diffuse_bounces = 2; cy.glossy_bounces = 3; cy.transmission_bounces = 0
    cy.volume_bounces = 0; cy.transparent_max_bounces = 4; cy.blur_glossy = 0.5
    cy.caustics_reflective = False; cy.caustics_refractive = False
    cy.sample_clamp_indirect = 4.0; cy.seed = 7
    full = PASSES == 'full'
    # 'depth'（终版默认）：Combined + Z（单通道；Blender 存数据通道总是 32 位，Position 三通道太大）| 'lean'：Combined + Position | 'full'：再加 Normal、DiffCol
    VL.use_pass_position = PASSES in ('lean', 'full'); VL.use_pass_z = PASSES in ('depth', 'full')
    VL.use_pass_normal = full; VL.use_pass_diffuse_color = full
    ims = R.image_settings
    if FORMAT == 'single':
        try: ims.media_type = 'IMAGE'
        except Exception: pass
        ims.file_format = 'OPEN_EXR'
    else:
        try:
            ims.media_type = 'MULTI_LAYER_IMAGE'      # Blender 5.x
            ims.file_format = 'OPEN_EXR'
        except Exception:
            ims.file_format = 'OPEN_EXR_MULTILAYER'   # Blender 4.x
    ims.color_depth = '16' if HALF else '32'; ims.exr_codec = CODEC
    try: ims.color_mode = 'RGBA'
    except Exception: pass
    od = OUT_DIR or bpy.app.tempdir.replace('\\', '/')
    tag = OUT_NAME or (VARIANT + '_' + LAYER)
    R.filepath = od.rstrip('/') + '/' + tag
    R.use_file_extension = True
    R.use_compositing = False; R.use_sequencer = False
    R.use_persistent_data = True       # 连续出图时不重复同步几何/位移
    fv, rv, uv_ = cam_basis(C['pos'])
    info = {'variant': VARIANT, 'layer': LAYER, 'pos': toThree(C['pos']), 'fwd': toThree(fv), 'right': toThree(rv), 'up': toThree(uv_),
            'vfov': C['vfov'], 'res': [R.resolution_x * R.resolution_percentage // 100, R.resolution_y * R.resolution_percentage // 100],
            'samples': SAMPLES, 'weights_baked': (not LIGHTGROUPS), 'weights': WEIGHTS,
            'lights': {'shaft': [list(scene['lan_shaft_pos']), list(scene['lan_shaft_dir']), 15.0],
                       'shaft2': [list(scene['lan_shaft2_pos']), list(scene['lan_shaft2_dir']), 9.0]}}
    return info, R.filepath + '.exr'

def do_render(scene, path):
    win = bpy.context.window
    prev = win.scene if win else None
    try:
        with bpy.context.temp_override(scene=scene):
            bpy.ops.render.render(write_still=True, scene=scene.name)
    finally:
        if win and prev and win.scene != prev: win.scene = prev      # 她的活动场景不变
    print('[scene] rendered ' + path, flush=True)
    if REPORT_SIZE:
        try:
            im = bpy.data.images.load(path, check_existing=False); im.name = PFX + 'sizecheck'
            im.pack(); print('LAN_FILESIZE', path, im.packed_file.size, flush=True)
            bpy.data.images.remove(im)
        except Exception as e:
            print('LAN_FILESIZE_FAIL', path, e, flush=True)

# ═════════════════════════ 入口 ═════════════════════════
INFO = None
if MODE == 'cleanup':
    cleanup(); print('[scene] cleaned up', flush=True)
else:
    scene = bpy.data.scenes.get(SCN)
    if MODE in ('build', 'all') or scene is None or bpy.data.objects.get(PFX + 'Cam') is None:
        scene = build()
    if MODE in ('render', 'all'):
        INFO, OUT_PATH = configure(scene)
        scene['lan_info'] = repr(INFO)
        print('LAN_INFO', repr(INFO).replace("'", '"').replace('True', 'true').replace('False', 'false'), flush=True)
        print('[scene] ' + VARIANT + '_' + LAYER + ' -> ' + OUT_PATH + ' at ' + str(INFO['res']) + ', ' + str(SAMPLES) + ' spp, device ' + DEVICE, flush=True)
        if RENDER: do_render(scene, OUT_PATH)
