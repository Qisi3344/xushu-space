# 虚数空间 · 背景板后期：降噪 → 大气雾 → 解析体积光柱 → 调色 → 反算 three.js ACES → 导出 WebP + 深度
#
#   python3 grade.py land  [rawdir] [outdir]
#   python3 grade.py port
#   python3 grade.py land rawdir --preview <tag>   只出 review/gpu/<tag>_*.png，不写 assets
#   终版一条命令：python3 ingest.py <放 EXR 的目录>（转 npy → 补相机 json → 调色 → 写 assets/）
#
# 两种输入都吃：
#   lean（v5.1，她 PC 上 Cycles+OptiX 出的）：{v}_{layer}.exr 只有 Combined（已降噪、灯光组权重已乘进灯）+ Position → 不再降噪、不再配比
#   full（v5 沙箱草稿）：还有 Normal / DiffCol / 灯光组 Combined_<组> → 按 grade_params.groups 配比 + 联合双边降噪
#
# 读 scene.py 的输出（raw/{v}_cave_*.npy, raw/{v}_city_*.npy, raw/{v}_cam.json），写：
#   assets/{v}_cave.webp     洞窟板（存的是"进 three.js ACES 之前的线性值"的 sRGB 编码，three 里 ACES 之后正好还原成调好的画面）
#   assets/{v}_city.webp     城层图集：上半 = 预乘颜色 C（pow 1/2.2 编码），下半 = 覆盖度 a。合成：L = C*k + L_cave*(1 - a*k)，k = city
#   assets/{v}_depth.png     深度：R/G = 16 位逆距离（高/低字节），B = 8 位逆距离（近 = 白，方便人看）
#   assets/{v}_city_depth.png 城层深度（8 位，近 = 白；仅作参考，运行时用洞窟深度）
#   review/{v}_*.png         给人看的预览（调色后的显示图）
import sys, os, json, math
import numpy as np, cv2
from PIL import Image

_args = sys.argv[1:]
PREVIEW = None
if '--preview' in _args:
    _i = _args.index('--preview'); PREVIEW = _args[_i + 1]; del _args[_i:_i + 2]
V = _args[0] if len(_args) > 0 else 'land'
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = _args[1] if len(_args) > 1 else os.path.join(HERE, 'raw')
OUT = _args[2] if len(_args) > 2 else os.path.join(HERE, '..', 'assets')
REV = os.path.join(HERE, 'review')
if PREVIEW:                                   # 预览：资产写进临时目录，review 图带前缀
    OUT = os.path.join(REV, 'preview_assets'); REV = os.path.join(REV, 'gpu')
os.makedirs(OUT, exist_ok=True); os.makedirs(REV, exist_ok=True)
PFXR = (PREVIEW + '_') if PREVIEW else ''
P = json.load(open(os.path.join(HERE, 'grade_params.json')))
for _k, _v in P.get('overrides', {}).get(V, {}).items():      # 每个版本可以单独覆盖参数，如 "shafts.shaft.k"
    _d = P
    for _kk in _k.split('.')[:-1]: _d = _d[_kk]
    _d[_k.split('.')[-1]] = _v
G = -2.596
NEAR, FAR = 2.0, 250.0           # 深度编码：v = (1/t - 1/FAR) / (1/NEAR - 1/FAR)，t = 到相机的直线距离

# ───────── three.js 的 ACESFilmic（TSL 版）及其反函数 ─────────
MIN = np.array([[0.59719, 0.35458, 0.04823], [0.07600, 0.90834, 0.01566], [0.02840, 0.13383, 0.83777]])
MOUT = np.array([[1.60475, -0.53108, -0.07367], [-0.10208, 1.10813, -0.00605], [-0.00327, -0.07276, 1.07602]])
iMIN, iMOUT = np.linalg.inv(MIN), np.linalg.inv(MOUT)
def fit(x): return (x * (x + 0.0245786) - 0.000090537) / (x * (x + 0.4329510) * 0.983729 + 0.238081)
def fit_inv(y):
    a = 1 - 0.983729 * y; b = 0.0245786 - 0.983729 * 0.4329510 * y; c = -(0.000090537 + 0.238081 * y)
    return (-b + np.sqrt(np.maximum(b * b - 4 * a * c, 0))) / (2 * a)
def aces(c): return np.clip(fit((c / 0.6) @ MIN.T) @ MOUT.T, 0, 1)
def aces_inv(d):
    y = np.clip(d, 0, 0.985) @ iMOUT.T
    return np.maximum((fit_inv(np.clip(y, -0.0003, 1.0)) @ iMIN.T) * 0.6, 0)
def srgb_enc(x): x = np.clip(x, 0, 1); return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
def srgb_dec(x): return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))
def sstep(e0, e1, x): t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)
def b2t(p): return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)       # blender → three

cam = json.load(open(os.path.join(RAW, f'{V}_cam.json')))
O, F, Rt, U = (np.array(cam[k], np.float64) for k in ('pos', 'fwd', 'right', 'up'))

def rays(H, W):
    th = math.tan(math.radians(cam['vfov'] / 2)); asp = W / H
    x = ((np.arange(W) + 0.5) / W * 2 - 1) * th * asp; y = (1 - (np.arange(H) + 0.5) / H * 2) * th
    d = F[None, None] + Rt[None, None] * x[None, :, None] + U[None, None] * y[:, None, None]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)

def load(layer):
    def g(n):
        p = os.path.join(RAW, f'{V}_{layer}_{n}.npy')
        return np.load(p).astype(np.float32) if os.path.exists(p) else None
    rgba, pos = g('rgba'), g('pos')
    if rgba.shape[-1] == 3: rgba = np.concatenate([rgba, np.ones_like(rgba[..., :1])], -1)
    if pos is not None: pos = b2t(pos)
    else:                                     # 只有平面深度 Z：沿像素射线重建位置（three 坐标）
        z = g('z'); h_, w_ = z.shape; dv = rays(h_, w_)
        z = np.where(z > 1e4, 1e4, z)
        pos = (O[None, None] + dv * (z / (dv @ F))[..., None]).astype(np.float32)
    return rgba, pos, g('nrm'), g('alb')

def denoise(rgb, nrm, dist, k):
    """联合双边：以法线 + 对数深度为引导，在 log 亮度里滤，保住边缘。"""
    if k <= 0: return rgb
    guide = np.concatenate([nrm[..., :2] * 0.5, (np.log(dist + 1) * 0.35)[..., None]], -1).astype(np.float32)
    lg = np.log(rgb + 1e-4).astype(np.float32)
    out = cv2.ximgproc.jointBilateralFilter(guide, lg, int(k), 0.12, k / 2.0)
    out = cv2.ximgproc.jointBilateralFilter(guide, out, int(k), 0.10, k / 2.0)
    return np.exp(out) - 1e-4

def vnoise(p):
    """便宜的 3D "灰尘"噪声（正弦叠加），给光柱一点不均匀。"""
    return (np.sin(p[..., 0] * 1.7 + np.sin(p[..., 1] * 0.9) * 2) * np.sin(p[..., 1] * 1.3 + p[..., 2] * 0.7) * 0.5
            + np.sin(p[..., 2] * 2.9 + p[..., 0] * 1.1 + 1.3) * np.sin(p[..., 1] * 3.3 - 0.7) * 0.3)

def shafts(dirs, tmax, origin=None):
    """单次散射的锥形光柱（不算岩石阴影：光路上是空的），沿视线 64 步积分。"""
    H, W, _ = dirs.shape; out = np.zeros((H, W, 3), np.float32)
    lights = cam.get('lights')
    if not lights:
        d = np.array([4.3, -9.8, 2.1]); d /= np.linalg.norm(d); S = np.array([0.9, G, 0.9]) - d * 23
        lights = {'shaft': [S.tolist(), d.tolist(), 15.0]}
    N = 64; tcap = 70.0
    for name, (S, D, ang) in lights.items():
        S, D = np.array(S), np.array(D); cfg = P['shafts'][name]
        co, ci = math.cos(math.radians(ang / 2 * cfg.get('widen', 1.0))), math.cos(math.radians(ang / 2 * 0.35))
        col = np.array(cfg['color'])
        ax1 = np.cross(D, [0, 0, 1.0]); ax1 /= np.linalg.norm(ax1); ax2 = np.cross(D, ax1)
        for r0 in range(0, H, 48):
            dv = dirs[r0:r0 + 48]; tm = np.minimum(tmax[r0:r0 + 48], tcap)
            acc = np.zeros(dv.shape[:2], np.float32)
            jit = (np.random.default_rng(r0).random(dv.shape[:2]) - 0.5)
            for i in range(N):
                t = (i + 0.5 + jit) / N * tm
                x = (O if origin is None else origin[r0:r0 + 48]) + dv * t[..., None]
                L = x - S; dist = np.linalg.norm(L, axis=-1); Ld = L / dist[..., None]
                cosa = (Ld * D).sum(-1)
                cone = sstep(co, ci, cosa)
                # 光从岩缝漏下来：绕轴向的条纹 + 细尘
                az = np.arctan2((Ld * ax2).sum(-1), (Ld * ax1).sum(-1))
                streak = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(az * 23 + 1.3)) * (0.5 + 0.5 * np.sin(az * 9 - 0.4 + dist * 0.05))
                cone = cone * (1 - cfg.get('streak', 0.0) + cfg.get('streak', 0.0) * streak)
                cos_th = -(Ld * dv).sum(-1)
                g = cfg.get('g', 0.4); ph = (1 - g * g) / np.power(1 + g * g - 2 * g * cos_th, 1.5)
                above = sstep(G - 0.05, G + 0.6, x[..., 1])
                dust = 0.72 + 0.28 * vnoise(x * 0.9)
                fogT = np.exp(-np.maximum(t + (0 if origin is None else 10) - P['fog']['start'], 0) * P['fog']['density'])
                acc += cone * ph * above * dust * fogT / (dist * dist * 0.004 + 1) * (tm / N)
            out[r0:r0 + 48] += acc[..., None] * col * cfg['k']
    return out

def shafts_lr(dirs, tmax, origin=None, f=3):
    H, W, _ = dirs.shape; h, w = H // f, W // f
    rs = lambda a: cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    d = rs(dirs); d /= np.linalg.norm(d, axis=-1, keepdims=True)
    tm = cv2.resize(tmax.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    o = None if origin is None else rs(origin)
    out = shafts(d, tm, o)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC).clip(0, None)

def grade_display(Ls):
    """场景线性 → 显示：three 的 ACES + 一点冷色分级（提一点点蓝色暗部、压中间调）。"""
    g = aces(Ls * P['exposure'])
    lift = np.array(P['lift']); g = g + lift * (1 - g) ** 3
    g = np.power(np.clip(g, 0, 1), P['gamma'])
    lum = (g * [0.2126, 0.7152, 0.0722]).sum(-1, keepdims=True)
    g = lum + (g - lum) * P['saturation']
    return np.clip(g, 0, 1)

def save_rgb(path, lin, q=90):
    enc = srgb_enc(lin)
    rng = np.random.default_rng(1)
    img = np.clip(enc * 255 + rng.random(enc.shape) - 0.5, 0, 255).round().astype(np.uint8)   # 抖动，防暗部色带
    Image.fromarray(img).save(path, quality=q, method=6) if path.endswith('.webp') else Image.fromarray(img).save(path)

def save_depth(path, dist, scale=0.5):
    v = np.clip((1 / np.maximum(dist, NEAR) - 1 / FAR) / (1 / NEAR - 1 / FAR), 0, 1).astype(np.float32)
    # 前景轮廓向外膨胀几像素再柔化：视差拉伸发生在背景一侧，岩石边缘本身保持干净，网格上也不会出现台阶
    k = max(3, int(round(v.shape[1] / 1920 * 7)) | 1)
    v = cv2.dilate(v, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    v = cv2.GaussianBlur(v, (0, 0), k / 2.2)
    if scale != 1: v = cv2.resize(v.astype(np.float32), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    q = np.round(v * 65535).astype(np.uint32)
    img = np.stack([(q >> 8).astype(np.uint8), (q & 255).astype(np.uint8), np.round(v * 255).astype(np.uint8)], -1)
    Image.fromarray(img).save(path, optimize=True)
    return v

# ───────── 洞窟层 ─────────
rgba, pos, nrm, alb = load('cave')
H, W = rgba.shape[:2]
dirs = rays(H, W)
dist = np.linalg.norm(pos - O, axis=-1)
lgs = {g: os.path.join(RAW, f'{V}_cave_lg_{g}.npy') for g in P['groups']}
LEAN = cam.get('weights_baked', False) or not all(os.path.exists(p) for p in lgs.values())
if not LEAN:
    rgb = sum(np.load(p).astype(np.float32) * w for (g, p), w in zip(lgs.items(), P['groups'].values()))
else:
    rgb = rgba[..., :3].copy()                # 灯光组权重已经乘进灯里了
if nrm is not None and not LEAN:
    rgb = denoise(rgb, nrm, dist, P['denoise'])
print('input', 'lean (baked weights, Cycles-denoised)' if LEAN else 'full (light groups + bilateral denoise)')
fo = P['fog']
hK = np.clip(np.exp(-(pos[..., 1] - G - 2.0) * fo['height']), 0.6, 2.5)
T = np.exp(-np.maximum(dist - fo['start'], 0) * fo['density'] * hK)[..., None]
Fc = np.array(fo['color'])
# 雾的内散射：越往洞底那团冷光越亮（空气透视：远处比近处的剪影更亮一点）
yy0, xx0 = np.mgrid[0:H, 0:W].astype(np.float32)
bg = fo['back_glow']
bglow = np.exp(-(((xx0 / W - bg[0]) / bg[2]) ** 2 + ((yy0 / H - bg[1]) / bg[3]) ** 2))[..., None] * np.array(fo['back_color'])
cave_s = rgb * P['surface_gain'] * T + (Fc + bglow) * (1 - T)
sh = shafts_lr(dirs, dist)
# 水面倒影里的光柱：水面像素按镜面方向再积一次，乘菲涅尔
water = np.abs(pos[..., 1] - G) < 0.02
if water.any():
    rd = dirs.copy(); rd[..., 1] *= -1
    cosi = np.clip(-dirs[..., 1], 0, 1)
    fres = 0.02 + 0.98 * (1 - cosi) ** 5
    rsh = shafts_lr(rd, np.full(dist.shape, 45.0), origin=pos + rd * 0.01)
    Tw = np.exp(-np.maximum(dist - fo['start'], 0) * fo['density'])[..., None]
    sh = sh + rsh * (water * fres * P['water_refl'])[..., None] * Tw
    cave_s = cave_s + sh
else:
    cave_s = cave_s + sh
if V == 'low':   # v8 低机位：贴着镜头的右侧石壁和近石压成剪影，只留一点冷边
    _nd = P.get('low_near', [2.5, 7.0, 0.18])
    _k = np.clip((dist - _nd[0]) / (_nd[1] - _nd[0]), 0, 1); _k = _k * _k * (3 - 2 * _k)
    cave_s = cave_s * (_nd[2] + (1 - _nd[2]) * _k)[..., None]
gA = grade_display(cave_s)
LA = aces_inv(gA)
save_rgb(os.path.join(REV, f'{PFXR}{V}_cave_display.png'), gA)
np.save(os.path.join(REV, f'{PFXR}{V}_LA.npy'), LA.astype(np.float16))
save_rgb(os.path.join(OUT, f'{V}_cave.webp'), LA, q=P['webp_q'])
save_depth(os.path.join(OUT, f'{V}_depth.png'), dist, scale=0.5)
Image.fromarray((np.clip(sh.mean(-1) * 6, 0, 1) ** 0.5 * 255).astype(np.uint8)).save(os.path.join(REV, f'{PFXR}{V}_shaft_only.png'))
print('cave', V, W, H, 'disp mean', gA.mean(), 'p99', np.percentile(gA, 99))

# ───────── 城层 ─────────
if os.path.exists(os.path.join(RAW, f'{V}_city_rgba.npy')):
    crgba, cpos, cnrm, _ = load('city')
    a = np.clip(crgba[..., 3], 0, 1)
    # 城脚贴着湖面的那一截直接淡出（否则城层和水面之间会留一条硬横线）
    a = a * sstep(G + 0.3, G + 5.0, cpos[..., 1])
    cdist = np.where(a > 1e-3, np.linalg.norm(cpos - O, axis=-1), FAR)
    cc = P['city']
    Tc = np.exp(-np.maximum(cdist - fo['start'], 0) * cc['fog_density'])[..., None]
    # 低处（湖面上的积雾）更淡，城的基座溶进雾里，只有上半截的塔和屋脊浮出来
    Tc = Tc * (0.15 + 0.85 * sstep(G + 1.0, G + cc.get('lift_h', 10.0), cpos[..., 1]))[..., None]
    # 远处冷光：屏幕空间里一团压在城后面的冷辉，只照远处（洞窟深度 > 40）
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    def proj(p):
        q = np.array(p) - O; z = q @ F; th = math.tan(math.radians(cam['vfov'] / 2))
        return [0.5 + (q @ Rt) / z / (th * W / H) * 0.5, 0.5 - (q @ U) / z / th * 0.5]
    cxy = proj(cc['glow_world']); sx, sy = cc['glow_size']
    gl = np.exp(-(((xx / W - cxy[0]) / sx) ** 2 + ((yy / H - cxy[1]) / sy) ** 2))
    if cam['variant'] == 'port':
        gl = np.exp(-(((xx / W - cxy[0]) / (sx * 2.2)) ** 2 + ((yy / H - cxy[1]) / (sy * 0.8)) ** 2))
    far = sstep(30, 95, dist)
    glow = (gl * far)[..., None] * np.array(cc['glow_color'])
    cityC = crgba[..., :3] * cc['gain']
    Fcity = Fc + bglow + glow * 0.5
    behind = cave_s + glow
    full_s = a[..., None] * (cityC / np.maximum(a[..., None], 1e-4) * Tc + Fcity * (1 - Tc)) + (1 - a[..., None]) * behind
    gB = grade_display(full_s)
    LB = aces_inv(gB)
    aa = a[..., None] * cc['alpha']
    C = np.clip(LB - (1 - aa) * LA, 0, None)
    save_rgb(os.path.join(REV, f'{PFXR}{V}_full_display.png'), gB)
    Cenc = np.clip(np.power(np.clip(C, 0, 1), 1 / 2.2) * 255 + np.random.default_rng(2).random(C.shape) - 0.5, 0, 255).round().astype(np.uint8)
    A8 = np.clip(aa[..., 0] * 255, 0, 255).round().astype(np.uint8)
    # 图集：上半 = C（预乘颜色，pow 1/2.2 编码），下半 = 覆盖度 a（灰度）。不用 alpha 通道：浏览器解码时可能预乘，a=0 处的颜色会丢
    atlas = np.concatenate([Cenc, np.repeat(A8[..., None], 3, -1)], 0)
    Image.fromarray(atlas).save(os.path.join(OUT, f'{V}_city.webp'), quality=P['city_q'], method=6)
    cv = np.clip((1 / np.maximum(cdist, NEAR) - 1 / FAR) / (1 / NEAR - 1 / FAR), 0, 1)
    Image.fromarray(np.round(cv2.resize(cv.astype(np.float32), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)).save(os.path.join(OUT, f'{V}_city_depth.png'), optimize=True)
    print('city', V, 'alpha px', (a > 0.5).mean(), 'C max', C.max())

json.dump({'variant': V, 'pos': cam['pos'], 'fwd': cam['fwd'], 'right': cam['right'], 'up': cam['up'], 'vfov': cam['vfov'],
           'aspect': W / H, 'depth': {'near': NEAR, 'far': FAR, 'encoding': 'inverse distance, R=hi G=lo (16-bit), B=8-bit, near=white'}},
          open(os.path.join(OUT, f'{V}.json'), 'w'), indent=1)
