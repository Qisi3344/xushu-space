# 虚数空间 v5.1 · Blender 背景板流水线

## 一条命令（她上传 EXR 之后）
把她 PC 上 `C:/Users/Public/Documents/lan_xushu/` 里的 4 张 EXR 放进同一个目录（如 `blender/pc/`），然后：

    cd /workspace/work/xushu/blender && python3 ingest.py pc

= exr2npy（Combined + Depth.Z / Position）→ 相机 json（目录里没有就用 `pc_cam/{v}_cam.json`）→ `grade.py land` + `grade.py port`
→ 写 `../assets/{v}_cave.webp / {v}_city.webp / {v}_depth.png / {v}_city_depth.png / {v}.json`，显示图在 `review/`。
只想看效果不覆盖 assets：`python3 ingest.py pc --no-assets`（出 `review/gpu/ingest_*`）。
之后：`python3 sheets.py v5`（先跑浏览器检查 diag5.mjs，见下）出 `../v5_check.jpg`、`../v5_plates.jpg`。

## 文件
- `scene.py` — 程序化搭场景 + 出图，**BLENDER_MCP_SAFE_MODE 兼容**。发送前必须 `python3 safecheck.py scene.py` 0 errors。
  - 只建/删 `Lan_X_*` 数据块、`Lan_Xushu` 场景与集合；不切她的活动场景、不保存她的 .blend。
  - `CFG = {}` 这一行由包装脚本**文本替换**（safe mode 禁 globals()）。主要键：
    `mode`(build/render/all/cleanup)、`variant`(land/port)、`layer`(cave/city)、`samples`、`device`('GPU')、`denoise`(True → OPTIX，失败退 OIDN)、
    `passes`('depth' 默认：Combined+Z | 'lean'：Combined+Position | 'full'：+Normal/DiffCol)、`res`([w,h] 预览)、`out_dir`、`out_name`、`format`('single' 预览)、
    `weights`（灯光组权重，直接乘进灯强度；lan_base × weight）、`lightgroups`(True = 旧流程出灯光组)、`subsurf`。
  - `mode:'build'` 一次搭好两版共用的几何（~20 s），之后 `mode:'render'` 每次只切相机/层/灯再渲。
- `safecheck.py` — 本地 AST 检查（实测规则见文件头）。
- `mcp/` — 驱动她 PC：`make_call.py <name> '<json cfg>'`（替换 CFG + safecheck）→ `bgq.sh <q> name file ...`（后台顺序调用、计时，日志 `mcp/logs/`）；
  `pull.sh <exr> <out.png> <rows> [3|4]` + `pull_tpl.py` + `assemble.py`：把她 PC 上的小预览以十进制文本拉回（`R<y>:...`），
  `prevgrade.py <tag> <v> [city_tag]` 用沙箱同分辨率 Position（`sbq.sh` → `sb/`）跑 grade 预览；`make_save.py` 生成存 .blend 的调用。
- `run_sandbox.py` — 沙箱 Blender 4.3 包装（同样文本替换 CFG）。`runq.sh` 顺序跑多张。
- `exr2npy.py` — 多层 EXR → npy：`{tag}_rgba`、`_z`（平面深度）、`_pos`、`_nrm`、`_alb`、`_lg_<组>`（有哪个转哪个）。
- `grade.py` + `grade_params.json` — 后期：（full 输入才做：灯光组配比 + 联合双边降噪）→ 深度雾 + 解析体积光柱（含水面倒影）→ 城层合成 → 调色 →
  **反算 three.js ACESFilmic**。lean/depth 输入（`cam.weights_baked`）直接用 Cycles 降噪后的 Combined；只有 Z 时沿像素射线重建位置。
  `--preview <tag>` 只出 review 图。
- `view.py`、`sheets.py` — 预览裁切、检查图拼版。

## 在她 PC 上出终版（Blender 5.2.1，RTX 5060 Ti，OptiX）
    cd mcp
    python3 make_call.py build '{"mode":"build","device":"GPU"}' && ./bgq.sh qb build out_build.py
    OD=C:/Users/Public/Documents/lan_xushu/
    python3 make_call.py lc "{\"mode\":\"render\",\"device\":\"GPU\",\"variant\":\"land\",\"layer\":\"cave\",\"samples\":384,\"denoise\":true,\"out_dir\":\"$OD\"}"
    （同理 land/city 128spp、port/cave 384spp、port/city 128spp）然后 ./bgq.sh qf lc out_lc.py ...
    python3 make_save.py && ./bgq.sh qs save out_save.py      # 独立 .blend：xushu_cave.blend
渲完日志里 `LAN_INFO {...}` 存成 `pc_cam/{v}_cam.json`（已存好），`LAN_FILESIZE` 是文件字节数。
用完删场景：`make_call.py clean '{"mode":"cleanup"}'`（只删 Lan_X_* / Lan_Xushu）。

## 资产约定（index.html 读，未变）
- `{v}_cave.webp`：sRGB 编码的"ACES 之前线性值"。
- `{v}_city.webp`：图集，上半 = 预乘颜色 C（pow 1/2.2），下半 = 覆盖度 a。合成 `L = C·k + L_cave·(1 − a·k)`，k = city。
- `{v}_depth.png`：16 位逆距离（R 高字节、G 低字节；B = 8 位，近=白），`v = (1/t − 1/250)/(1/2 − 1/250)`，t 为到相机直线距离。
- `{v}.json`：相机（three 坐标）、竖直 FOV、宽高比、深度编码。
