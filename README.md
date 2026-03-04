# TunaPred Demo（2023年6-7月）

本项目实现了一个从开源数据抓取、建模到前端可视化的端到端 Demo：

1. **三种金枪鱼习性分析**（黄鳍、鲣鱼/跳鲣、长鳍）。
2. 用 **GBIF 物种出现数据 + Open-Meteo 天气/海洋再分析数据** 预测活动范围和可捕获指数。
3. 模型使用支持 GPU 的 **LightGBM**（失败自动回退 CPU）。
4. 前端展示预测值与可信度（ensemble std 转换）。
5. 新增海洋多变量 NetCDF 数据集的下载配置与 dataloader。

## 快速启动（命令）

```bash
cd /workspace/TunaPred
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

浏览器访问：`http://localhost:8000`

## 页面操作顺序

1. 点击「拉取数据」
2. 点击「训练LightGBM」
3. 选择鱼种并点击「预测并可视化」

## 三种金枪鱼习性（简要）

- **黄鳍金枪鱼（Thunnus albacares）**：偏好热带与副热带暖水，常见于温跃层上方，活动与海表温度、涡旋/锋面结构相关。
- **鲣鱼/跳鲣（Katsuwonus pelamis）**：分布更广、群游明显，常在温暖海域表层活动，受海温和风浪影响明显。
- **长鳍金枪鱼（Thunnus alalunga）**：较黄鳍耐凉，活动深度可更深，常见于副热带至温带过渡海域。

## API

- `POST /api/fetch`：抓取/加载 2023-06/07 数据。
- `POST /api/train`：训练 LightGBM 集成模型，返回 MAE、R² 与天气相关性。
- `GET /api/predict?species=yellowfin|skipjack|albacore`：返回预测点位、预测值、可信度。

### 海洋数据集 API（新增）

- `GET /api/ocean/catalog`：返回支持的数据集元数据和文件名模板。
- `GET /api/ocean/path?dataset_key=CHL&date=2023-06-01`：给出某天文件落地路径及是否存在。
- `POST /api/ocean/download?dataset_key=CHL&date=2023-06-01`：按模板下载该天数据（需要为该 dataset 预配置 base URL）。

## 新增数据集与文件名模板

| 变量名 | 中文名称 | 单位 | 空间分辨率 | 深度层数 / 范围 | 数据类型 | 对应文件名模板 |
|---|---|---|---|---|---|---|
| CHL | 表层叶绿素 | mg/m³ | 0.0417° | 1层 / 表层 | 2D | `{date}_chla.nc` |
| analysed_sst | 海表温度 | K | 0.05° | 1层 / 表层 | 2D | `{date}120000-UKMO-...nc` |
| thetao | 深层海水温度 | °C | 0.0833° | 30层 / 0.5–380m | 3D | `{date}_seawater_temperature_deep.nc` |
| so | 海水盐度 | PSU | 0.0833° | 30层 / 0.5–380m | 3D | `{date}_so.nc` |
| mlotst | 混合层深度 | m | 0.0833° | 1层 / 表层 | 2D | `{date}_so_mld.nc` |
| chl | 3D 叶绿素 | mg/m³ | 0.25° | 38层 / 0.5–411m | 3D | `{date}_3d_chlorophy.nc` |
| o2 | 溶解氧浓度 | mmol/m³ | 0.25° | 38层 / 0.5–411m | 3D | `{date}_o2_pp.nc` |
| nppv | 净初级生产力 | mg C/m³/day | 0.25° | 38层 / 0.5–411m | 3D | `{date}_o2_pp.nc` |
| WVEL | 垂直流速 | m/s | 0.25° | 22层 / 5–410m | 3D | `WVEL.1440x720x50.{date}.nc` |
| sla | 海表高度异常 | m | 0.125° | 1层 / 表层 | 2D | `dt_global_allsat_phy_l4_{date}_20241017.nc` |
| adt | 绝对动力地形 | m | 0.125° | 1层 / 表层 | 2D | `dt_global_allsat_phy_l4_{date}_adt.nc` |

> `{date}` 在程序中按 `YYYYMMDD` 渲染。

## 下载命令示例（新增）

```bash
python scripts/fetch_ocean_data.py \
  --dataset CHL \
  --start 2023-06-01 \
  --end 2023-07-31 \
  --base-url "https://your-data-host/path/to/chl"
```

## dataloader 使用示例（Python）

```python
import datetime as dt
from backend.ocean_datasets import OceanDatasetManager

manager = OceanDatasetManager(data_root="data/ocean")

# 文件路径（无需打开）
print(manager.local_path("thetao", dt.date(2023, 6, 1)))

# 读取 netcdf（需已下载 + 安装 xarray/netCDF4）
ds = manager.load_dataset("thetao", dt.date(2023, 6, 1))

# 最近邻采样（3D可带 depth_m）
value = manager.sample_point("thetao", dt.date(2023, 6, 1), lat=5.0, lon=130.0, depth_m=100)
print(value)
```

## 文件结构

- `backend/data_pipeline.py`：GBIF + Open-Meteo 数据抓取与特征构建
- `backend/modeling.py`：LightGBM 训练、保存、推理
- `backend/ocean_datasets.py`：新增多海洋数据集元数据、下载与 NetCDF dataloader
- `backend/app.py`：FastAPI 后端 + 新 ocean API
- `scripts/fetch_ocean_data.py`：批量下载脚本
- `frontend/index.html`：交互前端与地图可视化
