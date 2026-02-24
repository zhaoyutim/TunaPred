# TunaPred Demo（2023年6-7月）

本项目实现了一个从开源数据抓取、建模到前端可视化的端到端 Demo：

1. **三种金枪鱼习性分析**（黄鳍、鲣鱼/跳鲣、长鳍）。
2. 用 **GBIF 物种出现数据 + Open-Meteo 天气/海洋再分析数据** 预测活动范围和可捕获指数。
3. 模型使用支持 GPU 的 **LightGBM**（失败自动回退 CPU）。
4. 前端展示预测值与可信度（ensemble std 转换）。

## 三种金枪鱼习性（简要）

- **黄鳍金枪鱼（Thunnus albacares）**：偏好热带与副热带暖水，常见于温跃层上方，活动与海表温度、涡旋/锋面结构相关。
- **鲣鱼/跳鲣（Katsuwonus pelamis）**：分布更广、群游明显，常在温暖海域表层活动，受海温和风浪影响明显。
- **长鳍金枪鱼（Thunnus alalunga）**：较黄鳍耐凉，活动深度可更深，常见于副热带至温带过渡海域。

> Demo 中将“捕鱼量”定义为 **可捕获指数（catch_index）**，它由观测活动强度（网格内出现次数）与天气适宜度共同构造，用于演示天气对捕捞结果的影响分析流程。

## 数据选择与预测目标

- **活动范围基础数据**：GBIF occurrence API（2023年6月、7月，三种金枪鱼，带经纬度）。
- **天气/海洋特征**：Open-Meteo Archive API（SST、10m最大风速、MSL气压、降水）。
- **时空粒度**：按 1° 网格 + 日聚合。
- **预测目标**：catch_index（可捕获指数）。

## 快速启动

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

浏览器访问：`http://localhost:8000`

## API

- `POST /api/fetch`：抓取/加载 2023-06/07 数据。
- `POST /api/train`：训练 LightGBM 集成模型，返回 MAE、R² 与天气相关性。
- `GET /api/predict?species=yellowfin|skipjack|albacore`：返回预测点位、预测值、可信度。

## 可信度定义

训练 5 个不同随机种子的 LightGBM 模型，对同一点进行预测：

- 均值 = 最终预测
- 标准差 std = 不确定性
- `confidence = 1 / (1 + std)`，std 越小可信度越高

## 文件结构

- `backend/data_pipeline.py`：抓取与特征构建
- `backend/modeling.py`：LightGBM 训练、保存、推理
- `backend/app.py`：FastAPI 后端
- `frontend/index.html`：交互前端与地图可视化
