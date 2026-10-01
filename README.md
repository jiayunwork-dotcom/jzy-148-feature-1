# 内部评分卡服务（自动建卡 + 在线打分）

把一份开发样本 CSV 丢进来，自动完成 **等频初箱 → 相邻合并（非零 / 占比 / WOE 单调）→ WOE/IV → 牛顿法逻辑回归 → 基准分/PDO 换分**，每一步中间结果随版本留存可查；之后在线打分走同一张卡。仅提供 HTTP 接口，无前端。

所有算法（分箱合并、WOE/IV、逻辑回归牛顿迭代、KS、AUC）均为自研，只依赖 NumPy 做矩阵/排序运算，**不使用 scikit-learn、statsmodels**。

## 目录结构

```
app/
  config.py              # 全局阈值（最小样本500、默认PDO/基准、IV阈值、牛顿参数）
  core/
    sample.py            # CSV 解析、缺失记号识别、数值/类别类型推断
    binning.py           # 分箱引擎：等频初箱、相邻合并、WOE/IV、落箱
    transform.py         # 原始值 -> WOE，构造回归设计矩阵
    regression.py        # 逻辑回归 + 自研牛顿迭代（含分离发散检测、迭代历史）
    scoring.py           # 基准分/基准odds/PDO 换分、常量分摊、PD 反算
    metrics.py           # KS、AUC（平均秩 / 累计分布，自研）
    pipeline.py          # 建卡流水线编排 + 作业开始前校验 + 产物组装
    exceptions.py        # ValidationError（作业前拒绝）/ BuildError（作业失败）
  jobs/scheduler.py      # 后台作业线程池，作业间零共享，失败原因落库
  storage/
    repository.py        # 仓储抽象 + 内存实现（测试用）
    postgres.py          # PostgreSQL 16 实现（行锁分配版本号）
  scoring/engine.py      # 在线打分：原始值落箱、未见类别标注、批量隔离
  api/schemas.py         # 请求/响应模型
  main.py                # FastAPI 路由
data/
  dev_sample.csv         # 确定性开发样本（1200 行，generate 脚本可复现）
  handcalc_sample.csv    # 小样本，前两个特征 WOE 能手算核对
tests/                   # pytest，覆盖需求列出的全部性质
docker-compose.yml       # db: postgres:16-alpine + api: python:3.12-slim
```

## 运行（Docker Compose）

```bash
docker compose up --build
# API: http://localhost:8000  文档: /docs   健康检查: /health
# PostgreSQL: localhost:5432 (scorecard/scorecard)
```

本地无数据库调试（进程内内存存储，重启即失）：

```bash
pip install -r requirements.txt
STORAGE_BACKEND=memory uvicorn app.main:app
```

运行测试：

```bash
pip install -r requirements.txt pytest
pytest
```

## 建卡

`POST /cards/{card_name}/build`，`multipart/form-data`：

| 字段 | 默认 | 说明 |
|---|---|---|
| `file` | 必填 | 开发样本 CSV，一列为 0/1 标签，其余为特征（数值/类别，允许缺失） |
| `label_col` | `label` | 标签列名 |
| `features` | 空 | 逗号分隔入模特征清单；留空则按 IV 阈值自动筛选 |
| `iv_threshold` | `0.02` | 自动筛选阈值 |
| `max_bins` | `20` | 数值特征等频初始箱上限 |
| `min_bin_pct` | `0.05` | 每箱样本占比下限 |
| `base_score` | `600` | 基准分 |
| `base_odds` | `20` | 基准分处的 好/坏 odds |
| `pdo` | `50` | Point to Double Odds，必须为正 |

返回 `202 {job_id, card_name, status}`，作业在后台跑。用 `GET /jobs/{job_id}` 轮询：
`succeeded`（带 `version`）或 `failed`（带 `error`，回归不收敛时还有 `newton_history`）。

**作业开始前同步拒绝（400，不产生作业）**：样本 < 500 行；标签不是 0/1；标签全为同一类；PDO ≤ 0；指定入模特征不存在；以及参数越界。

版本与产物：
- `GET /cards/{name}/versions`：版本摘要（参数、KS/AUC、各特征 IV、校准校验）。
- `GET /cards/{name}/versions/{v}/artifact`：**完整产物**——每个特征的分箱边界、各箱好/坏计数、WOE、IV 贡献、合并日志 `journal`、回归系数与逐轮牛顿迭代历史、每箱分值表、常量分摊说明。
- 同名卡可多次建卡，每次一个新版本；打分 `version` 缺省取最新。

## 打分

- 单条：`POST /cards/{name}/score`
  ```json
  {"version": 1, "features": {"age": 35, "city": "BJ", "job_grade": "A", "income": 9000}}
  ```
  返回 `total_score`、`pd`、每个特征的落箱 `bin_label` / `woe` / `score` / `status`。
  - `status`：`ok` / `missing`（落缺失箱）/ `unseen`（训练未见类别）/ `out_of_range`（数值越界，夹到首/末箱）。
  - **未见类别不静默归入任何箱**：WOE 按 0、得分取该特征常量分摊分，并显式标注 `has_unseen` 与该取值。
- 批量：`POST /cards/{name}/score/batch`，body `{"applicants":[{"applicant_id","features"}, ...]}`；单条出错只影响那一条（`ok:false` + `error`），其余照常出分。

## 算法与可复现要点

- **数值特征**：不同取值 ≤ `max_bins` 时逐取值成箱，否则按样本累计计数取等频分位点（打结值不拆开）切出 ≤20 个非空初箱；随后三阶段相邻合并，每次合并后重算计数与 WOE：
  1. 非零：任一箱坏或好为 0，与坏样率最接近的相邻箱合并；
  2. 占比：小于 `min_bin_pct` 的最小箱与坏样率最接近的相邻箱合并；
  3. 单调：方向由 WOE 与特征值的加权协方差符号决定，沿该方向把逆序相邻对中 WOE 落差最小的一对合并，直到单调。
- **类别特征**：按坏样率升序排（WOE 是坏样率的严格增函数，排序后天然单调），执行同样的非零/占比合并。
- **缺失值**：始终单独成一箱，不参与任何合并。
- **WOE/IV**：`WOE=ln((坏/总坏)/(好/总好))`，`IV=Σ(坏占比-好占比)·WOE`；单边为 0 时对该侧补 0.5 做对称平滑（保证标签取反 WOE 变号、IV 不变）。
- **回归**：WOE 为自变量 + 截距，自研牛顿（IRLS）迭代；记录每轮梯度、负对数似然、步长、系数。海森奇异加递减脊兜底；完全分离（系数沿固定方向线性增长、步长不衰减）判失败并写明原因。
- **换分**：`factor=PDO/ln2`，`Score=base_score+factor·ln(odds/base_odds)`，其中 `odds=(1-PD)/PD`。常量 `base_score-factor·ln(base_odds)-factor·b0` 按入模特征数等分到各特征，故**总分 = Σ 各箱分值**，odds=base_odds 时总分恰为 base_score；分值随系数与 WOE 正负可正可负（故总分不局限在某区间，属正常）。

## 已验证的性质（pytest）

- 收敛后全样本平均预测 PD = 样本违约率（误差 < 1e-6，带截距逻辑回归的必然）；
- 标签整体取反重建：各箱 WOE 变号、IV 不变；
- 样本原样复制一份重建：分箱、WOE、系数都不变；
- 各箱分值之和 = 总分；
- 总分 +PDO ⇔ odds（好/坏）翻倍；基准分处 odds = 基准 odds；
- PD 随总分严格单调下降且始终在 (0,1)；
- 数值特征合并后 WOE 序列确实单调，每箱好坏非零、占比 ≥ 下限、缺失单独成箱；
- 在线落箱反算 PD 与回归 `predict_proba` 一致（到机器精度）；
- 完全分离时建卡作业失败、原因与迭代历史落库、不产生版本；
- 样本不足/标签非法/单类/PDO 非正/特征不存在 → 作业前 400 拒绝；
- 同名卡并发建卡版本号不重号（行锁）；多卡并发互不串台；批量打分单条隔离；
- 手算样本前两个特征 WOE/IV 与公式逐值核对（见 `tests/test_binning_woe.py`）。
