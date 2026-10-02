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
  monitor/               # 投产后监控层（新增）
    audit.py             # 打分留痕 + 请求标识幂等（同标识同内容重放、不同内容拒绝）
    baseline.py          # 建卡时总分基准（训练样本逐条经在线引擎打分留存）
    stability.py         # PSI 手写、特征/总分分布对比、三档结论
    performance.py       # 回填后坏率/平均PD/KS/AUC/分段（KS、AUC 复用 core.metrics）
    backfill.py          # 表现回填后台作业（拒收清单/先到为准/幂等重放）
  api/schemas.py         # 请求/响应模型
  api/monitor.py         # 监控层路由（稳定性/补基准/回填/表现）
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
- 批量：`POST /cards/{name}/score/batch`，body `{"applicants":[{"applicant_id","features","request_id?"}, ...]}`；单条出错只影响那一条（`ok:false` + `error`），其余照常出分。

## 投产后监控

### 1) 打分留痕与请求标识幂等

每次成功打分（单条，及批量中成功的每一条）都追加一条**不可变**留痕 `score_records`：卡名、版本、调用方 `request_id`（可空）、入参原始特征、逐特征落箱（箱下标/箱标签/缺失/未见/越界）、总分、PD、UTC 打分时间。批量中失败的那一条不产生留痕。

单条 body 与批量 item 都可带 `request_id`：

| 情形 | 行为 | HTTP |
|---|---|---|
| 同标识 + 同内容重放 | 直接返回第一次的**完整原结果**（不再打分），计数只算 1 次 | 200 |
| 同标识 + 不同内容（或显式指定不同版本） | **明确拒绝，不覆盖原记录** | 409 |
| 显式指定不存在的版本 | 先做版本存在性校验 | 404 |
| 不带标识 | 照常打分、每条都留痕（部分唯一索引，NULL 不参与） | 200 |
| 标识超长（>128 字符） | 拒绝 | 400 |

「内容」指纹为 `{"version":解析后版本,"features":入参特征}` 规范化 JSON 的 SHA-256；注意整数 `35` 与浮点 `35.0` 视为不同内容，调用方应保持类型一致。打分响应的字段与数值与升级前完全一致（内部落箱下标 `bin_index` 只进留痕、不进响应）。

### 2) 人群稳定性 PSI

`GET /cards/{name}/versions/{version}/stability?start=&end=&warning=0.1&significant=0.25`

- 时间为 ISO 8601（`YYYY-MM-DD` 按 UTC；区间半开 `[start, end)`），按打分时间过滤；不同版本各算各的。
- 返回每个入模特征各常规箱**和缺失箱**的基准/实际计数与占比、PSI，以及总分分布同样一份；未见类别、数值越界单独报 `unseen_rate` / `out_of_range_rate`，**绝不并进任何箱**，也不进 PSI 求和。
- PSI 公式（评分卡行业通用）：`Σ (实际占比-基准占比)·ln(实际占比/基准占比)`。
- **零侧约定**：某箱一侧占比为 0 时，对该零侧按 **0.5 条样本**平滑后代入（与建卡 WOE 的单边平滑同一惯例）；两侧都为 0 的箱跳过。新箱涌入贡献有限为正、随样本量同阶、不会给 0/0 或无穷大。
- 三档：`PSI<0.10` stable；`[0.10,0.25)` warning；`>=0.25` significant（阈值查询参数或 `app/core/config.py` 可配，报告取各特征与总分中最差档）。

**总分基准与老版本**：总分是各特征 WOE 的联合函数，边际箱计数推不出总分联合分布。新版本建卡时即把训练样本经**在线引擎逐条打分**、按等频箱（打结不拆）把总分基准存入产物 `artifacts.score_baseline`。升级前已存在的老版本没有该基准：

- 特征层 PSI 只依赖分箱计数，**照常计算**；打分完全不受影响；
- 总分层返回 `total_score.available=false` 并说明原因；
- 若老卡原始建卡样本仍在，可 `POST /cards/{name}/versions/{v}/score-baseline`（multipart CSV）补录：先逐行在线落箱核对**每个入模特征每箱（含缺失箱）计数与样本量**和产物完全一致，任何不符整体 400 拒绝，防止拿错样本；已有基准不可覆盖（409）。

### 3) 表现回填与区分能力复核

- `POST /cards/{name}/backfills`，body `{"items":[{"request_id","label"}, ...]}`（`label` 必须 0/1，`"0"/"1"` 可）→ 202 后台作业。
- `GET /cards/{name}/backfills/{id}`：`total/processed/applied/duplicates` 与 `rejected`（每项带序号、标识、原因）。**找不到标识、标签非法**进拒收清单，不影响其它条目；只在路径卡名的留痕里找，跨卡同名标识互不干扰。
- **重复回填**：同标识同标签 = 幂等重放（计 `duplicates`，不改数据）；同标识不同标签 = **先到为准，后进拒收**（原因 `label_conflict`），绝不改标签、绝不重复计入好坏。
- `GET /cards/{name}/versions/{version}/performance?start=&end=`：只统计区间内已回填标签的放款，返回实际坏率、平均预测 PD、**KS、AUC**、按总分分段的预测/实际坏率对比。分段优先用总分基准等频箱（越界单列 below/above）；老版本无基准时按本批总分等频现切，仅用于分段对比。
- **KS/AUC 口径**：直接调用建卡同一套 `app.core.metrics.ks_stat/roc_auc`（平均秩 AUC、阈值阶梯 KS），不另写实现；标签单类时无法计算，如实返回 `null`。

### 统计口径取舍：只留痕、查询全量现算

本服务**不做**按时间桶的增量计数，唯一事实来源是不可变的 `score_records`，所有区间统计在查询时从留痕扫描聚合：

- **查询延迟**：与区间内打分条数线性相关（当前量级无压力）；日后量大可加按天/按箱物化物化视图，语义不变（物化由留痕重算并对账）。
- **写入开销**：每次打分仅一次 insert（含 `request_id` 时走部分唯一索引的 upsert），无计数更新、无计数竞争——8 线程并发不会把计数加错。
- **任意起止时间**：天然支持，桶粒度不牺牲边界精度，区间半开 `[start,end)`。
- **对账**：任何时候都能从留痕逐条重算（PSI 直方图、坏率、KS/AUC），与接口结果逐位核对；测试固化了「2000 次并发 = 留痕重算 = 2000」。
- **重启**：统计不依赖进程内状态，内存后端重启即失（与既有行为一致），PostgreSQL 后端重启后查询结果不变。

### 数据库平滑升级

PostgreSQL 侧新增 `score_records`、`backfill_jobs` 两表及索引，全部 `CREATE ... IF NOT EXISTS`；幂等靠 `score_records(card_name, request_id) WHERE request_id IS NOT NULL` 部分唯一索引。服务在已有数据的老库上启动即自动升级，不清库、不动既有表（集成测试覆盖旧 schema → 新 schema 后老数据完好、新能力可用）。

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

监控层（`tests/test_monitor_logic.py`、`tests/test_monitor_api.py`、
`tests/test_postgres_integration.py`）：

- 建卡样本原样逐条回放：各特征与总分 PSI 为 0（1e-12 内），各箱占比/计数与建卡产物逐位吻合；
- 8 线程并发 2000 次打分：统计正好 2000，且与留痕逐条重算一致；
- 同一 `request_id` 并发/串行重放 100 次：只留 1 条、返回逐位相同；不同内容 409 且原记录不被覆盖；
- 服务重启（PostgreSQL 后端重连）后同一查询结果不变；
- 版本 1/2 交替打分，两版本计数与稳定性互不渗入；批量失败条目不进统计；
- 未见类别/越界单独报占比，不并入任何箱；
- 回填：找不到/非法标签进拒收清单；同标识不同标签先到为准；KS/AUC 与建卡同一套 `core.metrics` 结果一致（1e-12）；
- 老版本特征层 PSI 照算、总分层明确 unavailable，补基准需通过逐箱计数校验；
- 旧 schema 老库平滑升级后老数据完好、新能力可用；
- 以上在内存与真实 PostgreSQL 16 上行为一致。

## 测试

```bash
pip install -r requirements.txt pytest httpx
pytest                                              # 内存后端；PG 用例在无库时自动 skip
docker compose up -d db                             # 或任意可用的 PostgreSQL 16
pytest tests/test_postgres_integration.py           # 默认连 localhost:5432
# 自定义连接：TEST_DATABASE_URL=... TEST_ADMIN_DSN=... pytest tests/test_postgres_integration.py
```
