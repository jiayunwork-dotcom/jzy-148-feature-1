# 内部评分卡服务（自动建卡 + 在线打分）

把一份开发样本 CSV 丢进来，自动完成 **等频初箱 → 相邻合并（非零 / 占比 / WOE 单调）→ WOE/IV → 牛顿法逻辑回归 → 基准分/PDO 换分**，每一步中间结果随版本留存可查；之后在线打分走同一张卡。仅提供 HTTP 接口，无前端。

所有算法（分箱合并、WOE/IV、逻辑回归牛顿迭代、KS、AUC）均为自研，只依赖 NumPy 做矩阵/排序运算，**不使用 scikit-learn、statsmodels**。

## 目录结构

```
app/
  config.py              # 全局阈值（最小样本500、默认PDO/基准、IV阈值、牛顿参数、PSI三档阈值）
  core/
    sample.py            # CSV 解析、缺失记号识别、数值/类别类型推断
    binning.py           # 分箱引擎：等频初箱、相邻合并、WOE/IV、落箱
    transform.py         # 原始值 -> WOE，构造回归设计矩阵
    regression.py        # 逻辑回归 + 自研牛顿迭代（含分离发散检测、迭代历史）
    scoring.py           # 基准分/基准odds/PDO 换分、常量分摊、PD 反算
    metrics.py           # KS、AUC（平均秩 / 累计分布，自研）——建卡与投后监控同一套口径
    pipeline.py          # 建卡流水线编排 + 作业开始前校验 + 产物组装（含投后监控基准）
    exceptions.py        # ValidationError（作业前拒绝）/ BuildError（作业失败）
  audit/service.py       # 打分留痕 + 请求标识幂等（重放只算一次、异内容拒绝）
  monitoring/
    psi.py               # 稳定性指标 PSI（手写，零占比地板口径、三档结论）
    baseline.py          # 建卡时逐条打分留存总分等频基准；总分落箱
    service.py           # 人群稳定性查询（按时间区间从留痕现算）
  performance/
    scheduler.py         # 表现回填后台作业、标签合法性校验
    service.py           # 实际坏率/平均PD/KS/AUC/总分分段（KS/AUC 调 core.metrics）
  jobs/scheduler.py      # 建卡后台作业线程池，作业间零共享，失败原因落库
  storage/
    repository.py        # 仓储抽象 + 内存实现（测试用，含全部监控表）
    postgres.py          # PostgreSQL 16 实现（行锁版本号 + 幂等登记 + 留痕/回填表）
  scoring/engine.py      # 在线打分：原始值落箱、未见类别标注、批量隔离
  api/schemas.py         # 请求/响应模型
  main.py                # FastAPI 路由（建卡/打分/监控/回填）
data/
  dev_sample.csv         # 确定性开发样本（1200 行，generate 脚本可复现）
  handcalc_sample.csv    # 小样本，前两个特征 WOE 能手算核对
tests/                   # pytest，覆盖需求列出的全部性质
  test_postgres_integration.py  # docker compose 真实库用例（无库自动 skip）
docker-compose.yml       # db: postgres:16-alpine + api + tests(profile=test)
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

## 投产后监控

监控是在既有能力之上加的一层，建卡作业、分箱/回归产物、版本管理、单条/批量打分的行为与响应字段均未改动；新增能力按职责放在 `audit/`、`monitoring/`、`performance/` 三个模块。

### 1. 打分留痕与请求标识幂等

每一次**成功**打分（单条、批量里成功的条目）都在 `score_logs` 留下一条记录：卡名、版本、每个入模特征的落箱标签与状态（`ok/missing/unseen/out_of_range` 及原始值）、总分、PD、打分时间。批量里失败的条目不落痕。

调用方可在单条 body 里带 `request_id`，或在批量条目中带 `request_id`（字符串）。标识按「卡 + 标识」全局唯一，单条与批量共用同一命名空间：

| 情形 | 行为 |
|---|---|
| 同标识 + 同内容重放 | 返回与第一次**逐字段完全相同**的结果，只落一条记录，统计只加 1（响应条目标 `replayed:true`，批量汇总另有 `replayed` 计数） |
| 同标识 + 不同内容 | 明确拒绝：单条 HTTP **409**，批量条目 `ok:false, conflict:true`（不影响同批其它条目），原记录永不被覆盖 |
| 同标识 + 不同内容并发到达（首个还在打分） | 同样立即拒绝（PG 用行锁判定，不等待错误重放） |
| 首次打分失败 | 回滚 pending 登记，修正内容后可用同标识重试 |
| 不带标识 | 照常打分、照常落痕，不参与幂等 |

内容相同与否按规范化 JSON（键排序）的 SHA-256 判定，键序不同、`1` 与 `1.0` 视为相同；NaN/Inf 归一为 null。打分响应原有的字段和数值一个未变（仅批量结果增加了 `replayed`/`conflict` 两个默认值字段、批量汇总增加 `replayed`/`conflicts` 计数）。

留痕可按时间区间对账查询：`GET /cards/{name}/versions/{v}/score-logs?start=&end=&limit=`。

### 2. 人群稳定性

`GET /cards/{name}/versions/{v}/stability?start=<ISO8601>&end=<ISO8601>`（时间为 `[start, end)`；不带时区按 UTC；区间可省略）。返回：

- 每个**入模特征**：各常规箱 + 缺失箱的基准占比（建卡样本）/实际占比（区间进件）/计数、PSI、档位；**未见类别占比、越界占比单独报出**（`unseen_pct` / `out_of_range_pct`），绝不并进任何箱；
- 总分分布：同样的逐箱占比对比 + PSI；越出基准分数范围的记录单独报 `out_of_range_pct`，不夹到边箱；
- `overall_rating`：所有特征与总分 PSI 的最大值对应的档位；
- 版本严格隔离：查询必须指定版本，统计只含该版本的打分记录。

**PSI 口径（手写，`app/monitoring/psi.py`）**：`Σ (p_actual − p_expected)·ln(p_actual/p_expected)`，交换两侧不变。某一侧某箱占比为 0 时公式发散，采用评分卡行业常见的**最小占比地板 1e-4**（万分之一）：两侧都先 `p' = max(p, 1e-4)` 再代入；地板只影响 PSI 求和，对比表里的原始占比仍如实显示 0；两侧都为 0 的箱贡献为 0。阈值三档（`app/core/config.py` 可配）：**stable `<0.10`，warning `[0.10, 0.25)`，drift `>=0.25`**。区间内没有正常落箱记录时该指标 PSI 返回 null 而不是硬给 0。

**总分基准怎么来的**：各特征的边际计数推不出总分联合分布，因此——
- **新版本**：建卡流水线收尾时用建好的卡对建卡样本**逐条打分**，把总分按等频（默认 10 箱，`score_baseline_bins` 可配，打结不拆、每箱非空）切分并留存计数到版本产物 `artifacts.monitoring.score_baseline`；
- **升级前的老版本**：产物里没有这块基准，接口总分部分返回 `status:"unavailable"` 并写明原因（无法从边际计数回推，可重新建卡获得），但**特征层稳定性照常计算**（基准直接取版本产物逐箱好坏计数），**打分也不受影响**。

### 3. 表现回填

`POST /cards/{name}/performance/backfill`，body `{"items":[{"request_id","label"}, ...]}`，标签只接受 0/1（`true/"1"/2` 等一律拒收）。同步完成格式校验后返回 202 与 `job_id`，作业后台执行；`GET /performance/backfills/{job_id}` 查进度/结果（`applied/duplicated/rejected` 与逐条拒收原因），`GET /performance/backfills?card_name=` 列作业。

- 找不到的标识 → 作业拒收清单 `not_found`；标签/标识非法 → 提交响应里 `invalid_rejected_items`，均不影响其它条目；
- 同一标识回填两次且**标签相同** → `duplicated`，幂等只算一遍；**标签不同** → 拒收 `label_conflict` 并保留首次标签（服务不替业务决定以哪次为准，交人工核对），同批次内重复同理；
- 查询：`GET /cards/{name}/versions/{v}/performance?start=&end=&n_bands=10`，返回这批**已有表现**的放款上的实际坏率、坏/好计数、平均预测 PD、**KS、AUC**，以及按总分分段的各段计数、实际坏率与平均预测 PD。分段默认与稳定性同口径、对齐建卡总分基准箱；只有老版本（无基准）才用该批分数现场切 `n_bands`（默认 10）个等频箱；标签全为同一类时 KS/AUC 按建卡评估同样的规则返回 null。
- **KS/AUC 口径唯一**：`app/performance/service.py` 直接调用建卡用的 `app/core/metrics.ks_stat / roc_auc`，输入就是留痕里逐条保存的 PD 与回填的 0/1 标签，没有第二套实现。

### 监控统计的实现取舍：只留痕、查询时现算（不做增量计数桶）

打分记录 `score_logs` 是唯一事实源，稳定性与表现统计一律在查询时按 `(卡, 版本, 时间区间)` 现算。

- **查询延迟**：随区间内打分量线性增长；用 `(card_name, version, scored_at)` 复合索引与 `monitor_query_limit`（默认 100 万）兜底。月度监控（万级到几十万级）为毫秒到几十毫秒量级；不适合做超低延迟大屏实时刷新。
- **写入开销**：打分路径只多一次 insert（带标识的是一次事务内 insert 记录 + 发布幂等结果），没有计数维护，写入恒定且不会出现计数热点。
- **任意起止时间**：现算天然支持任意 `[start,end)`；增量时间桶只能查桶边界对齐的区间，且桶粒度一旦定死无法改。
- **对账/出错恢复**：任何统计都能用打分记录逐条重算复核（测试里就同时跑「服务聚合」与「逐条重算」两条路径比对）；计数逻辑出错不会污染数据，修代码后查询即修复；记录补录/回填修正后统计自动正确。重启后结果不变（无内存统计态）。
- 代价是历史无限增长时扫描变贵：可按保留周期归档 `score_logs`（归档前先把需要的长期汇总落表），本期不实现。

### 数据库平滑升级

四张新表（`score_logs`、`idempotency`、`backfill_jobs`、`perf_labels`）与新索引全部 `CREATE ... IF NOT EXISTS`，服务启动时在已有库上直接执行即可，**不清库、不丢老版本与老作业数据**；新版本产物多一个增量字段 `monitoring`，老版本没有该字段时按上面的规则降级。内存后端（`STORAGE_BACKEND=memory`）与 PostgreSQL 后端实现同一套仓储接口，行为一致。

### 真实库集成测试

```bash
docker compose up -d db
RUN_PG_TESTS=1 DATABASE_URL=postgresql://scorecard:scorecard@localhost:5432/scorecard \
    pytest tests/test_postgres_integration.py
# 或一键：docker compose run --build tests   （profile=test）
```

无数据库时该文件自动 skip。用例覆盖：老库重复初始化不丢数据、跨连接并发重放只插一条、异内容并发冲突、训练样本自比 PSI=0、版本隔离、时间区间、回填拒收/幂等/冲突、KS/AUC 与建卡一致、重连（重启）后查询不变。

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

## 投后监控已验证的性质（pytest）

- 建卡样本原样逐条打分后再查稳定性：各特征与总分 PSI 全为 0（1e-12 内），逐箱占比/计数与建卡产物完全吻合；
- 8 线程并发打 2000 次分，留痕与统计中的打分次数正好 2000，且与从记录逐条重算一致；同标识并发/串行重放 100 次计数只加 1，重放响应逐字段等于首次；
- 同标识不同内容（含首个请求仍在打分的并发情形）明确 409/冲突，原记录不被覆盖；失败打分释放标识可重试；不带标识照常打分留痕；
- 批量失败条目不进留痕；批量条目独立幂等、冲突隔离；
- 未见类别、越界占比单独报出且不进 PSI；PSI 交换两侧不变、零占比走 1e-4 地板且结果有限；
- 服务重启（内存：统计无内存态；PG：新建连接重查）同一查询结果不变；版本 1/2 交替打分统计互不渗入；
- 老版本（无 `monitoring` 产物）总分稳定性返回 unavailable 且原因可读，特征层稳定性正常、打分正常；
- 回填：not_found/非法标签进拒收清单不影响其它条目；同标识同标签重放只算一遍、异标签拒收且首次标签保留；
- 投后 KS/AUC 与把同一批分数标签直接交给建卡 `core.metrics` 的结果逐位相等，也与建卡训练指标一致；总分分段计数合计等于放款数；
- 内存与 PostgreSQL 两后端共用同一仓储接口与服务层；真实 PG16（Compose）用例见 `tests/test_postgres_integration.py`，覆盖老库平滑升级。
