# 第五周 任务3：个人知识短板诊断与补强计划

## 一、短板诊断（基于前4周复盘）

| 短板领域 | 具体表现 | 影响程度 | 优先级 |
|----------|----------|----------|--------|
| Python 编码与兼容性 | Windows 下 GBK 编码报错、emoji 输出异常 | 高 | P0 |
| 类型检查与工程规范 | pyright 大量 `MissingTypeStubs` 警告、不了解类型注解 | 中 | P1 |
| 数据清洗进阶技巧 | 只会简单均值/中位数填充，不会插值、回归填充、孤立森林检测 | 中 | P1 |
| 模型评估指标 | 仅关注准确率，对精确率、召回率、F1、混淆矩阵理解不深 | 高 | P0 |
| PyTorch 工程实践 | 不会自定义 Dataset、迁移学习、模型保存策略 | 高 | P0 |
| Git 版本管理 | 会 add/commit，但不熟悉分支、标签、提交规范 | 低 | P2 |
| 文件与数据格式 | CSV/JSON 读写不熟练，未接触 XML/YAML | 低 | P2 |

---

## 二、补强方案

### P0：必须补齐（直接影响后续作业）

1. **编码与跨平台兼容**
   - 掌握 `encoding='utf-8'` 在 `open()`、`pandas.read_csv()`、`sys.stdout` 中的应用
   - 理解 Python 3 字符串 `str` 与字节 `bytes` 的区别
   - 学会处理 `UnicodeEncodeError` 和 `UnicodeDecodeError`

2. **模型评估指标**
   - 理解混淆矩阵的 4 个象限：TP / FP / TN / FN
   - 掌握 Precision、Recall、F1-score 公式与适用场景
   - 学会 sklearn 的 `classification_report` 和 `confusion_matrix`

3. **PyTorch 工程化**
   - 学会自定义 `torch.utils.data.Dataset`
   - 掌握 `state_dict` 保存最佳模型的策略
   - 了解 `DataLoader` 的 `num_workers`、`pin_memory`、`drop_last` 等参数

### P1：重要提升（增强代码质量）

4. **类型注解**
   - 学习 Python 3.9+ 内置类型：`list[int]`、`dict[str, float]`、`tuple[int, str]`
   - 学习 `typing.Optional`、`typing.Union` 等用法
   - 理解 pyright 警告类型与如何配置 `pyrightconfig.json`

5. **数据清洗进阶**
   - 时间序列缺失：前向填充 `ffill`、后向填充 `bfill`、线性插值 `interpolate`
   - 异常检测：`sklearn.ensemble.IsolationForest`、`scipy.stats.zscore`
   - 重复值处理：`df.duplicated()` + `df.drop_duplicates()`

### P2：拓展能力（长期积累）

6. **Git 进阶**
   - 分支操作：`git branch`、`git checkout -b feature-xxx`
   - 提交规范：`type(scope): subject`（feat/fix/docs/style/refactor）
   - 标签管理：`git tag -a v1.0 -m "release v1.0"`

7. **文件格式处理**
   - JSON：`json.load/dump`
   - YAML：`PyYAML`
   - Excel：`openpyxl`、`xlsxwriter` 进阶

---

## 三、练习目标

完成 `gap_filling_practice.py` 练习脚本，覆盖：
- 列表/字典推导式
- 类型注解
- pandas 高级清洗（插值、重复值、异常值）
- sklearn 混淆矩阵与分类报告
- 自定义 PyTorch Dataset 模板
