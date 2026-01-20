# Pipeline DAG 学习推荐（工业界标准）

**目的**：学习工业界最常用的 Pipeline DAG 编排系统设计

**要求**：
- 工业界最常用、通用性最强
- 文档完善、易于学习
- 设计模式经典、可复用

---

## 🎯 精选 3 个最佳项目（工业界标准）

### 1. **Apache Airflow** ⭐⭐⭐⭐⭐（首选）

**为什么选它**：
- ✅ **工业界标准**：最广泛使用的 pipeline 编排系统（Airbnb 开源，被 Google、Netflix、Uber 等大厂使用）
- ✅ **设计经典**：DAG（有向无环图）编排模式，是 pipeline 设计的经典范式
- ✅ **文档完善**：官方文档详细，社区活跃，学习资源丰富
- ✅ **功能全面**：任务调度、依赖管理、监控告警、可观测性一应俱全

**GitHub**: https://github.com/apache/airflow  
**文档**: https://airflow.apache.org/docs/

**核心设计点（值得学习）**：

1. **DAG 编排系统**
   ```python
   # 示例：类似 rad-linter 的 Step 0-5
   from airflow import DAG
   from airflow.operators.python import PythonOperator
   
   with DAG('rad_linter_pipeline', schedule_interval='@daily') as dag:
       step0 = PythonOperator(task_id='subset_data', python_callable=subset_data)
       step1 = PythonOperator(task_id='align_openi', python_callable=align_openi)
       step2 = PythonOperator(task_id='extract_visual', python_callable=extract_visual)
       step3 = PythonOperator(task_id='construct_labels', python_callable=construct_labels)
       step4 = PythonOperator(task_id='train_lora', python_callable=train_lora)
       step5 = PythonOperator(task_id='evaluate', python_callable=evaluate)
       
       step0 >> step1 >> step2 >> step3 >> step4 >> step5
   ```
   - **学习点**：任务依赖管理、并行执行、失败重试、调度系统

2. **可观测性设计**
   - **Web UI**：实时查看 DAG 执行状态、任务日志、任务依赖图
   - **Metrics**：任务执行时间、成功率、重试次数
   - **Logs**：结构化日志、日志聚合、日志检索
   - **Monitoring**：告警机制、健康检查、任务监控
   - **学习点**：Dashboard 设计、监控告警系统、可观测性集成

3. **扩展性设计**
   - **Plugins**：丰富的插件系统（Kubernetes、Docker、AWS、GCP 等）
   - **Operators**：可自定义操作符（Python、Bash、SQL、Kubernetes 等）
   - **XComs**：任务间数据传递
   - **学习点**：插件化架构、组件可扩展性、数据流管理

**与 rad-linter 的对应关系**：
- Step 0-5 → Airflow DAG 的 task
- Judge Server → 可以作为 Airflow Operator（ExternalTaskSensor）
- Docker 部署 → 可以使用 KubernetesPodOperator
- 版本固定 → Airflow 的 DAG 版本管理

**学习路径**：
1. 阅读官方教程：https://airflow.apache.org/docs/apache-airflow/stable/tutorial/
2. 运行示例：https://github.com/apache/airflow/tree/main/airflow/example_dags
3. 理解架构：https://airflow.apache.org/docs/apache-airflow/stable/concepts/overview.html
4. 应用到 rad-linter：设计 rad-linter 的 DAG

---

### 2. **MLflow** ⭐⭐⭐⭐⭐（ML 领域标准）

**为什么选它**：
- ✅ **ML 领域标准**：实验管理 + 模型注册的标准工具（Databricks 开源）
- ✅ **设计完整**：涵盖实验跟踪、模型版本、模型注册、模型部署全生命周期
- ✅ **文档完善**：官方文档详细，示例丰富
- ✅ **通用性强**：适用于任何 ML 项目，不只是医学领域

**GitHub**: https://github.com/mlflow/mlflow  
**文档**: https://mlflow.org/docs/latest/

**核心设计点（值得学习）**：

1. **实验跟踪系统**
   ```python
   # 示例：类似 rad-linter 的 Step 4-5
   import mlflow
   
   with mlflow.start_run():
       # Step 4: Train LoRA
       mlflow.log_param("lora_r", 16)
       mlflow.log_param("lora_alpha", 32)
       mlflow.log_param("learning_rate", 2e-4)
       train_lora_model(...)
       
       # Step 5: Evaluate
       metrics = evaluate_model(...)
       mlflow.log_metrics({
           "rule_adherence": 1.0,
           "silver_agreement": 0.8874,
           "judge_rule_gap": 0.8874,
           "f1_score": 0.80
       })
       
       # Save model and artifacts
       mlflow.pytorch.log_model(model, "model")
       mlflow.log_artifact("results/step5_evaluation.json")
   ```
   - **学习点**：实验版本管理、参数/指标记录、可复现性保证

2. **模型注册系统（Model Registry）**
   - **模型版本**：支持模型版本管理、模型标签、模型阶段（Staging → Production）
   - **模型部署**：支持多种部署方式（REST API、Docker、Spark UDF、Kubernetes 等）
   - **模型监控**：支持模型性能监控、模型漂移检测
   - **学习点**：模型生命周期管理、模型版本控制、模型部署策略

3. **可复现性设计**
   - **环境记录**：自动记录 Python 版本、依赖包版本
   - **代码版本**：关联 Git commit、代码路径
   - **数据版本**：记录输入数据路径/版本
   - **学习点**：实验可复现性、版本管理策略、环境管理

4. **UI 和可视化**
   - **Experiment UI**：实验对比、指标可视化、参数对比
   - **Model Registry UI**：模型版本浏览、模型部署状态
   - **学习点**：Dashboard 设计、数据可视化、用户体验设计

**与 rad-linter 的对应关系**：
- Step 4（LoRA 训练）→ MLflow 实验跟踪
- Step 5（评估）→ MLflow 指标记录
- Judge Server 版本 → MLflow 模型版本
- Docker 版本固定 → MLflow 环境记录
- 三面板评估结果 → MLflow 指标可视化

**学习路径**：
1. 阅读快速开始：https://mlflow.org/docs/latest/quickstart.html
2. 运行示例：https://github.com/mlflow/mlflow/tree/master/examples
3. 理解架构：https://mlflow.org/docs/latest/concepts.html
4. 应用到 rad-linter：用 MLflow 跟踪 Step 4-5 的实验

---

### 3. **Kubeflow Pipelines** ⭐⭐⭐⭐⭐（容器化 + Kubernetes 原生）

**为什么选它**：
- ✅ **容器化标准**：Kubernetes 原生的 ML pipeline，容器化设计经典
- ✅ **Google 开源**：基于 Google 内部 pipeline 系统（TFX）开源
- ✅ **设计现代**：云原生架构、微服务设计、可扩展性强
- ✅ **生产就绪**：适合大规模、生产环境部署

**GitHub**: https://github.com/kubeflow/pipelines  
**文档**: https://www.kubeflow.org/docs/components/pipelines/

**核心设计点（值得学习）**：

1. **容器化 Pipeline 设计**
   ```python
   # 示例：类似 rad-linter 的 Step 0-5
   from kfp import dsl
   from kfp.dsl import ContainerOp
   
   @dsl.pipeline(
       name='rad-linter-pipeline',
       description='Radiology report quality assessment pipeline'
   )
   def rad_linter_pipeline(
       input_data_path: str = 'data/indiana.parquet',
       judge_server_url: str = 'http://judge-server:8000'
   ):
       # Step 0: Subset data
       step0_op = dsl.ContainerOp(
           name='subset-data',
           image='rad-linter:step0',
           arguments=['--input', input_data_path],
           file_outputs={'output': '/output/subset.jsonl'}
       )
       
       # Step 1: Align OpenI
       step1_op = dsl.ContainerOp(
           name='align-openi',
           image='rad-linter:step1',
           arguments=['--input', step0_op.outputs['output']],
           file_outputs={'output': '/output/aligned.jsonl'}
       )
       
       # Step 2: Extract visual features
       step2_op = dsl.ContainerOp(
           name='extract-visual',
           image='rad-linter:step2',
           arguments=['--input', step1_op.outputs['output']],
           file_outputs={'output': '/output/visual_facts.jsonl'}
       )
       
       # ... 其他步骤
       
       # 定义依赖关系
       step0_op >> step1_op >> step2_op >> step3_op >> step4_op >> step5_op
   ```
   - **学习点**：容器化设计、Pipeline 编排、组件可复用、资源管理

2. **可观测性集成**
   - **Pipeline UI**：图形化 Pipeline 执行视图、步骤状态、日志查看
   - **Metrics**：Prometheus 集成、自定义指标
   - **Logs**：集中式日志（Elasticsearch、Stackdriver 等）
   - **Traces**：分布式追踪（Jaeger、Zipkin）
   - **学习点**：云原生可观测性、分布式系统监控、链路追踪

3. **版本管理和可复现性**
   - **Pipeline 版本**：支持 Pipeline 版本管理
   - **组件版本**：每个组件（Container）独立版本
   - **数据版本**：支持数据版本管理
   - **实验跟踪**：与 MLflow 集成
   - **学习点**：容器版本管理、Pipeline 版本控制、可复现性设计

4. **扩展性和资源管理**
   - **资源限制**：CPU/内存/GPU 限制
   - **并行执行**：支持并行任务、分布式执行
   - **自动缩放**：根据负载自动缩放
   - **学习点**：资源管理、可扩展性设计、弹性伸缩

**与 rad-linter 的对应关系**：
- Step 0-5 → Kubeflow Pipeline 的 component
- Docker 部署 → 每个 step 是一个容器
- Judge Server → 可以作为 Pipeline 的一个 component
- Docker 版本固定 → Container 镜像版本固定
- 三面板评估 → 可以作为 Pipeline 的 output artifact

**学习路径**：
1. 阅读官方教程：https://www.kubeflow.org/docs/components/pipelines/tutorials/
2. 运行示例：https://github.com/kubeflow/pipelines/tree/master/samples
3. 理解架构：https://www.kubeflow.org/docs/components/pipelines/concepts/
4. 应用到 rad-linter：将 rad-linter 的 Step 0-5 容器化为 Kubeflow Pipeline

---

## 💡 核心设计模式总结

从这 3 个项目中，你学到的最重要的设计模式：

### 1. **Pipeline 编排模式**（从 Airflow 学）
- DAG（有向无环图）编排
- 任务依赖管理
- 并行执行和资源调度
- 错误处理和重试机制
- 调度系统设计

### 2. **实验管理模式**（从 MLflow 学）
- 实验跟踪和版本管理
- 参数和指标记录
- 模型生命周期管理
- 可复现性保证
- 模型注册和部署

### 3. **容器化架构模式**（从 Kubeflow 学）
- 容器化组件设计
- 云原生架构
- 可观测性集成
- 资源管理和自动缩放
- 分布式系统设计

---

## 🔗 快速链接

### 代码仓库
- **Apache Airflow**: https://github.com/apache/airflow
- **MLflow**: https://github.com/mlflow/mlflow
- **Kubeflow Pipelines**: https://github.com/kubeflow/pipelines

### 文档和教程
- **Airflow Docs**: https://airflow.apache.org/docs/
- **MLflow Docs**: https://mlflow.org/docs/latest/
- **Kubeflow Docs**: https://www.kubeflow.org/docs/components/pipelines/

### 快速开始
- **Airflow 教程**: https://airflow.apache.org/docs/apache-airflow/stable/tutorial/
- **MLflow 快速开始**: https://mlflow.org/docs/latest/quickstart.html
- **Kubeflow 教程**: https://www.kubeflow.org/docs/components/pipelines/tutorials/

---

**最后更新**：2025-01-19
