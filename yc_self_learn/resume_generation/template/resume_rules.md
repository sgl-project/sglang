# Resume Generation Rules (LaTeX)

## 0. 总原则

**只做两种输出**：
1. `不合适，下一个`（如果触发硬拒绝条件）
2. **输出一整段 LaTeX**：只改 3 段 Experience（HydroSense / Seno / Roswell），其他简历内容一律不动

**不解释、不点评、不分析你行不行**，除非触发硬拒绝条件。

**核心目标**：
- **通过 AI 审查（ATS）**：确保关键词匹配，让简历能够通过 Applicant Tracking System
- **让简历出众**：通过精准的关键词匹配和合理的经验迁移，提高简历在 recruiter 和 hiring manager 眼中的匹配度
- **考虑人类阅读可读性**：不能只是给 AI 看，要兼顾人类阅读体验。用词不要太高大上，专业名词可以高大上，但语言逻辑和写法要按照 STAR 格式（Situation/Task-Action-Result），保持简洁直接、平实易懂
- **合理推测和迁移**：从推测规则中提取相关技能和经验，适度调整表述以贴近 JD 要求，但不能乱改大改

---

## 1. 硬条件筛选（唯一会"拒绝"的条件）

**只要命中下面任意一条 → 直接输出：`不合适，下一个`**

### 1.1 身份/签证硬性限制

JD 明确写：
- `US Citizen only` / `Citizens only`
- `Green card` / `Permanent resident only`
- `No visa sponsorship` / `不提供 sponsorship` / `现在或未来都不 sponsor`（这种属于硬拒绝）

### 1.2 必须法定执照且你不可能满足

例如：
- RN license（注册护士执照）
- board-certified radiologist（放射科医师认证）
- 律师执照
- 注册会计师（CPA）

这种属于完全不同法定赛道 → 直接拒绝

### 1.3 额外硬规则

如果 JD 明确写 `10+ years` 且你觉得是硬门槛 → 也直接拒绝

**注意**：`7–10 年`、`8+ 年` 这种不自动拒绝

---

## 2. 模版选择（只在两套 base 模版中二选一）

- **JD 偏 Data/ML/LLM/DE/Analytics** → 用 **LLM/ML 模版**
- **JD 偏 Device/Imaging/Embedded/Robotics/System** → 用 **Hardware/Imaging 模版**

混合就按岗位主任务倾向选其一，不造第三种结构。

---

## 3. Title 规则

### 3.1 三家公司每家只保留 1 个 title

- 不允许一行堆两个 title
- **Title 策略：3选2**——一个高度相似，一个近似相似，一个稍微偏离一些
  - **高度相似**：title 高度贴合 JD 关键词（通常是 HydroSense 或 Seno）
  - **近似相似**：title 部分贴合 JD，但保持技术栈的宽度（通常是另一家公司）
  - **稍微偏离**：title 不完全贴合 JD，但展现技术深度和广度（通常是 Roswell，保持 Research Scientist 等）
- 不能虚高到不合理（不把你从 engineer 写成 director）
- 大部分工程岗位的 job title 没那么严格，可以灵活调整以匹配 JD 关键词

### 3.2 HydroSense 的 "Co-founder" 锁死

**HydroSense title 必须包含 Co-founder（永远不动）**

允许后半段随 JD 微调：
- `Co-founder / Machine Learning Engineer`
- `Co-founder / Systems & Verification Engineer`
- `Co-founder / Data Engineer`
- ...

但 **Co-founder 必须保留**

### 3.3 Title 策略——3选2匹配

**策略**：3选2——一个高度相似，一个近似相似，一个稍微偏离一些

**执行原则**：
- **高度相似的公司**（通常是 HydroSense 或 Seno）：把 JD 核心词塞进 title
  - 例如：JD 是 "Backend Engineer" → `Co-founder / Backend Engineer` 或 `Backend Engineer`
  - 例如：JD 是 "AI Infrastructure Engineer" → `Co-founder / AI Infrastructure Engineer` 或 `AI Infrastructure Engineer`
- **近似相似的公司**：title 部分贴合 JD，但保持技术栈的宽度
  - 例如：JD 是 "Backend Engineer" → `Software Engineer` 或 `Full Stack Engineer`
  - 例如：JD 是 "AI Infrastructure Engineer" → `Machine Learning Engineer` 或 `AI Engineer`
- **稍微偏离的公司**（通常是 Roswell）：title 不完全贴合 JD，但展现技术深度和广度
  - 例如：保持 `Research Scientist` 以展现研究能力和学术深度
  - 例如：保持 `Machine Learning Engineer` 以展现 ML 技术栈的广度

**目标**：既能通过 ATS 关键词匹配（2个高度/近似相似的 title），又能展现技术能力和经验广度（1个稍微偏离的 title）

---

## 4. Bullet 数量规则——锁死 5-4-4（544）

- **HydroSense**：5 条
- **Seno**：4 条
- **Roswell**：4 条

**永久锁死 544**，不是 555、也不是 443。

---

## 5. "Inventor/专利"位置规则——锁死在 HydroSense 第 5 条

**`Inventor on 3 granted patents` 必须出现在 HydroSense 的第 5 条 bullet**

不能放到第 1 条。

---

## 6. Bullet 长度规则——按模板类型区分

### 6.1 Software 模板（LLM/ML）

- 每条尽量控制在 **~25-26 个词上下**（不写小作文，精简1-2个词）
- 如果为塞关键词变长，就通过删形容词/合并短语来拉回长度

### 6.2 Hardware 模板（Device/Imaging/Embedded）

- 每条尽量控制在 **~33-35 个词上下**（可以适当放宽）
- Hardware 项目通常涉及更多技术细节，允许稍长

### 6.3 总体原则

- 总体 Experience 段落字数 ≈ 模版原本长度，不明显变长
- 优先保证可读性和自然流畅，长度是参考值而非硬性限制

---

## 6.5 STAR 格式和写作风格规则

### 6.5.1 STAR 结构要求

每条 bullet 应遵循 **STAR 格式**（Situation/Task-Action-Result）：

- **Situation/Task**：隐含在开头，用动作词直接说明做了什么（如 "Built", "Designed", "Implemented"）
- **Action**：具体的技术细节和方法（如 "using PyTorch training + INT8 quantization", "via FastAPI and containerized with Docker"）
- **Result**：量化的成果和影响（如 "reducing weight-estimation MAE by 15%", "increased Recall@5 to 90%"）

### 6.5.2 写作风格要求

**参考 software.tex 和 hardware.tex 的写作风格**：

1. **动作词开头**：使用强动作词（Built, Designed, Implemented, Led, Developed, Engineered, Set up）
2. **语言简洁直接**：不过度修饰，避免冗长的从句和复杂的句式
3. **专业名词可以高大上**：技术术语（如 FAISS, LoRA, RAG, photodiode amplification）可以使用专业名词
4. **描述语言平实**：用词不要太高大上，用简单直接的词汇描述动作和结果
5. **量化结果**：尽可能包含具体的数字、百分比、规模（如 "5000+ devices", "30% to 15%", "sub-percent repeatability"）
6. **技术细节具体**：包含具体的技术栈和方法（如 "C++ firmware on STM32-based controllers", "Python/MLflow", "LabVIEW–C++ architecture"）

### 6.5.3 示例对比

**❌ 不符合风格（过于高大上、缺乏STAR）**：
```
Leveraged cutting-edge machine learning paradigms to architect a sophisticated data processing infrastructure that synergistically integrates advanced neural network architectures, resulting in substantial performance enhancements across multiple dimensions.
```

**✅ 符合风格（严格STAR格式、清晰pipeline、明确impact）**：
```
Built an end-to-end device management pipeline to address remote monitoring challenges across distributed fleet, 
by developing C++ firmware on STM32 controllers plus Python services on edge gateway, with secure OTA updates 
via RS-485 and Bluetooth/Wi-Fi, resulting in fleet-wide remote monitoring and configuration for 5000+ devices 
with 95% update success rate and 60% reduction in on-site maintenance visits.
```

**STAR格式分解**：
- **Situation/Task**（隐含在开头）："to address remote monitoring challenges across distributed fleet" - 明确了问题和任务
- **Action**（pipeline搭建）："Built... by developing C++ firmware... plus Python services... with secure OTA updates" - 清晰说明了搭建的pipeline组件（firmware layer + gateway layer + update mechanism）
- **Result**（量化impact）："resulting in... 5000+ devices, 95% update success rate, 60% reduction in maintenance" - 明确的量化成果和业务价值

### 6.5.4 执行标准

- **动作词 + 具体技术 + 量化结果**：每条 bullet 都要有这三要素
- **专业名词可以高大上，描述语言要平实**：技术术语用专业词汇，但描述动作和结果的词汇要简单直接
- **兼顾 AI 和人类阅读**：既要通过 ATS 关键词匹配，又要让人类读起来自然流畅

---

## 7. 工具名堆叠规则——优先保证可读性

**原则**：工具名数量不硬性限制，但必须保证人类阅读时的感受和可读性

### 7.1 核心原则

- **可读性优先**：考虑 recruiter 扫描时的阅读感受，避免工具名堆叠造成混淆
- **自然流畅**：工具名应该自然融入句子，不突兀
- **信息密度**：避免单条 bullet 中工具名过多导致信息密度过高，阅读负担重
- **核心优先**：优先选择最核心、与 JD 最匹配的工具名，其余可在面试时补充

### 7.2 违反示例（可读性差）

```
RAGAS, TruLens, FAISS, text-embedding-3, BGE, E5, LangChain, LangGraph, Langfuse, OpenTelemetry...
```

**问题**：单条 bullet 中出现 6-10 个工具名，堆叠过多，扫描时容易混淆，可读性差

### 7.3 符合示例（可读性好）

```
RAGAS and FAISS-based evaluation pipeline with custom monitoring tools
LangChain and vector database integration for RAG system
```

**优点**：工具名自然融入句子，清晰易读，不造成阅读负担

**注**：工具名数量不是绝对限制，重点是保证可读性和自然流畅。如果 4-5 个工具名能自然融入且不造成混淆，也是可以接受的。

---

## 8. 编程语言匹配规则（JD 要求必须体现）

**要求**：如果 JD 明确要求特定编程语言（如 `Go + Python`），需要在简历中至少一条 bullet 体现

### 8.1 执行原则

- **扫描 JD** 中的语言要求（如 "Python and Go", "Go + Python"）
- **自然融入**：在相关 bullet 中自然地提及该语言组合
- **真实性边界**：
  - 如果确实有实际经验：在 bullet 中自然融入（如 "Python/Go services" 或 "Python + Go microservices"）
  - 如果没有经验：不硬加，避免面试露馅
  - Go 和 Python 比较接近，可以自然结合使用

### 8.2 示例

**JD 要求**：`Strong proficiency in Python and Go`

**简历体现**：
- ✅ "Built Python/Go services for production LLM systems..."
- ✅ "Developed microservices in Python + Go for distributed systems..."
- ❌ 完全不提 Go（即使 JD 明确要求）

---

## 9. JD 关键词策略——"能原样复用就原样复用"

### 9.1 抓高价值关键词（优先级）

1. **岗位/职责关键词**：
   - `requirements`, `verification`, `design control`, `V&V`, `QMS`, `traceability`, `test plans`, `SOP`, `DHF/DMR/ECO`, `CAPA`, etc.

2. **技术栈**：
   - `Python/MATLAB/C++/SQL/Docker/CI/CD/OpenCV/ITK/VTK`, etc.
   - **注意**：如果 JD 明确要求特定语言（如 Go），需要在 bullet 中体现（参考规则 8）

3. **交付物/流程**：
   - `traceability`, `test plans`, `SOP`, `DHF/DMR/ECO`, `CAPA`, etc.

### 9.2 插入位置优先级和策略

**Title 策略**：3选2——一个高度相似，一个近似相似，一个稍微偏离一些（见规则 3.1 和 3.3）

**Bullet 内容策略**：3选2——一个高度相似，一个近似相似，一个稍微偏离一些

- **高度相似的公司**（通常是 HydroSense 或 Seno）：
  - 每段第 1 条 bullet 放"角色+影响力+JD核心词"
  - 其余 bullet 分散覆盖 JD 技术栈/流程词
  - 高度贴合 JD 关键词，确保 ATS 匹配
  
- **近似相似的公司**：
  - Bullet 部分贴合 JD，但保持技术栈的宽度
  - 覆盖部分 JD 关键词，同时展现相关技术能力
  - 例如：JD 是 "Backend Engineer"，可以写 "Full Stack" 相关但包含后端技术栈
  
- **稍微偏离的公司**（通常是 Roswell）：
  - Bullet 内容保持原有技术栈的宽度和深度，不完全贴合 JD
  - 展现研究能力、学术深度和技术广度
  - 例如：保持 multi-modal data pipeline、diffusion model 等研究性内容，展现高水平

**目标**：既能通过 ATS 关键词匹配（2个高度/近似相似的 title + bullet），又能展现技术能力和经验广度（1个稍微偏离的 title + bullet）

### 9.3 真实性边界

- 不编没做过的东西
- 可用"范式共性词"贴近（例如 `workflow` / `verification` / `reproducible pipeline`），但不写具体你没碰过的专有系统

### 9.4 经验迁移和合理推测规则

**原则**：根据 JD 适当修改内容，通过过往经历合理推测相关经验，但不能乱改大改

#### 9.4.1 允许的迁移和推测

- **技术栈迁移**：如果做过 LLM，可以合理推测会 CV、ML、post-training、RNN、LSTM 等相关技术
- **领域迁移**：如果做过医疗 AI，可以合理推测会理解其他垂直领域的 AI 应用
- **工具迁移**：如果用过 PyTorch，可以合理推测会用 TensorFlow；如果用过 Docker，可以合理推测会用 Kubernetes
- **流程迁移**：如果做过 MLOps，可以合理推测会 CI/CD、monitoring、evaluation 等流程

#### 9.4.2 不允许的改动

- **不能乱改大改**：不能把没做过的项目说成做过
- **不能编造专有系统**：不能写具体没碰过的专有系统名称（如特定的内部工具、未接触过的框架）
- **不能虚高职位**：不能把 engineer 写成 director，不能把 contributor 写成 lead

#### 9.4.3 执行标准

- **适度调整**：根据 JD 需求，在已有经验基础上适度调整表述，使其更贴合岗位要求
- **合理推测**：基于已有技术栈和项目经验，合理推测相关技能和经验
- **保持真实**：所有改动必须基于真实经历，不能凭空编造

**示例**：
- ✅ 做过 LLM → 可以写 "experience with deep learning, neural networks, and transformer architectures"
- ✅ 做过 RAG → 可以写 "familiarity with information retrieval and vector databases"
- ✅ 做过医疗 AI → 可以写 "experience applying AI to domain-specific applications"
- ❌ 没做过 Kubernetes → 不能写 "extensive experience with Kubernetes"（但可以写 "familiarity with containerization and orchestration"）

#### 9.4.4 基于实际技术栈和项目经验的推测指南

**你的实际技术栈**（参考 skill_set.tex）：
- **Languages**: Python, C/C++, SQL, Bash, MATLAB, JavaScript, LabVIEW
- **Libraries**: NumPy, Pandas, Scikit-learn, OpenCV, Matplotlib, Seaborn, SciPy, PyVISA
- **ML Frameworks**: PyTorch, TensorFlow, Keras, CUDA
- **ML Techniques**: RAG, LoRA, QLoRA, Quantization, CNN, UNet, LSTM, Transformer, SVM, LASSO, PCA
- **Infrastructure**: Docker, Kubernetes, AWS, GCP, FastAPI, Flask, gRPC, Git, Linux
- **Data**: PostgreSQL, BigQuery, MLflow, CI/CD, REST API, Container Orchestration, Monitoring
- **Signal & Imaging**: DSP filtering, spectral analysis, image reconstruction, calibration, 3D visualization
- **Hardware/Embedded**: Embedded C, microcontrollers, sensor interfacing (laser, photodiode, BLE, RS-485), data acquisition, system automation

**你的实际项目经验**（参考 project_experience.tex）：
- **SGLang Contributions** (2025): router performance, scalability, concurrency, scheduling bottlenecks, multi-worker inference, cross-GPU latency, data-parallel scheduling, message streaming, load reporting, task batching, distributed inference
- **KYC Document Intelligence Pipeline** (2025): SGLang + Fireworks AI, Llama 3.2 11B Vision, structured JSON outputs, document understanding, risk assessment, confidence-based routing, corner case handling, FSI deployment, serverless/on-demand/batch scaling
- **Rad-Linter** (2025): Cross-modal quality control, hospital deployment, SGLang serving, local computing, observability, production optimization
- **Baby Monitoring System** (2022): Edge AI, AWS IoT Greengrass, PyTorch inference, embedded sensors, Docker, AWS Lambda, CloudWatch, offline operation, data caching, AWS S3
- **Photoacoustic Imaging Research** (2021-Present): Medical imaging, AI-assisted diagnostics, computer vision, real-time processing, multiphoton microscopy, biomedical optics, clinical diagnostics, image denoising (Noise2Noise), vascular imaging, wound assessment (mmWave, RF)
- **Publications**: 8+ papers in IEEE, Photoacoustics, Med-X, Biomedical Optics Express on medical imaging, AI diagnostics, RF sensing
- **Robotics Projects**: Autonomous navigation, obstacle avoidance, vision-based path tracking, motor control, sensor integration, target identification, prediction algorithms
- **Signal Processing**: GNU Radio, MATLAB, real-time signal acquisition, demodulation, signal classification, RF systems, embedded signal processing
- **Startup Experience** (2015-Present): Co-founder, high-precision sensors (weather & medical), firmware (C, Python), business expansion to 5+ countries
- **Research Experience**: CAS (rocket signal communication, embedded software, real-time telemetry), Cornell (multiphoton microscopy, biomechanical protocols), UKY (GUIs, image processing, C#, LabVIEW, neonatal devices)

**合理推测规则**（基于以上实际经验）：

**LLM/AI 相关**：
- ✅ **做过 LLM/RAG** → 可以推测：CV (OpenCV, UNet, CNN), ML (各种算法), post-training (LoRA, QLoRA, Quantization), RNN/LSTM, Transformer architectures, vision-language models
- ✅ **做过 SGLang (LLM serving)** → 可以推测：vLLM, Triton, DeepSpeed-Inference, inference optimization, distributed systems, router performance, scheduling, structured outputs, multi-worker inference, cross-GPU communication
- ✅ **做过 KYC Document Intelligence** → 可以推测：document understanding, structured extraction, risk assessment, confidence-based routing, schema-first design, production deployment, FSI compliance, serverless architectures
- ✅ **做过 Rad-Linter** → 可以推测：cross-modal QA, quality control systems, hospital IT integration, local deployment, observability for clinical systems

**ML/DL 相关**：
- ✅ **做过 PyTorch** → 可以推测：TensorFlow, Keras, JAX（已在技能栈中）
- ✅ **做过 CNN/UNet** → 可以推测：ResNet, VGG, U-Net variants, segmentation networks, image classification
- ✅ **做过 LSTM** → 可以推测：RNN, GRU, sequence modeling, time series analysis
- ✅ **做过 Transformer** → 可以推测：BERT, GPT variants, attention mechanisms, self-supervised learning
- ✅ **做过 Quantization** → 可以推测：INT8, FP16, model compression, pruning, distillation

**Infrastructure/DevOps 相关**：
- ✅ **做过 Docker** → 可以推测：Kubernetes, container orchestration, Helm, Terraform（已在技能栈中）
- ✅ **做过 AWS** → 可以推测：GCP, Azure, AWS IoT Greengrass, Lambda, S3, CloudWatch, EC2, EKS（已在技能栈中）
- ✅ **做过 FastAPI** → 可以推测：Flask, Django, gRPC, REST API, GraphQL（已在技能栈中）
- ✅ **做过 CI/CD** → 可以推测：GitHub Actions, Jenkins, GitLab CI, automated testing, deployment pipelines

**Hardware/Embedded 相关**：
- ✅ **做过 Embedded C/C++** → 可以推测：RTOS, firmware development, microcontroller programming, real-time systems, device drivers
- ✅ **做过 Sensor Interfacing** → 可以推测：laser control, photodiode amplification, BLE, RS-485, I2C, SPI, UART, ADC/DAC
- ✅ **做过 Data Acquisition** → 可以推测：DAQ systems, signal conditioning, sampling synchronization, real-time data processing
- ✅ **做过 System Automation** → 可以推测：LabVIEW, industrial automation, PLC programming, control systems

**Signal Processing/Imaging 相关**：
- ✅ **做过 GNU Radio** → 可以推测：RF systems, SDR, signal demodulation, spectrum analysis, wireless communication
- ✅ **做过 DSP Filtering** → 可以推测：digital filters, FIR/IIR, spectral analysis, frequency domain processing
- ✅ **做过 Image Reconstruction** → 可以推测：tomography, inverse problems, calibration, 3D visualization, medical imaging algorithms
- ✅ **做过 Photoacoustic Imaging** → 可以推测：ultrasound imaging, optical imaging, biomedical optics, clinical imaging systems

**Robotics/Autonomous Systems 相关**：
- ✅ **做过 Autonomous Navigation** → 可以推测：SLAM, path planning, obstacle avoidance, sensor fusion, localization
- ✅ **做过 Vision-based Tracking** → 可以推测：computer vision, object detection, tracking algorithms, camera calibration
- ✅ **做过 Motor Control** → 可以推测：servo control, stepper motors, PID control, motion planning, robotics kinematics

**Medical/Clinical 相关**：
- ✅ **做过 Medical Imaging** → 可以推测：clinical diagnostics, FDA regulations, medical device development, hospital IT systems, DICOM, PACS
- ✅ **做过 Biomedical Research** → 可以推测：NIH-funded projects, clinical trials, IRB compliance, research protocols
- ✅ **做过 Wound Assessment (mmWave/RF)** → 可以推测：IoT sensors, wearable devices, RF sensing, through-dressing monitoring

**Startup/Business 相关**：
- ✅ **做过 Co-founder/Startup** → 可以推测：0-to-1 product development, business acumen, full-stack development, customer acquisition, international expansion, fundraising
- ✅ **做过 Firmware Development** → 可以推测：hardware-software integration, product development, manufacturing support, field deployment

**Research/Academic 相关**：
- ✅ **做过 PhD Research** → 可以推测：research methodology, publication writing, grant writing, collaboration, mentoring
- ✅ **做过 Teaching Assistant** → 可以推测：curriculum development, student mentoring, technical communication, lab supervision

---

## 10. 输出格式规则（LaTeX-only）

**只输出 LaTeX 段落**

结构固定：
- **HydroSense**（5 bullets）
- **Seno**（4 bullets）
- **Roswell**（4 bullets）

**不输出解释文字**（除非你单独问"规则复述"）

---

## 11. 文件命名规则

**文件命名格式**：`resume_XX_template_公司名字[_岗位名字].tex`

### 11.1 命名规则

- **XX**：数字序号（递增，如 01, 02, 03, 04, 05...）
- **template**：模板类型（`software` 或 `hardware`）
- **公司名字**：从 JD 中提取的公司名称（简化形式，去除特殊字符）
- **岗位名字**（可选）：如果同一公司有多个不同岗位，需要在文件名中包含岗位名称（简化形式，去除特殊字符和空格，用下划线连接）

### 11.2 示例

**单一公司单一岗位**：
- JD 公司：`FriendliAI` → 文件名：`resume_05_software_FriendliAI.tex`
- JD 公司：`ShelterZoom` → 文件名：`resume_02_software_ShelterZoom.tex`
- JD 公司：`Interface` → 文件名：`resume_03_software_Interface.tex`
- JD 公司：`Exelon Corporation` → 文件名：`resume_01_hardware_Exelon.tex`

**同一公司多个岗位**（必须包含岗位名）：
- JD 公司：`Thinking Machines Lab`，岗位：`Software Engineer, Data Infrastructure` → 文件名：`resume_40_software_ThinkingMachines_DataInfrastructure.tex`
- JD 公司：`Thinking Machines Lab`，岗位：`Infrastructure Research Engineer` → 文件名：`resume_41_software_ThinkingMachines_InfrastructureResearchEngineer.tex`
- JD 公司：`xAI`，岗位：`Member of Technical Staff - RL Infrastructure` → 文件名：`resume_38_software_xAI_RLInfrastructure.tex`
- JD 公司：`xAI`，岗位：`RLHF Evaluation Engineer` → 文件名：`resume_39_software_xAI_RLHFEvaluation.tex`

### 11.3 执行

- 从 JD 中识别公司名称（通常在 "Company:"、"About [Company]" 等字段）
- 从 JD 中识别岗位名称（通常在标题、第一行或 "About the Role" 部分）
- 简化公司名称（去除 "Corporation"、"Inc."、"LLC"、"Lab" 等后缀，去除特殊字符）
- 简化岗位名称（去除标点符号、空格用下划线连接，如 "Software Engineer, Data Infrastructure" → "DataInfrastructure"）
- **如果同一公司有多个不同岗位，必须在文件名中包含岗位名称**
- 生成文件时使用此命名格式

---

## 12. "不要分析我行不行"规则（执行口径）

**以后你贴 JD**：除非触发 `10+ years` 硬门槛 / 签证硬拒绝 / 必须执照

**否则一律直接出改写结果**，不再判断匹配度、不再劝你申不申。

---

## 示例输出格式

### LaTeX 输出示例：

```latex
\cventry{2021--Present}{Co-founder / Machine Learning Engineer}{HydroSense}{City, State}{}{
  \begin{itemize}
    \item Bullet 1 (~28 words): 角色+影响力+JD核心词
    \item Bullet 2 (~28 words): 技术栈/流程词
    \item Bullet 3 (~28 words): 技术栈/流程词
    \item Bullet 4 (~28 words): 技术栈/流程词
    \item Inventor on 3 granted patents related to medical imaging and signal processing.
  \end{itemize}}

\cventry{2019--2021}{Senior Software Engineer}{Seno}{City, State}{}{
  \begin{itemize}
    \item Bullet 1 (~28 words): 角色+影响力+JD核心词
    \item Bullet 2 (~28 words): 技术栈/流程词
    \item Bullet 3 (~28 words): 技术栈/流程词
    \item Bullet 4 (~28 words): 技术栈/流程词
  \end{itemize}}

\cventry{2017--2019}{Software Engineer}{Roswell}{City, State}{}{
  \begin{itemize}
    \item Bullet 1 (~28 words): 角色+影响力+JD核心词
    \item Bullet 2 (~28 words): 技术栈/流程词
    \item Bullet 3 (~28 words): 技术栈/流程词
    \item Bullet 4 (~28 words): 技术栈/流程词
  \end{itemize}}
```

### 拒绝输出示例：

```
不合适，下一个
```

---

## 执行流程

1. **输入**：JD（Job Description）
2. **判断**：检查硬拒绝条件（签证、执照、10+ years）
3. **选择**：选择模版（LLM/ML 或 Hardware/Imaging）
4. **提取**：
   - 提取公司名称（用于文件命名）
   - 提取 JD 关键词（岗位/技术栈/流程）
5. **改写**：
   - 修改 3 段 Experience 的 title（HydroSense 必须包含 Co-founder）
   - 改写 bullets（544 结构，每条 ~25-26 words for software，~33-35 words for hardware）
   - 确保 HydroSense 第 5 条是专利
6. **生成**：
   - 创建文件，文件名格式：`resume_XX_template_公司名字.tex`（XX 为递增序号）
   - 输出 LaTeX 段落（不解释）

---

## 注意事项

- **不分析匹配度**：只执行改写，不判断你是否适合
- **不劝申不申**：除非触发硬拒绝条件
- **真实性边界**：不编造未做过的事情
- **格式严格**：只输出 LaTeX，不输出其他文字

---

## 13. Cover Letter 生成规则（LaTeX格式）

### 13.1 文件命名规则

**文件命名格式**：`cover_letter_公司名字.tex`

- **公司名字**：从 JD 中提取的公司名称（简化形式，去除特殊字符）
- 如果同一公司有多个不同岗位，文件名中也需要包含岗位名称：`cover_letter_公司名字_岗位名字.tex`

### 13.2 LaTeX 格式要求

**使用标准 LaTeX letter 文档类**：

```latex
\documentclass[11pt,a4paper]{letter}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{url}
\usepackage{hyperref}

\geometry{margin=1in}

% Sender information
\signature{Yanda Cheng}
\address{Yanda Cheng \\ yc2675@cornell.edu \\ 8593383379}
```

### 13.3 Cover Letter 结构

**必须包含以下部分**：

1. **Opening**：称呼和开头段落
   - 表达对职位的兴趣
   - 简要介绍背景

2. **Preferred Research Areas of Interest**（如果是研究/实习岗位）
   - 列出 3-4 个研究兴趣方向
   - 每个方向说明为什么感兴趣，以及相关经验

3. **Relevant Experience and Contributions**
   - 列出与职位相关的经验
   - 使用项目符号列表
   - 突出量化成果（如 "reducing hallucination rates from 30% to 15%"）

4. **Why [Company Name]**
   - 说明为什么选择这家公司
   - 强调价值观和目标的契合

5. **Relevant Links**
   - GitHub 链接
   - Publications 链接
   - Projects 链接
   - Blog 链接（如果有）

6. **Closing**：结尾段落和签名
   - 说明可用时间（如果是实习）
   - 感谢考虑

### 13.4 写作风格要求

- **专业但热情**：保持专业语调，但表达对职位和公司的热情
- **具体量化**：尽可能使用具体数字和成果（如 "8+ publications", "30% to 15%"）
- **匹配关键词**：自然融入 JD 中的关键词和术语
- **真实经验**：只写真实经历，不编造
- **长度适中**：通常 1-2 页（A4）

### 13.5 执行流程

1. **输入**：Cover letter description（包含职位信息、要求、公司背景）
2. **提取**：
   - 提取公司名称（用于文件命名）
   - 提取职位类型（研究/工程/实习等）
   - 提取关键要求（研究兴趣、技能要求等）
3. **生成**：
   - 创建 LaTeX 格式的 cover letter
   - 使用标准 letter 文档类
   - 包含所有必需部分
   - 文件名格式：`cover_letter_公司名字.tex`

### 13.6 示例

**文件命名示例**：
- JD 公司：`Together AI` → 文件名：`cover_letter_TogetherAI.tex`
- JD 公司：`Thinking Machines Lab`，岗位：`Data Infrastructure` → 文件名：`cover_letter_ThinkingMachines_DataInfrastructure.tex`
