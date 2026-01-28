from sglang.test.doc_patch import launch_server_cmd, execute_shell_command
from sglang.utils import wait_for_server, print_highlight, terminate_process
import os
import subprocess
import requests
import json
import time
import threading

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['SGLANG_ENABLE_TORCH_COMPILE'] = '0'
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
os.environ['SGLANG_DISABLE_CUDNN_CHECK'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PYTHONPATH'] = '/workspace/pod/sglang/python:$PYTHONPATH'
os.environ['SGLANG_USE_MODELSCOPE'] = 'true'


# This is equivalent to running the following command in your terminal

server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server  --model-path Qwen/Qwen3-8B \
 --host 0.0.0.0 --attention-backend flashinfer --tp=1 --disable-cuda-graph \
 --skip-server-warmup  --enable-mixed-chunk \
 --enable-flashinfer-pod  --chunked-prefill-size 512
"""
)

wait_for_server(f"http://localhost:{port}")

# 定义发送请求的函数
def send_chat_request(messages, max_tokens=1000, temperature=0.0):
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    return result

'''# 测试 1: 简单请求验证 POD attention 基本功能
print("=== 测试 1: 简单请求验证 POD attention ===")
result1 = send_chat_request([{"role": "user", "content": "你好，请用一句话介绍 POD attention。"}], max_tokens=200)
print("响应:", result1["choices"][0]["message"]["content"])

# 测试 2: 长中文请求测试
print("\n=== 测试 2: 长中文请求测试 ===")
long_prompt = """你是一位专业的内容创作者，请根据以下要求生成一段原创文字：主题围绕人工智能如何改变现代教育，内容需兼具信息性与可读性，适合普通读者理解。请避免使用过于专业的术语，若必须使用，请附带简明解释。文章应包含三个核心部分：首先，简要介绍当前教育体系面临的主要挑战（如资源不均、个性化不足等）；其次，具体说明人工智能在解决这些问题中的实际应用（例如智能辅导系统、自适应学习平台、自动化评估工具等），并列举一两个真实或典型的案例加以佐证；最后，探讨 AI 融入教育可能带来的潜在风险或伦理问题（如数据隐私、算法偏见、师生关系弱化等），并提出建设性的应对建议。整体语气应保持客观、理性但不失温度，字数控制在 600 字以内。请确保逻辑连贯、段落分明，并以中文输出。不要添加标题或小标题，直接以正文形式呈现。此外，请勿引用未经核实的数据或虚构不存在的研究成果，所有陈述应基于当前（截至2025年）已知的主流技术发展和教育实践。如果你对某些细节不确定，请优先选择概括性描述而非具体数字。最终目标是让读者在阅读后，既能理解 AI 对教育的积极影响，也能意识到其中需要谨慎对待的问题，从而形成全面而平衡的认知。"""
result2 = send_chat_request([{"role": "user", "content": long_prompt}], max_tokens=3000)
print("响应:", result2["choices"][0]["message"]["content"])

# 测试 3: Mixed Chunk 模式测试（并发请求触发 POD mixed mode）
print("\n=== 测试 3: Mixed Chunk 模式测试 ===")

# 用于存储结果的容器
long_request_result = {"done": False, "response": None}

def long_request_thread():
    """长请求：会进入 decode 阶段"""
    print("[线程1] 启动长请求...")
    response = send_chat_request([{"role": "user", "content": long_prompt}], max_tokens=100)
    long_request_result["response"] = response
    long_request_result["done"] = True
    print("[线程1] 长请求完成！")

# 启动长请求线程
thread1 = threading.Thread(target=long_request_thread)
thread1.start()

# 等待 1 秒，确保长请求已经完成 prefill 并进入 decode 阶段
print("[主线程] 等待 1 秒，让长请求进入 decode 阶段...")
time.sleep(0.5)  # 增加到 0.5 秒，确保长请求完成 prefill 并开始 decode

# 此时发送第二个请求，会触发 mixed mode（新请求 prefill + 旧请求 decode）
print("[主线程] 发送短请求，触发 mixed mode...")
short_prompt = "请用一句话解释什么是机器学习。"
result3 = send_chat_request([{"role": "user", "content": short_prompt}], max_tokens=1000)
print("[主线程] 短请求响应:", result3["choices"][0]["message"]["content"])

# 等待长请求完成
thread1.join()
long_content = long_request_result["response"]["choices"][0]["message"]["content"]
print(f"\n[主线程] 长请求最终响应:")
print(f"  完整长度: {len(long_content)} 字符")
print(f"  内容预览: {long_content[:300]}...")
'''
# 测试 4: 多次并发请求测试 mixed mode 稳定性
'''print("\n=== 测试 4: 多次并发 Mixed Chunk 测试 ===")'''

def concurrent_mixed_test():
    """测试多次并发请求是否稳定"""
    results = []

    def worker(prompt, max_tokens, delay):
        if delay > 0:
            time.sleep(delay)
        resp = send_chat_request([{"role": "user", "content": prompt}], max_tokens=max_tokens)
        results.append(resp)
        '''print(f"  [完成] prompt 长度: {len(prompt)}, tokens: {max_tokens}")'''

    # 启动多个并发请求
    threads = []
    prompts = [
        ("请以系统化、循序渐进的方式解释量子计算的基础原理：先从“量子计算与经典计算的差异”切入，说明为什么经典比特只能表示0或1，而量子计算使用量子比特能够带来新的信息表示与计算范式；随后详细阐述量子比特（qubit）的数学表示与物理实现思路，解释基态|0⟩、|1⟩、振幅与相位的含义，以及测量导致的概率结果与塌缩过程；进一步重点说明量子叠加的概念、直观理解与其在计算中形成“量子并行性”的原因，同时澄清叠加并不等同于一次得到所有答案；再深入介绍量子纠缠的定义、典型纠缠态示例、纠缠所带来的非经典相关性，以及它为何是量子算法、量子通信与量子误差校正的重要资源；最后简要概括量子门、干涉与典型算法如何利用叠加与纠缠产生潜在加速，并提及噪声、退相干与纠错等现实挑战。", 1000, 0.0),    # 第一个立即开始，长文本
        ("请用面向初学者但保持技术严谨的方式解释“什么是深度学习”：先说明深度学习与机器学习、传统统计建模之间的关系，解释“深度”一词为何对应多层神经网络的层级结构；再描述神经网络如何通过多层非线性变换学习从低级到高级的特征表示，强调端到端训练与自动特征学习相对于手工特征工程的差异；要求介绍常见网络结构及其适用场景，例如CNN在图像中的局部感受野与权重共享思想、RNN/LSTM在序列建模中的记忆机制、Transformer的自注意力如何捕捉长程依赖；同时概括训练流程（损失函数、反向传播、梯度下降、正则化与早停等）和对数据与算力的需求；最后请列举深度学习典型应用（视觉、语音、NLP、推荐、生成式任务）并点出局限性（可解释性、数据偏差、泛化与鲁棒性、成本与能耗）", 1000, 0.0),         # 延迟 0.0 秒，触发 mixed
        ("请从语言设计理念、运行环境、生态系统与工程实践四个维度，全面比较Python与JavaScript的差异：先说明两者各自最典型的应用场景（Python在数据分析、AI、自动化、后端与科研；JavaScript在浏览器端交互、前端工程化以及Node.js后端）；再比较语法与代码组织方式（缩进与花括号、模块系统、包管理、常见编码风格）；进一步解释两者的执行模型与并发机制差异（Python解释器与GIL对多线程的影响、asyncio；JavaScript事件循环、回调/Promise/async-await），以及这对I/O密集与CPU密集任务的意义；同时对比类型系统与开发体验（动态类型、类型标注/TypeScript、调试与工具链）；最后从工程落地角度讨论性能、部署方式、跨平台能力、社区库质量与学习曲线，并给出在“做Web应用、做数据科学、做脚本自动化、做全栈开发”这几类目标下的选型建议与原因。", 1000, 0.0),  # 延迟 0.0 秒
        ("请用通俗但完整的方式介绍区块链技术的基本概念及其应用：首先解释区块链作为“分布式账本”的含义，说明数据如何以区块为单位组织、通过哈希指针形成链式结构，从而具备难以篡改与可追溯特性；其次阐述共识机制在无中心化信任场景中的作用，要求对比PoW、PoS等机制的基本思路、优缺点与安全假设；再说明公链、联盟链、私链的差异，以及节点、钱包、地址、私钥签名、交易广播与确认的基本流程；进一步介绍智能合约的概念、可编程性带来的业务自动执行能力，以及其常见风险（漏洞、可升级性、预言机问题）；最后请列举区块链在数字资产与支付清算、供应链溯源、跨境结算、身份与凭证、数据确权、去中心化金融（DeFi）、NFT与数字内容等领域的代表性应用，并指出现实挑战，如吞吐量、扩容、隐私保护、监管合规与能源消耗等。", 1000, 0.0),  # 延迟 0.0 秒
        ("请围绕“机器学习中的过拟合问题如何解决”给出结构化回答：先定义过拟合与欠拟合，并解释它们在训练集与验证/测试集表现上的典型特征；再从成因角度分析为什么模型容量过大、数据量不足、噪声或标签错误、训练过久、特征泄漏等会导致过拟合；随后系统梳理常用对策，包括数据层面（更多数据、数据增强、清洗异常与纠正标签）、模型层面（降低复杂度、特征选择、剪枝、限制树深、减少参数）、训练层面（正则化L1/L2、Dropout、早停Early Stopping、交叉验证、学习率与训练轮数控制）、集成方法（Bagging、随机森林、Boosting的正则化形式）以及超参数调优策略；同时说明如何正确划分数据集、如何使用验证集与K折交叉验证评估泛化；最后给出在分类、回归与深度学习场景下的具体做法示例，并提醒注意数据泄漏与评估指标选择对结论的影响。", 1000, 0.0),   # 延迟 0.0 秒
        ("请简要但覆盖全面地说明自然语言处理（NLP）的主要任务有哪些，并要求按“基础处理—理解—生成—应用系统”层级组织：先列出基础文本处理任务，如分词/词形还原、词性标注、命名实体识别、句法分析（依存/成分）与核心指代消解，并说明这些任务为更高级应用提供结构化信息；再概括语义理解类任务，如文本分类、情感分析、主题识别、语义相似度、信息抽取（关系抽取、事件抽取）、阅读理解与问答，以及它们在实际业务中的用途；随后说明文本生成与对话相关任务，包括机器翻译、摘要生成、对话系统、文本改写与风格迁移、代码生成与多模态生成中的文本部分；同时提及检索与推荐中的NLP任务，如语义检索、搜索排序、RAG中的检索与重写；最后要求给出典型应用场景（客服、舆情、办公自动化、教育、医疗、法律）并点出关键挑战，如歧义、长文本、事实一致性、偏见与安全、低资源语言与领域迁移等。", 1000, 0.0),  # 延迟 0.0 秒
        ("请解释“什么是计算机视觉”并列举其常见应用：首先给出定义，说明计算机视觉旨在让机器从图像与视频中感知、理解并进行决策或生成内容；接着按任务类别介绍核心问题，如图像分类、目标检测、语义/实例分割、关键点与姿态估计、跟踪、光流、3D重建与深度估计，以及OCR与文档理解等；要求说明这些任务分别解决什么问题、输出形式是什么（类别、框、掩码、关键点、轨迹等）；再概述常用技术路线，从传统特征到深度学习模型（CNN、ViT、检测框架如Faster R-CNN/YOLO、分割网络等）的演进；最后列出实际应用：人脸识别与门禁、安防监控、工业质检、医疗影像辅助诊断、无人机巡检、零售盘点、自动驾驶感知、AR/VR、内容审核与生成式视觉，并补充挑战，如光照与遮挡、实时性、数据标注成本、泛化与鲁棒性、隐私合规等。", 1000, 0.0) ,   # 延迟 0.0 秒
        ("请解释强化学习的基本概念并说明其在游戏中的应用：先用“智能体—环境—状态—动作—奖励”框架定义强化学习，解释目标是学习一项策略使长期累计回报最大化；再区分基于价值的方法（Q-learning、DQN）、基于策略的方法（Policy Gradient、PPO）以及Actor-Critic结构，并说明探索与利用的矛盾、折扣因子与回合式/连续式任务的差异；同时介绍马尔可夫决策过程（MDP）的基本要素，解释为什么状态表示、奖励设计与终止条件对学习效果至关重要；随后结合游戏场景阐述强化学习如何通过自博弈、模拟器交互与经验回放提升策略，并举例说明在Atari、围棋/象棋、MOBA或策略游戏中训练时面临的稀疏奖励、长时序信用分配、多智能体协作与对抗、以及样本效率问题；最后讨论强化学习从游戏走向现实控制（机器人、推荐、调度）时的挑战，如安全探索、仿真到现实迁移与可解释性。", 1000, 0.0) ,   # 延迟 0.0 秒
        ("请介绍生成对抗网络（GAN）是什么以及它如何工作：首先说明GAN是一类生成模型，由生成器（Generator）与判别器（Discriminator）构成，通过对抗训练学习数据分布；要求用直观语言解释两者的分工：生成器从随机噪声或条件输入生成样本，判别器判断样本来自真实数据还是生成数据；再解释训练过程为何是一个最小最大博弈，生成器试图“骗过”判别器、判别器不断提高鉴别能力，最终在理想状态下生成样本与真实分布难以区分；同时介绍条件GAN、CycleGAN等变体的基本思路与适用任务（图像到图像转换、风格迁移、超分辨率）；要求说明常见训练难点，如模式崩溃、梯度消失、训练不稳定，以及WGAN、谱归一化、标签平滑等改进方法的动机；最后列举GAN在图像生成、数据增强、图像修复、虚拟人物与内容创作中的应用，并提醒其潜在风险，如深度伪造、版权与安全治理问题。", 1000, 0.0)  ,  # 延迟 0.0 秒
        ("请介绍自动驾驶技术的核心原理和挑战，并按“感知—定位—预测—规划—控制—系统安全”链路展开：先说明自动驾驶如何通过多传感器（摄像头、毫米波雷达、激光雷达、超声波、IMU/GNSS）获取环境信息，并进行时间同步与融合；再解释感知模块的关键任务（车道线、交通灯标志、车辆行人检测与跟踪、占用网格/BEV表征），以及定位与建图（高精地图、SLAM、融合定位）的作用；随后说明对周围交通参与者行为的预测如何为规划提供先验，规划如何在规则、舒适性与安全约束下生成轨迹（全局路径、局部轨迹、决策），控制模块如何将轨迹转化为转向、油门、制动；进一步要求讨论数据闭环、仿真测试、OTA与监控体系；最后重点列出挑战：长尾场景与极端天气、传感器噪声与遮挡、实时性与算力、可解释与验证困难、功能安全与冗余、网络与对抗安全、法规与责任界定、以及规模化落地的成本与商业模式问题。", 1000, 0.0)    # 延迟 0.0 秒
    ]

    for prompt, tokens, delay in prompts:
        t = threading.Thread(target=worker, args=(prompt, tokens, delay))
        threads.append(t)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()

# 输出所有结果
    for i, resp in enumerate(results):
        content = resp['choices'][0]['message']['content']
        print(f"\n结果 {i+1}:")
        print(f"{content}")

concurrent_mixed_test()

'''# 清理
print("\n=== 测试完成 ===")'''

terminate_process(server_process)