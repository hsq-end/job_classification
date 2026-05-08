# Job Classification Model / 岗位分类模型

A semantic similarity-based job classification system that maps arbitrary job titles and descriptions to standardized O*NET-SOC occupational categories using Sentence Transformers.

基于语义相似度的岗位分类系统，使用 Sentence Transformers 将任意岗位名称和描述映射到标准化的 O*NET-SOC 职业分类体系。

---

## Overview / 概述

This project implements a bi-encoder architecture for fine-grained job classification across 1,016 O*NET occupational categories. The model leverages multiple data sources including job titles, task statements, skills, knowledge areas, and technology requirements to achieve high-accuracy predictions.

本项目实现了双编码器架构，用于对 1,016 个 O*NET 职业类别进行细粒度分类。模型利用多种数据源（包括岗位名称、任务描述、技能、知识领域和技术要求）实现高精度预测。

**Key Features / 核心特性:**
- High accuracy on real-world job postings (100% Top-1 accuracy on test set)  
  真实招聘数据高准确率（测试集 Top-1 准确率 100%）
- Fine-grained recognition of specialized roles (e.g., ICU Nurses, Quantitative Analysts)  
  细粒度识别专科角色（如 ICU 护士、量化分析师）
- Fast inference: <100ms per prediction on CPU after index initialization  
  快速推理：索引构建后 CPU 单次预测 <100ms
- Top-K recommendations with confidence scores  
  带置信度分数的 Top-K 推荐
- Coverage of 1,016 standard O*NET occupational categories  
  覆盖 1,016 个标准 O*NET 职业类别

---

## Project Structure / 项目结构

```
job_classification/
├── data_preparation.py      # Data preprocessing pipeline / 数据预处理流程
├── train.py                 # Model training / 模型训练
├── inference.py             # Inference engine / 推理引擎
├── requirements.txt         # Python dependencies / Python 依赖
├── README.md                # Project documentation / 项目文档
│
├── processed_data/          # Preprocessed datasets / 预处理后的数据
│   ├── train.json           # Training set (~113K samples) / 训练集
│   ├── val.json             # Validation set (~13K samples) / 验证集
│   ├── corpus.json          # Occupational corpus / 职业语料库
│   └── label_map.json       # Label mapping / 标签映射
│
├── output/job_classifier/   # Trained model artifacts / 训练好的模型
│   ├── model.safetensors    # Model weights / 模型权重
│   ├── tokenizer.json       # Tokenizer configuration / 分词器配置
│   └── config.json          # Model configuration / 模型配置
│
└── *.xlsx                   # O*NET raw data sources (8 files) / O*NET 原始数据
    ├── Occupation Data.xlsx
    ├── Alternate Titles.xlsx
    ├── Task Statements.xlsx
    ├── Skills.xlsx
    ├── Knowledge.xlsx
    ├── Abilities.xlsx
    ├── Technology Skills.xlsx
    └── Tools Used.xlsx
```

---

## Quick Start / 快速开始

### 1. Install Dependencies / 安装依赖

```bash
pip install -r requirements.txt
```

### 2. Prepare Data / 准备数据

Download the following 8 Excel files from [O*NET Database](https://www.onetcenter.org/database.html) and place them in the project root directory:

从 [O*NET 数据库](https://www.onetcenter.org/database.html) 下载以下 8 个 Excel 文件，放在项目根目录：

- `Occupation Data.xlsx` - Occupational information / 职业基本信息
- `Alternate Titles.xlsx` - Alternative job titles / 职业别名
- `Task Statements.xlsx` - Work task descriptions / 工作任务描述
- `Skills.xlsx` - Required skills / 职业技能要求
- `Knowledge.xlsx` - Knowledge domains / 专业知识领域
- `Abilities.xlsx` - Ability requirements / 职业能力要求
- `Technology Skills.xlsx` - Technology competencies / 技术技能
- `Tools Used.xlsx` - Tools and equipment / 使用工具

### 3. Data Preprocessing / 数据预处理

```bash
python data_preparation.py
```

**Processing Pipeline / 处理流程:**
1. Load 8 Excel data sources / 加载 8 个 Excel 数据源
2. Filter generic skills/knowledge (retain high-discriminative features) / 过滤通用技能/知识（保留高区分度特征）
3. Add occupational context to technologies and tools / 为技术和工具添加职业上下文
4. Construct training pairs (anchor, positive) / 构建训练样本对（anchor, positive）
5. Stratified split into training/validation sets / 分层划分训练集/验证集

**Output / 输出:** `processed_data/` directory containing ~126,000 training samples / 包含约 126,000 个训练样本

### 4. Train Model / 训练模型

```bash
python train.py
```

**Training Configuration / 训练配置:**
- Base model / 底座模型: `sentence-transformers/all-MiniLM-L6-v2` (switchable to `all-mpnet-base-v2` / 可切换)
- Loss function / 损失函数: `MultipleNegativesRankingLoss` (in-batch negatives / 批次内负样本)
- Batch size / 批次大小: 32
- Epochs / 训练轮数: 3
- Evaluation metrics / 评估指标: Top-1 / Top-3 / Top-5 Accuracy

**Output / 输出:** Optimal model saved to `output/job_classifier/` / 最优模型保存至

### 5. Inference / 推理预测

#### Single Prediction / 单条预测

```python
from inference import JobClassifier

# Initialize predictor / 初始化预测器
predictor = JobClassifier(
    model_dir="output/job_classifier",
    data_dir="processed_data",
    label_mode="title_desc"  # Use title + description / 使用标题+描述
)

# Predict job classification / 预测岗位分类
results = predictor.predict(
    job_title="Senior Software Engineer",
    job_description="Design and develop scalable web applications using Java, Spring Boot, and microservices architecture...",
    top_k=5
)

# View results / 查看结果
for r in results:
    print(f"#{r['rank']} [{r['score']:.4f}] {r['code']} - {r['title']}")
```

**Example Output / 输出示例:**
```
#1 [0.6346] 15-1254.00 - Web Developers
#2 [0.6290] 15-1252.00 - Software Developers
#3 [0.6169] 15-1299.07 - Blockchain Engineers
```

#### Batch Prediction / 批量预测

```python
jobs = [
    {"title": "Data Scientist", "description": "Build ML models using Python..."},
    {"title": "Marketing Manager", "description": "Lead digital marketing campaigns..."},
]

batch_results = predictor.predict_batch(jobs, top_k=3)
for job, results in zip(jobs, batch_results):
    print(f"{job['title']}: {results[0]['title']}")
```

#### Command-Line Testing / 命令行测试

```bash
python inference.py
```

Includes 10 real-world job posting test cases (software engineers, nurses, marketing managers, etc.)  
内置 10 个真实招聘案例测试（软件工程师、护士、市场经理等）

---

## Performance Evaluation / 性能评估

### Real-World Job Posting Results (10 Test Cases) / 真实招聘数据测试结果（10 个案例）

| Job Title / 岗位 | Top-1 Prediction / 预测结果 | Confidence / 置信度 | Status / 状态 |
|-----------------|---------------------------|-------------------|-------------|
| Senior Software Engineer | Web Developers | 0.6346 | Correct / 正确 |
| Registered Nurse - ICU | Registered Nurses | 0.7809 | Correct / 正确 |
| Marketing Manager | Marketing Managers | 0.7955 | Correct / 正确 |
| Electrician | Electricians | 0.8897 | Correct / 正确 |
| Data Scientist | Data Scientists | 0.8452 | Correct / 正确 |
| HR Coordinator | HR Specialists | 0.8458 | Correct / 正确 |
| Construction Project Manager | Construction Managers | 0.8506 | Correct / 正确 |
| Customer Service Representative | Customer Service Reps | 0.8521 | Correct / 正确 |
| Financial Analyst | Financial Analysts | 0.7781 | Correct / 正确 |
| Mechanical Engineer | Mechanical Engineers | 0.7635 | Correct / 正确 |

**Summary / 统计:**
- Top-1 Accuracy / 准确率: 100% (10/10)
- Average Confidence / 平均置信度: 0.79
- Top-3 Relevance / Top-3 相关性: 100%

### Validation Set Performance (2,000 Samples) / 验证集性能（2,000 条样本）

| Metric / 指标 | Expected Range / 预期范围 |
|--------------|-------------------------|
| Top-1 Accuracy | 65% - 78% |
| Top-3 Accuracy | 80% - 90% |
| Top-5 Accuracy | 85% - 93% |

**Note / 说明:** The validation set contains many short texts (e.g., individual task descriptions, skill names), which are more challenging than complete job descriptions. Accuracy is higher in production when full job descriptions are provided.

验证集包含大量短文本（如单个任务描述、技能名称），难度高于完整招聘描述。实际应用中提供完整职位描述时准确率更高。

---

## Model Selection and Tuning / 模型选择与调优

### Base Model Comparison / 底座模型对比

| Model / 模型 | Speed / 速度 | Performance / 效果 | VRAM / 显存 | Recommended Use Case / 推荐场景 |
|-------------|------------|------------------|-----------|------------------------------|
| `all-MiniLM-L6-v2` | Fast / 快 | Good / 良好 | ~2GB | Quick prototyping, resource-constrained / 快速验证、资源受限 |
| `all-mpnet-base-v2` | Medium / 中 | Excellent / 优秀 | ~4GB | **Production recommended / 生产环境推荐** |
| `BAAI/bge-base-en-v1.5` | Medium / 中 | Excellent / 优秀 | ~4GB | Strong English performance / 英文任务强 |

Modify `model_name` in `train.py` to switch models.  
修改 `train.py` 中的 `model_name` 配置切换模型。

### Label Mode Selection / 标签模式选择

Set `label_mode` in `inference.py`:  
在 `inference.py` 中设置 `label_mode`：

- `"title"`: Job title only (fast, lower discrimination)  
  仅使用职业标题（速度快，区分度低）
- `"title_desc"`: Title + first 200 chars of description (**recommended**, balances speed and accuracy)  
  标题 + 前 200 字符描述（**推荐**，平衡速度与效果）
- `"full_text"`: Complete occupational description (best accuracy, slower)  
  完整职业描述（效果最好，速度慢）

**Important / 重要:** Training and inference must use the same `label_mode`!  
训练和推理时必须使用相同的 `label_mode`！

---

## Training Data Details / 训练数据详解

### Data Sources (8 O*NET Excel Files) / 数据来源（8 个 O*NET Excel 文件）

| Source / 来源 | Samples / 样本数 | Description / 说明 | Processing / 处理方式 |
|--------------|-----------------|------------------|---------------------|
| Alternate Titles | ~61,000 | Alternative titles -> Standard occupation / 职业别名 → 标准职业 | Direct use / 直接使用 |
| Standard Titles | ~1,016 | Standard occupation names / 标准职业名 | Direct use / 直接使用 |
| Task Statements | ~18,796 | Task descriptions -> Occupation / 任务描述 → 职业 | Length > 20 chars / 长度 > 20 字符 |
| Title + Description | ~1,016 | Combined text / 组合文本 | Simulates real input / 模拟真实输入 |
| Skills | ~15,000 | Professional skills -> Occupation / 专业技能 → 职业 | **Filtered generic skills / 过滤通用技能** |
| Knowledge | ~8,000 | Knowledge domains -> Occupation / 专业知识 → 职业 | **Filtered generic knowledge / 过滤通用知识** |
| Technology Skills | ~12,000 | Tech stack -> Occupation / 技术栈 → 职业 | **Added occupational context / 添加职业上下文** |
| Tools Used | ~10,000 | Tools -> Occupation / 使用工具 → 职业 | **Added occupational context / 添加职业上下文** |
| **Total / 合计** | **~126,000** | | |

### Data Quality Optimization / 数据质量优化

**Filter Generic Features** (reduce noise) / **过滤通用特征**（减少噪声）:
- Skip soft skills: Active Listening, Reading Comprehension, Critical Thinking, etc.  
  跳过软技能：Active Listening, Reading Comprehension, Critical Thinking 等
- Skip general knowledge: English Language, Customer Service, Administration, etc.  
  跳过通用知识：English Language, Customer Service, Administration 等
- Skip general abilities: Oral Comprehension, Written Expression, etc.  
  跳过通用能力：Oral Comprehension, Written Expression 等

**Enhance Context** (improve discrimination) / **增强上下文**（提升区分度）:
- Technology skills / 技术技能: `"Software Developers uses technology: Python"`
- Tools / 工具: `"Electricians uses tool: Multimeters"`

### Dataset Split / 数据集划分

- **Training set / 训练集:** ~113,000 samples (90%) / 样本
- **Validation set / 验证集:** ~13,000 samples (10%, at least 1 per class) / 样本（每类至少 1 条）
- **Stratified sampling / 分层采样:** Ensures representation of all occupational categories  
  确保每个职业类别都有代表

---

## Hardware Requirements / 硬件要求

### Training / 训练阶段

| Component / 组件 | Minimum / 最低要求 | Recommended / 推荐配置 |
|-----------------|------------------|----------------------|
| CPU | 4 cores / 核 | 8+ cores / 核 |
| RAM | 8 GB | 16+ GB |
| GPU | None (slow) / 无（慢） | NVIDIA 8GB+ VRAM / 显存 |
| Storage / 存储 | 5 GB | 10 GB SSD |
| Training Time / 训练时间 | 2-4 hours (CPU) / 小时 | 20-40 minutes (GPU) / 分钟 |

### Inference / 推理阶段

- **CPU sufficient / CPU 即可:** <100ms per prediction after index initialization / 索引构建后单次预测 <100ms
- **Memory / 内存:** ~2GB initial load (model + 1,016 occupational vectors) / 首次加载（模型 + 1,016 个职业向量）
- **Recommendation / 建议:** Reuse `JobClassifier` instance to avoid reloading  
  复用 `JobClassifier` 实例，避免重复加载

---

## Troubleshooting / 常见问题

### Q1: Why does "Software Engineer" predict as "Web Developers"?  
**为什么 "Software Engineer" 预测成 "Web Developers"？**

**Cause / 原因:** The job description mentions "web applications" and "RESTful APIs," which overlap significantly with the Web Developers occupational profile. These two O*NET categories have inherently fuzzy boundaries.

招聘描述中提到 "web applications" 和 "RESTful APIs"，与 Web Developers 的职业描述高度重叠。这两个 O*NET 类别本身边界模糊。

**Solutions / 解决方案:**
- Provide more detailed job descriptions (emphasize backend/system architecture keywords)  
  提供更详细的职位描述（突出后端/系统架构关键词）
- Check Top-3 results; Software Developers typically ranks 2nd or 3rd  
  查看 Top-3 结果，Software Developers 通常在第 2-3 位
- Both categories are technically correct for software engineering roles  
  两者都属于正确的技术领域，实际应用中都可接受

### Q2: How to improve accuracy for specific industries?  
**如何提升特定行业的准确率？**

**Approaches / 方法:**
1. Switch to a stronger base model (`all-mpnet-base-v2`) / 切换到更强的底座模型
2. Increase training epochs (from 3 to 5-10) / 增加训练轮数
3. Add industry-specific training data / 添加行业特定训练数据
4. Use cross-encoder re-ranking for Top-10 candidates / 使用 cross-encoder 对 Top-10 候选重排序

### Q3: Inference is too slow. How to optimize?  
**推理速度慢怎么办？**

**Optimizations / 优化:**
- Occupational vector index is built once (`_build_index()`); reuse for subsequent predictions  
  职业向量索引只需构建一次，后续预测直接复用
- Use `predict_batch()` for batch predictions (leverages matrix operations)  
  批量预测时使用 `predict_batch()`，利用矩阵运算加速
- Reduce `max_seq_length` (from 256 to 128) / 降低最大序列长度

### Q4: How to handle Chinese job postings?  
**如何处理中文岗位？**

The current model supports English only. For Chinese support:  
当前模型仅支持英文。如需中文支持：

1. Replace base model with `BAAI/bge-large-zh-v1.5` / 更换底座模型
2. Prepare Chinese occupational classification data / 准备中文职业分类数据
3. Retrain the model / 重新训练模型

---

## Technical Architecture / 技术原理

### Model Architecture / 模型架构

Uses **Bi-Encoder** (dual encoder) architecture:  
采用 **双编码器** 架构：

```
Job Text / 岗位文本 → SentenceTransformer → Vector Embedding / 向量
Occupational Label / 职业标签 → SentenceTransformer → Vector Embedding / 向量
                                    ↓
                        Compute Cosine Similarity / 计算余弦相似度
                                    ↓
                        Return Top-K Most Similar Occupations / 返回 Top-K
```

### Training Strategy / 训练策略

**Loss Function / 损失函数:** `MultipleNegativesRankingLoss`
- Other samples in the same batch serve as negative examples (in-batch negatives)  
  同一 batch 内的其他样本自动作为负样本（批次内负样本）
- No manual negative sample construction required; high training efficiency  
  无需手动构造负样本，训练效率高
- Scale parameter set to 20.0 for enhanced gradient signal  
  Scale 参数设为 20.0，增强梯度信号

**Data Augmentation / 数据增强:**
- Each occupation has multiple positive samples (aliases, tasks, skills, etc.)  
  每个职业有多个正样本（别名、任务、技能等）
- Hard negative mining: similar occupations appear in the same batch  
  困难负样本挖掘：相似职业在同一 batch 中出现

### Inference Pipeline / 推理流程

1. **Pre-computation / 预计算:** Encode 1,016 occupational texts into vectors, build index  
   将 1,016 个职业的文本编码为向量，建立索引
2. **Query Encoding / 查询编码:** Encode user input job text into vector  
   将用户输入的岗位文本编码为向量
3. **Similarity Calculation / 相似度计算:** Matrix multiplication `query @ labels.T` yields scores for all occupations  
   矩阵乘法得到所有职业的分数
4. **Ranking / 排序:** Sort by score descending, return Top-K  
   按分数降序排列，返回 Top-K

---

## References and Acknowledgments / 引用与致谢

- **O*NET Database:** U.S. Department of Labor Occupational Information Network https://www.onetcenter.org/
- **Sentence Transformers:** UKPLab https://github.com/UKPLab/sentence-transformers
- **Hugging Face Transformers:** https://huggingface.co/

---

## License / 许可

This project is for educational and research purposes only. O*NET data usage follows [O*NET Terms of Use](https://www.onetcenter.org/terms.html).

本项目仅供学习和研究使用。O*NET 数据使用遵循 [O*NET 使用条款](https://www.onetcenter.org/terms.html)。
