"""
模型训练模块 - Bi-Encoder with MultipleNegativesRankingLoss

原理：
  - 将 (岗位文本, 标准职业名) 作为正样本对
  - 同一 batch 内其他样本自动作为负样本（in-batch negatives）
  - 训练后：相似职业的向量距离近，不同职业的向量距离远
"""

import json
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ── 配置 ──────────────────────────────────────────────────────────────
CONFIG = {
    # 底座模型（英文岗位分类推荐以下之一）
    # 'sentence-transformers/all-MiniLM-L6-v2'   轻量快速，适合快速验证
    # 'sentence-transformers/all-mpnet-base-v2'  效果最好，推荐生产使用
    # 'BAAI/bge-base-en-v1.5'                    BAAI 出品，英文效果强
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",  # 改用轻量模型加速训练

    "output_dir": "output/job_classifier",
    "data_dir": "processed_data",

    "batch_size": 32,        # CPU 训练降低 batch size
    "num_epochs": 3,         # 减少 epoch 数
    "warmup_ratio": 0.1,
    "max_seq_length": 256,   # 岗位名+简介截断长度

    # 推理时使用 label 侧的文本
    "label_text_mode": "title_desc",  # 使用 title+description 提升效果
}


def load_processed_data(data_dir: str):
    with open(f"{data_dir}/train.json") as f:
        train = json.load(f)
    with open(f"{data_dir}/val.json") as f:
        val = json.load(f)
    with open(f"{data_dir}/corpus.json") as f:
        corpus = json.load(f)
    with open(f"{data_dir}/label_map.json") as f:
        label_map = json.load(f)
    return train, val, corpus, label_map


def build_label_texts(corpus: dict, mode: str) -> dict:
    """构建每个职业类别的标准文本表示（用于推理时的向量库）"""
    label_texts = {}
    for code, info in corpus.items():
        if mode == "title":
            label_texts[code] = info['title']
        elif mode == "title_desc":
            desc = info['description'][:200] if info['description'] else ''
            label_texts[code] = f"{info['title']}. {desc}"
        else:
            label_texts[code] = info.get('full_text', info['title'])
    return label_texts


def create_training_examples(train_data: list, corpus: dict, mode: str):
    """
    创建 InputExample 对：(anchor_text, positive_text)
    anchor = 输入的岗位文本（别名/任务描述等）
    positive = 对应的标准职业文本
    """
    label_texts = build_label_texts(corpus, mode)
    examples = []

    for sample in train_data:
        code = sample['label_code']
        if code not in label_texts:
            continue
        examples.append(InputExample(
            texts=[sample['text'], label_texts[code]]
        ))

    # 对 task 来源样本做下采样，避免它主导训练
    task_samples = [e for e in examples if True]  # 可加来源过滤
    random.shuffle(examples)
    print(f"训练样本对数: {len(examples)}")
    return examples


class JobClassificationEvaluator:
    """
    自定义评估器：计算 Top-1 / Top-3 / Top-5 Accuracy
    使用向量相似度检索
    """
    def __init__(self, val_data: list, corpus: dict, label_map: dict,
                 mode: str, name: str = "val"):
        self.val_data = val_data[:2000]  # 验证集取前2000条加速评估
        self.corpus = corpus
        self.label_map = label_map
        self.mode = mode
        self.name = name
        self.label_texts = build_label_texts(corpus, mode)
        self.codes = list(self.label_texts.keys())
        self.code_to_idx = {c: i for i, c in enumerate(self.codes)}

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        model.eval()

        # 编码所有标准职业向量
        label_sentences = [self.label_texts[c] for c in self.codes]
        label_embeddings = model.encode(
            label_sentences, batch_size=128,
            show_progress_bar=False, convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 编码验证集 query
        queries = [s['text'] for s in self.val_data]
        true_codes = [s['label_code'] for s in self.val_data]
        query_embeddings = model.encode(
            queries, batch_size=128,
            show_progress_bar=False, convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 计算相似度
        scores = query_embeddings @ label_embeddings.T  # (N, num_labels)

        top1, top3, top5 = 0, 0, 0
        for i, true_code in enumerate(true_codes):
            if true_code not in self.code_to_idx:
                continue
            true_idx = self.code_to_idx[true_code]
            top_k = np.argsort(scores[i])[::-1][:5]
            if top_k[0] == true_idx:
                top1 += 1
            if true_idx in top_k[:3]:
                top3 += 1
            if true_idx in top_k[:5]:
                top5 += 1

        n = len(true_codes)
        top1_acc = top1 / n
        top3_acc = top3 / n
        top5_acc = top5 / n

        print(f"\n[Epoch {epoch}] {self.name} | "
              f"Top-1: {top1_acc:.4f} | Top-3: {top3_acc:.4f} | Top-5: {top5_acc:.4f}")

        return top1_acc  # 主要指标，用于保存最优模型


def train(config: dict = CONFIG):
    print("=" * 50)
    print("加载处理后的数据...")
    train_data, val_data, corpus, label_map = load_processed_data(config['data_dir'])

    print(f"加载底座模型: {config['model_name']}")
    model = SentenceTransformer(config['model_name'])
    model.max_seq_length = config['max_seq_length']

    print("构建训练样本对...")
    train_examples = create_training_examples(train_data, corpus, config['label_text_mode'])

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config['batch_size']
    )

    # MultipleNegativesRankingLoss: 同 batch 内其他样本自动作为负样本
    # scale=20 是经验值，效果好于默认值
    train_loss = MultipleNegativesRankingLoss(model, scale=20.0)

    evaluator = JobClassificationEvaluator(
        val_data, corpus, label_map,
        mode=config['label_text_mode']
    )

    warmup_steps = int(
        len(train_dataloader) * config['num_epochs'] * config['warmup_ratio']
    )

    output_dir = config['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n开始训练...")
    print(f"  训练样本对: {len(train_examples)}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Warmup steps: {warmup_steps}")
    print("=" * 50)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=config['num_epochs'],
        warmup_steps=warmup_steps,
        output_path=output_dir,
        save_best_model=True,
        show_progress_bar=True,
        evaluation_steps=500,
        checkpoint_save_steps=1000,
        checkpoint_path=f"{output_dir}/checkpoints",
    )

    print(f"\n训练完成！最优模型保存至: {output_dir}")


if __name__ == "__main__":
    train()
