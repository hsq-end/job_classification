"""
数据准备模块
将三个 O*NET 数据集整合成训练数据
"""

import pandas as pd
import json
import random
from pathlib import Path

random.seed(42)


def load_data(data_dir: str = "."):
    alt = pd.read_excel(f"{data_dir}/Alternate Titles.xlsx")
    occ = pd.read_excel(f"{data_dir}/Occupation Data.xlsx")
    task = pd.read_excel(f"{data_dir}/Task Statements.xlsx")
    skills = pd.read_excel(f"{data_dir}/Skills.xlsx")
    knowledge = pd.read_excel(f"{data_dir}/Knowledge.xlsx")
    abilities = pd.read_excel(f"{data_dir}/Abilities.xlsx")
    tech_skills = pd.read_excel(f"{data_dir}/Technology Skills.xlsx")
    tools = pd.read_excel(f"{data_dir}/Tools Used.xlsx")
    return alt, occ, task, skills, knowledge, abilities, tech_skills, tools


def build_label_map(occ: pd.DataFrame) -> dict:
    """构建 O*NET Code -> label index 的映射"""
    codes = sorted(occ['O*NET-SOC Code'].unique())
    return {code: idx for idx, code in enumerate(codes)}


def build_occupation_corpus(occ: pd.DataFrame, task: pd.DataFrame) -> dict:
    """
    为每个职业构建丰富的文本表示（用于推理时的标准向量库）
    格式: {code: {"title": ..., "description": ..., "tasks": ...}}
    """
    corpus = {}

    # 聚合每个职业的任务描述
    tasks_grouped = task.groupby('O*NET-SOC Code')['Task'].apply(
        lambda x: ' '.join(x.tolist())
    ).to_dict()

    for _, row in occ.iterrows():
        code = row['O*NET-SOC Code']
        title = row['Title']
        description = str(row['Description']) if pd.notna(row['Description']) else ''
        tasks_text = tasks_grouped.get(code, '')

        corpus[code] = {
            'title': title,
            'description': description,
            'tasks': tasks_text,
            # 完整的标准文本表示（推理时用）
            'full_text': f"{title}. {description}"
        }

    return corpus


def build_training_pairs(alt: pd.DataFrame, occ: pd.DataFrame,
                          task: pd.DataFrame, skills: pd.DataFrame,
                          knowledge: pd.DataFrame, abilities: pd.DataFrame,
                          tech_skills: pd.DataFrame, tools: pd.DataFrame,
                          min_alt_count: int = 5):
    """
    构建训练样本对 (anchor, positive_label_code)

    数据来源：
    1. Alternate Titles: 别名 → 标准职业
    2. Task Statements: 任务描述 → 标准职业（增强语义）
    3. Occupation Description: 职业描述 → 标准职业
    4. Skills: 职业技能 → 标准职业
    5. Knowledge: 职业知识 → 标准职业
    6. Abilities: 职业能力 → 标准职业
    7. Technology Skills: 技术技能 → 标准职业
    8. Tools Used: 使用工具 → 标准职业
    """
    label_map = build_label_map(occ)
    corpus = build_occupation_corpus(occ, task)

    # 过滤样本太少的类别
    alt_counts = alt.groupby('O*NET-SOC Code').size()
    valid_codes = set(alt_counts[alt_counts >= min_alt_count].index)
    valid_codes = valid_codes.intersection(set(label_map.keys()))

    print(f"有效类别数: {len(valid_codes)} / {len(label_map)}")

    training_samples = []

    # ── 来源1：Alternate Titles ──────────────────────────────────────
    alt_filtered = alt[alt['O*NET-SOC Code'].isin(valid_codes)]
    for _, row in alt_filtered.iterrows():
        code = row['O*NET-SOC Code']
        alt_title = str(row['Alternate Title'])
        if pd.isna(row['Alternate Title']):
            continue
        training_samples.append({
            'text': alt_title,
            'label_code': code,
            'label_idx': label_map[code],
            'source': 'alt_title'
        })

    # ── 来源2：标准 Title 本身 ───────────────────────────────────────
    for _, row in occ[occ['O*NET-SOC Code'].isin(valid_codes)].iterrows():
        code = row['O*NET-SOC Code']
        training_samples.append({
            'text': row['Title'],
            'label_code': code,
            'label_idx': label_map[code],
            'source': 'std_title'
        })

    # ── 来源3：Task Statements（每条任务 → 对应职业）────────────────
    task_filtered = task[task['O*NET-SOC Code'].isin(valid_codes)]
    for _, row in task_filtered.iterrows():
        code = row['O*NET-SOC Code']
        task_text = str(row['Task'])
        if len(task_text) < 20:
            continue
        training_samples.append({
            'text': task_text,
            'label_code': code,
            'label_idx': label_map[code],
            'source': 'task'
        })

    # ── 来源4：Title + Description 组合（模拟真实输入）──────────────
    for _, row in occ[occ['O*NET-SOC Code'].isin(valid_codes)].iterrows():
        code = row['O*NET-SOC Code']
        desc = str(row['Description']) if pd.notna(row['Description']) else ''
        if len(desc) > 50:
            # 截断描述，模拟岗位简介场景
            short_desc = desc[:300]
            combined = f"{row['Title']}. {short_desc}"
            training_samples.append({
                'text': combined,
                'label_code': code,
                'label_idx': label_map[code],
                'source': 'title_desc'
            })

    # ── 来源5：Skills（职业技能）────────────────────────────────────
    skills_filtered = skills[skills['O*NET-SOC Code'].isin(valid_codes)]
    for _, row in skills_filtered.iterrows():
        code = row['O*NET-SOC Code']
        skill_name = str(row['Element Name'])
        if pd.isna(row['Element Name']) or len(skill_name) < 3:
            continue
        training_samples.append({
            'text': f"Skill: {skill_name}",
            'label_code': code,
            'label_idx': label_map[code],
            'source': 'skill'
        })

    # ── 来源6：Knowledge（职业知识）─────────────────────────────────
    knowledge_filtered = knowledge[knowledge['O*NET-SOC Code'].isin(valid_codes)]
    for _, row in knowledge_filtered.iterrows():
        code = row['O*NET-SOC Code']
        knowledge_name = str(row['Element Name'])
        if pd.isna(row['Element Name']) or len(knowledge_name) < 3:
            continue
        training_samples.append({
            'text': f"Knowledge: {knowledge_name}",
            'label_code': code,
            'label_idx': label_map[code],
            'source': 'knowledge'
        })

    # ── 来源7：Abilities（职业能力）─────────────────────────────────
    abilities_filtered = abilities[abilities['O*NET-SOC Code'].isin(valid_codes)]
    for _, row in abilities_filtered.iterrows():
        code = row['O*NET-SOC Code']
        ability_name = str(row['Element Name'])
        if pd.isna(row['Element Name']) or len(ability_name) < 3:
            continue
        training_samples.append({
            'text': f"Ability: {ability_name}",
            'label_code': code,
            'label_idx': label_map[code],
            'source': 'ability'
        })

    # ── 来源8：Technology Skills（技术技能）─────────────────────────
    tech_filtered = tech_skills[tech_skills['O*NET-SOC Code'].isin(valid_codes)]
    for _, row in tech_filtered.iterrows():
        code = row['O*NET-SOC Code']
        tech_example = str(row['Example'])
        if pd.isna(row['Example']) or len(tech_example) < 3:
            continue
        training_samples.append({
            'text': f"Technology: {tech_example}",
            'label_code': code,
            'label_idx': label_map[code],
            'source': 'technology'
        })

    # ── 来源9：Tools Used（使用工具）────────────────────────────────
    tools_filtered = tools[tools['O*NET-SOC Code'].isin(valid_codes)]
    for _, row in tools_filtered.iterrows():
        code = row['O*NET-SOC Code']
        tool_name = str(row['Example'])
        if pd.isna(row['Example']) or len(tool_name) < 3:
            continue
        training_samples.append({
            'text': f"Tool: {tool_name}",
            'label_code': code,
            'label_idx': label_map[code],
            'source': 'tool'
        })

    print(f"总训练样本数: {len(training_samples)}")
    print("各来源分布:")
    df_samples = pd.DataFrame(training_samples)
    print(df_samples['source'].value_counts().to_string())

    return training_samples, label_map, corpus


def split_data(samples: list, val_ratio: float = 0.1):
    """按职业类别分层划分训练/验证集，确保每个类别在验证集中都有样本"""
    from collections import defaultdict

    by_label = defaultdict(list)
    for s in samples:
        by_label[s['label_code']].append(s)

    train, val = [], []
    for code, items in by_label.items():
        random.shuffle(items)
        # 每个类别至少保留1条在验证集
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    random.shuffle(train)
    random.shuffle(val)
    print(f"训练集: {len(train)} 条 | 验证集: {len(val)} 条")
    return train, val


def save_data(train, val, corpus, label_map, output_dir: str = "processed_data"):
    Path(output_dir).mkdir(exist_ok=True)

    with open(f"{output_dir}/train.json", 'w') as f:
        json.dump(train, f, ensure_ascii=False, indent=2)

    with open(f"{output_dir}/val.json", 'w') as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    with open(f"{output_dir}/corpus.json", 'w') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    with open(f"{output_dir}/label_map.json", 'w') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"数据已保存至 {output_dir}/")


if __name__ == "__main__":
    print("加载数据...")
    alt, occ, task, skills, knowledge, abilities, tech_skills, tools = load_data(".")

    print("构建训练样本...")
    samples, label_map, corpus = build_training_pairs(alt, occ, task, skills, knowledge, abilities, tech_skills, tools)

    print("划分训练/验证集...")
    train, val = split_data(samples)

    print("保存数据...")
    save_data(train, val, corpus, label_map)
