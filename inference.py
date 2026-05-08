"""
推理模块 - 给定岗位名称+简介，返回 O*NET 职业分类

使用方式：
    predictor = JobClassifier("output/job_classifier")
    result = predictor.predict("Software Engineer", "Design and develop scalable backend systems...")
    print(result)
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


class JobClassifier:
    def __init__(self, model_dir: str, data_dir: str = "processed_data",
                 label_mode: str = "title"):
        """
        model_dir: 训练好的模型路径
        data_dir:  处理后的数据目录（含 corpus.json, label_map.json）
        label_mode: 标签侧文本模式，需与训练时一致
        """
        print(f"加载模型: {model_dir}")
        self.model = SentenceTransformer(model_dir)

        with open(f"{data_dir}/corpus.json") as f:
            self.corpus = json.load(f)
        with open(f"{data_dir}/label_map.json") as f:
            self.label_map = json.load(f)

        self.label_mode = label_mode
        self._build_index()

    def _build_index(self):
        """预计算所有职业类别的向量索引（只需做一次）"""
        print("构建职业向量索引...")
        self.codes = list(self.corpus.keys())
        label_texts = []

        for code in self.codes:
            info = self.corpus[code]
            if self.label_mode == "title":
                text = info['title']
            elif self.label_mode == "title_desc":
                desc = info['description'][:200] if info['description'] else ''
                text = f"{info['title']}. {desc}"
            else:
                text = info.get('full_text', info['title'])
            label_texts.append(text)

        self.label_embeddings = self.model.encode(
            label_texts,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        print(f"索引构建完成，共 {len(self.codes)} 个职业类别")

    def predict(self, job_title: str, job_description: str = "",
                top_k: int = 5) -> list[dict]:
        """
        预测岗位分类

        Args:
            job_title: 岗位名称，例如 "Software Engineer"
            job_description: 岗位简介（可选但推荐提供）
            top_k: 返回前 K 个候选

        Returns:
            [
              {"rank": 1, "code": "15-1252.00", "title": "Software Developers",
               "score": 0.92},
              ...
            ]
        """
        # 拼接输入文本
        if job_description:
            query = f"{job_title}. {job_description[:300]}"
        else:
            query = job_title

        # 编码 query
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # 计算相似度
        scores = query_embedding @ self.label_embeddings.T
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            code = self.codes[idx]
            results.append({
                "rank": rank,
                "code": code,
                "title": self.corpus[code]['title'],
                "score": float(scores[idx]),
                "description": self.corpus[code]['description'][:150] + "..."
            })

        return results

    def predict_batch(self, jobs: list[dict], top_k: int = 3) -> list[list[dict]]:
        """
        批量预测，jobs 格式: [{"title": ..., "description": ...}, ...]
        """
        queries = []
        for job in jobs:
            title = job.get('title', '')
            desc = job.get('description', '')
            if desc:
                queries.append(f"{title}. {desc[:300]}")
            else:
                queries.append(title)

        query_embeddings = self.model.encode(
            queries,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(queries) > 100
        )

        scores_matrix = query_embeddings @ self.label_embeddings.T
        all_results = []

        for scores in scores_matrix:
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for rank, idx in enumerate(top_indices, 1):
                code = self.codes[idx]
                results.append({
                    "rank": rank,
                    "code": code,
                    "title": self.corpus[code]['title'],
                    "score": float(scores[idx])
                })
            all_results.append(results)

        return all_results


# ── 使用示例 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = JobClassifier(
        model_dir="output/job_classifier",
        data_dir="processed_data",
        label_mode="title_desc"  # 包含职业描述
    )

    # 真实招聘案例测试
    test_cases = [
        {
            "title": "Senior Software Engineer",
            "description": "We are looking for a Senior Software Engineer to join our team. You will design, develop, and maintain scalable web applications using Java, Spring Boot, and microservices architecture. Responsibilities include code review, mentoring junior developers, and collaborating with product teams. Requirements: 5+ years of experience, strong knowledge of RESTful APIs, database design (MySQL/PostgreSQL), and cloud platforms (AWS/Azure)."
        },
        {
            "title": "Registered Nurse - ICU",
            "description": "Seeking an experienced RN for our Intensive Care Unit. Provide direct patient care to critically ill patients, administer medications, monitor vital signs, operate medical equipment (ventilators, infusion pumps), and collaborate with physicians and healthcare team. Must have BSN, current RN license, BLS/ACLS certification, and minimum 2 years ICU experience."
        },
        {
            "title": "Marketing Manager",
            "description": "Lead our marketing team to develop and execute comprehensive marketing strategies. Manage digital marketing campaigns (SEO/SEM, social media, email marketing), analyze market trends, coordinate with creative teams, manage budget of $500K+, and report ROI metrics. Bachelor's degree in Marketing, 7+ years experience, proficiency in Google Analytics, HubSpot, and Adobe Creative Suite required."
        },
        {
            "title": "Electrician",
            "description": "Install, maintain, and repair electrical systems in residential and commercial buildings. Read blueprints and technical diagrams, troubleshoot electrical issues, ensure compliance with NEC codes, and conduct safety inspections. Must have valid electrician license, 3+ years experience, knowledge of conduit bending, wiring methods, and ability to work at heights."
        },
        {
            "title": "Data Scientist",
            "description": "Join our analytics team to build predictive models and extract insights from large datasets. Develop machine learning algorithms using Python (scikit-learn, TensorFlow, PyTorch), perform statistical analysis, create data visualizations, and present findings to stakeholders. PhD or Master's in Computer Science/Statistics, experience with SQL, Spark, and cloud-based ML platforms preferred."
        },
        {
            "title": "Human Resources Coordinator",
            "description": "Support HR operations including recruitment coordination, onboarding new employees, maintaining employee records, processing payroll, coordinating training programs, and ensuring compliance with labor laws. Handle employee inquiries, assist with benefits administration, and organize company events. Bachelor's degree in HR or related field, 2+ years HR experience, knowledge of HRIS systems required."
        },
        {
            "title": "Construction Project Manager",
            "description": "Oversee construction projects from planning to completion. Manage subcontractors, coordinate schedules, ensure quality control, maintain safety standards (OSHA compliance), manage project budgets ($1M-$10M), and communicate with clients and architects. Bachelor's in Construction Management or Civil Engineering, PMP certification preferred, 5+ years experience in commercial construction."
        },
        {
            "title": "Customer Service Representative",
            "description": "Handle customer inquiries via phone, email, and chat. Resolve complaints, process orders and returns, provide product information, maintain customer records in CRM system, and achieve satisfaction targets. High school diploma required, excellent communication skills, ability to multitask, experience with Zendesk or Salesforce preferred. Bilingual (English/Spanish) a plus."
        },
        {
            "title": "Financial Analyst",
            "description": "Conduct financial modeling, variance analysis, and forecasting to support business decisions. Prepare monthly/quarterly reports, analyze revenue trends, evaluate investment opportunities, and present recommendations to senior management. Bachelor's in Finance/Accounting, CFA Level 1 preferred, advanced Excel skills, experience with SAP or Oracle Financials, 3+ years in corporate finance."
        },
        {
            "title": "Mechanical Engineer",
            "description": "Design and develop mechanical systems for HVAC applications. Perform thermal analysis, create CAD drawings (SolidWorks/AutoCAD), conduct prototype testing, collaborate with manufacturing teams, and ensure compliance with ASME standards. Bachelor's in Mechanical Engineering, PE license preferred, 4+ years experience in building systems design, knowledge of energy efficiency regulations."
        },
    ]

    print("\n" + "=" * 60)
    print("岗位分类预测结果")
    print("=" * 60)

    for job in test_cases:
        results = predictor.predict(job['title'], job['description'], top_k=3)
        print(f"\n输入: {job['title']}")
        print(f"简介: {job['description'][:80]}...")
        print("预测结果:")
        for r in results:
            print(f"  #{r['rank']} [{r['score']:.4f}] {r['code']} - {r['title']}")
