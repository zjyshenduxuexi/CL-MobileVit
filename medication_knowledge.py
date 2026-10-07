"""Safety-gated pesticide knowledge retrieval for CL-MobileViT predictions.

The module deliberately stays outside the neural network. It reads only the
small subset of pesticide rows that can be mapped exactly to the model's
disease labels and returns traceable candidate registration records.

It does not verify that a registration is currently valid because the source
CSV does not contain a registration expiry/status field. The returned records
must therefore be checked against the current official label before field use.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


MODEL_DISEASE_RULES = {
    "Healthy": {
        "disease_cn": "健康小麦叶片",
        "targets": (),
        "status": "healthy",
        "message": "模型判断为健康叶片，不进入农药候选检索。",
    },
    "Brown Rust": {
        "disease_cn": "小麦叶锈病",
        "targets": ("叶锈病",),
        "status": "supported",
        "message": "仅检索防治对象精确标记为‘叶锈病’的登记记录。",
    },
    "Yellow Rust": {
        "disease_cn": "小麦条锈病",
        "targets": ("条锈病",),
        "status": "supported",
        "message": "仅检索防治对象精确标记为‘条锈病’的登记记录。",
    },
    "Mildew": {
        "disease_cn": "小麦白粉病",
        "targets": ("白粉病",),
        "status": "supported",
        "message": "仅检索防治对象精确标记为‘白粉病’的登记记录。",
    },
    "Leaf Blight": {
        "disease_cn": "小麦叶枯病（待标准化）",
        "targets": (),
        "status": "unsupported",
        "message": (
            "当前CSV没有与Leaf Blight严格对应的‘叶枯病’登记对象；"
            "纹枯病和叶斑病不会被自动映射，请由植保专家确认标准病名。"
        ),
    },
    "Septoria": {
        "disease_cn": "小麦Septoria病害（待标准化）",
        "targets": (),
        "status": "unsupported",
        "message": (
            "当前CSV没有与Septoria严格对应的‘斑枯病/叶斑枯病’登记对象；"
            "普通叶斑病记录不会被自动映射。"
        ),
    },
}


REQUIRED_COLUMNS = {
    "登记证号",
    "农药名称",
    "农药类别",
    "登记证持有人",
    "剂型",
    "毒性",
    "总有效成分含量",
    "有效成分",
    "有效成分英文名",
    "作物/场所",
    "防治对象",
    "用药量（制剂量/亩）",
    "施用方法",
    "批准日期",
    "最新批准日期",
    "使用技术要求",
    "注意事项",
    "中毒急救措施",
    "储存和运输方法",
    "备注",
    "使用方法序号",
}


SAFETY_NOTICES = (
    "候选记录来自本地CSV，CSV缺少登记有效截止日期和实时登记状态。",
    "输出仅用于科研原型和信息检索，不能替代当前官方标签或植保专家意见。",
    "实际使用前必须核验登记证号、适用作物、防治对象、剂量、施用方法、安全间隔期和当地规定。",
    "低置信度、健康类别、无精确病名映射或无合格记录时，系统拒绝自动推荐。",
)


@dataclass(frozen=True)
class PesticideCandidate:
    registration_number: str
    pesticide_name: str
    pesticide_category: str
    registration_holder: str
    formulation: str
    toxicity: str
    total_active_ingredient_content: str
    active_ingredient: str
    active_ingredient_english: str
    crop_or_site: str
    target: str
    dosage: str
    application_method: str
    approval_date: str
    latest_approval_date: str
    technical_requirements: str
    precautions: str
    first_aid: str
    storage_and_transport: str
    remarks: str
    use_sequence: str
    match_level: str
    registration_status: str = "not_verified"


class MedicationKnowledgeBase:
    """Load and query a safety-filtered subset of the wheat pesticide CSV."""

    def __init__(
        self,
        csv_path: Path | str,
        confidence_threshold: float = 0.80,
        allow_generic_rust: bool = False,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        self.csv_path = Path(csv_path)
        self.confidence_threshold = confidence_threshold
        self.allow_generic_rust = allow_generic_rust
        self._index: Dict[str, List[PesticideCandidate]] = {}
        self._load_index()

    @staticmethod
    def _clean(value: Optional[str]) -> str:
        return (value or "").strip()

    @staticmethod
    def _is_wheat_crop(value: str) -> bool:
        value = value.strip()
        if any(other in value for other in ("大麦", "燕麦", "荞麦", "青稞")):
            return False
        return "小麦" in value or "麦田" in value

    @staticmethod
    def _toxicity_rank(value: str) -> int:
        """Conservative order: lower values are preferred during display ranking."""
        if "剧毒" in value:
            return 5
        if "高毒" in value:
            return 4
        if "中等毒" in value:
            return 3
        if "低毒" in value:
            return 2
        if "微毒" in value:
            return 1
        return 6

    @staticmethod
    def _date_rank(value: str) -> int:
        """Return a platform-independent sortable date value.

        ``datetime.min.timestamp()`` raises ``OSError`` on Windows, so empty or
        malformed approval dates are represented by zero rather than by a
        pre-1970 Unix timestamp.
        """
        try:
            return datetime.strptime(value, "%Y/%m/%d").toordinal()
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _candidate_from_row(cls, row: Dict[str, str], match_level: str) -> PesticideCandidate:
        return PesticideCandidate(
            registration_number=cls._clean(row.get("登记证号")),
            pesticide_name=cls._clean(row.get("农药名称")),
            pesticide_category=cls._clean(row.get("农药类别")),
            registration_holder=cls._clean(row.get("登记证持有人")),
            formulation=cls._clean(row.get("剂型")),
            toxicity=cls._clean(row.get("毒性")),
            total_active_ingredient_content=cls._clean(row.get("总有效成分含量")),
            active_ingredient=cls._clean(row.get("有效成分")),
            active_ingredient_english=cls._clean(row.get("有效成分英文名")),
            crop_or_site=cls._clean(row.get("作物/场所")),
            target=cls._clean(row.get("防治对象")),
            dosage=cls._clean(row.get("用药量（制剂量/亩）")),
            application_method=cls._clean(row.get("施用方法")),
            approval_date=cls._clean(row.get("批准日期")),
            latest_approval_date=cls._clean(row.get("最新批准日期")),
            technical_requirements=cls._clean(row.get("使用技术要求")),
            precautions=cls._clean(row.get("注意事项")),
            first_aid=cls._clean(row.get("中毒急救措施")),
            storage_and_transport=cls._clean(row.get("储存和运输方法")),
            remarks=cls._clean(row.get("备注")),
            use_sequence=cls._clean(row.get("使用方法序号")),
            match_level=match_level,
        )

    @staticmethod
    def _row_is_safe_candidate(row: Dict[str, str]) -> bool:
        registration_number = (row.get("登记证号") or "").strip()
        category = (row.get("农药类别") or "").strip()
        formulation = (row.get("剂型") or "").strip()
        crop = (row.get("作物/场所") or "").strip()
        dosage = (row.get("用药量（制剂量/亩）") or "").strip()
        method = (row.get("施用方法") or "").strip()

        return all(
            (
                registration_number.startswith("PD"),
                "杀菌剂" in category,
                formulation not in {"原药", "母药"},
                MedicationKnowledgeBase._is_wheat_crop(crop),
                bool(dosage),
                bool(method),
            )
        )

    def _load_index(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Pesticide CSV not found: {self.csv_path}")

        supported_targets = {
            target
            for rule in MODEL_DISEASE_RULES.values()
            for target in rule["targets"]
        }
        if self.allow_generic_rust:
            supported_targets.add("锈病")

        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or ())
            missing_columns = sorted(REQUIRED_COLUMNS - fieldnames)
            if missing_columns:
                raise ValueError(
                    "Pesticide CSV is missing required columns: "
                    + ", ".join(missing_columns)
                )

            for row in reader:
                target = self._clean(row.get("防治对象"))
                if target not in supported_targets:
                    continue
                if not self._row_is_safe_candidate(row):
                    continue

                match_level = "generic_requires_expert_review" if target == "锈病" else "exact"
                candidate = self._candidate_from_row(row, match_level=match_level)
                self._index.setdefault(target, []).append(candidate)

        for target, candidates in self._index.items():
            candidates.sort(key=self._sort_key)

    def _sort_key(self, candidate: PesticideCandidate) -> Tuple[int, int, float, str, str]:
        match_rank = 0 if candidate.match_level == "exact" else 1
        approval = candidate.latest_approval_date or candidate.approval_date
        approval_rank = self._date_rank(approval)
        return (
            match_rank,
            self._toxicity_rank(candidate.toxicity),
            -approval_rank,
            candidate.registration_number,
            candidate.use_sequence,
        )

    def lookup(self, predicted_class: str, confidence: float, top_k: int = 3) -> Dict[str, object]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        rule = MODEL_DISEASE_RULES.get(predicted_class)
        base_report: Dict[str, object] = {
            "predicted_class": predicted_class,
            "confidence": round(float(confidence), 6),
            "confidence_threshold": self.confidence_threshold,
            "data_source": str(self.csv_path.resolve()),
            "data_file_modified_at": datetime.fromtimestamp(
                self.csv_path.stat().st_mtime
            ).isoformat(timespec="seconds"),
            "safety_notices": list(SAFETY_NOTICES),
            "candidates": [],
        }

        if rule is None:
            base_report.update(
                {
                    "status": "unknown_model_class",
                    "disease_cn": "未知类别",
                    "message": "模型类别没有配置知识图谱映射，系统拒绝推荐。",
                }
            )
            return base_report

        base_report["disease_cn"] = rule["disease_cn"]
        base_report["mapping_message"] = rule["message"]

        if predicted_class == "Healthy":
            base_report.update(
                {
                    "status": "healthy_no_pesticide",
                    "message": "健康类别不推荐使用农药，建议继续监测并保持常规田间管理。",
                }
            )
            return base_report

        if confidence < self.confidence_threshold:
            base_report.update(
                {
                    "status": "low_confidence_manual_review",
                    "message": "预测置信度低于安全阈值，系统拒绝自动给出用药候选。",
                }
            )
            return base_report

        if rule["status"] != "supported":
            base_report.update(
                {
                    "status": "unsupported_disease_manual_review",
                    "message": rule["message"],
                }
            )
            return base_report

        targets: Sequence[str] = rule["targets"]
        candidate_pool: List[PesticideCandidate] = []
        for target in targets:
            candidate_pool.extend(self._index.get(target, ()))

        if predicted_class == "Brown Rust" and self.allow_generic_rust:
            candidate_pool.extend(self._index.get("锈病", ()))

        candidate_pool.sort(key=self._sort_key)
        candidates = candidate_pool[:top_k]

        if not candidates:
            base_report.update(
                {
                    "status": "no_safe_candidate_manual_review",
                    "message": "未找到同时满足精确病名、小麦用途和基础安全字段要求的记录。",
                }
            )
            return base_report

        has_generic_match = any(
            candidate.match_level != "exact" for candidate in candidates
        )
        base_report.update(
            {
                "status": (
                    "candidate_records_with_generic_match"
                    if has_generic_match
                    else "candidate_records_found"
                ),
                "message": (
                    "已返回可追溯的候选登记记录；请在实际使用前核验当前官方标签。"
                ),
                "matched_targets": sorted({candidate.target for candidate in candidates}),
                "candidate_count": len(candidates),
                "candidates": [asdict(candidate) for candidate in candidates],
            }
        )
        return base_report


def format_text_report(report: Dict[str, object]) -> str:
    lines = [
        "CL-MobileViT 安全用药知识推理报告",
        "=" * 42,
        f"预测类别: {report.get('predicted_class', '')}",
        f"中文病名: {report.get('disease_cn', '')}",
        f"预测置信度: {float(report.get('confidence', 0.0)):.4f}",
        f"安全阈值: {float(report.get('confidence_threshold', 0.0)):.4f}",
        f"决策状态: {report.get('status', '')}",
        f"决策说明: {report.get('message', '')}",
    ]

    mapping_message = report.get("mapping_message")
    if mapping_message:
        lines.append(f"病名映射: {mapping_message}")

    candidates = report.get("candidates") or []
    if candidates:
        lines.extend(["", "候选登记记录（非自动处方）:"])
        for index, candidate in enumerate(candidates, start=1):
            lines.extend(
                [
                    f"  [{index}] 登记证号: {candidate['registration_number']}",
                    f"      农药名称: {candidate['pesticide_name']}",
                    f"      有效成分: {candidate['active_ingredient']}",
                    f"      剂型/毒性: {candidate['formulation']} / {candidate['toxicity']}",
                    f"      作物/对象: {candidate['crop_or_site']} / {candidate['target']}",
                    f"      标签用量: {candidate['dosage']}",
                    f"      施用方法: {candidate['application_method']}",
                    f"      批准日期: {candidate['latest_approval_date'] or candidate['approval_date'] or '未提供'}",
                    f"      匹配级别: {candidate['match_level']}",
                ]
            )

    lines.extend(["", "安全声明:"])
    for notice in report.get("safety_notices") or []:
        lines.append(f"  - {notice}")
    return "\n".join(lines) + "\n"


def save_report(report: Dict[str, object], output_dir: Path | str) -> Tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "medication_recommendation.json"
    text_path = output_path / "medication_recommendation.txt"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    text_path.write_text(format_text_report(report), encoding="utf-8")
    return json_path, text_path


def build_parser() -> argparse.ArgumentParser:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Query the wheat pesticide knowledge layer")
    parser.add_argument("--disease", required=True, choices=sorted(MODEL_DISEASE_RULES))
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--csv",
        type=Path,
        default=project_dir / "data" / "knowledge" / "China_pesticides_wheat.csv",
    )
    parser.add_argument("--allow-generic-rust", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    knowledge_base = MedicationKnowledgeBase(
        csv_path=args.csv,
        confidence_threshold=args.confidence_threshold,
        allow_generic_rust=args.allow_generic_rust,
    )
    report = knowledge_base.lookup(
        predicted_class=args.disease,
        confidence=args.confidence,
        top_k=args.top_k,
    )
    json_path, text_path = save_report(report, args.output_dir)
    print(format_text_report(report))
    print(f"JSON report saved to: {json_path}")
    print(f"Text report saved to: {text_path}")


if __name__ == "__main__":
    main()
