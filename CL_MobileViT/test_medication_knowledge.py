import unittest
from pathlib import Path

from medication_knowledge import MedicationKnowledgeBase


PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "data" / "knowledge" / "China_pesticides_wheat.csv"


class MedicationKnowledgeBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.knowledge_base = MedicationKnowledgeBase(
            CSV_PATH,
            confidence_threshold=0.80,
        )

    def test_healthy_class_never_returns_pesticides(self):
        report = self.knowledge_base.lookup("Healthy", confidence=0.99)
        self.assertEqual(report["status"], "healthy_no_pesticide")
        self.assertEqual(report["candidates"], [])

    def test_low_confidence_is_rejected(self):
        report = self.knowledge_base.lookup("Mildew", confidence=0.60)
        self.assertEqual(report["status"], "low_confidence_manual_review")
        self.assertEqual(report["candidates"], [])

    def test_unsupported_disease_is_rejected(self):
        report = self.knowledge_base.lookup("Leaf Blight", confidence=0.99)
        self.assertEqual(report["status"], "unsupported_disease_manual_review")
        self.assertEqual(report["candidates"], [])

    def test_septoria_is_rejected_without_exact_registration_target(self):
        report = self.knowledge_base.lookup("Septoria", confidence=0.99)
        self.assertEqual(report["status"], "unsupported_disease_manual_review")
        self.assertEqual(report["candidates"], [])

    def test_powdery_mildew_returns_exact_traceable_records(self):
        report = self.knowledge_base.lookup("Mildew", confidence=0.99, top_k=3)
        self.assertEqual(report["status"], "candidate_records_found")
        self.assertEqual(len(report["candidates"]), 3)
        for candidate in report["candidates"]:
            self.assertEqual(candidate["target"], "白粉病")
            self.assertEqual(candidate["match_level"], "exact")
            self.assertTrue(candidate["registration_number"].startswith("PD"))
            self.assertIn("杀菌剂", candidate["pesticide_category"])
            self.assertTrue(candidate["dosage"])
            self.assertTrue(candidate["application_method"])

    def test_stripe_rust_uses_only_exact_target(self):
        report = self.knowledge_base.lookup("Yellow Rust", confidence=0.99, top_k=3)
        self.assertEqual(report["status"], "candidate_records_found")
        self.assertTrue(report["candidates"])
        self.assertTrue(
            all(candidate["target"] == "条锈病" for candidate in report["candidates"])
        )

    def test_leaf_rust_does_not_use_generic_rust_by_default(self):
        report = self.knowledge_base.lookup("Brown Rust", confidence=0.99, top_k=20)
        self.assertEqual(report["status"], "candidate_records_found")
        self.assertTrue(report["candidates"])
        self.assertTrue(
            all(candidate["target"] == "叶锈病" for candidate in report["candidates"])
        )


if __name__ == "__main__":
    unittest.main()
