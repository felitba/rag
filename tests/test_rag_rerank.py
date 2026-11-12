import unittest
from pathlib import Path
from types import SimpleNamespace

from home_made_rag.rag import RAG


class FakeCatalog:
    def __init__(self):
        self.outfits = {
            "outfit_1": ["A", "B", "C"],
            "outfit_2": ["A", "D"],
            "outfit_3": ["D", "B"],
        }
        self.items = {
            item_id: {
                "category": "test",
                "title": "",
                "price": None,
                "image_path": f"data/images/{item_id}.jpg",
            }
            for item_id in {"A", "B", "C", "D"}
        }

    def image_path(self, item_id):
        return Path(self.items[item_id]["image_path"])

    def image_url(self, item_id):
        return f"/images/{item_id}.jpg"


class FakeChroma:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def get_embeddings(self, ids):
        return {
            item_id: self.embeddings[item_id]
            for item_id in ids
            if item_id in self.embeddings
        }


class RerankTests(unittest.TestCase):
    def setUp(self):
        self.embeddings = {
            "A": [1.0, 0.0, 0.0],
            "B": [0.0, 1.0, 0.0],
            "C": [0.0, 0.0, 1.0],
            "D": [1.0, 0.0, 0.0],
        }
        self.rag = RAG(
            embedding=None,
            chroma=FakeChroma(self.embeddings),
            catalog=FakeCatalog(),
        )
        self.query = [1.0, 0.0, 0.0]
        self.hypotheses = [
            (SimpleNamespace(category="jacket", description_en="jacket"), [0.0, 1.0, 0.0]),
            (SimpleNamespace(category="pants", description_en="pants"), [0.0, 0.0, 1.0]),
        ]
        self.candidates = [
            {"outfit_id": outfit_id, "visual_matches": []}
            for outfit_id in ("outfit_1", "outfit_2", "outfit_3")
        ]

    def test_rerank_prefers_outfit_with_both_complements(self):
        outfits, coverage = self.rag._rerank_outfits(
            self.candidates,
            self.query,
            self.hypotheses,
            max_outfits=2,
        )

        self.assertEqual([outfit["outfit_id"] for outfit in outfits], ["outfit_1", "outfit_3"])
        self.assertEqual(len(coverage), 2)
        self.assertTrue(all(item["covered"] for item in coverage))

    def test_coverage_requires_distinct_items(self):
        embeddings = {
            "E": [0.0, 0.7071067811865476, 0.7071067811865476],
        }
        chroma = FakeChroma(embeddings)
        rag = RAG(embedding=None, chroma=chroma, catalog=FakeCatalog())
        rag.catalog.outfits = {"outfit_e": ["E"]}
        rag.catalog.items = {
            "E": {
                "category": "test",
                "title": "",
                "price": None,
                "image_path": "data/images/E.jpg",
            }
        }

        coverage = rag._hypothesis_coverage(
            [{"outfit_id": "outfit_e"}],
            self.hypotheses,
            embeddings,
        )

        self.assertEqual(sum(1 for item in coverage if item["covered"]), 1)

    def test_explanation_requires_minimum_coverage(self):
        expansion = SimpleNamespace(stylist_summary_es="Resumen del estilista")

        covered = [
            {"covered": True},
            {"covered": True},
        ]
        partial = [
            {"covered": True},
            {"covered": False},
        ]

        self.assertEqual(
            self.rag._select_explanation(expansion, covered),
            ("Resumen del estilista", "expansion"),
        )
        self.assertEqual(
            self.rag._select_explanation(expansion, partial)[1],
            "generic",
        )

    def test_cosine_and_min_max(self):
        self.assertAlmostEqual(self.rag._cosine([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertAlmostEqual(self.rag._cosine([1.0, 0.0], [2.0, 0.0]), 1.0)
        self.assertEqual(self.rag._min_max([5.0, 5.0]), [0.5, 0.5])
        self.assertEqual(self.rag._min_max([0.0, 10.0]), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
