import unittest

from langchain_core.messages import AIMessage

from home_made_rag.expansion import (
    ExpansionError,
    ExpansionResult,
    HypotheticalGarment,
    ObservedGarment,
    QueryExpander,
)


class FakeStructuredLLM:
    def __init__(self, result):
        self.result = result

    def invoke(self, messages):
        return self.result


def expansion_result(complementary):
    return ExpansionResult(
        observed=ObservedGarment(
            category="t-shirt",
            color="white",
            style="casual",
            confidence=0.9,
            anchor_description_en="",
        ),
        complementary=complementary,
        stylist_summary_es="",
    )


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.expander = QueryExpander(model_name="gpt-4o-mini")

    def test_expand_validates_and_fills_defaults(self):
        self.expander.structured_llm = FakeStructuredLLM({
            "raw": AIMessage(
                content="",
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                },
            ),
            "parsed": expansion_result([
                HypotheticalGarment(
                    category="jacket",
                    description_en="a black leather jacket",
                ),
                HypotheticalGarment(
                    category="pants",
                    description_en="blue straight-leg jeans",
                ),
            ]),
            "parsing_error": None,
        })

        result = self.expander.expand(text="white t-shirt")

        self.assertEqual(result.observed.anchor_description_en, "white casual t-shirt")
        self.assertTrue(result.stylist_summary_es)
        self.assertEqual(len(result.complementary), 2)

    def test_empty_complements_raise(self):
        self.expander.structured_llm = FakeStructuredLLM({
            "raw": AIMessage(content=""),
            "parsed": expansion_result([]),
            "parsing_error": None,
        })

        with self.assertRaises(ExpansionError):
            self.expander.expand(text="white t-shirt")

    def test_parsing_error_raises(self):
        self.expander.structured_llm = FakeStructuredLLM({
            "raw": AIMessage(content=""),
            "parsed": None,
            "parsing_error": ValueError("invalid json"),
        })

        with self.assertRaises(ExpansionError):
            self.expander.expand(text="white t-shirt")


if __name__ == "__main__":
    unittest.main()
