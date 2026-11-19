import json
import unittest

from prompting import PromptError, parse_prompt


class ParsePromptTests(unittest.TestCase):
    def test_wrapped_json_keeps_prompt_contents(self):
        base = 'snow,\nred sign reading "{北}",'
        plan = {"base": base, "characters": [{"prompt": "white fox", "x": "0.2", "y": "0.5"}]}
        reply = '<think>草稿 {"base":"wrong"}</think>提示词如下：\n```json\n' + json.dumps(plan) + '\n```\n完成。'
        self.assertEqual(
            parse_prompt(reply),
            {"base": base, "negative": "", "characters": [{"prompt": "white fox", "x": 0.2, "y": 0.5}]},
        )

    def test_multiline_and_optional_fields(self):
        result = parse_prompt('{"base_prompt":"white fox,\nsnow","negative_prompt":"frame"}')
        self.assertEqual(result, {"base": "white fox,\nsnow", "negative": "frame", "characters": []})

    def test_ambiguous_or_incomplete_plans_are_rejected(self):
        for raw in ('{"base":"a"} {"base":"b"}', '[{"base":"a"}]', '{"base":"a"',
                    '{"base":"a","character_prompts":["girl"]}', '普通聊天回复'):
            with self.subTest(raw=raw), self.assertRaises(PromptError):
                parse_prompt(raw)

    def test_coordinate_errors_report_the_character_and_axis(self):
        for value in (None, False, -0.1, "NaN", 2):
            raw = json.dumps({"base": "scene", "characters": [{"prompt": "fox", "x": value, "y": 0.5}]})
            with self.subTest(value=value), self.assertRaisesRegex(PromptError, "第 1 个角色的 x"):
                parse_prompt(raw)
