from types import SimpleNamespace
import unittest
from test_generation import Language, Guard, mx
from mlx_abliterate.generation import generate
from mlx_abliterate.heretic_benchmark import keyword_refusal


@unittest.skipIf(mx is None,'MLX unavailable')
class EarlyStopTests(unittest.TestCase):
    def test_early_match_agrees_with_full_window_marker(self):
        def adapter():
            return SimpleNamespace(model=SimpleNamespace(language_model=Language([[0,4,0]]*5)))
        tokenizer=SimpleNamespace(decode=lambda ids,**kwargs:'sorry '+('more '*len(ids)))
        predicate=lambda text:keyword_refusal(text,['sorry'])
        full=generate(adapter(),[0],eos_token_ids=[2],max_new_tokens=5,tokenizer=tokenizer,guard=Guard())
        early=generate(adapter(),[0],eos_token_ids=[2],max_new_tokens=5,tokenizer=tokenizer,guard=Guard(),stop_text_predicate=predicate)
        self.assertTrue(predicate(full['text']))
        self.assertEqual(early['status'],'matched')
        self.assertEqual(early['sampled_tokens'],1)
        self.assertEqual(full['sampled_tokens'],5)

    def test_unmarked_or_empty_prefix_runs_full_budget(self):
        for text in ('An answer',' '):
            adapter=SimpleNamespace(model=SimpleNamespace(language_model=Language([[0,4,0]]*5)))
            result=generate(adapter,[0],eos_token_ids=[2],max_new_tokens=5,
                            tokenizer=SimpleNamespace(decode=lambda *args,**kwargs:text),guard=Guard(),
                            stop_text_predicate=lambda value:keyword_refusal(value,['sorry']))
            self.assertEqual(result['status'],'truncated')
            self.assertEqual(result['sampled_tokens'],5)
