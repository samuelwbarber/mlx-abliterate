import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mlx_abliterate.heretic_collection import load_training_pair, chat_tokens, estimate_directions
from mlx_abliterate.edits import mx


class TrainingPairTests(unittest.TestCase):
    def fixture(self, root):
        lock={'datasets':{}}
        for group in ('harmful','harmless'):
            prompts=[f'{group} fixture {i}' for i in range(400)]
            path=root/f'{group}-train400.json'
            raw=json.dumps({'prompts':prompts}).encode();path.write_bytes(raw)
            lock['datasets'][group]={'repository':'fixture/'+group,'revision':'a'*40,
                'subsets':{'train[:400]':{'sha256':hashlib.sha256(raw).hexdigest(),
                    'ordered_prompt_sha256':[hashlib.sha256(p.encode()).hexdigest() for p in prompts]}}}
        (root/'dataset-lock.json').write_text(json.dumps(lock))

    def test_exact_pinned_training_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            groups,lock=load_training_pair(root)
            self.assertEqual(len(groups['harmful']),400)
            self.assertEqual(lock['harmful']['indices'],list(range(400)))
            (root/'harmful-train400.json').write_text('{"prompts":[]}')
            with self.assertRaises(ValueError):load_training_pair(root)

    def test_system_prompt_and_token_contract(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                self.messages=messages;self.kwargs=kwargs
                return {'input_ids':[[1,2]]}
        tokenizer=Tokenizer()
        self.assertEqual(chat_tokens(tokenizer,'fixture'),[1,2])
        self.assertEqual(tokenizer.messages[0]['role'],'system')
        self.assertFalse(tokenizer.kwargs['enable_thinking'])


@unittest.skipIf(mx is None,'optional MLX unavailable')
class MeanTests(unittest.TestCase):
    def test_online_contrast_and_orthogonalization(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                return [1 if messages[-1]['content']=='positive' else 0]
        class Adapter:
            def collect_last_hidden(self,tokens):
                vector=mx.concatenate([mx.array([1.,float(tokens[0])]),mx.zeros((5118,))])
                return {i:vector for i in range(65)}
        class Guard:
            def check(self,mx):return {}
        directions,lengths=estimate_directions(Adapter(),Tokenizer(),
            {'harmful':['positive','positive'],'harmless':['negative']},mx,Guard())
        vector=directions['layer.0']
        self.assertAlmostEqual(vector[0].item(),0)
        self.assertAlmostEqual(vector[1].item(),1)
        self.assertEqual(lengths['harmful']['count'],2)
        self.assertEqual(set(directions),{f'layer.{i}' for i in range(64)})

    def test_incomplete_collection_rejected(self):
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):return [1]
        class Adapter:
            def collect_last_hidden(self,tokens):return {0:mx.ones((5120,))}
        class Guard:
            def check(self,mx):return {}
        with self.assertRaises(ValueError):
            estimate_directions(Adapter(),Tokenizer(),{'harmful':['x'],'harmless':['y']},mx,Guard())
