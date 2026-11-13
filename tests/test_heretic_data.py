import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlx_abliterate import heretic_data as data


class HereticDataTests(unittest.TestCase):
    def fixture(self):
        config=b'system_prompt="fixture"\nmax_response_length=100\n[scorer.KeywordRate]\nkeyword_markers=["fixture"]\n'
        specs={};downloads={data.CONFIG_URL:config}
        for group in ('harmful','harmless'):
            specs[group]={'repository':'fixture/'+group,'revision':'a'*40,'files':{}}
            for split,count in [('train',400),('test',100)]:
                raw=f'{group}-{split}'.encode()
                specs[group]['files'][split]=(hashlib.sha256(raw).hexdigest(),count)
                url=f'https://huggingface.co/datasets/fixture/{group}/resolve/'+('a'*40)+f'/data/{split}-00000-of-00001.parquet'
                downloads[url]=raw
        return config,specs,downloads

    def prepare_fixture(self,root):
        config,specs,downloads=self.fixture()
        def rows(path,count):return [f'{path.stem} fixture {i}' for i in range(count)]
        with patch.object(data,'DATASETS',specs),patch.object(data,'CONFIG_SHA256',hashlib.sha256(config).hexdigest()), \
             patch.object(data,'_download',side_effect=lambda url:downloads[url]) as download, \
             patch.object(data,'_parquet_rows',side_effect=rows):
            result=data.prepare_standard_data(root)
            self.assertEqual(download.call_count,5)
            before={p.name:p.read_bytes() for p in root.iterdir()}
            second=data.prepare_standard_data(root)
            self.assertEqual(download.call_count,5)
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})
            self.assertEqual(result,second)
        return result

    def test_preparation_order_hashes_and_idempotency(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);result=self.prepare_fixture(root)
            self.assertTrue(result['ready'])
            lock=json.loads((root/'dataset-lock.json').read_text())
            subset=json.loads((root/'harmful-test100.json').read_text())['prompts']
            self.assertEqual(len(subset),100)
            self.assertTrue(subset[0].endswith('fixture 0'))
            self.assertTrue(subset[-1].endswith('fixture 99'))
            self.assertEqual(lock['datasets']['harmful']['subsets']['test[:100]']['sha256'],
                hashlib.sha256((root/'harmful-test100.json').read_bytes()).hexdigest())
            development=json.loads((root/'harmful-development32.lock.json').read_text())
            self.assertTrue(development['overlaps_direction_training'])
            self.assertFalse(development['overlaps_standard_test100'])

    def test_hash_failure_never_publishes_download(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'artifact'
            with patch.object(data,'_download',return_value=b'wrong'):
                with self.assertRaises(ValueError):data._pinned_file(path,'https://fixture','0'*64)
            self.assertFalse(path.exists())

    def test_existing_artifact_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'artifact';path.write_bytes(b'first')
            with self.assertRaises(ValueError):data._save_identical(path,b'second')
            self.assertEqual(path.read_bytes(),b'first')
            link=Path(folder)/'link';link.symlink_to(path)
            with self.assertRaises(ValueError):data._save_identical(link,b'first')

    def test_bounded_download(self):
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,n):return b'x'*n
        with patch.object(data,'MAX_DOWNLOAD_BYTES',8),patch.object(data.urllib.request,'urlopen',return_value=Response()):
            with self.assertRaises(ValueError):data._download('https://fixture')
