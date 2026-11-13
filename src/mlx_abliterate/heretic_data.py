"""Download and prepare immutable upstream Heretic data, without logging prompts.

No corpus or upstream implementation is embedded here. Known source file hashes
are verified before parsing, and existing outputs are never replaced.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import tomllib
import urllib.request

UPSTREAM_REVISION = 'a14b8543537b50ce8db4db7eaae4eb295d5d5142'
CONFIG_SHA256 = '0b468a26580ed14ee4331707baebde2725deaeb011cfc5fb764e0a6a66bad5e2'
CONFIG_URL = f'https://raw.githubusercontent.com/p-e-w/heretic/{UPSTREAM_REVISION}/config.default.toml'
MAX_DOWNLOAD_BYTES = 16 * 1024**2
DATASETS = {
    'harmful': {'repository':'mlabonne/harmful_behaviors',
                'revision':'01cead01398926d81f7c52bdb790ee8cf77ebba7',
                'files':{'train':('ad845a73ae983b456961e020a53a08551d46019daa30530c8ccf4a4859140fb9',416),
                         'test':('5b3d4baa7639ef78e043efe56fab40ffdcf3a38d49251db3448b3e92f802ca2a',104)}},
    'harmless': {'repository':'mlabonne/harmless_alpaca',
                 'revision':'02c6a92cfcf11bb0c387334f8146d149d65b587f',
                 'files':{'train':('b13d6c60adef58f41897e5ae50245cb171fd0145768300ffe5857b382e080c46',25058),
                          'test':('8df830a54e63adbf6e8bdf8143cb8f0cd86a9fe8dbddd21737f4cbdf22d5c08e',6265)}}}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value):
    return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


def _read_bounded(path):
    if path.is_symlink():
        raise ValueError('Refusing symlink artifact')
    with path.open('rb') as stream:
        raw=stream.read(MAX_DOWNLOAD_BYTES+1)
    if len(raw)>MAX_DOWNLOAD_BYTES:
        raise ValueError('Artifact exceeds size limit')
    return raw


def _download(url):
    request=urllib.request.Request(url,headers={'User-Agent':'mlx-abliterate-data-preparation'})
    with urllib.request.urlopen(request,timeout=60) as response:
        raw=response.read(MAX_DOWNLOAD_BYTES+1)
    if len(raw)>MAX_DOWNLOAD_BYTES:
        raise ValueError('Download exceeds size limit')
    return raw


def _save_identical(path,raw):
    if path.exists() or path.is_symlink():
        if _read_bounded(path)!=raw:
            raise ValueError('Existing artifact differs from pinned preparation')
        return
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.data-',delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        try:
            os.link(temporary,path)
        except FileExistsError:
            if _read_bounded(path)!=raw:
                raise ValueError('Concurrent artifact differs from pinned preparation')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _pinned_file(path,url,expected):
    raw=_read_bounded(path) if path.exists() or path.is_symlink() else _download(url)
    if _sha(raw)!=expected:
        raise ValueError('Pinned source SHA256 mismatch')
    _save_identical(path,raw)
    return raw


def _parquet_rows(path,expected_count):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError('Install the data extra: pip install -e ".[data]"') from error
    parquet=pq.ParquetFile(path)
    if parquet.metadata.num_rows!=expected_count:
        raise ValueError('Pinned dataset row count mismatch')
    rows=[]
    for batch in parquet.iter_batches(batch_size=512,columns=['text']):
        values=batch.column(0).to_pylist()
        if any(not isinstance(row,str) for row in values):
            raise ValueError('Dataset text column must contain strings')
        rows.extend(values)
    return rows


def prepare_standard_data(destination):
    destination=Path(destination)
    if destination.is_symlink():
        raise ValueError('Refusing symlink destination')
    destination.mkdir(parents=True,exist_ok=True)
    configraw=_pinned_file(destination/'config.default.toml',CONFIG_URL,CONFIG_SHA256)
    config=tomllib.loads(configraw.decode())
    lock={'schema_version':1,'upstream_revision':UPSTREAM_REVISION,'datasets':{}}
    for label,spec in DATASETS.items():
        repo,revision=spec['repository'],spec['revision']
        record={'repository':repo,'revision':revision,'column':'text','shuffle':False,'files':{},'subsets':{}}
        for split,(expected_hash,expected_count) in spec['files'].items():
            remote=f'data/{split}-00000-of-00001.parquet'
            url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{remote}'
            local=destination/f'{label}-{split}.parquet'
            raw=_pinned_file(local,url,expected_hash)
            rows=_parquet_rows(local,expected_count)
            record['files'][split]={'upstream_path':remote,'sha256':_sha(raw),'row_count':len(rows)}
            for count in ([100,400] if split=='train' else [100]):
                if len(rows)<count:
                    raise ValueError('Insufficient dataset rows')
                selected=rows[:count];name=f'{label}-{split}{count}.json'
                encoded=_json_bytes({'prompts':selected})
                _save_identical(destination/name,encoded)
                record['subsets'][f'{split}[:{count}]']={'filename':name,'count':count,'sha256':_sha(encoded),
                    'ordered_prompt_sha256':[_sha(x.encode()) for x in selected]}
            if split=='train' and label=='harmless':
                _save_identical(destination/'harmless-train256.json',_json_bytes({'prompts':rows[:256]}))
            if split=='train' and label=='harmful':
                for count in (32,100):
                    filename=destination/f'harmful-development{count}.json'
                    encoded=_json_bytes({'prompts':rows[:count]})
                    _save_identical(filename,encoded)
                    development_lock={'dataset':repo,'revision':revision,'split':f'train[:{count}]',
                        'indices':list(range(count)),'overlaps_direction_training':True,
                        'overlaps_standard_test100':False,'sha256':_sha(encoded)}
                    _save_identical(filename.with_suffix('.lock.json'),_json_bytes(development_lock))
        lock['datasets'][label]=record
    benchmark={'schema_version':1,'keywords':config['scorer']['KeywordRate']['keyword_markers'],
        'system_prompt':config['system_prompt'],'max_new_tokens':config['max_response_length'],
        'provenance':{'upstream_repository':'p-e-w/heretic','upstream_revision':UPSTREAM_REVISION,
          'upstream_config_sha256':_sha(configraw),'upstream_source_license':'AGPL-3.0-or-later',
          'dataset_repository':DATASETS['harmful']['repository'],'dataset_revision':DATASETS['harmful']['revision'],
          'split':'test[:100]','column':'text','shuffle':False,
          'dataset_file_sha256':_sha(_read_bounded(destination/'harmful-test100.json')),
          'matcher_semantics':'Empty whitespace-only response is a match; otherwise lowercase, remove asterisks, normalize U+2019 apostrophe, collapse whitespace, then case-insensitive substring match.',
          'generation_semantics':'Upstream greedy100 continuation tokens after model template and automatically determined response prefix. Ollama adapter differences must be recorded.'}}
    _save_identical(destination/'benchmark-config.json',_json_bytes(benchmark))
    _save_identical(destination/'dataset-lock.json',_json_bytes(lock))
    return {'ready':True,'datasets':{k:{'revision':v['revision'],'rows':{s:f['row_count'] for s,f in v['files'].items()}}
                                  for k,v in lock['datasets'].items()},
            'keyword_count':len(benchmark['keywords']),
            'dataset_lock_sha256':_sha(_json_bytes(lock)),
            'benchmark_config_sha256':_sha(_json_bytes(benchmark))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    print(json.dumps(prepare_standard_data(args.destination)))


if __name__=='__main__':
    main()
