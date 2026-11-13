"""CPU-only comparison of four verified checkpoint evaluation artifacts."""
from __future__ import annotations

import argparse
import array
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys

IDENTITY_FIELDS = ('source_identity', 'dataset_fingerprint', 'rubric_sha256', 'edit_spec_sha256', 'runtime')
MATCH_FIELDS = ('dataset_fingerprint', 'rubric_sha256', 'runtime', 'row_order', 'splits', 'max_new_tokens')
LABELS = ('stock', 'dynamic', 'edited', 'control')
MAX_FILE_BYTES = 16 * 1024**2


def _bytes(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError('Evaluation artifact exceeds size bound')
    return raw


def _npy(raw, shape):
    """Read only the narrow little-endian float32 one-row NPY output contract."""
    if raw[:6] != b'\x93NUMPY' or raw[6:8] not in (b'\x01\x00', b'\x02\x00'):
        raise ValueError('Unsupported NPY format')
    size_bytes = 2 if raw[6] == 1 else 4
    header_size = int.from_bytes(raw[8:8+size_bytes], 'little')
    offset = 8 + size_bytes + header_size
    if header_size > 65536 or offset > len(raw):
        raise ValueError('Invalid NPY header size')
    header = ast.literal_eval(raw[8+size_bytes:offset].decode('ascii').strip())
    # A single-row matrix has the same element order in C and Fortran layout;
    # MLX serializes these one-row arrays with fortran_order=True.
    if (header.get('descr') != '<f4' or type(header.get('fortran_order')) is not bool or
            tuple(shape) != header.get('shape') or len(shape) != 2 or shape[0] != 1 or
            type(shape[1]) is not int or not 1 <= shape[1] <= 2_000_000):
        raise ValueError('Expected one nonempty float32 logit row')
    if len(raw)-offset != shape[1]*4:
        raise ValueError('NPY payload length differs from shape')
    values = array.array('f')
    values.frombytes(raw[offset:])
    if sys.byteorder != 'little':
        values.byteswap()
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Nonfinite checkpoint logits')
    return values


def logit_metrics(base, candidate):
    if len(base) != len(candidate) or not base:
        raise ValueError('Logit widths differ or are empty')
    if not all(math.isfinite(v) for values in (base,candidate) for v in values):
        raise ValueError('Nonfinite logits')
    peak_p, peak_q = max(base), max(candidate)
    sum_p = math.fsum(math.exp(v-peak_p) for v in base)
    sum_q = math.fsum(math.exp(v-peak_q) for v in candidate)
    log_sum_p, log_sum_q = math.log(sum_p), math.log(sum_q)
    kl = math.fsum(math.exp(p-peak_p)/sum_p * ((p-peak_p-log_sum_p)-(q-peak_q-log_sum_q))
                   for p,q in zip(base,candidate))
    if kl < -1e-10 or not math.isfinite(kl):
        raise ValueError('Invalid KL computation')
    differences = [abs(p-q) for p,q in zip(base,candidate)]
    return {'kl_base_to_candidate': max(kl,0.), 'max_abs_logit_delta':max(differences),
            'mean_abs_logit_delta':math.fsum(differences)/len(differences),
            'exact_logits': all(v == 0 for v in differences)}


def _load_artifact(report_path, folder):
    report = json.loads(_bytes(report_path))
    folder = Path(folder).resolve()
    manifest_bytes = _bytes(folder/'manifest.json')
    if hashlib.sha256(manifest_bytes).hexdigest() != report['logits']['manifest_sha256']:
        raise ValueError('Logit manifest SHA mismatch')
    manifest = json.loads(manifest_bytes)
    if report.get('schema_version') != 1 or manifest.get('schema_version') != 1:
        raise ValueError('Unsupported evaluation schema')
    for field in IDENTITY_FIELDS:
        if report[field] != manifest[field]:
            raise ValueError(f'Report/manifest identity mismatch: {field}')
    records = manifest['records']
    ids = [r['id'] for r in records]
    filenames = [r['file'] for r in records]
    if (not records or len(set(ids)) != len(ids) or len(set(filenames)) != len(filenames) or
            ids != report['row_order'] or ids != [r['id'] for r in report['results']] or
            report['logits']['count'] != len(records)):
        raise ValueError('Missing, duplicate, or misordered evaluation rows')
    for record in records:
        filename = record['file']
        path = (folder/filename).resolve()
        if Path(filename).name != filename or path.parent != folder or not path.is_file():
            raise ValueError('Unsafe logit file path')
    if {p.name for p in folder.iterdir()} != set(filenames)|{'manifest.json'}:
        raise ValueError('Undeclared logit artifact files')
    return report, records, folder


def _retention(base, candidate):
    if len(base) != len(candidate):
        raise ValueError('Result row counts differ')
    regressions, failures, proxy = [], [], []
    for before, after in zip(base,candidate):
        if before['id'] != after['id'] or before['kind'] != after['kind']:
            raise ValueError('Result row identity differs')
        identifier = before['id']
        if before['status'] != 'completed' or after['status'] != 'completed':
            failures.append(identifier)
        if before['kind'] == 'refusal_likelihood':
            if before['status'] == after['status'] == 'completed':
                values = (before.get('mean_nll'),after.get('mean_nll'))
                if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
                    raise ValueError('Invalid completed proxy score')
                proxy.append({'id':identifier,'mean_nll_gain':values[1]-values[0]})
        elif before.get('task_check_passed') and not after.get('task_check_passed'):
            regressions.append(identifier)
    return {'retained_baseline_correct_tasks':not regressions and not failures,
            'regressed_task_ids':regressions,'failed_row_ids':failures,
            'proxy_nll_gains':proxy, 'mean_proxy_nll_gain':statistics.mean(r['mean_nll_gain'] for r in proxy) if proxy else None,
            'exact_result_records':base == candidate}


def compare_checkpoints(artifacts):
    if set(artifacts) != set(LABELS):
        raise ValueError('Require stock, dynamic, edited, and control artifacts')
    loaded = {label:_load_artifact(*artifacts[label]) for label in LABELS}
    base = loaded['stock'][0]
    for label in LABELS:
        report = loaded[label][0]
        for field in MATCH_FIELDS:
            if report[field] != base[field]:
                raise ValueError(f'Checkpoint comparison mismatch: {field}')
        if report['source_identity']['source_manifest_digest'] != base['source_identity']['source_manifest_digest']:
            raise ValueError('Different ancestral source model')
    if loaded['dynamic'][0]['source_identity'] != base['source_identity']:
        raise ValueError('Dynamic evaluation did not use identical base')
    if base['source_identity']['modified'] or not loaded['dynamic'][0]['edit_spec_sha256']:
        raise ValueError('Expected unchanged stock and explicit dynamic edits')
    if any(loaded[label][0]['edit_spec_sha256'] is not None for label in ('stock','edited','control')):
        raise ValueError('Unexpected dynamic edits in checkpoint evaluation')
    pairs = [('stock','dynamic'),('stock','edited'),('dynamic','edited'),('stock','control')]
    comparisons = {f'{a}_to_{b}': {'rows':[], 'tasks':_retention(loaded[a][0]['results'],loaded[b][0]['results'])} for a,b in pairs}
    for index, record in enumerate(loaded['stock'][1]):
        arrays = {}
        for label in LABELS:
            other = loaded[label][1][index]
            for field in ('id','prompt_sha256','tokens_sha256','shape'):
                if other[field] != record[field]:
                    raise ValueError(f'Paired logit row mismatch: {field}')
            raw = _bytes(loaded[label][2]/other['file'])
            if hashlib.sha256(raw).hexdigest() != other['sha256']:
                raise ValueError('Logit file SHA mismatch')
            arrays[label] = _npy(raw, other['shape'])
        for a,b in pairs:
            comparisons[f'{a}_to_{b}']['rows'].append({'id':record['id'],**logit_metrics(arrays[a],arrays[b])})
    for comparison in comparisons.values():
        rows = comparison['rows']
        comparison['summary'] = {'row_count':len(rows),'all_logits_exact':all(r['exact_logits'] for r in rows)}
        for field in ('kl_base_to_candidate','max_abs_logit_delta','mean_abs_logit_delta'):
            values = [r[field] for r in rows]
            comparison['summary'][field] = {'mean':statistics.mean(values),'median':statistics.median(values),'max':max(values)}
    control = comparisons['stock_to_control']
    return {'schema_version':1,'dataset_fingerprint':base['dataset_fingerprint'],
            'rubric_sha256':base['rubric_sha256'],'runtime':base['runtime'],
            'identities':{label:{field:loaded[label][0][field] for field in ('source_identity','edit_spec_sha256')} for label in LABELS},
            'evaluation_statuses':{label:loaded[label][0]['status'] for label in LABELS},
            'comparisons':comparisons,
            'quantization_control_exact':control['summary']['all_logits_exact'] and control['tasks']['exact_result_records'],
            'scope':'Paired first-token drift and authored exact-task retention diagnostics; not broad capability or refusal effectiveness validation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('runs'))
    parser.add_argument('--prefix',default='checkpoint')
    parser.add_argument('--suffix',default='v1')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    artifacts = {label:(args.root/f'{args.prefix}-{label}-{args.suffix}.json',
                        args.root/f'{args.prefix}-{label}-{args.suffix}-logits') for label in LABELS}
    report = compare_checkpoints(artifacts)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'quantization_control_exact':report['quantization_control_exact'],
                     'stock_to_edited':report['comparisons']['stock_to_edited']['tasks'],
                     'dynamic_to_edited':report['comparisons']['dynamic_to_edited']['summary']}))


if __name__ == '__main__':
    main()
