"""Bounded Ollama serving measurements; no generated code is executed.

Reports contain hashes and timing metadata, never prompt or generated text.
The speed gate is only a serving-throughput gate, not a fidelity/quality verdict.
"""
from __future__ import annotations

import hashlib
import json
import math
import socket
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def load_prompts(path: str | Path) -> list[str]:
    """Read a JSON array of nonblank strings (maximum file size 8 MiB)."""
    with Path(path).open('rb') as source:
        raw = source.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError('prompt file exceeds 8 MiB')
    return _validate_prompts(json.loads(raw))


def _validate_prompts(prompts: Any) -> list[str]:
    if not isinstance(prompts, (list, tuple)) or not prompts:
        raise ValueError('prompts must be a nonempty list of strings')
    if any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError('prompts must contain nonblank strings')
    return list(prompts)


def _endpoint(base_url: str, path: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError('base_url must be an HTTP(S) origin')
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
        raise ValueError('base_url must not contain credentials, path, query or fragment')
    return base_url.rstrip('/') + path


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('HTTP redirects are not allowed')


def _request(url: str, payload: dict | None, timeout: float):
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    return urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout)


def _json_request(url: str, payload: dict | None, timeout: float) -> dict:
    with _request(url, payload, timeout) as response:
        raw = response.read(MAX_LINE_BYTES + 1)
    if len(raw) > MAX_LINE_BYTES:
        raise ValueError('metadata response exceeds limit')
    result = json.loads(raw)
    if not isinstance(result, dict) or 'error' in result:
        raise ValueError('invalid metadata response')
    return result


def _stream_events(response, start: float, timeout: float):
    """Bound both framing memory and wall time, including trickling partial lines."""
    buffer = bytearray()
    received = 0
    while True:
        remaining = timeout - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError('request deadline exceeded')
        # urllib exposes an HTTPResponse backed by a socket. Bound its next read
        # to the remaining deadline, rather than restarting the full timeout.
        try:
            response.fp.raw._sock.settimeout(remaining)
        except AttributeError:
            pass  # Fixture/custom responses retain the original bounded timeout.
        chunk = response.read1(8192)
        if not chunk:
            if buffer:
                yield json.loads(buffer)
            return
        received += len(chunk)
        if received > MAX_RESPONSE_BYTES:
            raise ValueError('stream exceeds response size limit')
        buffer.extend(chunk)
        while b'\n' in buffer:
            line, _, rest = buffer.partition(b'\n')
            buffer = bytearray(rest)
            if len(line) > MAX_LINE_BYTES:
                raise ValueError('stream line exceeds size limit')
            if line.strip():
                yield json.loads(line)
        if len(buffer) > MAX_LINE_BYTES:
            raise ValueError('stream line exceeds size limit')


def _measure(base_url: str, payload: dict, timeout: float) -> dict:
    start = time.monotonic()
    output = hashlib.sha256()
    thinking = hashlib.sha256()
    result: dict[str, Any] = {'status': 'runtime_error', 'first_token_seconds': None,
                              'final_answer_bytes': 0, 'thinking_bytes': 0}
    final = None
    try:
        with _request(_endpoint(base_url, '/api/generate'), payload, timeout) as response:
            for event in _stream_events(response, start, timeout):
                if not isinstance(event, dict) or 'error' in event:
                    raise ValueError('invalid or failed generation event')
                for key, digest, counter in [('response', output, 'final_answer_bytes'),
                                              ('thinking', thinking, 'thinking_bytes')]:
                    value = event.get(key, '')
                    if not isinstance(value, str):
                        raise ValueError('non-text generation content')
                    raw = value.encode()
                    digest.update(raw)
                    result[counter] += len(raw)
                    if raw and result['first_token_seconds'] is None:
                        result['first_token_seconds'] = time.monotonic() - start
                if event.get('done') is True:
                    final = event
                    break
        if final is None:
            raise ValueError('stream ended without completion event')
        metrics = {}
        for key in ('eval_count', 'eval_duration', 'prompt_eval_count', 'prompt_eval_duration',
                    'load_duration', 'total_duration'):
            value = final.get(key)
            if type(value) is not int or value < 0:
                raise ValueError('missing or invalid timing counter: ' + key)
            metrics[key] = value
        if metrics['eval_count'] == 0 or metrics['eval_duration'] == 0:
            raise ValueError('generation timing counters must be positive')
        if metrics['total_duration'] < metrics['eval_duration']:
            raise ValueError('generation exceeds total duration')
        result.update(metrics)
        result['decode_tokens_per_second'] = metrics['eval_count'] * 1e9 / metrics['eval_duration']
        result['prefill_tokens_per_second'] = (metrics['prompt_eval_count'] * 1e9 /
            metrics['prompt_eval_duration'] if metrics['prompt_eval_duration'] else None)
        reason = final.get('done_reason')
        result['done_reason'] = reason
        limit = payload['options'].get('num_predict')
        if reason == 'length' or (isinstance(limit, int) and limit > 0 and metrics['eval_count'] >= limit):
            result['status'] = 'truncated'
        elif not result['final_answer_bytes']:
            result['status'] = 'empty'
        elif reason != 'stop':
            result['status'] = 'unknown_completion'
        else:
            result['status'] = 'completed'
    except (OSError, ValueError, TimeoutError, urllib.error.URLError) as exc:
        # Exception messages may contain server responses, prompts or credentials.
        result['error_type'] = type(exc).__name__
        if isinstance(exc, (TimeoutError, socket.timeout)):
            result['status'] = 'timeout'
    result['wall_seconds'] = time.monotonic() - start
    result['response_sha256'] = output.hexdigest()
    result['thinking_sha256'] = thinking.hexdigest()
    return result


def _summary(samples: list[dict]) -> dict:
    valid = [s['decode_tokens_per_second'] for s in samples if s['status'] == 'completed']
    timed = [s['decode_tokens_per_second'] for s in samples
             if s['status'] in ('completed', 'truncated') and 'decode_tokens_per_second' in s]
    return {'performance_only_median_including_truncated': statistics.median(timed) if timed else None,
            'samples': len(samples), 'outcomes': dict(Counter(s['status'] for s in samples)),
            'valid_samples': len(valid),
            'median_decode_tokens_per_second': statistics.median(valid) if valid else None,
            'min_decode_tokens_per_second': min(valid) if valid else None,
            'max_decode_tokens_per_second': max(valid) if valid else None}


def run_benchmark(model: str, prompts: list[str], *, base_url: str = 'http://127.0.0.1:11434',
                  repeats: int = 5, warmups: int = 1, options: dict | None = None,
                  think: bool = False, timeout: float = 120, cold: bool = False,
                  environment: dict | None = None) -> dict:
    """Measure each prompt repeatedly; cold=True unloads only the named model.

    Other resident models must be managed by the caller. MTP is not toggled via
    undocumented options. Pass measured power/thermal/MTP settings in environment.
    Counts include thinking tokens when Ollama includes them; split token counts
    and cache-hit state are unavailable from this API and are reported as unknown.
    """
    prompts = _validate_prompts(prompts)
    _endpoint(base_url, '/api/generate')
    if not isinstance(model, str) or not model.strip():
        raise ValueError('model must be nonblank')
    if type(repeats) is not int or repeats < 1 or type(warmups) is not int or warmups < 1:
        raise ValueError('repeats and warmups must be positive integers')
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError('timeout must be finite and between 0 and 3600 seconds')
    if type(think) is not bool or type(cold) is not bool:
        raise ValueError('think and cold must be booleans')
    settings = {'temperature': 0, 'seed': 0, 'num_predict': 256, 'num_ctx': 4096}
    if options is not None:
        if not isinstance(options, dict):
            raise ValueError('options must be a mapping')
        settings.update(options)
    if type(settings['num_predict']) is not int or not 0 < settings['num_predict'] <= 32768:
        raise ValueError('num_predict must be bounded between 1 and 32768')
    if type(settings['num_ctx']) is not int or not 0 < settings['num_ctx'] <= 32768:
        raise ValueError('num_ctx must be bounded between 1 and 32768')
    json.dumps(settings, allow_nan=False)
    metadata = _json_request(_endpoint(base_url, '/api/version'), None, timeout)
    report = {'schema_version': 1, 'model': model, 'ollama_version': metadata.get('version'),
              'settings': {'options': settings, 'think': think, 'warmups': warmups,
                           'repeats': repeats, 'cold': cold, 'keep_alive': '5m'},
              'prompt_sha256': [_hash(p) for p in prompts], 'environment': environment or {},
              'limitations': ['Quality and tensor fidelity are not scored.',
                  'Reasoning/final token split, cache-hit state, MTP acceptance and memory are not observed.',
                  'Warm requests may reuse prefixes; this is not a cache-miss benchmark.',
                  'Caller must ensure only one model is resident and matched power/thermal conditions.'],
              'cold_samples': [], 'warmup_samples': [], 'samples': []}
    if cold:
        unloaded = _json_request(_endpoint(base_url, '/api/generate'),
                                 {'model': model, 'keep_alive': 0, 'stream': False}, timeout)
        if unloaded.get('done') is not True:
            raise ValueError('model unload did not complete')
    def measure(prompt_index: int, phase: str, repetition: int) -> dict:
        payload = {'model': model, 'prompt': prompts[prompt_index], 'stream': True,
                   'think': think, 'options': settings, 'keep_alive': '5m'}
        sample = _measure(base_url, payload, timeout)
        sample.update(prompt_index=prompt_index, phase=phase, repetition=repetition)
        return sample
    if cold:
        report['cold_samples'].append(measure(0, 'cold', 0))
    for repetition in range(warmups):
        for index in range(len(prompts)):
            report['warmup_samples'].append(measure(index, 'warmup', repetition))
    for repetition in range(repeats):
        for index in range(len(prompts)):
            report['samples'].append(measure(index, 'warm', repetition))
    report['summary'] = _summary(report['samples'])
    report['status'] = ('complete' if all(s['status'] == 'completed' for s in
        report['cold_samples'] + report['warmup_samples'] + report['samples']) else 'incomplete')
    report['environment_sha256'] = _hash(json.dumps(report['environment'], sort_keys=True, allow_nan=False))
    report['serving_configuration_sha256'] = _hash(json.dumps({
        key: report[key] for key in ('ollama_version', 'settings', 'prompt_sha256', 'environment')
    }, sort_keys=True, allow_nan=False))
    return report


def compare_reports(baseline: dict, candidate: dict, *, minimum_samples: int = 5,
                    minimum_ratio: float = .9) -> dict:
    """A conservative aggregate speed gate; cannot pass invalid/unmatched runs."""
    if type(minimum_samples) is not int or minimum_samples < 5:
        raise ValueError('minimum_samples must be at least five per prompt')
    if not isinstance(minimum_ratio, (int, float)) or not math.isfinite(minimum_ratio) or not 0 < minimum_ratio <= 1:
        raise ValueError('minimum_ratio must be in (0, 1]')
    reasons = []
    for key in ('schema_version', 'ollama_version', 'settings', 'prompt_sha256', 'environment'):
        if baseline.get(key) != candidate.get(key) or key not in baseline or key not in candidate:
            reasons.append('mismatched or missing ' + key)
    if not baseline.get('ollama_version'):
        reasons.append('Ollama version is unknown')
    prompts = baseline.get('prompt_sha256', [])
    if not isinstance(prompts, list) or len(set(prompts)) < 2:
        reasons.append('at least two distinct prompts are required')
    medians = []
    for label, report in [('baseline', baseline), ('candidate', candidate)]:
        samples = report.get('samples', [])
        valid = []
        counts = Counter()
        for sample in samples:
            rate = sample.get('decode_tokens_per_second')
            if (sample.get('status') != 'completed' or sample.get('phase') != 'warm'
                    or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0):
                reasons.append(label + ' contains invalid or non-warm samples')
                continue
            counts[sample.get('prompt_index')] += 1
            valid.append(rate)
        if any(counts[i] < minimum_samples for i in range(len(prompts))) or not prompts:
            reasons.append(label + ' has insufficient valid samples per prompt')
        medians.append(statistics.median(valid) if valid else None)
    left, right = baseline.get('samples', []), candidate.get('samples', [])
    paired = lambda samples: [(s.get('prompt_index'), s.get('repetition'), s.get('prompt_eval_count')) for s in samples]
    if paired(left) != paired(right):
        reasons.append('sample schedule or tokenized prompt lengths differ')
    ratio = medians[1] / medians[0] if all(medians) else None
    return {'status': 'inconclusive' if reasons else ('pass' if ratio >= minimum_ratio else 'fail'),
            'reasons': sorted(set(reasons)), 'minimum_ratio': minimum_ratio,
            'decode_ratio': ratio, 'baseline_median': medians[0], 'candidate_median': medians[1],
            'scope': 'Aggregate warm decode throughput only; not quality, fidelity, or all-context release approval.'}
