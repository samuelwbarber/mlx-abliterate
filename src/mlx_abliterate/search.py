"""Sequential, resumable callback orchestration; not a model/search integration.

The caller owns the model and scoring rubric. reset_state() MUST discard all KV,
Gated DeltaNet and prefix caches and restore the immutable base/edit state. It is
called before and after every attempt, including failures. A reset failure stops
further evaluation. No callback is allowed to launch concurrent model workers.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

STATUSES = {'completed', 'runtime_error', 'invalid', 'empty', 'truncated', 'malformed', 'cancelled'}


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _hash(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_new(path: Path, value: dict) -> None:
    """Atomic create, durable file+directory sync, never replace existing records."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.pending-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_canonical(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read(path: Path):
    with path.open('rb') as source:
        raw = source.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError('study record exceeds size limit')
    return json.loads(raw)


@contextmanager
def _exclusive(directory: Path):
    # flock is released by the OS on process death; no stale-lock deletion needed.
    with (directory / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('study is already owned by another worker') from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _validate_result(result: dict) -> dict:
    """Keep metrics only; never serialize callback text, traces or generated code."""
    if not isinstance(result, dict) or result.get('status') not in STATUSES:
        raise ValueError('callback must return a recognized status')
    status = result['status']
    failures = result.get('failures', 0 if status == 'completed' else 1)
    if type(failures) is not int or failures < 0:
        raise ValueError('failures must be a nonnegative integer')
    metrics = result.get('metrics', {})
    if not isinstance(metrics, dict) or any(
        not isinstance(key, str) or not key or len(key) > 100 or
        isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
        for key, value in metrics.items()
    ):
        raise ValueError('metrics must be finite numeric values with short string names')
    if len(metrics) > 100:
        raise ValueError('too many callback metrics')
    score = result.get('observed_score', result.get('score'))
    constraints = result.get('constraints_met', False)
    if type(constraints) is not bool:
        raise ValueError('constraints_met must be boolean')
    if status == 'completed':
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score):
            raise ValueError('completed callback requires finite score (lower is better)')
        if 'constraints_met' not in result:
            raise ValueError('completed callback requires explicit constraints_met')
    else:
        score = None
    eligible = status == 'completed' and failures == 0 and constraints
    return {'status': status, 'failures': failures, 'metrics': dict(metrics),
            'observed_score': score, 'score': score if eligible else None, 'constraints_met': constraints,
            'eligible': eligible}


def run_candidates(run_dir: str | Path, candidates: list[dict], provenance: dict, *,
                   evaluate, reset_state, cancelled=None) -> dict:
    """Evaluate fixed candidates sequentially and resume an identical study.

    Candidate dictionaries require unique nonblank `id`; other JSON parameters
    are opaque. Zero-edit candidates are supported without special handling.
    Provenance must include nonempty model/dataset/runtime/edit_schema values.
    evaluate(candidate) returns status, finite lower-is-better score, explicit
    constraints_met, optional failures and numeric metrics. Any failures remove
    eligibility. Non-completed outcomes never receive a favorable score.

    Cancellation before an attempt leaves it pending. KeyboardInterrupt or
    InterruptedError during evaluation writes a cancelled attempt, reset_state
    runs, and a later invocation retries it in a NEW immutable attempt file.
    Other terminal outcomes are not retried implicitly. A reset failure blocks
    this study; investigate and start a new run instead of resuming invalid state.
    """
    if not callable(evaluate) or not callable(reset_state) or (cancelled is not None and not callable(cancelled)):
        raise ValueError('evaluate/reset_state/cancelled must be callable')
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('candidates must be a nonempty list')
    ids = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get('id'), str) or not candidate['id'].strip():
            raise ValueError('each candidate needs a nonblank id')
        if candidate['id'] in ids:
            raise ValueError('candidate IDs must be unique')
        ids.add(candidate['id'])
        if len(_canonical(candidate)) > 64 * 1024:
            raise ValueError('candidate exceeds size limit')
    if not isinstance(provenance, dict) or any(not provenance.get(key) for key in ('model', 'dataset', 'runtime', 'edit_schema')):
        raise ValueError('provenance requires model, dataset, runtime and edit_schema')
    manifest = {'schema_version': 1, 'provenance': provenance, 'candidates': candidates}
    if len(_canonical(manifest)) > 4 * 1024 * 1024:
        raise ValueError('study manifest exceeds size limit')
    # Freeze caller-owned mutable input before invoking callbacks.
    manifest = json.loads(_canonical(manifest))
    candidates = manifest['candidates']
    study_hash = _hash(manifest)
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with _exclusive(directory):
        manifest_path = directory / 'manifest.json'
        if manifest_path.exists():
            if _read(manifest_path) != manifest:
                raise ValueError('resume refused: model/dataset/runtime/edit schema or candidate schedule changed')
        else:
            if (directory / 'trials').exists():
                raise ValueError('trial directory exists without a study manifest')
            _write_new(manifest_path, manifest)
        trials = directory / 'trials'
        trials.mkdir(exist_ok=True)
        histories: dict[int, list[dict]] = {index: [] for index in range(len(candidates))}
        for path in sorted(trials.iterdir()):
            if path.name.startswith('.pending-'):
                continue  # Incomplete temp files never count as committed trials.
            match = re.fullmatch(r'(\d{6})-(\d{6})\.json', path.name)
            if not match:
                raise ValueError('unrecognized file in trial journal')
            index, attempt = map(int, match.groups())
            if index not in histories or attempt != len(histories[index]) + 1:
                raise ValueError('invalid trial index or non-contiguous attempt history')
            record = _read(path)
            if (record.get('study_sha256') != study_hash or record.get('candidate_sha256') != _hash(candidates[index])
                    or record.get('index') != index or record.get('attempt') != attempt):
                raise ValueError('trial provenance mismatch')
            if _validate_result(record['result']) != record['result']:
                raise ValueError('invalid stored trial result')
            if record.get('reset_failed'):
                raise RuntimeError('study blocked by a prior cache-reset failure; investigate before a new study')
            if histories[index] and histories[index][-1]['result']['status'] != 'cancelled':
                raise ValueError('attempt exists after a terminal result')
            histories[index].append(record)
        stopped = False
        blocked = False
        for index, candidate in enumerate(candidates):
            history = histories[index]
            if history and history[-1]['result']['status'] != 'cancelled':
                continue
            if cancelled is not None and cancelled():
                stopped = True
                break
            started = time.monotonic()
            interrupted = False
            reset_failed = False
            error_type = None
            try:
                reset_state()
            except (Exception, KeyboardInterrupt) as exc:
                reset_failed = True
                error_type = type(exc).__name__
            if reset_failed:
                result = _validate_result({'status': 'runtime_error'})
            else:
                try:
                    result = _validate_result(evaluate(json.loads(_canonical(candidate))))
                except (KeyboardInterrupt, InterruptedError) as exc:
                    interrupted = True
                    error_type = type(exc).__name__
                    result = _validate_result({'status': 'cancelled'})
                except Exception as exc:
                    error_type = type(exc).__name__
                    result = _validate_result({'status': 'runtime_error'})
                finally:
                    try:
                        reset_state()
                    except (Exception, KeyboardInterrupt) as exc:
                        reset_failed = True
                        error_type = type(exc).__name__
                        result = _validate_result({'status': 'runtime_error'})
            attempt = len(history) + 1
            record = {'schema_version': 1, 'study_sha256': study_hash,
                      'candidate_sha256': _hash(candidate), 'candidate_id': candidate['id'],
                      'index': index, 'attempt': attempt, 'result': result,
                      'wall_seconds': time.monotonic() - started, 'reset_failed': reset_failed,
                      'error_type': error_type}
            _write_new(trials / f'{index:06d}-{attempt:06d}.json', record)
            history.append(record)
            if reset_failed:
                blocked = True
                break
            if interrupted or result['status'] == 'cancelled':
                stopped = True
                break
        latest = [history[-1] for history in histories.values() if history]
        eligible = [record for record in latest if record['result']['eligible']]
        best = min(eligible, key=lambda record: record['result']['score']) if eligible else None
        return {'schema_version': 1, 'study_sha256': study_hash,
                'status': 'blocked' if blocked else ('cancelled' if stopped else 'complete'),
                'evaluated_candidates': len(latest), 'planned_candidates': len(candidates),
                'eligible_candidates': len(eligible),
                'best_candidate_id': best['candidate_id'] if best else None,
                'best_score': best['result']['score'] if best else None,
                'trials': latest,
                'scope': 'Callback orchestration only; no model adapter or refusal/capability judge is supplied.'}
