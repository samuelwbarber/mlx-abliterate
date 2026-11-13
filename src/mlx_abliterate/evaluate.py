"""Dataset split locks and numerical drift, without an automatic refusal judge."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import unicodedata
from types import MappingProxyType

SPLITS = ('direction', 'search', 'final', 'capability')
MAX_DATASET_BYTES = 16 * 1024 * 1024
MAX_RECORD_BYTES = 64 * 1024


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class DatasetSplits:
    """Prompt records are local-only; use lock() for a content-free manifest."""
    splits: dict[str, tuple[dict, ...]]
    fingerprint: str
    _records: tuple[dict, ...]

    def lock(self) -> dict:
        return {'schema_version': 1, 'fingerprint': self.fingerprint,
                'normalization': 'NFKC then whitespace collapse; case preserved',
                'counts': {key: len(value) for key, value in self.splits.items()},
                'records': [dict(record) for record in self._records]}


def load_dataset_splits(path: str | Path) -> DatasetSplits:
    """Load JSONL with id, prompt, split and optional group/source/license/revision.

    All four split purposes must be present. Direction records need >=2 distinct
    named groups. Duplicate normalized prompts anywhere are rejected. This catches
    text duplicates, not paraphrases or semantic leakage. Source/license fields
    are provenance supplied by the caller, not a claim of verified permissions.
    """
    splits: dict[str, list[dict]] = {name: [] for name in SPLITS}
    seen_prompts: set[str] = set()
    seen_ids: set[str] = set()
    locked = []
    total = 0
    with Path(path).open('rb') as source:
        for number, line in enumerate(iter(lambda: source.readline(MAX_RECORD_BYTES + 1), b''), 1):
            total += len(line)
            if len(line) > MAX_RECORD_BYTES or total > MAX_DATASET_BYTES:
                raise ValueError('dataset exceeds record or file size limit')
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f'invalid dataset JSON on line {number}') from exc
            allowed = {'id', 'prompt', 'split', 'group', 'source', 'license', 'revision'}
            if not isinstance(row, dict) or set(row) - allowed:
                raise ValueError(f'unknown dataset fields on line {number}')
            if any(not isinstance(row.get(key), str) or not row[key].strip()
                   for key in ('id', 'prompt', 'split')):
                raise ValueError(f'missing nonblank dataset field on line {number}')
            if any(not isinstance(value, str) or not value.strip() for value in row.values()):
                raise ValueError(f'dataset metadata must be nonblank strings on line {number}')
            if row['split'] not in splits:
                raise ValueError(f'unknown split on line {number}')
            if row['id'] in seen_ids:
                raise ValueError(f'duplicate example ID on line {number}')
            if row['split'] == 'direction' and not row.get('group'):
                raise ValueError(f'direction example missing group on line {number}')
            normalized = ' '.join(unicodedata.normalize('NFKC', row['prompt']).split())
            normalized_hash = _digest(normalized)
            if normalized_hash in seen_prompts:
                raise ValueError(f'duplicate normalized prompt on line {number}; split leakage rejected')
            seen_ids.add(row['id'])
            seen_prompts.add(normalized_hash)
            splits[row['split']].append(dict(row))
            lock_row = {key: value for key, value in row.items() if key != 'prompt'}
            lock_row.update(prompt_sha256=_digest(row['prompt']), normalized_sha256=normalized_hash)
            locked.append(lock_row)
    if any(not rows for rows in splits.values()):
        raise ValueError('all direction/search/final/capability splits must be nonempty')
    if len({row['group'] for row in splits['direction']}) < 2:
        raise ValueError('direction split must contain at least two named contrast groups')
    # Row order matters for reproducible candidate evaluation and resume.
    fingerprint = _digest(_canonical({'schema_version': 1, 'records': locked}))
    return DatasetSplits(
        MappingProxyType({key: tuple(MappingProxyType(row) for row in rows) for key, rows in splits.items()}),
        fingerprint, tuple(MappingProxyType(row) for row in locked),
    )


def first_token_kl(base_logits, candidate_logits):
    """Stable float32 KL(base || candidate), per row, for finite last-token logits.

    Accept [vocabulary] or [batch, vocabulary]. No vocabulary-wide sequence cache
    is created. The result is a drift diagnostic, not a capability score.
    """
    from .edits import _require_mlx, mx
    _require_mlx()
    base = mx.array(base_logits, dtype=mx.float32)
    candidate = mx.array(candidate_logits, dtype=mx.float32)
    if base.shape != candidate.shape or base.ndim not in (1, 2) or any(n == 0 for n in base.shape):
        raise ValueError('logits must have matching nonempty [vocab] or [batch, vocab] shape')
    if not (mx.all(mx.isfinite(base)).item() and mx.all(mx.isfinite(candidate)).item()):
        raise ValueError('logits must be finite')
    base = base - mx.max(base, axis=-1, keepdims=True)
    candidate = candidate - mx.max(candidate, axis=-1, keepdims=True)
    log_p = base - mx.logsumexp(base, axis=-1, keepdims=True)
    log_q = candidate - mx.logsumexp(candidate, axis=-1, keepdims=True)
    kl = mx.sum(mx.exp(log_p) * (log_p - log_q), axis=-1)
    if not mx.all(mx.isfinite(kl)).item():
        raise ValueError('KL overflow or invalid distribution')
    # Tiny negatives can arise from roundoff; materially negative values are errors.
    if mx.any(kl < -1e-5).item():
        raise ValueError('KL was materially negative')
    result = mx.maximum(kl, 0)
    mx.eval(result)
    return result
