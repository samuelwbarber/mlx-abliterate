"""Bounded merge component checks and explicit edited/control export preparation.

These measurements quantify re-quantization drift on deterministic projection
inputs. They do not establish whole-model behavior or approve a candidate.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile

from .export import ExportError, verify_export
from .merge import (_source_entries, _validate_spec, _write_bundle,
                    merge_export, merge_nvfp4)


def quantization_control_spec(spec: dict) -> dict:
    """Force re-quantization of precisely the candidate's selected projections."""
    control = copy.deepcopy(spec)
    control['force_requantize'] = True
    for edit in control['projections'].values():
        edit['strength'] = 0.0
    return control


def _metrics(mx, reference, actual):
    error = actual.astype(mx.float32) - reference.astype(mx.float32)
    if not bool(mx.all(mx.isfinite(error)).item()):
        raise ExportError('Nonfinite component output drift')
    rms = mx.sqrt(mx.mean(error * error)).item()
    reference_rms = mx.sqrt(mx.mean(reference.astype(mx.float32) ** 2)).item()
    return {'max_abs_error': mx.max(mx.abs(error)).item(), 'rms_error': rms,
            'reference_rms': reference_rms,
            'relative_rms_error': rms / reference_rms if reference_rms else None}


def validate_components(source: Path, spec: dict, *, scratch_limit_bytes=1024**3) -> dict:
    """Check selected projections sequentially, never loading the whole model.

    Inputs are deterministic synthetic float32 vectors; ordinary additive bias
    is omitted equally on both sides because this edit leaves bias unchanged.
    Packed GPU matmul is used on both sides. This is an export drift diagnostic,
    not an independent proof of native quantization arithmetic.
    """
    import mlx.core as mx
    from .edits import directional_output, normalize_direction
    if not mx.metal.is_available():
        raise ExportError('Metal GPU required; CPU fallback is unsupported')
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(4 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    source = Path(source).resolve()
    identity = verify_export(source, require_unmodified=True)
    receipt, entries = _source_entries(source)
    edits, policy, force = _validate_spec(spec, receipt, entries)
    by_name = {entry['name']: entry for entry in entries}
    reports = {}
    with tempfile.TemporaryDirectory(prefix='mlx-merge-components-') as folder:
        path = Path(folder) / 'bundle.safetensors'
        for prefix, edit in edits.items():
            keys = [prefix + suffix for suffix in ('.weight', '.weight_scale', '.weight_scale_2')]
            _write_bundle(path, [by_name[key] for key in keys])
            arrays = mx.load(str(path))
            weight, scales, factor = (arrays[key] for key in keys)
            width = weight.shape[1] * 8
            inputs = mx.sin(mx.arange(4 * width, dtype=mx.float32) * .013).reshape(4, width)
            def project(q, s, g):
                return mx.quantized_matmul(inputs, q, s, transpose=True,
                                          group_size=16, bits=4, mode='nvfp4') * g
            baseline = project(weight, scales, factor)
            strength = edit['strength']
            dynamic = (directional_output(baseline, normalize_direction(edit['direction']), strength)
                       if strength else baseline)
            mx.eval(baseline, dynamic)
            mx.reset_peak_memory()
            starting = mx.get_active_memory()
            variants = {}
            for label, amount, requantize in (('edited', strength, force), ('quantization_only', 0., True)):
                q, s, g, merge_report = merge_nvfp4(weight, scales, factor, edit.get('direction'), amount,
                    scale_policy=policy, force_requantize=requantize,
                    scratch_limit_bytes=scratch_limit_bytes)
                actual = project(q, s, g)
                variants[label] = {'drift': _metrics(mx, dynamic if label == 'edited' else baseline, actual),
                                   'merge': merge_report}
                del q, s, g, actual
                mx.clear_cache()
            reports[prefix] = {'input_rows': 4, 'input_width': width, 'output_width': weight.shape[0],
                               'variants': variants, 'starting_active_bytes': starting,
                               'peak_active_bytes': mx.get_peak_memory()}
            del arrays, weight, scales, factor, inputs, baseline, dynamic
            mx.clear_cache()
            path.unlink()
    return {'schema_version': 1, 'source_identity': identity, 'projections': reports,
            'quality_validated': False,
            'scope': 'Synthetic projection-output drift only; full-model dynamic/export drift and quality remain required.'}


def export_pair(source: Path, edited: Path, control: Path, spec: dict) -> dict:
    """Create distinct new artifacts; leave any completed artifact on failure."""
    source, edited, control = map(Path, (source, edited, control))
    if edited.resolve().is_relative_to(control.resolve()) or control.resolve().is_relative_to(edited.resolve()):
        raise ExportError('Edited and quantization-control destinations must be distinct and disjoint')
    for destination in (edited, control):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
    # merge_export performs the full spec/source validation before writes.
    candidate_report = merge_export(source, edited, spec)
    candidate_identity = verify_export(edited)
    control_report = merge_export(source, control, quantization_control_spec(spec))
    control_identity = verify_export(control)
    return {'schema_version': 1, 'edited': {'merge': candidate_report, 'identity': candidate_identity},
            'quantization_only': {'merge': control_report, 'identity': control_identity},
            'quality_validated': False, 'imported': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('components', 'export-pair'))
    parser.add_argument('source', type=Path)
    parser.add_argument('--edit-spec', required=True, type=Path)
    parser.add_argument('--edited', type=Path)
    parser.add_argument('--control', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = json.loads(args.edit_spec.read_text())
    if args.mode == 'components':
        report = validate_components(args.source, spec)
    else:
        if args.edited is None or args.control is None:
            parser.error('export-pair requires --edited and --control')
        report = export_pair(args.source, args.edited, args.control, spec)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
