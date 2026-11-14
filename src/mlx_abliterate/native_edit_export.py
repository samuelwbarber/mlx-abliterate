"""Byte-preserving native directional checkpoints for an explicit custom runtime.

No MLX import, decoding, or re-quantization occurs here. The custom architecture
must be registered by the isolated directional runner; stock runtimes must reject
it. Directions are stored as supplied float32 values, with unit-norm admission.
The native wrapper performs the same additional normalization as the dynamic
Python wrapper before using these vectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile

from .export import ExportError, _copy_range, _dump, _publish_directory, verify_export
from .inventory import read_safetensors_header
from .merge import read_edit_spec, _source_entries, _validate_spec, _validate_strength
from .mlx_adapter import preflight_export

ARCHITECTURE = 'Qwen3_5DirectionalForConditionalGeneration'
TENSOR_PREFIX = 'mlx_abliterate.direction.'
METHOD = 'weight_only_rank_one'


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(path, limit=32*1024**2):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:
                raise ExportError('Duplicate JSON key')
            result[key]=value
        return result
    with Path(path).open('rb') as stream:
        raw=stream.read(limit+1)
    if len(raw)>limit:
        raise ExportError('Native export metadata exceeds size bound')
    def invalid(_):
        raise ExportError('Nonfinite JSON value')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid),_sha(raw)


def _vector_bytes(values):
    if (not isinstance(values,list) or len(values)!=5120 or
            any(type(v) not in (int,float) or not math.isfinite(v) for v in values)):
        raise ExportError('Direction must contain 5120 finite numbers')
    try:
        raw=struct.pack('<5120f',*values)
    except (OverflowError,struct.error) as error:
        raise ExportError('Direction is not representable as float32') from error
    decoded=struct.unpack('<5120f',raw)
    norm=math.sqrt(math.fsum(v*v for v in decoded))
    if not math.isfinite(norm) or abs(norm-1.)>1e-5:
        raise ExportError('Float32 direction must have unit norm within 1e-5')
    return raw


def _admit_selection(spec, report, identity, paths):
    if (report.get('status')!='complete' or report.get('export_gate_passed') is not True or
            report.get('source_identity')!=identity or spec.get('source_content_fingerprint')!=identity['content_fingerprint']):
        raise ExportError('Successful selection and matching verified source identity required')
    selected=report.get('selected')
    if not isinstance(selected,dict):
        raise ExportError('Missing selected candidate')
    strength=_validate_strength(selected.get('strength'))
    if strength==0:
        raise ExportError('Selected candidate must be nonzero')
    start,stop=selected.get('start'),selected.get('stop')
    if type(start) is not int or type(stop) is not int or not 0<=start<stop<=64:
        raise ExportError('Invalid selected layer range')
    expected={p for p in paths if start<=int(p.split('.')[3])<stop}
    if not expected or set(spec['projections'])!=expected or any(edit['strength']!=strength for edit in spec['projections'].values()):
        raise ExportError('Specification differs from selected projection schedule')
    standard=report.get('standard',{})
    trials=[t for t in standard.get('trials',[]) if t.get('candidate_id')==selected.get('id')]
    if standard.get('status')!='complete' or standard.get('best_candidate_id')!=selected.get('id') or len(trials)!=1:
        raise ExportError('Missing completed selected standard trial')
    result=trials[0].get('result',{})
    refusals=result.get('metrics',{}).get('refusals')
    if (result.get('eligible') is not True or result.get('constraints_met') is not True or
            result.get('status')!='completed' or result.get('failures')!=0 or
            type(refusals) is not int or not 0<=refusals<=5):
        raise ExportError('Selected trial does not satisfy standard/capability gates')
    return selected


def _write_directions(path, extras):
    header,offset={},0
    for name,raw in extras.items():
        header[name]={'dtype':'F32','shape':[5120],'data_offsets':[offset,offset+len(raw)]}
        offset+=len(raw)
    encoded=json.dumps(header,separators=(',',':')).encode()
    encoded+=b' '*(-len(encoded)%8)
    with path.open('xb') as stream:
        stream.write(struct.pack('<Q',len(encoded)));stream.write(encoded)
        for raw in extras.values():stream.write(raw)
        stream.flush();os.fsync(stream.fileno())


def _tensor_identity(record):
    return {key:record[key] for key in ('name','dtype','shape','size','payload_sha256')}


def export_native_edits(source_export, destination, edit_spec_path, selection_report_path, *, zero_strength=False):
    """Export only an admitted successful candidate, or its explicit zero control.

    Destination publication is exclusive and atomic. Base tensor and asset
    payloads stream with a 4 MiB buffer; only config.json is intentionally edited.
    """
    if type(zero_strength) is not bool:
        raise ExportError('zero_strength must be boolean')
    source=Path(source_export).resolve();destination=Path(destination).absolute()
    if destination.exists() or destination.is_symlink():raise FileExistsError(destination)
    if destination.resolve().is_relative_to(source):raise ExportError('Destination must be outside source export')
    identity=verify_export(source,require_unmodified=True)
    topology=preflight_export(source)
    receipt,entries=_source_entries(source)
    spec,spec_sha=_json(edit_spec_path,limit=16*1024**2)
    _validate_spec(spec,receipt,entries)
    if spec.get('force_requantize',False):raise ExportError('Native edits must not request re-quantization')
    report,report_sha=_json(selection_report_path)
    selected=_admit_selection(spec,report,identity,topology['projection_paths'])
    config,config_sha=_json(source/'config.json')
    if 'sha256:'+config_sha!=next(a['digest'] for a in receipt['assets'] if a['name']=='config.json'):
        raise ExportError('Source configuration changed after verification')
    if config.get('architectures')!=['Qwen3_5ForConditionalGeneration'] or 'mlx_abliterate' in config:
        raise ExportError('Expected original Qwen architecture without native edit extension')
    original_config_canonical_sha=_sha(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    if any(entry['name'].startswith('mlx_abliterate.') for entry in entries):
        raise ExportError('Source already contains native extension tensors')
    extras,targets={},[]
    for index,path in enumerate(sorted(spec['projections'])):
        edit=spec['projections'][path]
        name=TENSOR_PREFIX+str(index)
        extras[name]=_vector_bytes(edit['direction'])
        targets.append({'path':path.replace('model.language_model.layers.','model.layers.',1),
                        'direction_tensor':name,'strength':0. if zero_strength else float(edit['strength'])})
    if selected.get('direction_mode')=='global' and len(set(extras.values()))!=1:
        raise ExportError('Global selected candidate must use one identical direction')
    extension={'version':1,'method':METHOD,'base_content_fingerprint':identity['content_fingerprint'],'targets':targets}
    config['architectures']=[ARCHITECTURE];config['mlx_abliterate']=extension
    config_raw=(json.dumps(config,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    index,_=_json(source/'model.safetensors.index.json')
    weight_map=dict(index['weight_map'])
    filenames=set(weight_map.values())
    extra_filename='mlx-abliterate-directions.safetensors'
    if extra_filename in filenames:raise ExportError('Reserved native shard already present')
    destination.parent.mkdir(parents=True,exist_ok=True)
    needed=sum((source/name).stat().st_size for name in filenames)+sum(a['size'] for a in receipt['assets'])+sum(map(len,extras.values()))+32*1024**2
    if shutil.disk_usage(destination.parent).free<needed:raise ExportError('Insufficient disk space for native export')
    source_files=filenames|{a['name'] for a in receipt['assets']}|{'Modelfile','.mlx-abliterate/create-options.json','.mlx-abliterate/export-report.json','model.safetensors.index.json'}
    source_stats={name:(source/name).stat() for name in source_files}
    staging=Path(tempfile.mkdtemp(prefix='.'+destination.name+'-',dir=destination.parent))
    try:
        for name in sorted(filenames):
            with (source/name).open('rb') as origin,(staging/name).open('xb') as target:
                _copy_range(origin,target,0,(source/name).stat().st_size)
                target.flush();os.fsync(target.fileno())
        _write_directions(staging/extra_filename,extras)
        records=list(receipt['tensors'])
        for name,raw in extras.items():
            weight_map[name]=extra_filename
            records.append({'name':name,'original_name':name,'dtype':'F32','shape':[5120],
                            'size':len(raw),'payload_sha256':_sha(raw),'shard':extra_filename})
        assets=[]
        for asset in receipt['assets']:
            name=asset['name']
            if name=='config.json':
                with (staging/name).open('xb') as target:target.write(config_raw)
                assets.append({**asset,'size':len(config_raw),'digest':'sha256:'+_sha(config_raw)})
            else:
                with (source/name).open('rb') as origin,(staging/name).open('xb') as target:
                    digest=_copy_range(origin,target,0,asset['size'])
                if 'sha256:'+digest!=asset['digest']:raise ExportError('Source asset changed while copying')
                assets.append(asset)
        for name in ('Modelfile',):
            with (source/name).open('rb') as origin,(staging/name).open('xb') as target:
                _copy_range(origin,target,0,(source/name).stat().st_size)
        _dump(staging/'model.safetensors.index.json',{'metadata':{'total_size':sum(r['size'] for r in records)},'weight_map':weight_map})
        receipts=staging/'.mlx-abliterate';receipts.mkdir()
        shutil.copyfile(source/'.mlx-abliterate/create-options.json',receipts/'create-options.json')
        (receipts/'edit-spec.json').write_text(json.dumps(spec,separators=(',',':'),allow_nan=False)+'\n')
        _dump(receipts/'export-report.json',{**receipt,'format':'modelopt-nvfp4-native-directional-v1',
                    'modified':True,'validated_import':False,'tensors':records,'assets':assets})
        native={'schema_version':1,'architecture':ARCHITECTURE,'source_identity':identity,
                'source_tensors':[_tensor_identity(r) for r in receipt['tensors']],
                'source_assets':receipt['assets'],'source_config_sha256':config_sha,
                'source_config_canonical_sha256':original_config_canonical_sha,
                'source_options_sha256':_sha((source/'.mlx-abliterate/create-options.json').read_bytes()),
                'source_modelfile_sha256':_sha((source/'Modelfile').read_bytes()),
                'edit_spec_sha256':spec_sha,'selection_report_sha256':report_sha,'selected':selected,
                'zero_control':zero_strength,'extension':extension,
                'direction_tensors':[_tensor_identity(r) for r in records if r['name'] in extras],
                'base_weights_unchanged':True,'requires_custom_runtime':True,'quality_validated':False,
                'normalization':'Stored float32 spec vectors; custom runtime normalizes once as dynamic wrapper.'}
        # The receipt stores both original input-spec hash and the exact archived
        # serialization hash, since JSON formatting is intentionally independent.
        native['archived_edit_spec_sha256']=_sha((receipts/'edit-spec.json').read_bytes())
        _dump(receipts/'native-edit-report.json',native)
        for name,previous in source_stats.items():
            current=(source/name).stat()
            if (current.st_ino,current.st_size,current.st_mtime_ns)!=(previous.st_ino,previous.st_size,previous.st_mtime_ns):
                raise ExportError('Source changed during native export')
        verified=verify_native_export(staging)
        _publish_directory(staging,destination)
        return verified
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise


def verify_native_export(export_directory, source_export=None):
    """Verify bytes, extension topology, all preserved-base receipts and vectors."""
    root=Path(export_directory).resolve()
    identity=verify_export(root)
    native,_=_json(root/'.mlx-abliterate/native-edit-report.json')
    receipt,_=_json(root/'.mlx-abliterate/export-report.json')
    config,_=_json(root/'config.json')
    if (native.get('schema_version')!=1 or native.get('architecture')!=ARCHITECTURE or
            type(native.get('zero_control')) is not bool or native.get('base_weights_unchanged') is not True or
            native.get('requires_custom_runtime') is not True or
            config.get('architectures')!=[ARCHITECTURE] or identity['modified'] is not True):
        raise ExportError('Native export requires its explicit custom architecture')
    extension=config.get('mlx_abliterate')
    if (not isinstance(extension,dict) or extension!=native.get('extension') or
            type(extension.get('version')) is not int or extension.get('version')!=1 or extension.get('method')!=METHOD or
            extension.get('base_content_fingerprint')!=native['source_identity']['content_fingerprint']):
        raise ExportError('Native edit configuration/receipt mismatch')
    restored_config={k:v for k,v in config.items() if k!='mlx_abliterate'}
    restored_config['architectures']=['Qwen3_5ForConditionalGeneration']
    if _sha(json.dumps(restored_config,sort_keys=True,separators=(',',':'),allow_nan=False).encode())!=native['source_config_canonical_sha256']:
        raise ExportError('Native export altered unrelated base configuration')
    targets=extension.get('targets')
    if not isinstance(targets,list) or not 1<=len(targets)<=128:
        raise ExportError('Require 1–128 native targets')
    records={r['name']:_tensor_identity(r) for r in receipt['tensors']}
    originals={r['name']:r for r in native['source_tensors']}
    directions={r['name']:r for r in native['direction_tensors']}
    if (len(originals)!=len(native['source_tensors']) or len(directions)!=len(targets) or
            set(originals)&set(directions) or set(records)!=set(originals)|set(directions)):
        raise ExportError('Unexpected native tensor inventory')
    for name,record in {**originals,**directions}.items():
        if records[name]!=record:raise ExportError('Preserved base or direction tensor identity differs')
    if (not all(name.startswith(TENSOR_PREFIX) for name in directions) or
            any(name.startswith('mlx_abliterate.') for name in originals)):
        raise ExportError('Invalid native direction names')
    source_assets={a['name']:a for a in native['source_assets']}
    assets={a['name']:a for a in receipt['assets']}
    if set(source_assets)!=set(assets) or any(a!=assets[name] for name,a in source_assets.items() if name!='config.json'):
        raise ExportError('Native export changed a non-config asset')
    for filename,field in (('Modelfile','source_modelfile_sha256'),('.mlx-abliterate/create-options.json','source_options_sha256')):
        if _sha((root/filename).read_bytes())!=native[field]:raise ExportError('Native export changed runtime metadata')
    spec_path=root/'.mlx-abliterate/edit-spec.json'
    if _sha(spec_path.read_bytes())!=native['archived_edit_spec_sha256']:raise ExportError('Archived edit specification changed')
    spec,_=read_edit_spec(spec_path)
    index,_=_json(root/'model.safetensors.index.json')
    seen_paths,seen_directions=set(),set()
    for target in targets:
        path=target.get('path');name=target.get('direction_tensor')
        if not isinstance(path,str) or not path.startswith('model.layers.') or path in seen_paths or name not in directions or name in seen_directions:
            raise ExportError('Duplicate or invalid native target')
        original=path.replace('model.layers.','model.language_model.layers.',1)
        if original not in spec['projections'] or original+'.weight' not in originals:
            raise ExportError('Native target does not name an admitted base projection')
        strength=_validate_strength(target.get('strength'))
        expected=0. if native['zero_control'] else spec['projections'][original]['strength']
        if strength!=expected:raise ExportError('Native target strength mismatch')
        descriptor=directions[name]
        if descriptor['dtype']!='F32' or descriptor['shape']!=[5120] or descriptor['size']!=20480:
            raise ExportError('Native direction must be float32 width5120')
        shard=root/index['weight_map'][name]
        header=read_safetensors_header(shard)
        with shard.open('rb') as stream:
            base=8+struct.unpack('<Q',stream.read(8))[0]
            lo,hi=header[name]['data_offsets'];stream.seek(base+lo);raw=stream.read(hi-lo)
        expected_raw=_vector_bytes(spec['projections'][original]['direction'])
        if raw!=expected_raw:raise ExportError('Native direction payload differs from selected specification')
        seen_paths.add(path);seen_directions.add(name)
    if len(seen_paths)!=len(spec['projections']):raise ExportError('Missing native projection target')
    if source_export is not None:
        source_identity=verify_export(source_export,require_unmodified=True)
        if source_identity!=native['source_identity']:raise ExportError('Original source identity mismatch')
        source_receipt,_=_json(Path(source_export)/'.mlx-abliterate/export-report.json')
        if {r['name']:_tensor_identity(r) for r in source_receipt['tensors']}!=originals:
            raise ExportError('Original source tensors differ from preserved receipts')
        if source_receipt['assets']!=native['source_assets']:
            raise ExportError('Original source assets differ from preserved receipts')
        for filename,field in (('Modelfile','source_modelfile_sha256'),('.mlx-abliterate/create-options.json','source_options_sha256'),('config.json','source_config_sha256')):
            if _sha((Path(source_export)/filename).read_bytes())!=native[field]:
                raise ExportError('Original runtime/config metadata identity differs')
    return {'schema_version':1,'identity':identity,'source_identity':native['source_identity'],
            'source_identity_reverified':source_export is not None,'base_tensor_count':len(originals),
            'direction_tensor_count':len(directions),'extra_payload_bytes':sum(r['size'] for r in directions.values()),
            'zero_control':native['zero_control'],'architecture':ARCHITECTURE,'base_weights_unchanged':True,
            'requires_custom_runtime':True,'quality_validated':False,'imported':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path)
    parser.add_argument('--edit-spec',type=Path,required=True)
    parser.add_argument('--selection-report',type=Path,required=True)
    parser.add_argument('--zero-strength',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.is_symlink():raise FileExistsError(args.output)
    report=export_native_edits(args.source,args.destination,args.edit_spec,args.selection_report,zero_strength=args.zero_strength)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
