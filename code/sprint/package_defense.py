"""Build and seal the portable Windows Stage 2 runtime from the verified local assets."""

import argparse
import importlib.metadata as metadata
import json
import shutil
import sys
import sysconfig
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_STORED, ZipFile

from packaging.requirements import Requirement

from sprint.defense_artifacts import ASSETS, POLICY, SUBMITTED_SHA256, VERSION, verify_release
from sprint.score_push import fingerprint


MODULES = ('__init__', 'augment', 'bhasha_audit', 'build_bhasha_defense', 'data', 'defense',
           'defense_artifacts', 'encoder', 'live', 'muril', 'postprocess', 'runtime',
           'score_data', 'score_native', 'score_push', 'score_retrieval', 'train', 'train_gpu')
ROOT_PACKAGES = ('numpy', 'scipy', 'scikit-learn', 'onnxruntime', 'tokenizers', 'joblib',
                 'setuptools', 'cloudpickle')
LAUNCHERS = {'Run-Stage2.cmd': '--open --port 0', 'Run-Terminal.cmd': '--terminal',
             'Verify-Stage2.cmd': '--verify'}
EVIDENCE = ('policy.json', 'reproduction.json', 'template_fallback_holdout.json', 'notebook_reproduction.json')


def copy_dependencies(destination):
    site = Path(sysconfig.get_path('purelib')).resolve()
    pending = list(ROOT_PACKAGES)
    versions = {}
    while pending:
        package = pending.pop()
        dist = metadata.distribution(package)
        canonical = dist.metadata['Name'].lower().replace('_', '-')
        if canonical in versions:
            continue
        versions[canonical] = dist.version
        for requirement in map(Requirement, dist.requires or []):
            if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
                pending.append(requirement.name)
        for entry in dist.files or []:
            source = dist.locate_file(entry).resolve()
            if (not source.is_relative_to(site) or not source.is_file() or source.suffix == '.pyc'
                    or '__pycache__' in source.parts):
                continue
            target = destination / source.relative_to(site)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return dict(sorted(versions.items()))


def package_runtime(target):
    python = Path(sys.base_prefix)
    target.mkdir()
    tag = f'python{sys.version_info.major}{sys.version_info.minor}'
    for filename in ('python.exe', 'python3.dll', f'{tag}.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'LICENSE.txt'):
        shutil.copy2(python / filename, target / filename)
    shutil.copytree(python / 'DLLs', target / 'DLLs', ignore=shutil.ignore_patterns('*.ico', '__pycache__'))
    shutil.copytree(python / 'Lib', target / 'Lib',
                    ignore=shutil.ignore_patterns('site-packages', '__pycache__', 'test', 'tests', 'ensurepip',
                                                   'idlelib', 'tkinter', 'turtledemo', '*.pyc'))
    versions = copy_dependencies(target / 'Lib' / 'site-packages')
    (target / f'{tag}._pth').write_text('Lib\nDLLs\nLib\\site-packages\n..\\app\nimport site\n', encoding='ascii')
    return versions


def manifest(root, versions):
    files = {path.relative_to(root).as_posix(): {'bytes': path.stat().st_size, 'sha256': fingerprint(path)}
             for path in sorted(root.rglob('*')) if path.is_file() and path != root / 'manifest.json'}
    result = {'version': VERSION, 'platform': 'Windows x64', 'python': sys.version.split()[0],
              'submission_sha256': SUBMITTED_SHA256, 'packages': versions, 'files': files}
    (root / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return len(files)


def prepare(args):
    if sys.platform != 'win32' or sys.maxsize < 2**32 or args.output.exists():
        raise ValueError('Use a fresh output path from the 64-bit Windows project environment')
    proof = json.loads((args.model / 'verification.json').read_text(encoding='utf-8'))
    if proof['submission_sha256'] != SUBMITTED_SHA256 or proof['matching_predictions'] != 4912:
        raise ValueError('The model bundle does not reproduce the submitted Bhasha classifier')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix='stage2-assemble-') as scratch:
        release = Path(scratch) / 'release'
        release.mkdir()
        print('Assembling isolated Windows Python and model assets', flush=True)
        versions = package_runtime(release / 'runtime')
        model = release / 'model'
        model.mkdir()
        for name in (*ASSETS, 'verification.json'):
            shutil.copy2(args.model / name, model / name)
        app = release / 'app'
        (app / 'sprint').mkdir(parents=True)
        shutil.copy2(args.project / 'intent_model.py', app / 'intent_model.py')
        for name in MODULES:
            shutil.copy2(args.project / 'sprint' / f'{name}.py', app / 'sprint' / f'{name}.py')
        shutil.copytree(args.project / 'sprint' / 'defense_web', app / 'sprint' / 'defense_web')
        evidence = release / 'evidence'
        evidence.mkdir()
        for name in EVIDENCE:
            shutil.copy2(args.project / POLICY.parent / name, evidence / name)
        shutil.copy2(args.project / 'Bhasha-Stage2-Reproduction.ipynb', evidence / 'Bhasha-Stage2-Reproduction.ipynb')
        shutil.copy2(args.project / 'sprint' / 'STAGE2_RELEASE.md', release / 'START-HERE.md')
        for filename, mode in LAUNCHERS.items():
            command = ('@echo off\nsetlocal\ncd /d "%~dp0"\n'
                       '".\\runtime\\python.exe" -B -X utf8 -m sprint.defense --model ".\\model" '
                       f'--policy ".\\evidence\\policy.json" --release-root "." {mode} %*\nexit /b %errorlevel%\n')
            (release / filename).write_text(command, encoding='ascii', newline='\r\n')
        count = manifest(release, versions)
        release.rename(args.output)
    print(f'Prepared {count:,} files at {args.output}', flush=True)


def seal(args):
    if args.archive.exists():
        raise FileExistsError(args.archive)
    verify_release(args.output)
    proof = json.loads((args.output / 'manifest.json').read_text(encoding='utf-8'))
    rehearsal = json.loads(args.rehearsal.read_text(encoding='utf-8'))
    if rehearsal['passed'] is not True or rehearsal['timing']['model']['submission_sha256'] != SUBMITTED_SHA256:
        raise ValueError('The release must pass the local rehearsal on the verified model before sealing')
    shutil.copy2(args.rehearsal, args.output / 'evidence' / 'release_rehearsal.json')
    count = manifest(args.output, proof['packages'])
    temporary = args.archive.with_suffix('.partial.zip')
    with ZipFile(temporary, 'x', compression=ZIP_STORED) as bundle:
        for path in sorted(args.output.rglob('*')):
            if path.is_file():
                relative = path.relative_to(args.output)
                bundle.write(path, (Path('Anveshan-Stage2') / relative).as_posix())
    temporary.replace(args.archive)
    digest = fingerprint(args.archive)
    args.archive.with_suffix('.sha256').write_text(f'{digest}  {args.archive.name}\n', encoding='ascii')
    print(f'Sealed {count:,} files in {args.archive} ({args.archive.stat().st_size:,} bytes). SHA-256 {digest}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'seal'))
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--rehearsal', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        if args.model is None:
            parser.error('--model is required when preparing the release')
        prepare(args)
    else:
        if args.archive is None or args.rehearsal is None:
            parser.error('--archive and --rehearsal are required when sealing the release')
        seal(args)


if __name__ == '__main__':
    main()
