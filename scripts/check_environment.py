"""Validate all pinned runtime distributions and their active dependency requirements."""
import importlib.metadata as md,json
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
ROOT=Path(__file__).resolve().parents[1]
def main():
    pins={};failures=[]
    for line in (ROOT/'requirements-lock.txt').read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            n,v=line.split('==');pins[canonicalize_name(n)]=v
    for name,version in pins.items():
        try:d=md.distribution(name)
        except md.PackageNotFoundError:
            failures.append(f'Missing {name}');continue
        if d.version!=version:failures.append(f'{name}: expected {version}, found {d.version}')
        for expr in d.requires or []:
            req=Requirement(expr)
            if req.marker and not req.marker.evaluate({'extra':''}):continue
            dep=canonicalize_name(req.name)
            if dep not in pins:failures.append(f'{name}: unlocked dependency {dep}')
            elif pins[dep] not in req.specifier:failures.append(f'{name}: {dep}=={pins[dep]} violates {req.specifier}')
    print(json.dumps({'pinned_packages':len(pins),'failures':failures},indent=2))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
