"""
run_all_sanity.py
─────────────────
모든 sanity check를 순서대로 실행하고
결과 이미지를 sanity_results/ 폴더에 저장한다.

실행 방법:
    python run_all_sanity.py
"""
import sys, time, traceback, shutil
from pathlib import Path
from datetime import datetime

# Windows stdout 버퍼링 해제
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent))

timestamp = datetime.now().strftime("%m%d_%H%M%S")
out_dir   = Path(__file__).parent / 'sanity_results' / timestamp
out_dir.mkdir(parents=True, exist_ok=True)

def section(name):
    print(f"\n{'='*52}", flush=True)
    print(f"  {name}", flush=True)
    print(f"{'='*52}", flush=True)

def run_safe(name, func, out_dir):
    print(f"\n[{name}] 실행 중 ...", flush=True)
    t = time.time()
    try:
        func(out_dir)
        print(f"[{name}] 완료  ({time.time()-t:.1f}s)", flush=True)
    except Exception:
        print(f"[{name}] ERROR:", flush=True)
        traceback.print_exc()

section("Sanity Check – All Tests")
print(f"  output -> {out_dir}", flush=True)

t_total = time.time()

from sanity_01_flat_mirror        import run as run_01
from sanity_02_near_field_energy  import run as run_02
from sanity_03_far_field          import run as run_03
from sanity_04_deformation_patterns import run as run_04

run_safe('01 flat_mirror',          run_01, out_dir)
run_safe('02 near_field_energy',    run_02, out_dir)
run_safe('03 far_field',            run_03, out_dir)
run_safe('04 deformation_patterns', run_04, out_dir)

section("완료")
files = sorted(out_dir.glob('*.png'))
print(f"  총 소요: {time.time()-t_total:.1f}s", flush=True)
print(f"  저장된 파일 ({len(files)}개):", flush=True)
for f in files:
    print(f"    {f.name}", flush=True)