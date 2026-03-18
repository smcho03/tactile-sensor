"""
sanity_01_flat_mirror.py
────────────────────────
h = 0 (평면 거울)
  - 반사면 세기 균일 확인
  - CMOS 세기 / 위상 확인
출력: 1x3
"""
import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from me_0318 import forward_propagate
from sanity_utils import plot_1x3

# ── 파라미터 ──────────────────────────────────────────────────
wavelength = 500e-9;  A0, t = 1.0, 0.0
n1 = 1.0;  n2_complex = complex(0.97112, 1.8737)
r_width    = 500e-6        # 반사면 실제 폭
N          = 150            # 픽셀 개수

pixel_size = r_width / N   # 픽셀 간격 자동 계산

x_coords = np.linspace(1, N, N) * pixel_size       # x > 0
y_coords = np.linspace(-N//2, N//2-1, N) * pixel_size
h_ref    = np.zeros((N, N))

# 중심거리 10mm 고정
x_center = (x_coords[0] + x_coords[-1]) / 2
x_cmos   = x_center + 10e-3

z_center = (x_coords[0] + x_coords[-1]) / 2

cmos_width    = 600e-6        # CMOS 실제 폭
N_cmos     = 150          # CMOS 픽셀 개수

cmos_pixel = cmos_width / N_cmos   # CMOS 픽셀 간격 자동 계산

y_prime = np.linspace(-cmos_width/2, cmos_width/2, N_cmos)
z_prime = np.linspace(z_center - cmos_width/2, z_center + cmos_width/2, N_cmos)

# ── 실행 ──────────────────────────────────────────────────────
def run(out_dir: Path):
    print("[01] Flat mirror  h=0 ...")
    res = forward_propagate(
        h_ref, x_coords, y_coords, y_prime, z_prime,
        wavelength, A0, t, n1, n2_complex, x_cmos)

    # 터미널 정량 출력
    I_ref   = res['I_CMOS']
    R_tilde = res['R_tilde']
    dx = float(x_coords[1]-x_coords[0])
    dy = float(y_coords[1]-y_coords[0])
    E_in  = float(np.sum(np.abs(res['U_in'])**2))  * dx * dy
    E_ref = float(np.sum(np.abs(res['U_ref'])**2)) * dx * dy
    print(f"  |R|^2 mean   : {float(np.mean(np.abs(R_tilde)**2)):.4f}")
    print(f"  E_in         : {E_in:.4e}")
    print(f"  E_ref        : {E_ref:.4e}  ({E_ref/E_in*100:.2f}% of E_in)")
    print(f"  E_absorbed   : {(E_in-E_ref):.4e}  ({(E_in-E_ref)/E_in*100:.2f}%)")
    print(f"  I_CMOS CoV   : {I_ref.std()/I_ref.mean():.4f}")

    plot_1x3(
        title      = 'Sanity 01 – Flat Mirror  h = 0',
        h_nm       = h_ref * 1e9,
        x_coords   = x_coords,
        y_coords   = y_coords,
        I          = I_ref,
        phi        = np.angle(res['U_CMOS']),
        y_prime    = y_prime,
        z_prime    = z_prime,
        out_path   = out_dir / 'sanity_01_flat_mirror.png',
    )

if __name__ == '__main__':
    out_dir = Path(__file__).parent / 'sanity_results'
    out_dir.mkdir(exist_ok=True)
    run(out_dir)