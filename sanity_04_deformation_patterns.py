"""
sanity_04_deformation_patterns.py
──────────────────────────────────
5가지 변형 패턴  (각각 3x3 저장)
  1. 중앙 단일 픽셀
  2. 중앙 가우시안
  3. 대칭 두 점 가우시안
  4. y = k 직선 (한 행)
  5. x = k 직선 (한 열)
"""
import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from me_0318 import forward_propagate
from sanity_utils import plot_3x3

wavelength = 500e-9;  A0, t = 1.0, 0.0
n1 = 1.0;  n2_complex = complex(0.97112, 1.8737)
pixel_size = 10e-6;  N = 32

x_coords = np.linspace(1, N, N) * pixel_size
y_coords = np.linspace(-N//2, N//2-1, N) * pixel_size

x_center = (x_coords[0] + x_coords[-1]) / 2
x_cmos   = x_center + 10e-3

N_cmos   = 48
y_span   = y_coords[-1] - y_coords[0]
z_center = (x_coords[0] + x_coords[-1]) / 2
z_span   = (x_coords[-1] - x_coords[0]) * 1.5
y_prime  = np.linspace(y_coords[0]  - y_span*0.25,
                       y_coords[-1] + y_span*0.25, N_cmos)
z_prime  = np.linspace(z_center - z_span/2,
                       z_center + z_span/2, N_cmos)

X_tmp, Y_tmp = np.meshgrid(x_coords, y_coords)
h_amp  = -wavelength / 8
sigma  = 3 * pixel_size
h_ref  = np.zeros((N, N))

# 5가지 패턴 정의
h1 = np.zeros((N, N));  h1[N//2, N//2] = h_amp

h2 = h_amp * np.exp(
    -((X_tmp-x_coords[N//2])**2 + Y_tmp**2) / (2*sigma**2))

h3 = h_amp * (
    np.exp(-((X_tmp-x_coords[N//2])**2+(Y_tmp-y_coords[3*N//4])**2)/(2*sigma**2)) +
    np.exp(-((X_tmp-x_coords[N//2])**2+(Y_tmp-y_coords[N//4]  )**2)/(2*sigma**2)))

h4 = np.zeros((N, N));  h4[N//2, :] = h_amp

h5 = np.zeros((N, N));  h5[:, N//2] = h_amp

patterns = [
    ('single_pixel',    h1, '1. Single pixel (center)'),
    ('gaussian',        h2, '2. Gaussian (center)'),
    ('two_gaussians',   h3, '3. Two Gaussians (y-sym)'),
    ('y_line',          h4, '4. y=k line (row)'),
    ('x_line',          h5, '5. x=k line (col / z=k)'),
]


def run(out_dir: Path):
    print("[04] Deformation patterns ...")

    # reference 한 번만 계산
    res_ref = forward_propagate(
        h_ref, x_coords, y_coords, y_prime, z_prime,
        wavelength, A0, t, n1, n2_complex, x_cmos)

    for key, h_def, label in patterns:
        print(f"  [{key}] ...")
        res_def = forward_propagate(
            h_def, x_coords, y_coords, y_prime, z_prime,
            wavelength, A0, t, n1, n2_complex, x_cmos)

        plot_3x3(
            title    = f'Sanity 04 – {label}',
            h_ref_nm = h_ref * 1e9,
            h_def_nm = h_def * 1e9,
            x_coords = x_coords,
            y_coords = y_coords,
            I_ref    = res_ref['I_CMOS'],
            I_def    = res_def['I_CMOS'],
            phi_ref  = np.angle(res_ref['U_CMOS']),
            phi_def  = np.angle(res_def['U_CMOS']),
            y_prime  = y_prime,
            z_prime  = z_prime,
            out_path = out_dir / f'sanity_04_{key}.png',
        )

if __name__ == '__main__':
    out_dir = Path(__file__).parent / 'sanity_results'
    out_dir.mkdir(exist_ok=True)
    run(out_dir)