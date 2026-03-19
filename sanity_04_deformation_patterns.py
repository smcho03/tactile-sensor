"""
sanity_04_deformation_patterns.py
──────────────────────────────────
6가지 변형 패턴 (각각 3x3 저장)
  1. 중앙 단일 픽셀
  2. 중앙 가우시안
  3. 대칭 두 점 가우시안
  4. y = k 직선 (한 행)
  5. x = k 직선 (한 열)
  6. 랜덤 변형 (band-limited, 공간 스무딩)
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from me_0318       import forward_propagate
from sanity_utils  import plot_3x3
from sanity_params import (wavelength, A0, t, n1, n2_complex,
                           x_coords, y_coords, x_cmos,
                           y_prime, z_prime, pixel_size)

N     = len(x_coords)
h_amp = -wavelength /8
sigma = 5 * pixel_size
h_ref = np.zeros((len(y_coords), N))

X_tmp, Y_tmp = np.meshgrid(x_coords, y_coords)

h1 = np.zeros_like(h_ref)
h1[len(y_coords)//2, N//2] = h_amp

h2 = h_amp * np.exp(
    -((X_tmp - x_coords[N//2])**2 + Y_tmp**2) / (2*sigma**2))

h3 = h_amp * (
    np.exp(-((X_tmp-x_coords[N//2])**2+(Y_tmp-y_coords[3*len(y_coords)//4])**2)/(2*sigma**2)) +
    np.exp(-((X_tmp-x_coords[N//2])**2+(Y_tmp-y_coords[  len(y_coords)//4])**2)/(2*sigma**2)))

h4 = np.zeros_like(h_ref); h4[len(y_coords)//2, :] = h_amp
h5 = np.zeros_like(h_ref); h5[:, N//2]              = h_amp

# ── 랜덤 변형: 저주파 band-limited noise ─────────────────────
# aliasing 없이 샘플링되려면 공간적으로 충분히 완만해야 함
# → 푸리에 공간에서 고주파 성분 제거 후 역변환
rng      = np.random.default_rng(seed=42)
raw      = rng.standard_normal((len(y_coords), N))          # white noise
F        = np.fft.fft2(raw)
freq_y   = np.fft.fftfreq(len(y_coords))                    # normalized freq
freq_x   = np.fft.fftfreq(N)
FY, FX   = np.meshgrid(freq_y, freq_x, indexing='ij')
f_cutoff = 0.1                                              # Nyquist의 10%만 통과
mask     = (np.sqrt(FY**2 + FX**2) < f_cutoff).astype(float)
h6_raw   = np.real(np.fft.ifft2(F * mask))
h6       = h_amp * h6_raw / np.abs(h6_raw).max()           # h_amp로 정규화

patterns = [
    ('single_pixel',  h1, '1. Single pixel (center)'),
    ('gaussian',      h2, '2. Gaussian (center)'),
    ('two_gaussians', h3, '3. Two Gaussians (y-sym)'),
    ('y_line',        h4, '4. y=k line (row)'),
    ('x_line',        h5, '5. x=k line (col / z=k)'),
    ('random',        h6, '6. Random (band-limited)'),
]


def run(out_dir: Path):
    print("[04] Deformation patterns ...", flush=True)
    res_ref = forward_propagate(
        h_ref, x_coords, y_coords, y_prime, z_prime,
        wavelength, A0, t, n1, n2_complex, x_cmos)

    for key, h_def, label in patterns:
        print(f"  [{key}] ...", flush=True)
        res_def = forward_propagate(
            h_def, x_coords, y_coords, y_prime, z_prime,
            wavelength, A0, t, n1, n2_complex, x_cmos)
        plot_3x3(
            title      = f'Sanity 04 – {label}',
            h_ref_nm   = h_ref * 1e9,
            h_def_nm   = h_def * 1e9,
            x_coords   = x_coords,
            y_coords   = y_coords,
            I_ref      = res_ref['I_CMOS'],
            I_def      = res_def['I_CMOS'],
            U_ref_cmos = res_ref['U_CMOS'],
            U_def_cmos = res_def['U_CMOS'],
            y_prime    = y_prime,
            z_prime    = z_prime,
            out_path   = out_dir / f'sanity_04_{key}.png',
        )

if __name__ == '__main__':
    out_dir = Path(__file__).parent / 'sanity_results'
    out_dir.mkdir(exist_ok=True)
    run(out_dir)