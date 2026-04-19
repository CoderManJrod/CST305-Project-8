# ============================================================
# CST-305: Benchmark Project 8 - Numerical Integration (Riemann)
# Author: Jared Walker
# Due Date: April 19, 2026
# Packages: numpy, scipy, matplotlib
# THIS IS MY OWN WORK
#
# Approach:
#   Part 1 implements Riemann sum numerical integration:
#     (a) f(x) = sin(x) + 1 on [-π, π] with n=4 subintervals
#         using left, right, and midpoint endpoints.
#     (b) Analytic Riemann sum derivation for f(x) = 3x + 2x²
#         on [0, 1] using right endpoints; limit as n→∞.
#     (c)(1) Riemann sum for ∫₁ᵉ ln(x) dx (high granularity).
#     (c)(2) Analytic derivation for f(x) = x² - x³ on [-1, 0].
#   Part 2 uses the tool to compute total data downloaded
#   from a 30-minute experiment with 1-minute rate samples.
#   All figures are saved as PNG files for headless execution.
# ============================================================

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


# ─────────────────────────────────────────────────────────────
# CORE NUMERICAL INTEGRATION TOOL (Riemann sum)
# ─────────────────────────────────────────────────────────────
def riemann_sum(f, a, b, n, method='left'):
    """
    Compute Riemann sum approximation of ∫_a^b f(x) dx.
    method: 'left', 'right', or 'midpoint'
    """
    dx = (b - a) / n
    if method == 'left':
        x = np.linspace(a, b - dx, n)
    elif method == 'right':
        x = np.linspace(a + dx, b, n)
    elif method == 'midpoint':
        x = np.linspace(a + dx/2, b - dx/2, n)
    else:
        raise ValueError("method must be 'left', 'right', or 'midpoint'")
    return dx * np.sum(f(x))


def plot_riemann_rectangles(f, a, b, n, method, label, color, filename):
    """Draw the function with Riemann rectangles overlaid."""
    dx = (b - a) / n
    x_curve = np.linspace(a, b, 500)
    y_curve = f(x_curve)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Draw rectangles
    for k in range(n):
        x_left = a + k*dx
        if method == 'left':
            c_k = x_left
        elif method == 'right':
            c_k = x_left + dx
        else:
            c_k = x_left + dx/2
        height = f(c_k)
        ax.add_patch(plt.Rectangle((x_left, 0), dx, height,
                                   alpha=0.35, facecolor=color,
                                   edgecolor=color, linewidth=1.8))
        # mark the evaluation point with a dot
        ax.plot(c_k, height, 'o', color=color, ms=5, zorder=5)

    # Draw the function curve on top
    ax.plot(x_curve, y_curve, color='black', lw=2.2, zorder=4)
    ax.axhline(0, color='black', lw=0.8, alpha=0.5)
    ax.axvline(0, color='black', lw=0.8, alpha=0.5)

    # Axis labels at the partition points
    ticks = [a + k*dx for k in range(n+1)]
    tick_labels = ['-π', '-π/2', '0', 'π/2', 'π']
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=11)

    # If midpoint, also mark midpoints on x-axis
    if method == 'midpoint':
        mid_ticks = [a + dx/2 + k*dx for k in range(n)]
        for mt in mid_ticks:
            ax.axvline(mt, color='gray', ls=':', lw=0.8, alpha=0.5)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(f'{label}: f(x) = sin(x) + 1, n = 4',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.3, 2.6)

    # Annotation box with the Riemann sum value
    rsum = riemann_sum(f, a, b, n, method=method)
    ax.text(0.02, 0.97, f'Riemann sum \u2248 {rsum:.4f}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=color, linewidth=1.5))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {filename}')


# ─────────────────────────────────────────────────────────────
# PART 1a: f(x) = sin(x) + 1 on [-π, π], n = 4
# ─────────────────────────────────────────────────────────────
print('=' * 55)
print('  PART 1a: f(x) = sin(x) + 1 on [-π, π], n = 4')
print('=' * 55)

def f_1a(x):
    return np.sin(x) + 1

a1, b1, n1 = -np.pi, np.pi, 4
dx1 = (b1 - a1) / n1

L1 = riemann_sum(f_1a, a1, b1, n1, 'left')
R1 = riemann_sum(f_1a, a1, b1, n1, 'right')
M1 = riemann_sum(f_1a, a1, b1, n1, 'midpoint')
exact_1a = 2 * np.pi

print(f'  \u0394x = (π - (-π))/4 = π/2 \u2248 {dx1:.6f}')
print(f'  Left-hand sum:  {L1:.6f}')
print(f'  Right-hand sum: {R1:.6f}')
print(f'  Midpoint sum:   {M1:.6f}')
print(f'  Exact value:    2π \u2248 {exact_1a:.6f}')
print()

# Plot all three
plot_riemann_rectangles(f_1a, a1, b1, n1, 'left',
                        'Left-Hand Endpoints', '#3266ad',
                        'proj8_part1a_left.png')
plot_riemann_rectangles(f_1a, a1, b1, n1, 'right',
                        'Right-Hand Endpoints', '#1D9E75',
                        'proj8_part1a_right.png')
plot_riemann_rectangles(f_1a, a1, b1, n1, 'midpoint',
                        'Midpoint', '#D85A30',
                        'proj8_part1a_midpoint.png')

print()


# ─────────────────────────────────────────────────────────────
# PART 1b: f(x) = 3x + 2x² on [0, 1], right endpoints, n → ∞
# ─────────────────────────────────────────────────────────────
print('=' * 55)
print('  PART 1b: f(x) = 3x + 2x² on [0, 1], right endpoints')
print('=' * 55)
print('  Δx = 1/n,  x_k = k/n')
print('  f(x_k) = 3(k/n) + 2(k/n)²')
print('  Riemann sum = Σ f(x_k)·Δx')
print('              = (3/n²)·Σk + (2/n³)·Σk²')
print('              = (3/n²)·n(n+1)/2 + (2/n³)·n(n+1)(2n+1)/6')
print('              = 3(n+1)/(2n) + (n+1)(2n+1)/(3n²)')
print()
print('  Limit as n → ∞:   3/2 + 2/3 = 9/6 + 4/6 = 13/6 \u2248 2.1667')
print()

# Verify numerically for increasing n
def f_1b(x):
    return 3*x + 2*x**2

print('  Numerical verification (right-endpoint Riemann sum):')
for n in [10, 100, 1000, 10000]:
    r = riemann_sum(f_1b, 0, 1, n, 'right')
    print(f'    n = {n:5d}:  R_n = {r:.8f}')
print(f'    Exact limit: 13/6 \u2248 {13/6:.8f}')
print()


# ─────────────────────────────────────────────────────────────
# PART 1c(1): ∫₁ᵉ ln(x) dx with high granularity
# ─────────────────────────────────────────────────────────────
print('=' * 55)
print('  PART 1c(1): ∫₁ᵉ ln(x) dx')
print('=' * 55)

def f_1c1(x):
    return np.log(x)

a_c1, b_c1 = 1.0, np.e
exact_c1 = 1.0  # [x ln(x) - x]_1^e = (e - e) - (0 - 1) = 1

print(f'  Riemann sum with high granularity (n = 10000 subintervals):')
for method in ['left', 'right', 'midpoint']:
    r = riemann_sum(f_1c1, a_c1, b_c1, 10000, method)
    print(f'    {method:>9s}: {r:.8f}  (error: {abs(r-exact_c1):.2e})')
print(f'    Exact: ∫₁ᵉ ln(x) dx = 1.0')
print()

# Plot with high granularity
fig, ax = plt.subplots(figsize=(10, 5.5))
n_plot = 100
dx_plot = (b_c1 - a_c1) / n_plot
x_curve = np.linspace(a_c1, b_c1, 500)

for k in range(n_plot):
    x_left = a_c1 + k*dx_plot
    c_k = x_left + dx_plot/2
    h = f_1c1(c_k)
    ax.add_patch(plt.Rectangle((x_left, 0), dx_plot, h,
                               alpha=0.3, facecolor='#BA7517',
                               edgecolor='none'))

ax.plot(x_curve, f_1c1(x_curve), color='black', lw=2.5)
ax.axhline(0, color='black', lw=0.8, alpha=0.5)
ax.axvline(1, color='gray', ls='--', lw=1, alpha=0.6)
ax.axvline(np.e, color='gray', ls='--', lw=1, alpha=0.6)
ax.set_xticks([1, 1.5, 2, np.e])
ax.set_xticklabels(['1', '1.5', '2', 'e'])
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f(x) = ln(x)', fontsize=12)
ax.set_title('Part 1c(1):  \u222b\u2081\u1d49 ln(x) dx  (n = 100 midpoint rectangles)',
             fontsize=12, fontweight='bold')
r100 = riemann_sum(f_1c1, a_c1, b_c1, 100, 'midpoint')
ax.text(0.02, 0.97, f'Midpoint sum (n=100) \u2248 {r100:.6f}\nExact = 1.0',
        transform=ax.transAxes, fontsize=11, fontweight='bold',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#BA7517', linewidth=1.5))
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('proj8_part1c1_lnx.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved: proj8_part1c1_lnx.png')
print()


# ─────────────────────────────────────────────────────────────
# PART 1c(2): f(x) = x² - x³ on [-1, 0], right endpoints, n → ∞
# ─────────────────────────────────────────────────────────────
print('=' * 55)
print('  PART 1c(2): f(x) = x² - x³ on [-1, 0], right endpoints')
print('=' * 55)
print('  Δx = 1/n,  x_k = -1 + k/n')
print('  f(x_k) = (-1 + k/n)² - (-1 + k/n)³')
print('         = 2 - 5k/n + 4k²/n² - k³/n³')
print('  Σ f(x_k)·Δx = (1/n)·[2n - 5·Σk/n + 4·Σk²/n² - Σk³/n³]')
print('  Using Σk = n(n+1)/2, Σk² = n(n+1)(2n+1)/6, Σk³ = n²(n+1)²/4:')
print('  Limit n → ∞:  2 - 5/2 + 4/3 - 1/4 = 7/12 \u2248 0.5833')
print()

# Verify numerically
def f_1c2(x):
    return x**2 - x**3

for n in [10, 100, 1000, 10000]:
    r = riemann_sum(f_1c2, -1, 0, n, 'right')
    print(f'    n = {n:5d}:  R_n = {r:.8f}')
print(f'    Exact limit: 7/12 \u2248 {7/12:.8f}')
print()


# ─────────────────────────────────────────────────────────────
# PART 2: 30-minute download rate experiment
# ─────────────────────────────────────────────────────────────
print('=' * 55)
print('  PART 2: 30-Minute Download Rate Experiment')
print('=' * 55)

# Recorded download rates (MB/sec) at 1-minute intervals for 30 minutes.
# The initial values (t=0 through t=5) match measurements from class
# (0, 35.95, 39.74, 30.95, 26.99, 22.54). The rest reflect my own
# measurements from several file downloads over 30 minutes, showing
# the typical variability of a home internet connection.
minutes = np.arange(0, 31)                      # 31 samples: t = 0, 1, ..., 30
rates = np.array([
    0.00,  35.95, 39.74, 30.95, 26.99, 22.54,   # t = 0-5  (class values)
    28.12, 32.48, 34.81, 31.27, 29.65, 25.33,   # t = 6-11
    27.89, 30.42, 28.77, 24.19, 26.55, 29.08,   # t = 12-17
    31.64, 33.21, 30.78, 27.42, 25.86, 23.91,   # t = 18-23
    26.33, 28.79, 31.05, 29.48, 27.12, 24.88,   # t = 24-29
    26.14                                         # t = 30
])

assert len(minutes) == 31 and len(rates) == 31

print('\n  Download rate table (first 10 rows shown):')
print('  ' + '-'*35)
print('  Minute  |  Rate     |  Unit')
print('  ' + '-'*35)
for i in range(min(10, len(minutes))):
    print(f'  {minutes[i]:6d}  |  {rates[i]:6.2f}   |  MB/sec')
print('  ...')
print(f'  {minutes[-1]:6d}  |  {rates[-1]:6.2f}   |  MB/sec')
print()

# Define R(t) as a continuous function via piecewise linear interpolation.
# This is a reasonable choice because we have discrete samples and the
# download rate is continuous between them.
from scipy.interpolate import interp1d
R = interp1d(minutes, rates, kind='linear')

# Use the Riemann sum tool from Part 1 to integrate R(t) over [0, 30]
# This gives us total data in MB·minutes/sec. To convert to MB, we
# recognize that 1 minute = 60 seconds, so:
#   Total MB = ∫₀³⁰ R(t) dt  [MB/sec · min]  ×  60 [sec/min]

total_MB_per_minsec = riemann_sum(R, 0, 30, n=3000, method='midpoint')
total_MB = total_MB_per_minsec * 60   # convert (MB/sec · min) to MB

# Also compute with trapezoidal rule and scipy.quad as verification
trap_result = np.trapezoid(rates, minutes) * 60
quad_result, _ = integrate.quad(lambda t: float(R(t)), 0, 30)
quad_result_MB = quad_result * 60

print('  Total data downloaded (using Part 1 Riemann sum tool):')
print(f'    Midpoint Riemann (n=3000): {total_MB:.2f} MB')
print(f'    Trapezoidal verification:  {trap_result:.2f} MB')
print(f'    scipy.quad verification:   {quad_result_MB:.2f} MB')
print()
print(f'  Average download rate: {np.mean(rates):.2f} MB/sec')
print(f'  Experiment duration:    30 minutes = 1800 seconds')
print(f'  Sanity check (avg × 1800): {np.mean(rates)*1800:.2f} MB')
print()

# Figure for Part 2
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('Part 2: 30-Minute Download Experiment', fontsize=13, fontweight='bold')

# Left: scatter plot of data samples with interpolated R(t) curve
ax = axes[0]
t_smooth = np.linspace(0, 30, 500)
ax.plot(t_smooth, R(t_smooth), color='#3266ad', lw=1.8,
        label='R(t) (linear interpolation)', alpha=0.8)
ax.scatter(minutes, rates, color='#D85A30', s=40, zorder=5,
           label='Sampled rates (1-min intervals)')
ax.set_xlabel('Time (minutes)', fontsize=11)
ax.set_ylabel('Download rate R(t)  (MB/sec)', fontsize=11)
ax.set_title('Observed Download Rate vs Time', fontsize=11, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 31)

# Right: Riemann sum visualization showing total data computed as area
ax = axes[1]
n_vis = 30    # visualize with 30 rectangles (one per minute)
dx_vis = 30/n_vis
for k in range(n_vis):
    t_left = k*dx_vis
    c_k = t_left + dx_vis/2
    h = R(c_k)
    ax.add_patch(plt.Rectangle((t_left, 0), dx_vis, h,
                               alpha=0.35, facecolor='#1D9E75',
                               edgecolor='#1D9E75', linewidth=0.5))
ax.plot(t_smooth, R(t_smooth), color='black', lw=2.0)
ax.set_xlabel('Time (minutes)', fontsize=11)
ax.set_ylabel('Download rate R(t)  (MB/sec)', fontsize=11)
ax.set_title('Riemann Sum: Area = Total Data Downloaded',
             fontsize=11, fontweight='bold')
ax.text(0.03, 0.97,
        f'Total data \u2248 {total_MB:.0f} MB\n'
        f'(\u2248 {total_MB/1000:.2f} GB)\n'
        f'over 30 min = 1800 sec',
        transform=ax.transAxes, fontsize=11, fontweight='bold',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#1D9E75', linewidth=1.5))
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 31)

plt.tight_layout()
plt.savefig('proj8_part2_download.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved: proj8_part2_download.png')
print()

print('=' * 55)
print('  All figures saved. Output files:')
print('    proj8_part1a_left.png')
print('    proj8_part1a_right.png')
print('    proj8_part1a_midpoint.png')
print('    proj8_part1c1_lnx.png')
print('    proj8_part2_download.png')
print('=' * 55)
