"""Draws docs/benchmark.png from the benchmark CSVs in bench/: nanoseconds
per value for Goose, NumPy, Rust and C++, a row of thin bars per case, on a
log scale.

    python tools/plot_bench.py
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, 'bench')

SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT2 = '#52514e'
GRID = '#e4e3df'

# (label, CSV, NumPy generator, colour, marker), in a fixed order.
SERIES = [
    ('Goose', 'bench_goose_base.csv', None, '#2a78d6', 'o'),
    ('NumPy', 'bench_numpy.csv', 'PCG64', '#eb6834', 's'),
    ('Rust rand_distr', 'bench_rust_base.csv', None, '#1baf7a', '^'),
    ('C++ <random>', 'bench_cpp.csv', None, '#eda100', 'D'),
]

CASES = [
    ('raw', 'raw 64-bit output'),
    ('uniform', 'uniform'),
    ('normal', 'normal'),
    ('exponential', 'exponential'),
    ('below 1000', 'below(1000)'),
    ('below 3e18', 'below(3e18)'),
    ('gamma 0.5', 'gamma, shape 0.5'),
    ('gamma 3', 'gamma, shape 3'),
    ('beta 2 5', 'beta(2, 5)'),
    ('beta 0.5 0.5', 'beta(0.5, 0.5)'),
    ('chi_squared 5', 'chi_squared(5)'),
    ('student_t 5', 'student_t(5)'),
    ('lognormal', 'lognormal'),
    ('cauchy', 'cauchy'),
    ('weibull 1.5', 'weibull(1.5)'),
    ('von_mises 4', 'von_mises(4)'),
    ('poisson 3', 'poisson(3)'),
    ('poisson 1000', 'poisson(1000)'),
    ('binomial 20 0.3', 'binomial(20, 0.3)'),
    ('binomial 1000 0.3', 'binomial(1000, 0.3)'),
    ('geometric 0.01', 'geometric(0.01)'),
    ('hypergeometric 500 500 200', 'hypergeometric'),
    ('alias 6', 'alias table, 6 outcomes'),
    ('shuffle', 'shuffle, per element'),
]


def load(name, gen):
    out = {}
    for row in csv.reader(open(os.path.join(BENCH, name))):
        if not row:
            continue
        if len(row) == 3:
            g, k, v = row
            if g != gen:
                continue
        else:
            k, v = row
        # Each library's default generator is its "raw" row.
        if k in ('raw xoshiro256++', 'raw mt19937_64'):
            k = 'raw'
        out[k] = float(v)
    return out


def fade(color, keep=0.3):
    """color mixed with the surface, keeping that fraction of it."""
    c = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    s = [int(SURFACE[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(keep * a + (1 - keep) * b):02x}' for a, b in zip(c, s))


def main():
    data = [load(f, g) for _, f, g, _, _ in SERIES]
    n = len(CASES)
    fig, ax = plt.subplots(figsize=(8.5, 9.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Cases from Goose's fastest to its slowest. Each is a row of four thin
    # bars, one per library, from the left edge of the axis, with a gap of
    # surface between neighbours. The fastest in each row is in full colour,
    # the others faded towards the surface.
    cases = sorted(CASES, key=lambda c: data[0][c[0]])
    lane = 0.2
    left = 0.5
    for s, (label, _, _, color, _) in enumerate(SERIES):
        faded = fade(color)
        for i, (key, _) in enumerate(cases):
            if key not in data[s]:
                continue
            best = min(d[key] for d in data if key in d)
            ax.barh(i + (s - 1.5) * lane, data[s][key] - left, left=left, height=lane,
                    color=color if data[s][key] == best else faded, zorder=3,
                    edgecolor=SURFACE, linewidth=1.0)
        ax.barh(0, 0, color=color, label=label)     # the legend's swatch

    ax.set_xscale('log')
    ticks = [0.5, 1, 2, 5, 10, 20, 50, 100, 200]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.set_xlim(0.5, 200)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([c[1] for c in cases])
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#f3f2ef', zorder=0, lw=0)
    ax.grid(axis='x', color=GRID, lw=0.8, zorder=1)
    ax.tick_params(colors=TEXT2, labelsize=9, length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.set_xlabel('nanoseconds per value (log scale; lower is faster)', color=TEXT2, fontsize=9)

    fig.suptitle('goose-random against NumPy, Rust and C++', x=0.02, ha='left',
                 color=TEXT, fontsize=13, fontweight='bold', y=0.985)
    fig.text(0.02, 0.945, 'Filling 2²⁰ values, single-threaded, pinned to one core of an i5-12600K. '
             'Fastest in full colour; a missing bar: no such sampler.', color=TEXT2, fontsize=8.5, ha='left')
    leg = ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False,
                    fontsize=9, handletextpad=0.3, columnspacing=1.4, borderaxespad=0.4)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(os.path.join(ROOT, 'docs'), exist_ok=True)
    path = os.path.join(ROOT, 'docs', 'benchmark.png')
    fig.savefig(path, facecolor=SURFACE)
    print(path)


if __name__ == '__main__':
    main()
