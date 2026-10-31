"""NumPy's Generator on the cases of bench.goose: ns per value, filling
arrays of 2^20 values (so the per-call overhead is spread thin), the best
of at least 5 samples and 0.2 s. Default bit generator PCG64, and SFC64.

    python bench/bench_numpy.py [out.csv]
"""
import sys
import time

import numpy as np

N = 1 << 20


def time_it(f):
    best, total, samples = 1e30, 0.0, 0
    while samples < 5 or total < 0.2:
        t0 = time.perf_counter()
        f()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        total += dt
        samples += 1
    return best / N * 1e9


def cases(g, bg):
    out = np.empty(N)
    idx = np.arange(N)
    p = np.array([0.1, 0.2, 0.3, 0.15, 0.05, 0.2])
    return [
        ('raw', lambda: bg.random_raw(N)),
        ('uniform', lambda: g.random(N, out=out)),
        ('normal', lambda: g.standard_normal(N, out=out)),
        ('exponential', lambda: g.standard_exponential(N, out=out)),
        ('below 1000', lambda: g.integers(0, 1000, N)),
        ('below 3e18', lambda: g.integers(0, 3000000000000000000, N)),
        ('gamma 0.5', lambda: g.standard_gamma(0.5, N, out=out)),
        ('gamma 3', lambda: g.standard_gamma(3.0, N, out=out)),
        ('beta 2 5', lambda: g.beta(2.0, 5.0, N)),
        ('beta 0.5 0.5', lambda: g.beta(0.5, 0.5, N)),
        ('chi_squared 5', lambda: g.chisquare(5.0, N)),
        ('student_t 5', lambda: g.standard_t(5.0, N)),
        ('lognormal', lambda: g.lognormal(0.0, 1.0, N)),
        ('cauchy', lambda: g.standard_cauchy(N)),
        ('weibull 1.5', lambda: g.weibull(1.5, N)),
        ('von_mises 4', lambda: g.vonmises(0.0, 4.0, N)),
        ('poisson 3', lambda: g.poisson(3.0, N)),
        ('poisson 1000', lambda: g.poisson(1000.0, N)),
        ('binomial 20 0.3', lambda: g.binomial(20, 0.3, N)),
        ('binomial 1000 0.3', lambda: g.binomial(1000, 0.3, N)),
        ('geometric 0.01', lambda: g.geometric(0.01, N)),
        ('hypergeometric 500 500 200', lambda: g.hypergeometric(500, 500, 200, N)),
        ('alias 6', lambda: g.choice(6, N, p=p)),
        ('shuffle', lambda: g.shuffle(idx)),
    ]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'bench_numpy.csv'
    lines = []
    for name, bg in (('PCG64', np.random.PCG64(1)), ('SFC64', np.random.SFC64(1))):
        g = np.random.Generator(bg)
        for case, f in cases(g, bg):
            ns = time_it(f)
            print(f'{name} {case}: {ns:.3f} ns')
            lines.append(f'{name},{case},{ns}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'numpy {np.__version__}')


if __name__ == '__main__':
    main()
