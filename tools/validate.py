"""Goodness of fit of every distribution in the random package, with SciPy.

Builds and runs test/dump.goose, which writes n draws per case, then tests
each case against SciPy's distribution:

* continuous: Kolmogorov-Smirnov, and chi-squared on 200 bins equally
  likely under the target plus 8 tail bins of probability 1e-4 down to 1e-6
  on each side, so that a wrong tail shows even when KS cannot see it;
* discrete: chi-squared on the support, merging cells expected below 20.

    python tools/validate.py [n]        (default 1,000,000 draws per case)
    python tools/validate.py n case...  (only those cases)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOOSE = os.environ.get('GOOSE', 'goose')


def power_law(alpha, lo, hi):
    s = 1 - alpha

    class PL(stats.rv_continuous):
        def _cdf(self, x):
            if alpha == 1:
                return np.log(x / lo) / np.log(hi / lo)
            return (x ** s - lo ** s) / (hi ** s - lo ** s)

        def _ppf(self, q):
            if alpha == 1:
                return lo * np.exp(q * np.log(hi / lo))
            return (lo ** s + q * (hi ** s - lo ** s)) ** (1 / s)

    return PL(a=lo, b=hi)()


CONT = {
    'uniform': stats.uniform(),
    'uniform_dense': stats.uniform(),
    'normal': stats.norm(),
    'exponential': stats.expon(),
    'gamma_0.01': stats.gamma(0.01),
    'gamma_0.5': stats.gamma(0.5),
    'gamma_1.5': stats.gamma(1.5),
    'gamma_7': stats.gamma(7),
    'gamma_1000': stats.gamma(1000),
    'gamma_1e8': stats.gamma(1e8),
    'beta_0.2_0.3': stats.beta(0.2, 0.3),
    'beta_0.5_0.5': stats.beta(0.5, 0.5),
    'beta_2_5': stats.beta(2, 5),
    'beta_0.5_8': stats.beta(0.5, 8),
    'chi2_1': stats.chi2(1),
    'chi2_17': stats.chi2(17),
    't_1': stats.t(1),
    't_3.5': stats.t(3.5),
    't_50': stats.t(50),
    'f_4_9': stats.f(4, 9),
    'lognormal_0.3_1.2': stats.lognorm(1.2, scale=np.exp(0.3)),
    'cauchy': stats.cauchy(),
    'laplace': stats.laplace(),
    'logistic': stats.logistic(),
    'gumbel': stats.gumbel_r(),
    'weibull_0.7': stats.weibull_min(0.7),
    'weibull_3': stats.weibull_min(3),
    'rayleigh': stats.rayleigh(),
    'pareto_1.5': stats.pareto(1.5),
    'lomax_2.5': stats.lomax(2.5),
    'powerlaw_2.35': power_law(2.35, 0.08, 150.0),
    'powerlaw_1': power_law(1.0, 0.08, 150.0),
    'powerlaw_-2': power_law(-2.0, 0.08, 150.0),
    'triangular': stats.triang(1.5 / 4, loc=-1, scale=4),
    'wald_1_0.5': stats.invgauss(1 / 0.5, scale=0.5),
    'wald_3_20': stats.invgauss(3 / 20, scale=20),
    'vonmises_0.3': stats.vonmises(0.3),
    'vonmises_5': stats.vonmises(5),
    'vonmises_500': stats.vonmises(500),
}

DISC = {
    'poisson_0.5': stats.poisson(0.5),
    'poisson_7': stats.poisson(7),
    'poisson_12': stats.poisson(12),
    'poisson_1000': stats.poisson(1000),
    'poisson_1e7': stats.poisson(1e7),
    'binomial_30_0.2': stats.binom(30, 0.2),
    'binomial_200_0.4': stats.binom(200, 0.4),
    'binomial_100000_0.7': stats.binom(100000, 0.7),
    'binomial_1000_0.004': stats.binom(1000, 0.004),
    'geometric_0.5': stats.geom(0.5),
    'geometric_0.01': stats.geom(0.01),
    'negbinomial_2.5_0.3': stats.nbinom(2.5, 0.3),
    'hypergeom_40_60_30': stats.hypergeom(100, 40, 30),
    'hypergeom_5000_20000_8000': stats.hypergeom(25000, 5000, 8000),
    'zipf_1.7': stats.zipf(1.7),
    'logseries_0.95': stats.logser(0.95),
}


def chi2_p(obs, exp):
    keep = exp > 0
    obs, exp = obs[keep], exp[keep]
    chi = ((obs - exp) ** 2 / exp).sum()
    return chi, stats.chi2.sf(chi, len(obs) - 1)


def continuous(x, dist):
    ks = stats.kstest(x, dist.cdf).pvalue
    tails = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    qs = sorted(set(tails + list(np.linspace(0, 1, 201)[1:-1]) + [1 - t for t in tails]))
    edges = dist.ppf(qs)
    # A quantile below the smallest normal double (Gamma(0.01) puts 0.08% of
    # its mass there) has no usable edge: its cell merges into the next.
    keep = edges > 1e-300
    qs, edges = list(np.array(qs)[keep]), edges[keep]
    probs = np.diff([0.0] + qs + [1.0])
    obs = np.bincount(np.searchsorted(edges, x, side='right'), minlength=len(probs))
    _, p = chi2_p(obs.astype(float), probs * len(x))
    return ks, p


def discrete(k, dist):
    # The observed range; the mass outside it is a cell of its own.
    lo, hi = int(k.min()), int(k.max())
    support = np.arange(lo, hi + 1)
    pmf = dist.pmf(support)
    obs = np.bincount(k - lo, minlength=len(support)).astype(float)
    n = len(k)
    # Merge cells left to right until each expects at least 20.
    mo, me, o, e = [], [], 0.0, 0.0
    for oi, pi in zip(obs, pmf * n):
        o += oi
        e += pi
        if e >= 20:
            mo.append(o); me.append(e); o = e = 0.0
    if mo:
        mo[-1] += o; me[-1] += e
    # The mass outside [lo, hi].
    rest = n * max(0.0, 1.0 - pmf.sum())
    if rest >= 1:
        mo.append(0.0); me.append(rest)
    return chi2_p(np.array(mo), np.array(me))[1]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000
    only = set(sys.argv[2:])
    exe = os.path.join(ROOT, 'test', 'dump.exe' if os.name == 'nt' else 'dump')
    subprocess.run([GOOSE, 'build', os.path.join(ROOT, 'test', 'dump.goose')], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    worst = 1.0
    flagged = []
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([exe, d, str(n)], check=True)
        print(f'{n:,} draws per case; p-values (flagged below 1e-3)\n')
        print(f'{"case":28} {"KS":>9} {"chi2":>9}')
        for name, dist in CONT.items():
            if only and name not in only:
                continue
            x = np.fromfile(os.path.join(d, name + '.f64'), dtype='<f8')
            assert len(x) == n and np.all(np.isfinite(x)), name
            ks, p = continuous(x, dist)
            flag = ' <--' if min(ks, p) < 1e-3 else ''
            print(f'{name:28} {ks:9.4f} {p:9.4f}{flag}', flush=True)
            worst = min(worst, ks, p)
            if flag:
                flagged.append(name)
        for name, dist in DISC.items():
            if only and name not in only:
                continue
            k = np.fromfile(os.path.join(d, name + '.i64'), dtype='<i8')
            assert len(k) == n, name
            p = discrete(k, dist)
            flag = ' <--' if p < 1e-3 else ''
            print(f'{name:28} {"":>9} {p:9.4f}{flag}', flush=True)
            worst = min(worst, p)
            if flag:
                flagged.append(name)
    print(f'\nsmallest p {worst:.2g}; flagged: {", ".join(flagged) or "none"}')
    sys.exit(1 if flagged else 0)


if __name__ == '__main__':
    main()
