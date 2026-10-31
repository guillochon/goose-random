// Writes samples for tools/validate.py: one file of raw little-endian f64s
// (or i64s, for the discrete distributions) per case, named by the case.
//
//   goose build test/dump.goose && test/dump <dir> <n>

import std;
import os;
import random from "..";

var dir: u8[>..] = [];
var n = 1000000;
var g = random::seed(20261031);

fn put(name: const u8[:], xs: f64[:]) {
    var path: u8[>..] = [];
    path.append(dir);
    path.append("/");
    path.append(name);
    path.append(".f64");
    let ok = write_file(path, bytes_of(xs));
    assert(ok);
}

fn puti(name: const u8[:], xs: i64[:]) {
    var path: u8[>..] = [];
    path.append(dir);
    path.append("/");
    path.append(name);
    path.append(".i64");
    let ok = write_file(path, bytes_of(xs));
    assert(ok);
}

fn f<F>(name: const u8[:]) {
    var xs: f64[>..] = [];
    for i in 0..n { xs.push(F()); }
    put(name, xs);
}

fn d<F>(name: const u8[:]) {
    var xs: i64[>..] = [];
    for i in 0..n { xs.push(F()); }
    puti(name, xs);
}

fn main() {
    let a = args();
    dir.append(a[1]);
    n = parse_int(a[2]);
    f("uniform") { random::uniform(g) };
    f("uniform_dense") { random::uniform_dense(g) };
    f("normal") { random::normal(g) };
    f("exponential") { random::exponential(g) };
    f("gamma_0.01") { random::gamma(g, 0.01) };
    f("gamma_0.5") { random::gamma(g, 0.5) };
    f("gamma_1.5") { random::gamma(g, 1.5) };
    f("gamma_7") { random::gamma(g, 7.0) };
    f("gamma_1000") { random::gamma(g, 1000.0) };
    f("gamma_1e8") { random::gamma(g, 1e8) };
    f("beta_0.2_0.3") { random::beta(g, 0.2, 0.3) };
    f("beta_0.5_0.5") { random::beta(g, 0.5, 0.5) };
    f("beta_2_5") { random::beta(g, 2.0, 5.0) };
    f("beta_0.5_8") { random::beta(g, 0.5, 8.0) };
    f("chi2_1") { random::chi_squared(g, 1.0) };
    f("chi2_17") { random::chi_squared(g, 17.0) };
    f("t_1") { random::student_t(g, 1.0) };
    f("t_3.5") { random::student_t(g, 3.5) };
    f("t_50") { random::student_t(g, 50.0) };
    f("f_4_9") { random::fisher_f(g, 4.0, 9.0) };
    f("lognormal_0.3_1.2") { random::lognormal(g, 0.3, 1.2) };
    f("cauchy") { random::cauchy(g, 0.0, 1.0) };
    f("laplace") { random::laplace(g, 0.0, 1.0) };
    f("logistic") { random::logistic(g, 0.0, 1.0) };
    f("gumbel") { random::gumbel(g, 0.0, 1.0) };
    f("weibull_0.7") { random::weibull(g, 0.7, 1.0) };
    f("weibull_3") { random::weibull(g, 3.0, 1.0) };
    f("rayleigh") { random::rayleigh(g, 1.0) };
    f("pareto_1.5") { random::pareto(g, 1.0, 1.5) };
    f("lomax_2.5") { random::lomax(g, 2.5, 1.0) };
    f("powerlaw_2.35") { random::power_law(g, 2.35, 0.08, 150.0) };
    f("powerlaw_1") { random::power_law(g, 1.0, 0.08, 150.0) };
    f("powerlaw_-2") { random::power_law(g, -2.0, 0.08, 150.0) };
    f("triangular") { random::triangular(g, -1.0, 0.5, 3.0) };
    f("wald_1_0.5") { random::wald(g, 1.0, 0.5) };
    f("wald_3_20") { random::wald(g, 3.0, 20.0) };
    f("vonmises_0.3") { random::von_mises(g, 0.0, 0.3) };
    f("vonmises_5") { random::von_mises(g, 0.0, 5.0) };
    f("vonmises_500") { random::von_mises(g, 0.0, 500.0) };
    d("poisson_0.5") { random::poisson(g, 0.5) };
    d("poisson_7") { random::poisson(g, 7.0) };
    d("poisson_12") { random::poisson(g, 12.0) };
    d("poisson_1000") { random::poisson(g, 1000.0) };
    d("poisson_1e7") { random::poisson(g, 1e7) };
    d("binomial_30_0.2") { random::binomial(g, 30, 0.2) };
    d("binomial_200_0.4") { random::binomial(g, 200, 0.4) };
    d("binomial_100000_0.7") { random::binomial(g, 100000, 0.7) };
    d("binomial_1000_0.004") { random::binomial(g, 1000, 0.004) };
    d("geometric_0.5") { random::geometric(g, 0.5) };
    d("geometric_0.01") { random::geometric(g, 0.01) };
    d("negbinomial_2.5_0.3") { random::negative_binomial(g, 2.5, 0.3) };
    d("hypergeom_40_60_30") { random::hypergeometric(g, 40, 60, 30) };
    d("hypergeom_5000_20000_8000") { random::hypergeometric(g, 5000, 20000, 8000) };
    d("zipf_1.7") { random::zipf(g, 1.7) };
    d("logseries_0.95") { random::log_series(g, 0.95) };
}
