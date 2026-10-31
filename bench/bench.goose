// Times the generators and distributions, for comparison with
// bench_numpy.py (NumPy), rust/ (rand and rand_distr) and bench_cpp.cpp
// (C++ <random>). Every case fills an array of 2^20 values, as the others
// are timed; the result is nanoseconds per value, the best of at least 5
// samples and 0.2 s. Results go to stdout and to the CSV named on the
// command line (bench_goose.csv by default).
//
//   goose build bench/bench.goose && bench/bench [out.csv]

import std;
import os;
import math;
import random from "..";

let N = 1048576;

var csv: u8[>..]  = [];
var fs:  f64[>..] = [];
var is:  i64[>..] = [];
var us:  u64[>..] = [];
var sink = 0.0;

fn time_it<F>(name: const u8[:]) {
    var best = 1.0e30;
    var total = 0.0;
    var samples = 0;
    while samples < 5 || total < 0.2 {
        let t0 = clock();
        F();
        let dt = clock() - t0;
        best = min(best, dt);
        total += dt;
        samples++;
    }
    sink += fs[N - 1] + is[N - 1] + us[N - 1];
    let ns = best / N * 1.0e9;
    format(csv, name, ",", ns, "\n");
    var line: u8[>..] = [];
    format(line, name, ": ");
    format_flt(line, ns, 3);
    format(line, " ns");
    print(line);
}

fn floats<G, F>(g: G&, name: const u8[:]) {
    time_it(name) { for i in 0..N { fs[i] = F(g); } };
}

fn ints<G, F>(g: G&, name: const u8[:]) {
    time_it(name) { for i in 0..N { is[i] = F(g); } };
}

fn raw<G>(g: G&, name: const u8[:]) {
    time_it(name) { random::fill_u64(g, us); };
}

fn main() {
    let a = args();
    let out = if a.len > 1 { a[1] } else { "bench_goose.csv" };
    fs.resize(N, 0.0);
    is.resize(N, 0);
    us.resize(N, 0);

    var x = random::seed(1);
    var s = random::seed_xoshiro256ss(1);
    var c = random::seed_sfc64(1);
    var m = random::seed_splitmix64(1);
    raw(x, "raw xoshiro256++");
    raw(s, "raw xoshiro256**");
    raw(c, "raw sfc64");
    raw(m, "raw splitmix64");

    time_it("uniform") { random::fill_uniform(x, fs); };
    time_it("normal") { random::fill_normal(x, fs); };
    time_it("exponential") { random::fill_exponential(x, fs); };
    floats(x, "uniform_dense") { g => random::uniform_dense(g) };
    ints(x, "below 1000") { g => random::below(g, 1000) };
    ints(x, "below 3e18") { g => random::below(g, 3000000000000000000) };
    floats(x, "gamma 0.5") { g => random::gamma(g, 0.5) };
    floats(x, "gamma 3") { g => random::gamma(g, 3.0) };
    floats(x, "beta 2 5") { g => random::beta(g, 2.0, 5.0) };
    floats(x, "beta 0.5 0.5") { g => random::beta(g, 0.5, 0.5) };
    floats(x, "chi_squared 5") { g => random::chi_squared(g, 5.0) };
    floats(x, "student_t 5") { g => random::student_t(g, 5.0) };
    floats(x, "lognormal") { g => random::lognormal(g, 0.0, 1.0) };
    floats(x, "cauchy") { g => random::cauchy(g, 0.0, 1.0) };
    floats(x, "weibull 1.5") { g => random::weibull(g, 1.5, 1.0) };
    floats(x, "von_mises 4") { g => random::von_mises(g, 0.0, 4.0) };
    ints(x, "poisson 3") { g => random::poisson(g, 3.0) };
    ints(x, "poisson 1000") { g => random::poisson(g, 1000.0) };
    ints(x, "binomial 20 0.3") { g => random::binomial(g, 20, 0.3) };
    ints(x, "binomial 1000 0.3") { g => random::binomial(g, 1000, 0.3) };
    ints(x, "geometric 0.01") { g => random::geometric(g, 0.01) };
    ints(x, "hypergeometric 500 500 200") { g => random::hypergeometric(g, 500, 500, 200) };
    let t = random::alias_table_from([0.1, 0.2, 0.3, 0.15, 0.05, 0.2]);
    ints(x, "alias 6") { g => random::sample(g, t) };
    time_it("shuffle") { random::shuffle(x, is); };

    let ok = write_file(out, csv);
    assert(ok);
    print("sink ", sink);
}
