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
    time_it("uniform_dense") { for i in 0..N { fs[i] = random::uniform_dense(x); } };
    time_it("below 1000") { for i in 0..N { is[i] = random::below(x, 1000); } };
    time_it("below 3e18") { for i in 0..N { is[i] = random::below(x, 3000000000000000000); } };
    time_it("gamma 0.5") { for i in 0..N { fs[i] = random::gamma(x, 0.5); } };
    time_it("gamma 3") { for i in 0..N { fs[i] = random::gamma(x, 3.0); } };
    time_it("beta 2 5") { for i in 0..N { fs[i] = random::beta(x, 2.0, 5.0); } };
    time_it("beta 0.5 0.5") { for i in 0..N { fs[i] = random::beta(x, 0.5, 0.5); } };
    time_it("chi_squared 5") { for i in 0..N { fs[i] = random::chi_squared(x, 5.0); } };
    time_it("student_t 5") { for i in 0..N { fs[i] = random::student_t(x, 5.0); } };
    time_it("lognormal") { for i in 0..N { fs[i] = random::lognormal(x, 0.0, 1.0); } };
    time_it("cauchy") { for i in 0..N { fs[i] = random::cauchy(x, 0.0, 1.0); } };
    time_it("weibull 1.5") { for i in 0..N { fs[i] = random::weibull(x, 1.5, 1.0); } };
    time_it("von_mises 4") { for i in 0..N { fs[i] = random::von_mises(x, 0.0, 4.0); } };
    time_it("poisson 3") { for i in 0..N { is[i] = random::poisson(x, 3.0); } };
    time_it("poisson 1000") { for i in 0..N { is[i] = random::poisson(x, 1000.0); } };
    time_it("binomial 20 0.3") { for i in 0..N { is[i] = random::binomial(x, 20, 0.3); } };
    time_it("binomial 1000 0.3") { for i in 0..N { is[i] = random::binomial(x, 1000, 0.3); } };
    time_it("geometric 0.01") { for i in 0..N { is[i] = random::geometric(x, 0.01); } };
    time_it("hypergeometric 500 500 200") { for i in 0..N { is[i] = random::hypergeometric(x, 500, 500, 200); } };
    let t = random::alias_table_from([0.1, 0.2, 0.3, 0.15, 0.05, 0.2]);
    time_it("alias 6") { for i in 0..N { is[i] = random::sample(x, t); } };
    time_it("shuffle") { random::shuffle(x, is); };

    let ok = write_file(out, csv);
    assert(ok);
    print("sink ", sink);
}
