// C++ <random> on the cases of bench.goose that it has: ns per value,
// filling a vector of 2^20 values, the best of at least 5 samples and
// 0.2 s. std::mt19937_64, the usual choice, with the standard library's
// distributions (MSVC's STL under clang-cl on the test machine).
//
//   clang-cl /O2 /arch:AVX2 /EHsc bench_cpp.cpp && bench_cpp [out.csv]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static const size_t N = 1 << 20;
static FILE *csv;
static double sink;

template <class F> static void time_it(const char *name, F f) {
    double best = 1e30, total = 0;
    int samples = 0;
    while (samples < 5 || total < 0.2) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        best = std::min(best, dt);
        total += dt;
        samples++;
    }
    double ns = best / N * 1e9;
    std::printf("%s: %.3f ns\n", name, ns);
    std::fprintf(csv, "%s,%.6f\n", name, ns);
}

template <class D, class T> static void fill(const char *name, std::mt19937_64 &g, std::vector<T> &out, D d) {
    time_it(name, [&] {
        for (auto &x : out) x = d(g);
        sink += (double)out[N - 1];
    });
}

int main(int argc, char **argv) {
    csv = std::fopen(argc > 1 ? argv[1] : "bench_cpp.csv", "w");
    std::mt19937_64 g(1);
    std::vector<double> f(N);
    std::vector<uint64_t> u(N);
    std::vector<int64_t> s(N);
    for (size_t i = 0; i < N; i++) s[i] = (int64_t)i;
    time_it("raw mt19937_64", [&] {
        for (auto &x : u) x = g();
        sink += (double)u[N - 1];
    });
    fill("uniform", g, f, std::uniform_real_distribution<double>(0.0, 1.0));
    fill("normal", g, f, std::normal_distribution<double>());
    fill("exponential", g, f, std::exponential_distribution<double>());
    fill("below 1000", g, u, std::uniform_int_distribution<uint64_t>(0, 999));
    fill("below 3e18", g, u, std::uniform_int_distribution<uint64_t>(0, 2999999999999999999ull));
    fill("gamma 0.5", g, f, std::gamma_distribution<double>(0.5));
    fill("gamma 3", g, f, std::gamma_distribution<double>(3.0));
    fill("chi_squared 5", g, f, std::chi_squared_distribution<double>(5.0));
    fill("student_t 5", g, f, std::student_t_distribution<double>(5.0));
    fill("lognormal", g, f, std::lognormal_distribution<double>(0.0, 1.0));
    fill("cauchy", g, f, std::cauchy_distribution<double>(0.0, 1.0));
    fill("weibull 1.5", g, f, std::weibull_distribution<double>(1.5, 1.0));
    fill("poisson 3", g, s, std::poisson_distribution<int64_t>(3.0));
    fill("poisson 1000", g, s, std::poisson_distribution<int64_t>(1000.0));
    fill("binomial 20 0.3", g, s, std::binomial_distribution<int64_t>(20, 0.3));
    fill("binomial 1000 0.3", g, s, std::binomial_distribution<int64_t>(1000, 0.3));
    fill("geometric 0.01", g, s, std::geometric_distribution<int64_t>(0.01));
    std::discrete_distribution<int64_t> dd({0.1, 0.2, 0.3, 0.15, 0.05, 0.2});
    fill("alias 6", g, s, dd);
    time_it("shuffle", [&] {
        std::shuffle(s.begin(), s.end(), g);
        sink += (double)s[0];
    });
    std::fclose(csv);
    std::printf("sink %g\n", sink);
}
