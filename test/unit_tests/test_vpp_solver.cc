/**
 * @file test_vpp_solver.cc
 * @brief Checks the SE / FCSE integer solvers in vpp_solver.cc against
 * brute-force enumeration of x^T Q x + b^T x over a small integer box.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "armadillo"
#include "modulation.h"
#include "vpp_solver.h"
#include "vpp_utils.h"

namespace {

constexpr double kTol = 1e-7;

double Energy(const arma::mat& q, const arma::vec& b, const arma::ivec& x) {
  const arma::vec xv = arma::conv_to<arma::vec>::from(x);
  return arma::as_scalar(xv.t() * q * xv + b.t() * xv);
}

double BruteForceMin(const arma::mat& q, const arma::vec& b, int vmin,
                     int vmax) {
  const size_t n = b.n_elem;
  arma::ivec x(n);
  x.fill(vmin);
  double best = std::numeric_limits<double>::infinity();
  while (true) {
    best = std::min(best, Energy(q, b, x));
    size_t i = 0;
    while (i < n && x(i) == vmax) {
      x(i) = vmin;
      ++i;
    }
    if (i == n) {
      break;
    }
    ++x(i);
  }
  return best;
}

// Random SPD Q with a spread of eigenvalues, like a ZF Gram matrix, and a
// linear term large enough that the optimum is usually away from x = 0.
void RandomProblem(size_t n, arma::mat& q, arma::vec& b) {
  const arma::mat a = arma::randn<arma::mat>(n, n);
  q = a.t() * a + 0.05 * arma::eye<arma::mat>(n, n);
  b = 4.0 * arma::randn<arma::vec>(n);
}

void ExpectWellFormed(const arma::mat& q, const arma::vec& b,
                      const VppSolverResult& r, int vmin, int vmax) {
  ASSERT_EQ(r.solution.n_elem, b.n_elem);
  EXPECT_GE(r.solution.min(), vmin);
  EXPECT_LE(r.solution.max(), vmax);
  EXPECT_NEAR(r.energy, Energy(q, b, r.solution),
              kTol * (1.0 + std::fabs(r.energy)));
}

}  // namespace

TEST(TestVppSolver, SphereMatchesBruteForce) {
  arma::arma_rng::set_seed(1);
  for (size_t n : {2, 4, 6}) {
    for (int w : {1, 2}) {
      for (int trial = 0; trial < 40; ++trial) {
        arma::mat q;
        arma::vec b;
        RandomProblem(n, q, b);
        const VppSolverResult r = VppSolveSphere(
            q, b, arma::ivec(n, arma::fill::zeros), VppSphereConfig{-w, w});
        ExpectWellFormed(q, b, r, -w, w);
        const double opt = BruteForceMin(q, b, -w, w);
        EXPECT_NEAR(r.energy, opt, kTol * (1.0 + std::fabs(opt)))
            << "n=" << n << " w=" << w << " trial=" << trial;
      }
    }
  }
}

TEST(TestVppSolver, FcseExhaustiveMatchesBruteForce) {
  arma::arma_rng::set_seed(2);
  for (size_t n : {2, 4, 5}) {
    for (int trial = 0; trial < 30; ++trial) {
      arma::mat q;
      arma::vec b;
      RandomProblem(n, q, b);
      const VppSolverResult r = VppSolveFixedComplexity(
          q, b, arma::ivec(n, arma::fill::zeros),
          VppFixedComplexityConfig{-2, 2, n});
      ExpectWellFormed(q, b, r, -2, 2);
      const double opt = BruteForceMin(q, b, -2, 2);
      EXPECT_NEAR(r.energy, opt, kTol * (1.0 + std::fabs(opt)));
    }
  }
}

TEST(TestVppSolver, FcseDefaultIsFeasibleAndNotBelowOptimum) {
  arma::arma_rng::set_seed(3);
  const size_t n = 6;
  for (int trial = 0; trial < 40; ++trial) {
    arma::mat q;
    arma::vec b;
    RandomProblem(n, q, b);
    const VppSolverResult r = VppSolveFixedComplexity(
        q, b, arma::ivec(n, arma::fill::zeros),
        VppFixedComplexityConfig{-2, 2, 2});
    ExpectWellFormed(q, b, r, -2, 2);
    const double opt = BruteForceMin(q, b, -2, 2);
    EXPECT_GE(r.energy, opt - kTol * (1.0 + std::fabs(opt)));
  }
}

TEST(TestVppSolver, FcseZeroLayersIsBabaiRounding) {
  arma::arma_rng::set_seed(4);
  const size_t n = 6;
  const int w = 3;
  for (int trial = 0; trial < 40; ++trial) {
    arma::mat q;
    arma::vec b;
    RandomProblem(n, q, b);
    const arma::mat r = arma::chol(q, "upper");
    const arma::vec y = arma::solve(arma::trimatl(r.t()), -0.5 * b);
    arma::ivec x(n, arma::fill::zeros);
    for (int k = static_cast<int>(n) - 1; k >= 0; --k) {
      double s = y(k);
      for (size_t j = k + 1; j < n; ++j) {
        s -= r(k, j) * x(j);
      }
      x(k) = std::clamp(static_cast<int>(std::lround(s / r(k, k))), -w, w);
    }
    const VppSolverResult res = VppSolveFixedComplexity(
        q, b, arma::ivec(n, arma::fill::zeros),
        VppFixedComplexityConfig{-w, w, 0});
    EXPECT_TRUE(arma::all(res.solution == x));
  }
}

TEST(TestVppSolver, HeatBathNeverBeatsSphere) {
  arma::arma_rng::set_seed(5);
  const size_t n = 6;
  VppSolverConfig hb;
  hb.value_min = -2;
  hb.value_max = 2;
  hb.num_replicas = 16;
  hb.runs = 2;
  hb.iters = 50;
  hb.neighbor_window = 1;
  hb.beta_min = 5.0;
  hb.beta_max = 20.0;
  for (int trial = 0; trial < 40; ++trial) {
    arma::mat q;
    arma::vec b;
    RandomProblem(n, q, b);
    const double sc = arma::abs(q).max();
    const arma::mat qs = q / sc;
    const arma::vec bs = b / sc;
    const arma::ivec zero(n, arma::fill::zeros);
    const VppSolverResult se = VppSolveSphere(qs, bs, zero, VppSphereConfig{-2, 2});
    const VppSolverResult heat = VppSolveInteger(qs, bs, zero, hb);
    EXPECT_GE(heat.energy, se.energy - kTol);
  }
}

TEST(TestVppSolver, NonPositiveDefiniteFallsBackToZero) {
  const size_t n = 4;
  const arma::vec v = {1.0, -2.0, 0.5, 3.0};
  const arma::mat q = v * v.t();  // rank 1
  const arma::vec b = {1.0, 2.0, -1.0, 0.5};
  const arma::ivec zero(n, arma::fill::zeros);
  const VppSolverResult se = VppSolveSphere(q, b, zero, VppSphereConfig{-2, 2});
  const VppSolverResult fc = VppSolveFixedComplexity(
      q, b, zero, VppFixedComplexityConfig{-2, 2, 2});
  EXPECT_TRUE(arma::all(se.solution == 0));
  EXPECT_TRUE(arma::all(fc.solution == 0));
  EXPECT_EQ(se.energy, 0.0);
  EXPECT_EQ(fc.energy, 0.0);
}

TEST(TestVppSolver, ZeroWindowReturnsZero) {
  arma::arma_rng::set_seed(6);
  arma::mat q;
  arma::vec b;
  RandomProblem(2, q, b);
  const arma::ivec zero(2, arma::fill::zeros);
  EXPECT_TRUE(arma::all(
      VppSolveSphere(q, b, zero, VppSphereConfig{0, 0}).solution == 0));
  EXPECT_TRUE(arma::all(VppSolveFixedComplexity(q, b, zero,
                                                VppFixedComplexityConfig{0, 0, 2})
                            .solution == 0));
}

namespace {

// Runs Demodulate() on num symbols (aligned buffers, as the AVX2 kernels
// require) and returns the LLR bytes.
std::vector<int8_t> SoftDemod(const std::vector<complex_float>& sym,
                              size_t mod_bits) {
  const size_t num = sym.size();
  auto* in = static_cast<float*>(std::aligned_alloc(64, 64 * ((num * 8 + 63) / 64)));
  auto* out = static_cast<int8_t*>(
      std::aligned_alloc(64, 64 * ((num * mod_bits + 63) / 64 + 1)));
  for (size_t i = 0; i < num; i++) {
    in[2 * i] = sym[i].re;
    in[2 * i + 1] = sym[i].im;
  }
  Demodulate(in, out, num, mod_bits, false);
  std::vector<int8_t> llr(out, out + num * mod_bits);
  std::free(in);
  std::free(out);
  return llr;
}

}  // namespace

// ApplyPeriodicMsbLlr assumes bytes 0/1 of each symbol are the real/imag MSB
// LLRs and are linear in x; check that against the real demodulator.
TEST(TestVppSolver, DemodMsbLayoutIsLinearInAxis) {
  for (size_t mod_bits : {2, 4, 6}) {
    const double slope = (mod_bits == 2) ? -20.0 * M_SQRT2 : 100.0;
    std::vector<complex_float> sym(32);
    for (size_t i = 0; i < sym.size(); i++) {
      sym[i] = {0.01f * (static_cast<float>(i) - 15.5f),
                -0.007f * (static_cast<float>(i) - 10.0f)};
    }
    const auto llr = SoftDemod(sym, mod_bits);
    for (size_t i = 0; i < sym.size(); i++) {
      EXPECT_NEAR(llr[i * mod_bits], slope * sym[i].re, 1.01)
          << "mod_bits=" << mod_bits << " i=" << i;
      EXPECT_NEAR(llr[i * mod_bits + 1], slope * sym[i].im, 1.01)
          << "mod_bits=" << mod_bits << " i=" << i;
    }
  }
}

TEST(TestVppSolver, PeriodicMsbLlr) {
  for (size_t mod_bits : {2, 4, 6}) {
    const double tau = ComputeVppTau(static_cast<CommsLib::ModulationOrder>(mod_bits));
    std::vector<complex_float> sym = {
        {0.1f, -0.05f},                                     // inner: unchanged
        {static_cast<float>(0.499 * tau), static_cast<float>(-0.499 * tau)},  // at the fold
        {static_cast<float>(0.375 * tau), static_cast<float>(-0.375 * tau)},  // halfway
    };
    sym.resize(16, complex_float{0.0f, 0.0f});
    const auto before = SoftDemod(sym, mod_bits);
    auto after = before;
    ApplyPeriodicMsbLlr(sym.data(), after.data(), sym.size(), mod_bits, tau);
    for (size_t b = 0; b < mod_bits; b++) {
      EXPECT_EQ(after[b], before[b]);  // |x| <= tau/4: untouched
    }
    for (size_t a = 0; a < 2; a++) {
      EXPECT_LE(std::abs(after[mod_bits + a]), 1);  // ambiguous at +-tau/2
      // tau*3/8 sits midway: triangle value is 1/3 of the linear one.
      EXPECT_NEAR(after[2 * mod_bits + a], before[2 * mod_bits + a] / 3.0, 1.0);
      EXPECT_EQ(after[2 * mod_bits + a] > 0, before[2 * mod_bits + a] > 0);
    }
    for (size_t i = 1; i < 3; i++) {
      for (size_t b = 2; b < mod_bits; b++) {
        EXPECT_EQ(after[i * mod_bits + b], before[i * mod_bits + b]);
      }
    }
  }
}

namespace {
// TEMPDEBUG (2026-10-28, solver-budget sweep): a realistic 16-BS-antenna,
// n_served-UE ZF-precoder Gram matrix (Q = tau^2 * G, normalized by max|Q|,
// exactly as doprecode.cc::ApplyVppPerturbation builds it), instead of
// RandomProblem's generic SPD matrix, so the sweep below reflects the real
// operating point's conditioning. Revert this whole TEMPDEBUG block before
// committing.
void RealisticVppProblem(size_t bs_ant, size_t ue_num, double tau,
                         arma::mat& q, arma::vec& b) {
  // h is (ue_num x bs_ant), matching Agora/QUIOVPPDOWNLINK's convention
  // (rows = UE receive antennas, cols = BS transmit antennas).
  arma::cx_mat h(arma::randn<arma::mat>(ue_num, bs_ant),
                 arma::randn<arma::mat>(ue_num, bs_ant));
  h *= std::sqrt(0.5);
  const arma::cx_mat hh = h * h.t();
  const arma::cx_mat p = h.t() * arma::inv(hh);  // (bs_ant x ue_num) precoder
  arma::mat f_real(2 * bs_ant, 2 * ue_num);
  f_real.submat(0, 0, bs_ant - 1, ue_num - 1) = arma::real(p);
  f_real.submat(0, ue_num, bs_ant - 1, 2 * ue_num - 1) = -arma::imag(p);
  f_real.submat(bs_ant, 0, 2 * bs_ant - 1, ue_num - 1) = arma::imag(p);
  f_real.submat(bs_ant, ue_num, 2 * bs_ant - 1, 2 * ue_num - 1) = arma::real(p);
  const arma::mat g = f_real.t() * f_real;
  q = (tau * tau) * g;
  q = 0.5 * (q + q.t());
  const arma::vec u = arma::randn<arma::vec>(2 * ue_num);  // unperturbed data
  b = (2.0 * tau) * (g * u);
  const double sc = arma::abs(q).max();
  q /= sc;
  b /= sc;
}

TEST(TestVppSolver, BudgetSweepAt16x8Dummy) {
  // Not a real assertion-based test -- prints a budget/exact-match-rate
  // table to stdout for the 16-BS-ant/8-served-UE, 64QAM operating point.
  arma::arma_rng::set_seed(42);
  const size_t bs_ant = 16;
  const size_t ue_num = 8;
  const size_t n = 2 * ue_num;
  const int window = 9;  // DefaultVppLllWindow(64QAM)
  const double tau = 2.0 * (8.0 - 1.0 + 1.0) / std::sqrt(2.0 * 63.0 / 3.0);
  const int kTrials = 1000;

  struct Budget {
    size_t replicas, runs, iters;
  };
  const std::vector<Budget> budgets = {
      {8, 2, 50},  {8, 1, 50},  {8, 2, 35},  {8, 2, 20},  {8, 2, 15},
      {6, 2, 50},  {6, 2, 35},  {6, 2, 25},  {10, 2, 25}, {10, 2, 20},
      {6, 1, 50},  {8, 1, 35},  {8, 1, 25},  {6, 2, 20},  {6, 1, 25},
  };

  for (const auto& bud : budgets) {
    VppSolverConfig cfg;
    cfg.value_min = -window;
    cfg.value_max = window;
    cfg.num_replicas = bud.replicas;
    cfg.runs = bud.runs;
    cfg.iters = bud.iters;
    cfg.neighbor_window = 1;
    cfg.beta_min = 5.0;
    cfg.beta_max = 20.0;
    int exact_matches = 0;
    double total_work = static_cast<double>(bud.replicas * bud.runs * bud.iters);
    for (int trial = 0; trial < kTrials; ++trial) {
      arma::mat q;
      arma::vec b;
      RealisticVppProblem(bs_ant, ue_num, tau, q, b);
      const arma::ivec zero(n, arma::fill::zeros);
      const VppSolverResult se =
          VppSolveSphere(q, b, zero, VppSphereConfig{-window, window});
      const VppSolverResult vpp = VppSolveInteger(q, b, zero, cfg);
      if (std::fabs(vpp.energy - se.energy) < 1e-6 * (1.0 + std::fabs(se.energy))) {
        exact_matches++;
      }
    }
    std::printf(
        "TEMPDEBUG budget_sweep: replicas=%zu runs=%zu iters=%zu "
        "work=%.0f exact_match_rate=%.4f\n",
        bud.replicas, bud.runs, bud.iters, total_work,
        static_cast<double>(exact_matches) / kTrials);
  }
}
}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
