/**
 * @file vpp_solver.cc
 * @brief See vpp_solver.h. Ported from solve_iqp / sample_neighbor / dl_lll in
 * QUIOVPPDOWNLINK/scripts/7_8_26_intmaxbm_downlink.cpp.
 */
#include "vpp_solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "logger.h"

namespace {

// PCG32 (https://www.pcg-random.org/), identical to Pcg32 in the reference
// (seeding, output permutation and uniform01), so both draw the same stream.
class PcgRng {
 public:
  explicit PcgRng(uint64_t seed) {
    state_ = seed + kIncrement;
    (void)NextU32();
  }

  uint32_t NextU32() {
    uint64_t x = state_;
    const unsigned count = static_cast<unsigned>(x >> 59);
    state_ = x * kMultiplier + kIncrement;
    x ^= x >> 18;
    return Rotr32(static_cast<uint32_t>(x >> 27), count);
  }

  double Uniform01() {
    return (static_cast<double>(NextU32()) + 0.5) / 4294967296.0;
  }

 private:
  static uint32_t Rotr32(uint32_t x, unsigned r) {
    return (x >> r) | (x << ((-r) & 31u));
  }

  uint64_t state_ = 0x4d595df4d0f33173ULL;
  static constexpr uint64_t kMultiplier = 6364136223846793005ULL;
  static constexpr uint64_t kIncrement = 1442695040888963407ULL;
};

struct Move {
  int delta;
  double energy;
};

// Energy after changing one variable by `delta`, given the local field
// lf = 2*Q*x + b and the diagonal entry q_ii.
inline double DeltaEnergy(double prev_energy, double local_field, double q_ii,
                          int delta) {
  return prev_energy + delta * local_field +
         static_cast<double>(delta * delta) * q_ii;
}

// Heat-bath proposal for variable x[idx]: enumerates every value within
// +-neighbors of the current one (clipped to [value_min, value_max]) and
// samples among them at inverse temperature beta. Same arithmetic and same
// single uniform draw as sample_neighbor in the reference.
Move SampleNeighbor(const double* x, const double* local_field,
                    const double* q_diag, double prev_energy, size_t idx,
                    double beta, int value_min, int value_max, int neighbors,
                    PcgRng& rng) {
  const int current = static_cast<int>(x[idx]);
  const double lf = local_field[idx];
  const double q_ii = q_diag[idx];

  if (neighbors == 1) {
    double min_energy = prev_energy;
    double e_minus = std::numeric_limits<double>::infinity();
    double e_plus = std::numeric_limits<double>::infinity();
    const bool has_minus = current > value_min;
    const bool has_plus = current < value_max;
    if (has_minus) {
      e_minus = DeltaEnergy(prev_energy, lf, q_ii, -1);
      min_energy = std::min(min_energy, e_minus);
    }
    if (has_plus) {
      e_plus = DeltaEnergy(prev_energy, lf, q_ii, 1);
      min_energy = std::min(min_energy, e_plus);
    }

    auto boltzmann_weight = [&](double e) {
      return e == min_energy ? 1.0 : std::exp(-(e - min_energy) * beta);
    };
    const double w_minus = has_minus ? boltzmann_weight(e_minus) : 0.0;
    const double w_zero = boltzmann_weight(prev_energy);
    const double w_plus = has_plus ? boltzmann_weight(e_plus) : 0.0;
    double draw = rng.Uniform01() * (w_minus + w_zero + w_plus);
    if (draw <= w_minus) {
      return {-1, e_minus};
    }
    draw -= w_minus;
    if (draw <= w_zero) {
      return {0, prev_energy};
    }
    return {1, e_plus};
  }

  const int minv = std::max(value_min, current - neighbors);
  const int maxv = std::min(value_max, current + neighbors);
  const int n = maxv - minv + 1;
  constexpr int kStackCapacity = 128;
  if (n <= 0 || n > kStackCapacity) {
    throw std::runtime_error("VppSolveInteger: unsupported proposal window");
  }
  std::array<double, kStackCapacity> energies;
  std::array<double, kStackCapacity> probs;
  double min_energy = std::numeric_limits<double>::infinity();
  for (int proposal = minv; proposal <= maxv; ++proposal) {
    const double e =
        DeltaEnergy(prev_energy, lf, q_ii, proposal - current);
    energies[proposal - minv] = e;
    min_energy = std::min(min_energy, e);
  }
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    probs[i] = std::exp(-(energies[i] - min_energy) * beta);
    total += probs[i];
  }
  const double sample = rng.Uniform01();
  double prefix = 0.0;
  int selected = n - 1;
  for (int i = 0; i < n; ++i) {
    prefix += probs[i] / total;
    if (sample <= prefix) {
      selected = i;
      break;
    }
  }
  return {(selected + minv) - current, energies[selected]};
}

// Geometric ladder, decreasing (index 0 = coldest / largest beta), both
// endpoints included -- the order solve_iqp sorts its ladder into.
std::vector<double> GeomBetas(double beta_max, double beta_min, size_t n) {
  std::vector<double> betas(n, beta_max);
  if (n > 1) {
    const double ratio = std::pow(beta_min / beta_max, 1.0 / (n - 1));
    double b = beta_max;
    for (size_t i = 0; i < n; ++i) {
      betas[i] = b;
      b *= ratio;
    }
  }
  return betas;
}

// Sphere form of the objective: with Q = R^T R (R upper triangular) and
// R^T y = -b/2, x^T Q x + b^T x = ||y - R x||^2 - y^T y. Layer k of R only
// involves x[k..n-1], so a tree search decides x[n-1] first and x[0] last.
struct SphereProblem {
  arma::mat r;
  arma::vec y;
  bool ok = false;
};

SphereProblem PrepareSphereProblem(const arma::mat& q, const arma::vec& b) {
  SphereProblem p;
  if (arma::chol(p.r, q, "upper")) {
    p.ok = arma::solve(p.y, arma::trimatl(p.r.t()), -0.5 * b);
  }
  return p;
}

VppSolverResult ZeroResult(size_t n) {
  VppSolverResult result;
  result.solution.zeros(n);
  result.energy = 0.0;
  return result;
}

class TreeSearch {
 public:
  TreeSearch(const SphereProblem& p, int value_min, int value_max)
      : r_(p.r),
        y_(p.y),
        n_(static_cast<int>(p.y.n_elem)),
        value_min_(value_min),
        value_max_(value_max),
        x_(n_, 0),
        best_x_(n_, 0) {}

  VppSolverResult Result() const {
    VppSolverResult result;
    result.solution.set_size(n_);
    for (int i = 0; i < n_; ++i) {
      result.solution(i) = best_x_[i];
    }
    result.energy = best_dist_ - arma::dot(y_, y_);
    return result;
  }

 protected:
  // Unconstrained minimizer of layer k's term given x[k+1..n-1].
  double Center(int k) const {
    double s = y_(k);
    for (int j = k + 1; j < n_; ++j) {
      s -= r_(k, j) * x_[j];
    }
    return s / r_(k, k);
  }

  double LayerCost(int k, double center, int v) const {
    const double e = r_(k, k) * (center - v);
    return e * e;
  }

  void Leaf(double dist) {
    if (dist < best_dist_) {
      best_dist_ = dist;
      best_x_ = x_;
    }
  }

  const arma::mat& r_;
  const arma::vec& y_;
  const int n_;
  const int value_min_;
  const int value_max_;
  std::vector<int> x_;
  std::vector<int> best_x_;
  double best_dist_ = std::numeric_limits<double>::infinity();
};

class SphereSearch : public TreeSearch {
 public:
  SphereSearch(const SphereProblem& p, int value_min, int value_max,
               const arma::ivec& init_kernel)
      : TreeSearch(p, value_min, value_max) {
    // Seed the radius with the (clamped) warm start so the first dive is
    // already pruned; strict improvement below keeps the result exact.
    for (int i = 0; i < n_ && i < static_cast<int>(init_kernel.n_elem); ++i) {
      x_[i] = std::clamp(static_cast<int>(init_kernel(i)), value_min_,
                         value_max_);
    }
    double dist = 0.0;
    for (int k = n_ - 1; k >= 0; --k) {
      dist += LayerCost(k, Center(k), x_[k]);
    }
    best_x_ = x_;
    best_dist_ = dist;
  }

  void Run() { Descend(n_ - 1, 0.0); }

 private:
  void Descend(int k, double dist) {
    if (k < 0) {
      Leaf(dist);
      return;
    }
    const double c = Center(k);
    const int floor_c = static_cast<int>(std::floor(c));
    // Schnorr-Euchner: visit values in non-decreasing |c - v|, so the first
    // one that leaves the sphere ends this layer.
    int down = std::min(floor_c, value_max_);
    int up = std::max(floor_c + 1, value_min_);
    while (true) {
      const bool down_ok = down >= value_min_;
      const bool up_ok = up <= value_max_;
      if (!down_ok && !up_ok) {
        break;
      }
      const int v =
          (down_ok && (!up_ok || (c - down) <= (up - c))) ? down-- : up++;
      const double d = dist + LayerCost(k, c, v);
      if (d >= best_dist_) {
        break;
      }
      x_[k] = v;
      Descend(k - 1, d);
    }
  }
};

class FixedComplexitySearch : public TreeSearch {
 public:
  FixedComplexitySearch(const SphereProblem& p, int value_min, int value_max,
                        size_t full_layers)
      : TreeSearch(p, value_min, value_max),
        full_layers_(static_cast<int>(full_layers)) {}

  void Run() { Descend(n_ - 1, 0.0); }

 private:
  void Descend(int k, double dist) {
    if (k < 0) {
      Leaf(dist);
      return;
    }
    const double c = Center(k);
    if (n_ - 1 - k < full_layers_) {
      for (int v = value_min_; v <= value_max_; ++v) {
        x_[k] = v;
        Descend(k - 1, dist + LayerCost(k, c, v));
      }
    } else {
      const int v = std::clamp(static_cast<int>(std::lround(c)), value_min_,
                               value_max_);
      x_[k] = v;
      Descend(k - 1, dist + LayerCost(k, c, v));
    }
  }

  const int full_layers_;
};

}  // namespace

VppSolverResult VppSolveSphere(const arma::mat& q, const arma::vec& b,
                               const arma::ivec& init_kernel,
                               const VppSphereConfig& cfg) {
  const SphereProblem p = PrepareSphereProblem(q, b);
  if (!p.ok || cfg.value_min > cfg.value_max) {
    return ZeroResult(b.n_elem);
  }
  SphereSearch search(p, cfg.value_min, cfg.value_max, init_kernel);
  search.Run();
  return search.Result();
}

VppSolverResult VppSolveFixedComplexity(const arma::mat& q, const arma::vec& b,
                                        const arma::ivec& /*init_kernel*/,
                                        const VppFixedComplexityConfig& cfg) {
  const SphereProblem p = PrepareSphereProblem(q, b);
  if (!p.ok || cfg.value_min > cfg.value_max) {
    return ZeroResult(b.n_elem);
  }
  const size_t radix = static_cast<size_t>(cfg.value_max - cfg.value_min) + 1;
  const size_t wanted = std::min(cfg.full_layers, static_cast<size_t>(b.n_elem));
  size_t full_layers = 0;
  for (size_t paths = 1;
       full_layers < wanted && paths <= kVppFcseMaxPaths / radix;
       ++full_layers) {
    paths *= radix;
  }
  if (full_layers < wanted) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
      AGORA_LOG_WARN(
          "VppSolveFixedComplexity: full_layers %zu needs more than %zu paths "
          "at value range %zu, using %zu\n",
          wanted, kVppFcseMaxPaths, radix, full_layers);
    }
  }
  FixedComplexitySearch search(p, cfg.value_min, cfg.value_max, full_layers);
  search.Run();
  return search.Result();
}

VppSolverResult VppSolveInteger(const arma::mat& q, const arma::vec& b,
                                const arma::ivec& init_kernel,
                                const VppSolverConfig& cfg) {
  const size_t n = b.n_elem;
  const size_t num_replicas = std::max<size_t>(cfg.num_replicas, 1);
  const std::vector<double> betas =
      GeomBetas(cfg.beta_max, cfg.beta_min, num_replicas);
  const int neighbors = static_cast<int>(std::max<size_t>(cfg.neighbor_window, 1));

  const double* q_mem = q.memptr();  // column-major, symmetric

  // Scratch buffers reused across calls on this worker thread instead of
  // heap-allocating fresh ones every call (this runs ~900x/frame): each
  // resize()/assign() below is a no-op once n/num_replicas settle to their
  // steady-state values (normally constant for the whole run).
  thread_local std::vector<double> q_diag;
  thread_local std::vector<double> x0;
  thread_local std::vector<double> lf0;
  thread_local std::vector<double> best_solution;
  thread_local std::vector<std::vector<double>> kernels;
  thread_local std::vector<std::vector<double>> local_fields;
  thread_local std::vector<double> energies;
  thread_local std::vector<double> sample_best;

  q_diag.resize(n);
  for (size_t i = 0; i < n; ++i) {
    q_diag[i] = q_mem[i + i * n];
  }

  // Common starting point of every replica: local field 2*Q*x + b and energy
  // x^T Q x + b^T x (just b and 0 for the usual all-zero start).
  x0.assign(n, 0.0);
  bool nonzero_init = false;
  for (size_t i = 0; i < n && i < init_kernel.n_elem; ++i) {
    x0[i] = static_cast<double>(init_kernel(i));
    nonzero_init = nonzero_init || init_kernel(i) != 0;
  }
  lf0.assign(b.memptr(), b.memptr() + n);
  double e0 = 0.0;
  if (nonzero_init) {
    const arma::vec xv(x0);
    const arma::vec lfv = 2.0 * (q * xv) + b;
    lf0.assign(lfv.memptr(), lfv.memptr() + n);
    e0 = arma::as_scalar(xv.t() * q * xv + b.t() * xv);
  }

  const uint64_t base_seed =
      cfg.seeded ? 0 : static_cast<uint64_t>(std::random_device{}());

  bool have_best = false;
  double best_energy = 0.0;
  best_solution.assign(n, 0.0);
  size_t stale_runs = 0;

  kernels.resize(num_replicas);
  local_fields.resize(num_replicas);
  energies.resize(num_replicas);

  for (size_t sample = 0; sample < std::max<size_t>(cfg.runs, 1); ++sample) {
    PcgRng rng(base_seed + 15485863ULL * sample);
    for (size_t r = 0; r < num_replicas; ++r) {
      kernels[r] = x0;
      local_fields[r] = lf0;
      energies[r] = e0;
    }

    double sample_best_energy = energies[0];
    sample_best = kernels[0];
    double energy_before_sweep = sample_best_energy;
    size_t stale_sweeps = 0;

    for (size_t sweep = 1; sweep <= cfg.iters; ++sweep) {
      for (size_t r = 0; r < num_replicas; ++r) {
        double* x = kernels[r].data();
        double* lf = local_fields[r].data();
        for (size_t idx = 0; idx < n; ++idx) {
          const Move mv = SampleNeighbor(x, lf, q_diag.data(), energies[r], idx,
                                         betas[r], cfg.value_min,
                                         cfg.value_max, neighbors, rng);
          if (mv.delta != 0) {
            x[idx] += mv.delta;
            const double step = 2.0 * mv.delta;
            const double* q_col = q_mem + idx * n;
            for (size_t k = 0; k < n; ++k) {
              lf[k] += step * q_col[k];
            }
            energies[r] = mv.energy;
            if (mv.energy < sample_best_energy) {
              sample_best_energy = mv.energy;
              sample_best = kernels[r];
            }
          }
        }
      }

      if (sweep >= 3 && cfg.swap_period > 0 && sweep % cfg.swap_period == 0) {
        for (size_t r = 0; r + 1 < num_replicas; ++r) {
          const double delta =
              (betas[r] - betas[r + 1]) * (energies[r] - energies[r + 1]);
          if (delta >= 0.0 || std::exp(delta) > rng.Uniform01()) {
            std::swap(kernels[r], kernels[r + 1]);
            std::swap(local_fields[r], local_fields[r + 1]);
            std::swap(energies[r], energies[r + 1]);
          }
        }
      }

      if (sample_best_energy < energy_before_sweep) {
        energy_before_sweep = sample_best_energy;
        stale_sweeps = 0;
      } else {
        ++stale_sweeps;
        if (cfg.early_stop_patience > 0 &&
            stale_sweeps >= cfg.early_stop_patience) {
          break;
        }
      }
    }

    const bool improved_global = !have_best || sample_best_energy < best_energy;
    if (improved_global) {
      have_best = true;
      best_energy = sample_best_energy;
      best_solution = sample_best;
      stale_runs = 0;
    } else {
      ++stale_runs;
      if (cfg.run_patience > 0 && stale_runs >= cfg.run_patience) {
        break;
      }
    }
  }

  VppSolverResult result;
  result.solution.set_size(n);
  for (size_t i = 0; i < n; ++i) {
    result.solution(i) = static_cast<arma::sword>(std::llround(best_solution[i]));
  }
  result.energy = best_energy;
  return result;
}

arma::mat LllReduce(const arma::mat& basis, double delta) {
  const int n = static_cast<int>(basis.n_cols);
  arma::mat b = basis;
  arma::mat t(n, n, arma::fill::eye);
  arma::mat mu(n, n, arma::fill::zeros);
  std::vector<double> bnorm(n, 0.0);

  {  // one-time full Gram-Schmidt
    arma::mat bs = b;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < i; ++j) {
        mu(i, j) = (bnorm[j] > 0.0) ? arma::dot(b.col(i), bs.col(j)) / bnorm[j] : 0.0;
        bs.col(i) -= mu(i, j) * bs.col(j);
      }
      bnorm[i] = arma::dot(bs.col(i), bs.col(i));
    }
  }

  // Size-reduce column k against column l (l<k): rewrites only mu row k.
  auto reduce = [&](int k, int l) {
    if (std::fabs(mu(k, l)) <= 0.5) {
      return;
    }
    const double q = std::round(mu(k, l));
    b.col(k) -= q * b.col(l);
    t.col(k) -= q * t.col(l);
    for (int j = 0; j < l; ++j) {
      mu(k, j) -= q * mu(l, j);
    }
    mu(k, l) -= q;
  };

  int k = 1;
  long guard = 0;
  const long guard_max = 4000L * n * n + 10000L;
  while (k < n && guard++ < guard_max) {
    for (int j = k - 1; j >= 0; --j) {
      reduce(k, j);
    }
    const double lhs = bnorm[k];
    const double rhs = (delta - mu(k, k - 1) * mu(k, k - 1)) * bnorm[k - 1];
    if (lhs >= rhs) {
      ++k;
    } else {
      // Swap columns k-1, k and update the Gram-Schmidt state locally
      // (Cohen, "A Course in Computational Algebraic Number Theory", Alg 2.6.3).
      const double m = mu(k, k - 1);
      const double bk = bnorm[k] + m * m * bnorm[k - 1];
      mu(k, k - 1) = (bk > 0.0) ? m * bnorm[k - 1] / bk : 0.0;
      bnorm[k] = (bk > 0.0) ? bnorm[k - 1] * bnorm[k] / bk : 0.0;
      bnorm[k - 1] = bk;
      b.swap_cols(k, k - 1);
      t.swap_cols(k, k - 1);
      for (int j = 0; j <= k - 2; ++j) {
        std::swap(mu(k, j), mu(k - 1, j));
      }
      const double mkk1 = mu(k, k - 1);
      for (int i = k + 1; i < n; ++i) {
        const double tmp = mu(i, k);
        mu(i, k) = mu(i, k - 1) - m * tmp;
        mu(i, k - 1) = tmp + mkk1 * mu(i, k);
      }
      k = std::max(k - 1, 1);
    }
  }
  return t;
}
