/**
 * @file vpp_solver.h
 * @brief Integer-quadratic-program solver used to search for the VPP
 * perturbation vector at the downlink precoder, plus the LLL lattice
 * reduction used by the "VPP" beamforming algorithm.
 *
 * VppSolveInteger is a port of the parallel-tempering solver `solve_iqp`
 * (swap_period >= 1 path) and its `sample_neighbor` heat-bath move in
 * QUIOVPPDOWNLINK/scripts/7_8_26_intmaxbm_downlink.cpp, rewritten against
 * Armadillo. The random-number stream, the move/swap acceptance rules and the
 * order in which they are consumed are kept identical, so for the same
 * (Q, b, ladder, seed) it returns the same solution as the reference. The
 * reference runs its N_s samples on N_s threads; here they run one after the
 * other because this is called from inside an already-pinned Agora worker.
 *
 * As in the reference, Q and b should be normalized by max|Q| before the call
 * (argmin is scale-invariant) so that one fixed inverse-temperature ladder
 * lands in the same regime for every instance -- see doprecode.cc.
 *
 * VppSolveSphere (exact) and VppSolveFixedComplexity (fixed cost) solve the
 * same problem for the "SE" and "FCSE" beamforming modes.
 *
 * NOT real-time-safe: every solver here runs a search per call. They are only
 * reached when "beamforming" is "VPP", "SE" or "FCSE"; the default
 * ZF/MMSE/MRC paths never touch this file.
 */
#ifndef VPP_SOLVER_H_
#define VPP_SOLVER_H_

#include <armadillo>
#include <cstddef>

struct VppSolverConfig {
  int value_min;  // variables live in [value_min, value_max]
  int value_max;
  size_t num_replicas;     // N_R: rungs of the inverse-temperature ladder
  size_t runs;             // N_s: independent restarts, best energy wins
  size_t iters;            // N_C: sweeps over all variables
  size_t neighbor_window;  // proposal half-width (1 = exact +-1 heat bath)
  double beta_min;         // geometric ladder, beta_max (cold) .. beta_min (hot)
  double beta_max;
  size_t swap_period = 1;  // replica exchange every swap_period-th sweep >= 3
  bool seeded = true;      // deterministic RNG (true) or random_device (false)
  // 0 (default) = always run all `iters` sweeps of every `run`, matching the
  // original behavior exactly. > 0: bail out of a run's sweep loop early once
  // `sample_best_energy` has gone this many consecutive sweeps (checked once
  // per sweep, i.e. after that sweep's replica-exchange step too) without
  // improving -- the remaining sweeps of that run are skipped, but later runs
  // (if cfg.runs > 1) still start fresh and get the full budget. See
  // VppSolveInteger.
  size_t early_stop_patience = 0;
  // 0 (default) = always run all `runs` restarts, matching the original
  // behavior exactly. > 0: stop issuing further restarts once this many
  // consecutive restarts in a row have failed to improve on the
  // best-energy-so-far (across all restarts, not just the current one). See
  // VppSolveInteger.
  size_t run_patience = 0;
};

struct VppSolverResult {
  arma::ivec solution;
  double energy;
};

/**
 * @brief Minimizes x^T Q x + b^T x over integer x in
 * [cfg.value_min, cfg.value_max]^n, starting every replica of every run from
 * init_kernel. Q must be symmetric. The reported energy is in the units of
 * the (Q, b) that were passed in.
 */
VppSolverResult VppSolveInteger(const arma::mat& q, const arma::vec& b,
                                const arma::ivec& init_kernel,
                                const VppSolverConfig& cfg);

struct VppSphereConfig {
  int value_min;  // variables live in [value_min, value_max]
  int value_max;
};

// Upper bound on FCSE's (value range)^full_layers path count; guards against a
// misconfigured full_layers overflowing the count rather than just running slow.
constexpr size_t kVppFcseMaxPaths = 10000000;

struct VppFixedComplexityConfig {
  int value_min;
  int value_max;
  size_t full_layers;  // p: first-decided tree layers expanded exhaustively
};

/**
 * @brief Sphere Encoding (SE): exact minimizer of x^T Q x + b^T x over
 * integer x in [cfg.value_min, cfg.value_max]^n, via depth-first
 * branch-and-bound with Schnorr-Euchner (zig-zag from the conditional centre)
 * ordering on chol(Q). init_kernel only seeds the initial sphere radius; the
 * result is the global optimum regardless. Unlike VppSolveInteger there is no
 * iteration budget, so runtime depends on the conditioning of Q (worst case
 * approaches full enumeration). Returns x = 0 if Q is not positive definite.
 */
VppSolverResult VppSolveSphere(const arma::mat& q, const arma::vec& b,
                               const arma::ivec& init_kernel,
                               const VppSphereConfig& cfg);

/**
 * @brief Fixed-Complexity Sphere Encoding (FCSE, Barbero & Thompson FSD):
 * same objective, but the first cfg.full_layers decided layers are expanded
 * over every value in the box and each resulting path is completed by greedy
 * (rounded conditional centre) decisions, with no pruning. Deterministic cost
 * (value range)^full_layers * O(n^2); full_layers >= n is exhaustive (exact),
 * full_layers == 0 is a single Babai rounding pass. full_layers is lowered
 * (with a warning) if the path count would exceed kVppFcseMaxPaths. Returns
 * x = 0 if Q is not positive definite. init_kernel is unused.
 */
VppSolverResult VppSolveFixedComplexity(const arma::mat& q, const arma::vec& b,
                                        const arma::ivec& init_kernel,
                                        const VppFixedComplexityConfig& cfg);

/**
 * @brief Classic LLL lattice reduction (Lovász condition, incrementally
 * maintained Gram-Schmidt) on the columns of `basis`. Returns the unimodular
 * integer matrix T such that basis*T is reduced -- i.e. for a quadratic form
 * Q = basis.t()*basis, T.t()*Q*T is the reduced (far more diagonal) form of
 * the same lattice. Ported from dl_lll in
 * QUIOVPPDOWNLINK/scripts/7_8_26_intmaxbm_downlink.cpp. Used by the "VPP"
 * beamforming algorithm as a precondition before searching -- see
 * doprecode.cc::ApplyVppPerturbation.
 */
arma::mat LllReduce(const arma::mat& basis, double delta = 0.99);

#endif  // VPP_SOLVER_H_
