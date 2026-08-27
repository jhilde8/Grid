/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: tests/Test_a2a_spatial_sum.cc

Copyright (C) 2015-2026

Author: Jonas Hildebrand <jonas.hildebrand@uconn.edu>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution directory
*************************************************************************************/

/*
  Regression and scaling harness for A2ASpatialSum's post-GEMM reduction
  variants.

  Every variant computes the identical object -- EMF[t][i][j][m] over the
  global time extent -- from the identical packed buffers, and differs only
  in how the GEMM output is turned into that object:

    SumAllMomentaCacheBlocked        baseline. Copies the whole GEMM output
                                     to the host, then per (i,j) tile fills
                                     a zero-padded nt_global buffer on the
                                     host and reduces it there.

    SumAllMomentaCacheBlockedDevice  fills and reduces each tile on the
                                     device, and copies down once per tile
                                     after the reduction.

  A ring-gather variant is planned and drops in here as a third column.

  Because the packing is shared and the reduction is the only thing that
  differs, the comparison is exact rather than statistical: on a build with
  no accelerator the two paths feed byte-identical buffers to the same
  GlobalSumVector on the same communicator, so they should agree bitwise.
  On an accelerator build MPI may pick a different reduction algorithm for a
  device pointer than for a host pointer, which reorders the summation and
  leaves round-off level differences. The test prints the relative
  difference either way so it is visible which regime you are in.

  The random field content is irrelevant to the timings that matter: the
  cost of the collective and of the staging copy depends only on buffer
  size and rank topology. So running this at production N_i / N_j / nmom on
  a production rank layout measures the reduction essentially exactly. What
  it does not reproduce is the HBM pressure of resident A2A vectors, which
  affects what fits rather than how fast the reduction runs.

  Usage:
    Test_a2a_spatial_sum --grid 16.16.16.32 --mpi 1.1.1.4 \
                         --Ni 8 --Nj 8 --nmom 2 --cacheBlock 4 --nrepeat 5

  cacheBlock <= 0 selects max(N_i, N_j), i.e. one tile, which is the
  degenerate "tiling off" case.
*/

#include <Grid/Grid.h>
#include <Grid/algorithms/blas/A2ASpatialSum.h>

using namespace Grid;

typedef vComplexD                          vector_type;
typedef typename vector_type::scalar_type  scalar_type;
typedef iSpinColourVector<vector_type>     SpinColourVector_v;
typedef Lattice<SpinColourVector_v>        FermionField;
typedef Eigen::Tensor<ComplexD, 4, Eigen::RowMajor> ResultTensor;

// Mirrors the "Sum detail" block that A2ANewMesonField logs, so a run of
// this test and a production log can be read side by side, with one
// addition: each slot is reduced across ranks and reported as mean and max.
//
// The max/mean ratio is the skew diagnostic. Rank skew does not show up as
// a gap between the Sum wall time and the sum of these slots -- every
// microsecond is accounted for either way. It hides inside slot 3, because
// a rank that arrives early at the collective blocks there until the last
// rank shows up, and that wait is charged to the collective. So it is only
// visible across ranks: a max/mean near 1 on slot 3 means the ranks are
// arriving together and the number is a real collective cost, while 2-3x
// means a large part of what looks like GSV time is actually waiting.
//
// Every rank must call this -- the reductions are collective -- even though
// only rank 0's output survives Grid_quiesce_nodes.
//
// timings and bytesMoved accumulate across calls, so dividing by nrepeat
// gives the per-call figure while the GB/s ratios stay correct without it.
// Returns the mean per-call time in each slot, for the caller to compare
// variants with.
template <size_t NSLOT>
static std::array<double, NSLOT> ReportTimings(GridBase *grid,
                                               const std::string &tag,
                                               const std::array<double, NSLOT> &t,
                                               const std::array<double, NSLOT> &b,
                                               const char *const *label,
                                               int nrepeat)
{
  // ProcessorCount, not RankCount: the sums below reduce over this grid's
  // communicator, whereas RankCount reports GlobalSharedMemory::WorldSize.
  // Equal for a full-machine grid, not for a split one.
  int nrank = grid->ProcessorCount();

  std::array<double, NSLOT> tmax = t, tsum = t, tmean = {};
  double tot = 0.;
  for (size_t k = 0; k < NSLOT; k++) {
    grid->GlobalMax(tmax[k]);
    grid->GlobalSum(tsum[k]);
    tmean[k] = tsum[k] / nrank;
    tot     += t[k];
  }

  double tot_max  = tot;
  double tot_sum  = tot;
  grid->GlobalMax(tot_max);
  grid->GlobalSum(tot_sum);
  double tot_mean = tot_sum / nrank;

  auto gbps = [](double bytes, double us) {
    return (us > 0.) ? bytes / us * 1.e6 / 1024. / 1024. / 1024. : 0.;
  };
  auto ratio = [](double mx, double mean) { return (mean > 0.) ? mx / mean : 0.; };

  std::cout << GridLogMessage << tag << " per call (us), over " << nrank
            << " ranks:" << std::endl;
  for (size_t k = 0; k < NSLOT; k++) {
    std::cout << GridLogMessage << "    " << label[k]
              << " mean " << tmean[k] / nrepeat
              << "  max " << tmax[k] / nrepeat
              << "  max/mean " << ratio(tmax[k], tmean[k]);
    // bytesMoved is this rank's own count; it is uniform across ranks here,
    // so pairing it with the mean time gives the mean throughput.
    if (k > 0) std::cout << "  (" << gbps(b[k], tmean[k]) << " GB/s)";
    std::cout << std::endl;
  }
  // Reduced from each rank's own total rather than summing the per-slot
  // maxima, which would overstate it whenever different ranks are the
  // slowest in different slots.
  std::cout << GridLogMessage << "    total          "
            << " mean " << tot_mean / nrepeat
            << "  max " << tot_max / nrepeat
            << "  max/mean " << ratio(tot_max, tot_mean) << std::endl;

  return tmean;
}

static const char *const Label5[5] = {
  "GEMM           ", "device<->host  ", "fill           ",
  "GlobalSumVector", "scatter        " };

static const char *const Label6[6] = {
  "GEMM           ", "device<->host  ", "gather to slab ",
  "spatial reduce ", "scatter        ", "temporal gather" };

static int OptInt(char **begin, char **end, const std::string &opt, int fallback)
{
  if (GridCmdOptionExists(begin, end, opt))
    return std::stoi(GridCmdOptionPayload(begin, end, opt));
  return fallback;
}

int main(int argc, char *argv[])
{
  Grid_init(&argc, &argv);

  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(4, vComplexD::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  GridCartesian grid(latt_size, simd_layout, mpi_layout);

  int nd  = grid.Nd();
  int nt  = latt_size[nd - 1];

  int N_i        = OptInt(argv, argv + argc, "--Ni",         8);
  int N_j        = OptInt(argv, argv + argc, "--Nj",         8);
  int nmom       = OptInt(argv, argv + argc, "--nmom",       2);
  int cacheBlock = OptInt(argv, argv + argc, "--cacheBlock", 4);
  int nrepeat    = OptInt(argv, argv + argc, "--nrepeat",    1);

  if (cacheBlock <= 0) cacheBlock = std::max(N_i, N_j);
  GRID_ASSERT(N_i > 0 && N_j > 0 && nmom > 0 && nrepeat > 0);

  int lt = grid.LocalDimensions()[nd - 1];
  int Pt = mpi_layout[nd - 1];

  std::cout << GridLogMessage << "A2ASpatialSum reduction-variant regression" << std::endl;
  std::cout << GridLogMessage << "  lattice     = " << latt_size  << std::endl;
  std::cout << GridLogMessage << "  mpi         = " << mpi_layout << std::endl;
  std::cout << GridLogMessage << "  nt global   = " << nt << ", nt local = " << lt
            << ", ranks in time = " << Pt << std::endl;
  std::cout << GridLogMessage << "  N_i         = " << N_i << ", N_j = " << N_j
            << ", nmom = " << nmom << std::endl;
  std::cout << GridLogMessage << "  cacheBlock  = " << cacheBlock
            << " (" << ((N_i + cacheBlock - 1) / cacheBlock)
                     * ((N_j + cacheBlock - 1) / cacheBlock)
            << " tiles per call)" << std::endl;
  std::cout << GridLogMessage << "  nrepeat     = " << nrepeat << std::endl;
  std::cout << GridLogMessage << "  reduced object per call = "
            << (double)nt * N_i * N_j * nmom * sizeof(scalar_type) / 1024. / 1024.
            << " MiB" << std::endl;

  GridParallelRNG pRNG(&grid);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});

  std::vector<FermionField> left (N_i, &grid);
  std::vector<FermionField> right(N_j, &grid);
  for (auto &f : left)  random(pRNG, f);
  for (auto &f : right) random(pRNG, f);

  // Momentum phases. Any valid set of phases exercises the path, so these
  // are built directly as p = (m,0,0) rather than replicating the module's
  // phase-difference bookkeeping -- ApplyAllPhaseRight wants absolute
  // phases, one per momentum.
  std::vector<LatticeComplexD> ph(nmom, &grid);
  {
    LatticeComplexD coor(&grid);
    ComplexD        I(0.0, 1.0);
    for (int m = 0; m < nmom; m++) {
      ph[m] = Zero();
      for (int mu = 0; mu < nd - 1; mu++) {
        LatticeCoordinate(coor, mu);
        RealD p_mu = (mu == 0) ? (RealD)m : 0.0;
        ph[m] = ph[m] + (p_mu / (RealD)latt_size[mu]) * coor;
      }
      ph[m] = exp((RealD)(2 * M_PI) * I * ph[m]);
    }
  }

  // Force accelerator initialisation outside every timed region: the first
  // kernel launch and the first device allocation of a run carry start-up
  // cost that would otherwise land in whichever variant happens to run
  // first and make it look artificially slow.
  { accelerator_for(i, 1, 1, { (void)i; }); }

  A2ASpatialSum<SpinColourVector_v> spatial_sum;

  spatial_sum.AllocateRight(N_j, &grid, nmom);
  spatial_sum.AllocateLeft(N_i, nmom);
  spatial_sum.PackLeftConj(left);
  spatial_sum.PackRight(right);

  std::vector<deviceVector<scalar_type>> ph_flat(nmom);
  for (int m = 0; m < nmom; m++)
    A2ASpatialSum<SpinColourVector_v>::PackPhase(&grid, ph[m], ph_flat[m]);

  // Reads LR_buf and writes LR_mom_buf, leaving LR_buf untouched, so one
  // call serves every variant below. Each variant then runs its own GEMM
  // with beta=0 over the same packed operands, which is why their
  // timings[0] should agree -- a free check that nothing upstream drifted
  // between them.
  spatial_sum.ApplyAllPhaseRight(ph_flat);

  // The Ring variant takes nmom BEFORE N_j, so that its tile agrees with
  // its swapped-operand GEMM and with a RowMajor result at the same time.
  // The comparison below maps between the two orderings.
  ResultTensor result_host(nt, N_i, N_j, nmom);
  ResultTensor result_dev (nt, N_i, N_j, nmom);
  ResultTensor result_ring(nt, N_i, nmom, N_j);
  result_host.setZero();
  result_dev.setZero();
  result_ring.setZero();

  // Untimed warm-up of each variant: the first call grows the persistent
  // buffers and lets MPI settle on an algorithm for this message size and
  // communicator.
  spatial_sum.SumAllMomentaCacheBlocked      (result_host, cacheBlock);
  spatial_sum.SumAllMomentaCacheBlockedDevice(result_dev,  cacheBlock);
  spatial_sum.SumAllMomentaCacheBlockedRing  (result_ring, cacheBlock);

  std::array<double, 5> t_host = {}, b_host = {};
  std::array<double, 5> t_dev  = {}, b_dev  = {};
  std::array<double, 6> t_ring = {}, b_ring = {};

  for (int r = 0; r < nrepeat; r++)
    spatial_sum.SumAllMomentaCacheBlocked(result_host, cacheBlock, &t_host, &b_host);

  for (int r = 0; r < nrepeat; r++)
    spatial_sum.SumAllMomentaCacheBlockedDevice(result_dev, cacheBlock, &t_dev, &b_dev);

  for (int r = 0; r < nrepeat; r++)
    spatial_sum.SumAllMomentaCacheBlockedRing(result_ring, cacheBlock, &t_ring, &b_ring);

  auto m_host = ReportTimings(&grid, "[SumAllMomentaCacheBlocked      ]", t_host, b_host, Label5, nrepeat);
  auto m_dev  = ReportTimings(&grid, "[SumAllMomentaCacheBlockedDevice]", t_dev,  b_dev,  Label5, nrepeat);
  auto m_ring = ReportTimings(&grid, "[SumAllMomentaCacheBlockedRing  ]", t_ring, b_ring, Label6, nrepeat);

  double norm2_host = 0.0, norm2_dev = 0.0, norm2_ring = 0.0;
  for (int t = 0; t < nt;   t++)
  for (int i = 0; i < N_i;  i++)
  for (int j = 0; j < N_j;  j++)
  for (int m = 0; m < nmom; m++) {
    ComplexD h = result_host(t, i, j, m);
    ComplexD d = result_dev (t, i, j, m);
    ComplexD r = result_ring(t, i, m, j);   // nmom before N_j
    norm2_host += norm2(h);
    norm2_dev  += norm2(h - d);
    norm2_ring += norm2(h - r);
  }

  double rel_dev  = (norm2_host > 0.0) ? std::sqrt(norm2_dev  / norm2_host) : 0.0;
  double rel_ring = (norm2_host > 0.0) ? std::sqrt(norm2_ring / norm2_host) : 0.0;

  std::cout << GridLogMessage << "norm2(host)       = " << norm2_host << std::endl;
  std::cout << GridLogMessage << "rel diff, device  = " << rel_dev
            << (norm2_dev == 0.0 ? "   (bitwise identical)" : "") << std::endl;
  std::cout << GridLogMessage << "rel diff, ring    = " << rel_ring
            << (norm2_ring == 0.0 ? "   (bitwise identical)" : "") << std::endl;

  // Compared on the rank-mean rather than this rank's own numbers, so a
  // single slow rank does not decide the ratio.
  double mt_host = 0., mt_dev = 0., mt_ring = 0.;
  for (int k = 0; k < 5; k++) { mt_host += m_host[k]; mt_dev += m_dev[k]; }
  for (int k = 0; k < 6; k++) { mt_ring += m_ring[k]; }

  // Comms-only comparison: slot 3 is the collective in every variant, plus
  // the ring's separate temporal gather in slot 5.
  double comms_host = m_host[3];
  double comms_dev  = m_dev[3];
  double comms_ring = m_ring[3] + m_ring[5];

  if (mt_dev  > 0.0)
    std::cout << GridLogMessage << "speedup vs host, device = " << mt_host / mt_dev
              << "x total, " << ((comms_dev > 0.0) ? comms_host / comms_dev : 0.0)
              << "x comms" << std::endl;
  if (mt_ring > 0.0)
    std::cout << GridLogMessage << "speedup vs host, ring   = " << mt_host / mt_ring
              << "x total, " << ((comms_ring > 0.0) ? comms_host / comms_ring : 0.0)
              << "x comms" << std::endl;

  // Loose enough to absorb a reordered reduction on an accelerator build,
  // tight enough that any indexing error shows up as a failure rather than
  // as a plausible-looking number. The ring is deterministic run to run
  // (see RingAllReduce.h) but its summation order differs from MPI's, so
  // it is not expected to match the host path bitwise.
  GRID_ASSERT(rel_dev  < 1e-12);
  GRID_ASSERT(rel_ring < 1e-12);

  std::cout << GridLogMessage << "A2ASpatialSum reduction variants agree." << std::endl;

  Grid_finalize();
  return EXIT_SUCCESS;
}
