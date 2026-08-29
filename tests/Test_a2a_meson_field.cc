/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: tests/Test_a2a_meson_field.cc

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
  Regression harness for the A2A meson field, in the same shape as
  Test_extended_meson_field and Test_chromo_magnetic_operator_field: the
  production contraction diffed against a CPU reference that shares none of
  its machinery.

    ref   Masaaki's CPU path, lifted from
          Hadrons/Modules/MContraction/A2AFewMesonField.hpp. Conjugates left
          and applies the gamma to right in host view loops, contracts the
          twelve spin-colour terms per site with a scalar accumulation,
          projects every momentum in the same loop, extracts SIMD lanes,
          zero-pads the local timeslices out to the global time extent, and
          finishes with one padded GlobalSumVector.

    gpu   NewMesonField<FImpl>::compute from A2Autils.h -- the same call
          sequence the Hadrons module runs, minus its blocking offsets and
          IO: gamma on device, packing and conjugation inside A2ASpatialSum,
          one batched GEMM over all momenta, then SumRing.

  The two agree on nothing but the answer. Different contraction (scalar,
  twelve unrolled terms, versus one batched GEMM), different momentum
  placement (inside the site loop versus applied to the GEMM's right operand
  beforehand), different reduction (zero-padded allreduce over the whole
  communicator versus a ring reduce over the spatial ranks plus a ring
  gather over the time ranks), different index layout. That is the point:
  the reference has been in production for years, so a disagreement points
  at the replacement rather than at an assumption they share.

  This is a correctness test. Both paths run once, on the same vectors, and
  it passes on the diff. The stage timings are printed because they are
  useful to look at, not because anything is judged on them.

  What this does NOT cover, by construction: the blocking offsets Hadrons
  passes to PackRight/PackLeftConj, and the IO fill. Those are module-level
  concerns, tested Hadrons-side.

  The two paths deliberately run at different cacheBlock -- the reference at
  the module's own default, the contraction at one tile -- so the comparison
  also exercises the tiling being irrelevant to the answer.

  Usage:
    Test_a2a_meson_field --grid 16.16.16.32 --mpi 1.1.1.4 \
                         --Ni 8 --Nj 8 --nmom 2 --cacheBlock 0

  cacheBlock <= 0 selects one tile for the contraction path.
*/

#include <Grid/Grid.h>
#include <Grid/algorithms/blas/A2ASpatialSum.h>

using namespace Grid;

typedef WilsonImplD                        FImpl;
typedef typename FImpl::FermionField       FermionField;
typedef typename FImpl::ComplexField       ComplexField;

typedef vComplexD                          vector_type;
typedef typename vector_type::scalar_type  scalar_type;
typedef iSpinColourVector<vector_type>     SpinColourVector_v;
typedef iSinglet<vector_type>              Scalar_v;
typedef iSinglet<scalar_type>              Scalar_s;

// result[t][i][m][j] throughout: the order SumRing writes, so the reference
// is the one doing the mapping rather than the path under test.
typedef Eigen::Tensor<ComplexD, 4, Eigen::RowMajor> ResultTensor;

static inline double Tms(double us) { return us * 1e-3; }

static const char *const Label6[6] = {
  "GEMM           ", "device<->host  ", "gather to slab ",
  "spatial reduce ", "scatter        ", "temporal gather" };

// Prints the SumRing stage breakdown in the same tagged form as the stages
// above, with each slot reduced across ranks.
//
// The max/mean ratio is the skew diagnostic. Rank skew does not show up as a
// gap between the wall time and the sum of these slots -- every microsecond
// is accounted for either way. It hides inside the reduce slot, because a
// rank that arrives early blocks there until the last rank shows up, and
// that wait is charged to the collective. So it is only visible across
// ranks: near 1 means the ranks are arriving together and the number is a
// real communication cost, while 2-3x means much of what looks like reduce
// time is waiting.
//
// Every rank must call this -- the reductions are collective -- even though
// only rank 0's output survives Grid_quiesce_nodes.
template <size_t NSLOT>
static void ReportStages(GridBase *grid,
                         const std::string &tag,
                         const std::string &block,
                         const std::array<double, NSLOT> &t,
                         const std::array<double, NSLOT> &b,
                         const char *const *label)
{
  // ProcessorCount, not RankCount: the sums below reduce over this grid's
  // communicator, whereas RankCount reports GlobalSharedMemory::WorldSize.
  // Equal for a full-machine grid, not for a split one.
  int nrank = grid->ProcessorCount();

  std::array<double, NSLOT> tmax = t, tsum = t;
  double tot = 0.;
  for (size_t k = 0; k < NSLOT; k++) {
    grid->GlobalMax(tmax[k]);
    grid->GlobalSum(tsum[k]);
    tot += t[k];
  }

  double tot_max = tot, tot_sum = tot;
  grid->GlobalMax(tot_max);
  grid->GlobalSum(tot_sum);

  auto gbps  = [](double bytes, double us) {
    return (us > 0.) ? bytes / us * 1.e6 / 1024. / 1024. / 1024. : 0.;
  };
  auto ratio = [](double mx, double mean) { return (mean > 0.) ? mx / mean : 0.; };

  for (size_t k = 0; k < NSLOT; k++) {
    double mean = tsum[k] / nrank;
    std::cout << GridLogMessage << tag << " " << label[k] << " "
              << Tms(mean) << " ms  max " << Tms(tmax[k]) << " ms"
              << "  max/mean " << ratio(tmax[k], mean);
    // bytesMoved is this rank's own count, uniform across ranks here, so
    // pairing it with the mean time gives the mean throughput. Slots with no
    // byte count (the compute steps) simply omit it. The two comms slots
    // count bytes on the wire, not payload -- see wire_ring_reduce in
    // A2ASpatialSum.h.
    if (b[k] > 0.) std::cout << "  (" << gbps(b[k], mean) << " GB/s)";
    std::cout << std::endl;
  }
  // Reduced from each rank's own total rather than summing the per-slot
  // maxima, which would overstate it whenever different ranks are the
  // slowest in different slots.
  std::cout << GridLogMessage << tag << " " << block << " "
            << Tms(tot_sum / nrank) << " ms  max " << Tms(tot_max) << " ms"
            << "  max/mean " << ratio(tot_max, tot_sum / nrank)
            << "  [TOTAL]" << std::endl;
}

static int OptInt(char **begin, char **end, const std::string &opt, int fallback)
{
  if (GridCmdOptionExists(begin, end, opt))
    return std::stoi(GridCmdOptionPayload(begin, end, opt));
  return fallback;
}

// ================================================================
// CPU reference: A2AFewMesonField's execute() with the Hadrons module
// scaffolding removed -- the two view loops that build the conjugated left
// and gamma-multiplied right vectors, then the four kernel stages.
// ================================================================
class A2AMesonFieldRef
{
public:
  static void compute(ResultTensor &result,
                      const std::vector<FermionField> &left,
                      const std::vector<FermionField> &right,
                      const std::vector<ComplexField> &ph,
                      Gamma::Algebra gamma,
                      int cacheBlock = 12)
  {
    GridBase *grid = left[0].Grid();
    const std::string tag = "[ref]";

    const int orthogdim = grid->Nd() - 1;
    int rd     = grid->_rdimensions[orthogdim];
    int ld     = grid->_ldimensions[orthogdim];
    int Nsimd  = grid->Nsimd();
    int e1     = grid->_slice_nblock[orthogdim];
    int e2     = grid->_slice_block [orthogdim];
    int stride = grid->_slice_stride[orthogdim];
    int ostr   = grid->_ostride[orthogdim];

    int nt   = (int)result.dimension(0);
    int N_i  = (int)left.size();
    int N_j  = (int)right.size();
    int nmom = (int)ph.size();

    if (cacheBlock <= 0) cacheBlock = std::max(N_i, N_j);

    double t0;

    // Left carries the conjugation and no gamma; the gamma goes on the
    // right. Both are plain host view loops, exactly as in the module.
    t0 = usecond();
    std::vector<SpinColourVector_v> zeroVec((size_t)rd * ostr, Zero());
    std::vector<std::vector<SpinColourVector_v>> leftv(N_i, zeroVec);
    for (int i = 0; i < N_i; i++) {
      autoView(tmp, left[i], CpuRead);
      thread_for(r, rd, {
        int so = r * ostr;
        for (int n = 0; n < e1; n++)
        for (int b = 0; b < e2; b++) {
          int ss = so + n * stride + b;
          leftv[i][ss] = conjugate(tmp[ss]);
        }
      });
    }
    std::cout << GridLogMessage << tag << " pack_left:       " << Tms(usecond()-t0) << " ms\n";

    t0 = usecond();
    std::vector<std::vector<SpinColourVector_v>> rightv(N_j, zeroVec);
    for (int j = 0; j < N_j; j++) {
      autoView(tmp, right[j], CpuRead);
      thread_for(r, rd, {
        int so = r * ostr;
        for (int n = 0; n < e1; n++)
        for (int b = 0; b < e2; b++) {
          int ss = so + n * stride + b;
          rightv[j][ss] = Gamma(gamma) * tmp[ss];
        }
      });
    }
    std::cout << GridLogMessage << tag << " pack_right:      " << Tms(usecond()-t0) << " ms\n";

    int pd = grid->_processors[orthogdim];
    int pc = grid->_processor_coor[orthogdim];

    // Stage times accumulate over the cacheBlock tiles and are reported once.
    double t_contract = 0., t_extract = 0., t_assemble = 0., t_gsv = 0., t_scatter = 0.;

    for (int ii = 0; ii < N_i; ii += cacheBlock)
    for (int jj = 0; jj < N_j; jj += cacheBlock) {
      int    Niii   = std::min(N_i - ii, cacheBlock);
      int    Njjj   = std::min(N_j - jj, cacheBlock);
      size_t MFrvol = (size_t)rd * nmom * Niii * Njjj;
      size_t MFlvol = (size_t)ld * nmom * Niii * Njjj;

      // Stage 1: spin-colour contract and momentum project, accumulating
      // into SIMD vectors indexed by the reduced time coordinate.
      t0 = usecond();
      std::vector<Scalar_v> lvSum(MFrvol, Zero());
      thread_for(r, rd, {
        int so = r * ostr;
        for (int n = 0; n < e1; n++)
        for (int b = 0; b < e2; b++) {
          int ss = so + n * stride + b;
          for (int iii = 0; iii < Niii; iii++)
          for (int jjj = 0; jjj < Njjj; jjj++) {
            Scalar_v tmp = Zero();
            for (int s = 0; s < Ns; s++)
            for (int c = 0; c < Nc; c++)
              tmp()()() = tmp()()()
                        + leftv[ii + iii][ss]()(s)(c) * rightv[jj + jjj][ss]()(s)(c);
            for (int m = 0; m < nmom; m++) {
              size_t idx = m + nmom * (jjj + Njjj * (iii + Niii * (size_t)r));
              autoView(mom_v, ph[m], CpuRead);
              auto phase = mom_v[ss];
              mac(&lvSum[idx], &tmp, &phase);
            }
          }
        }
      });
      t_contract += usecond() - t0;

      // Stage 2: unpack the SIMD lanes onto the local time coordinate.
      t0 = usecond();
      std::vector<Scalar_s> lsSum(MFlvol, scalar_type(0.0));
      thread_for(rt, rd, {
        Coordinate icoor(grid->Nd());
        ExtractBuffer<Scalar_s> extracted(Nsimd);
        for (int iii = 0; iii < Niii; iii++)
        for (int jjj = 0; jjj < Njjj; jjj++)
        for (int m = 0; m < nmom; m++) {
          size_t ij_rdx = m + nmom * (jjj + Njjj * (iii + Niii * (size_t)rt));
          extract(lvSum[ij_rdx], extracted);
          for (int idx = 0; idx < Nsimd; idx++) {
            grid->iCoorFromIindex(icoor, idx);
            int    ldx    = rt + icoor[orthogdim] * rd;
            size_t ij_ldx = m + nmom * (jjj + Njjj * (iii + Niii * (size_t)ldx));
            lsSum[ij_ldx] = lsSum[ij_ldx] + extracted[idx];
          }
        }
      });
      t_extract += usecond() - t0;

      // Stage 3: zero-pad to the global time extent. This rank writes its
      // own timeslices and zeros everywhere else, which is what lets the
      // allreduce below double as a concatenation along time -- the trick
      // SumRing replaces with an explicit gather.
      t0 = usecond();
      std::vector<ComplexD> mfCache((size_t)nmom * nt * Niii * Njjj, ComplexD(0.0, 0.0));
      thread_for(lt, ld, {
        for (int pt = 0; pt < pd; pt++) {
          if (pt != pc) continue;
          int t = lt + pt * ld;
          for (int iii = 0; iii < Niii; iii++)
          for (int jjj = 0; jjj < Njjj; jjj++)
          for (int m = 0; m < nmom; m++) {
            size_t ij_dx = m + nmom * (jjj + Njjj * (iii + Niii * (size_t)lt));
            mfCache[((size_t)m * nt + t) * Niii * Njjj + iii * Njjj + jjj]
                = lsSum[ij_dx]()()();
          }
        }
      });
      t_assemble += usecond() - t0;

      // Stage 4: the padded allreduce, over every rank in the communicator.
      t0 = usecond();
      grid->GlobalSumVector(mfCache.data(), (int)((size_t)nmom * nt * Niii * Njjj));
      t_gsv += usecond() - t0;

      t0 = usecond();
      for (int t   = 0; t   < nt;   t++)
      for (int iii = 0; iii < Niii; iii++)
      for (int jjj = 0; jjj < Njjj; jjj++)
      for (int m   = 0; m   < nmom; m++)
        result(t, ii + iii, m, jj + jjj)
            = mfCache[((size_t)m * nt + t) * Niii * Njjj + iii * Njjj + jjj];
      t_scatter += usecond() - t0;
    }

    std::cout << GridLogMessage << tag << " contract:        " << Tms(t_contract) << " ms\n";
    std::cout << GridLogMessage << tag << " lane_extract:    " << Tms(t_extract)  << " ms\n";
    std::cout << GridLogMessage << tag << " time_assemble:   " << Tms(t_assemble) << " ms\n";
    std::cout << GridLogMessage << tag << " globalsum:       " << Tms(t_gsv)      << " ms\n";
    std::cout << GridLogMessage << tag << " scatter:         " << Tms(t_scatter)  << " ms\n";
  }
};

int main(int argc, char *argv[])
{
  Grid_init(&argc, &argv);

  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(4, vComplexD::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  GridCartesian grid(latt_size, simd_layout, mpi_layout);

  int nd = grid.Nd();
  int nt = latt_size[nd - 1];

  int N_i        = OptInt(argv, argv + argc, "--Ni",         8);
  int N_j        = OptInt(argv, argv + argc, "--Nj",         8);
  int nmom       = OptInt(argv, argv + argc, "--nmom",       2);
  int cacheBlock = OptInt(argv, argv + argc, "--cacheBlock", 0);

  GRID_ASSERT(N_i > 0 && N_j > 0 && nmom > 0);

  int lt = grid.LocalDimensions()[nd - 1];
  int Pt = mpi_layout[nd - 1];

  Gamma::Algebra gamma = Gamma::Algebra::Gamma5;

  std::cout << GridLogMessage << "A2A meson field: CPU reference vs contraction" << std::endl;
  std::cout << GridLogMessage << "  lattice     = " << latt_size  << std::endl;
  std::cout << GridLogMessage << "  mpi         = " << mpi_layout << std::endl;
  std::cout << GridLogMessage << "  nt global   = " << nt << ", nt local = " << lt
            << ", ranks in time = " << Pt << std::endl;
  std::cout << GridLogMessage << "  N_i         = " << N_i << ", N_j = " << N_j
            << ", nmom = " << nmom << std::endl;
  std::cout << GridLogMessage << "  cacheBlock  = " << cacheBlock
            << (cacheBlock <= 0 ? "  (one tile)" : "") << std::endl;
  std::cout << GridLogMessage << "  meson field = "
            << (double)nt * N_i * N_j * nmom * sizeof(scalar_type) / 1024. / 1024.
            << " MiB" << std::endl;

  GridParallelRNG pRNG(&grid);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});

  std::vector<FermionField> left (N_i, &grid);
  std::vector<FermionField> right(N_j, &grid);
  for (auto &f : left)  random(pRNG, f);
  for (auto &f : right) random(pRNG, f);

  // Absolute phases, built once and handed to both paths: p = (m,0,0) with
  // the module's own formula, exp(2*pi*i*sum_mu (p_mu/L_mu)*x_mu). Building
  // them separately per path would only test the formula against itself.
  std::vector<ComplexField> ph(nmom, &grid);
  {
    ComplexField coor(&grid);
    ComplexD     I(0.0, 1.0);
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
  // cost that would otherwise land in whichever path runs first.
  { accelerator_for(i, 1, 1, { (void)i; }); }

  ResultTensor result_ref(nt, N_i, nmom, N_j);
  ResultTensor result_gpu(nt, N_i, nmom, N_j);
  result_ref.setZero();
  result_gpu.setZero();

  double start = usecond();
  A2AMesonFieldRef::compute(result_ref, left, right, ph, gamma);
  double t_ref = usecond() - start;

  typedef NewMesonField<FImpl> MF;

  std::array<double, 6>            sumT  = {}, sumB = {};
  std::array<double, MF::NCompute> compT = {}, compB = {};   // no byte counts
  start = usecond();
  MF::compute(result_gpu, left, right, ph, gamma, cacheBlock, sumT, sumB, compT);
  double t_gpu = usecond() - start;

  ReportStages(&grid, "[gpu]", "compute:       ", compT, compB, MF::ComputeLabel);
  ReportStages(&grid, "[gpu]", "SumRing:       ", sumT,  sumB,  Label6);

  double norm2_ref = 0.0, norm2_gpu = 0.0, norm2_diff = 0.0;
  for (int t = 0; t < nt;   t++)
  for (int i = 0; i < N_i;  i++)
  for (int m = 0; m < nmom; m++)
  for (int j = 0; j < N_j;  j++) {
    ComplexD r = result_ref(t, i, m, j);
    ComplexD c = result_gpu(t, i, m, j);
    norm2_ref  += norm2(r);
    norm2_gpu  += norm2(c);
    norm2_diff += norm2(r - c);
  }

  double rel = (norm2_ref > 0.0) ? std::sqrt(norm2_diff / norm2_ref) : 0.0;

  std::cout << GridLogMessage
            << "  norm2_ref=" << norm2_ref
            << "  norm2_gpu=" << norm2_gpu
            << std::endl;
  std::cout << GridLogMessage
            << "  rel_gpu=" << rel
            << std::endl;
  std::cout << GridLogMessage
            << "  t_ref=" << t_ref * 1e-6 << "s"
            << "  t_gpu=" << t_gpu * 1e-6 << "s"
            << std::endl;

  // Both paths sum the same terms in different orders, so they agree to
  // rounding rather than bitwise.
  GRID_ASSERT(rel < 1e-10);

  std::cout << GridLogMessage
            << "A2A meson field: reference and contraction agree." << std::endl;

  Grid_finalize();
  return EXIT_SUCCESS;
}
