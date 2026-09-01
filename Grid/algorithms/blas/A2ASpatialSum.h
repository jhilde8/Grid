/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: Grid/algorithms/blas/A2ASpatialSum.h

    Copyright (C) 2025

Author: Peter Boyle <pboyle@bnl.gov>
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
/*  END LEGAL */
#pragma once

NAMESPACE_BEGIN(Grid);

/*
  A2ASpatialSum

  Replaces the scalar spatial accumulation loop in A2A extended meson field
  contractions with a batched GEMM over local time slices, enabling GPU offload.

  Given:
    leftv[N_i][osite]    - conjugated left SpinColourVectors (SIMD-packed)
    loopRight[N_j][osite]- type-contracted right SpinColourVectors (SIMD-packed)

  Computes:
    EMF[i,j,t] = sum_{x,s,c} leftv[i][x,t,s,c] * loopRight[j][x,t,s,c]

  via batched GEMM over nt local time slices, then a ring reduction and a
  ring gather across MPI (see SumRing).

  Memory layout (all C row-major):
    W_buf      [nt][N_i][nxyz*Nsc]        W[t][i][x*Nsc+sc]  = leftv[i] at (x,t)
    LR_buf     [nt][N_j][nxyz*Nsc]        LR[t][j][x*Nsc+sc] = loopRight[j] at (x,t)
    LR_mom_buf [nt][nmom][N_j][nxyz*Nsc]  LRM[t][m][j][...]  = LR[t][j][...] * phase_m[x]
    EMF_mom_buf[nt][N_i][nmom*N_j]        GEMM output

  Momentum folds into the GEMM's N dimension rather than repeating the GEMM
  once per momentum: ApplyAllPhaseRight reads the unphased pack in LR_buf and
  writes nmom phase-multiplied copies into LR_mom_buf. K and M (nxyz*Nsc and
  N_i) are unchanged; N widens from N_j to nmom*N_j with m the slower-varying
  sub-index, so each timeslice is nmom contiguous [N_j][nxyz*Nsc] blocks --
  what gemmBatched wants, no repacking. At nmom=1 with no phase applied the
  GEMM reads LR_buf directly and LR_mom_buf is never allocated; see
  AllocateRight.

  Operand order and the index layout it produces are documented above SumRing.
*/
template<class vobj>
class A2ASpatialSum
{
public:
  typedef typename vobj::scalar_type   scalar;
  typedef typename vobj::scalar_object sobj;

  GridBase *grid;
  int N_i, N_j;
  int nt, nxyz, Nsc;
  int nmom;

  // Bytes the reduce puts on the wire per byte of payload: 2(P_d-1)/P_d
  // summed over the spatial dimensions only, matching
  // CartesianRingAllReduce(orthogDim = nd-1). Fixed by the process grid, so
  // set once in AllocateRight. Exact, being a count of sends this code
  // issues itself rather than an assumption about MPI.
  double wire_ring_reduce;

  deviceVector<scalar>   W_buf;
  deviceVector<scalar>   LR_buf;
  deviceVector<scalar *> W_ptrs;

  // Multi-momentum path: built alongside, not instead of, the buffers
  // above -- see class comment.
  deviceVector<scalar>   LR_mom_buf;
  deviceVector<scalar>   EMF_mom_buf;
  deviceVector<scalar *> LR_mom_ptrs;
  deviceVector<scalar *> EMF_mom_ptrs;

  // Staging for SumRing below. tile_buf holds one cacheBlock tile at full
  // nt_global extent. tile_host is the host landing buffer for the scatter
  // fallback only - the direct path copies into the caller's tensor and never
  // touches it, so it stays empty for a caller that always takes that path.
  // Both are sized lazily rather than in Allocate, because nt_global and
  // cacheBlock are only known at the call site.
  deviceVector<scalar>   tile_buf;
  std::vector<scalar>    tile_host;

  A2ASpatialSum() : grid(nullptr), N_i(0), N_j(0), nt(0), nxyz(0), Nsc(0), nmom(1),
                    wire_ring_reduce(0.0) {}

  // Aim LR_mom_ptrs at the buffer holding the GEMM's right operand. The
  // per-timeslice stride is nmom*N_j*nxyz*Nsc; when base is LR_buf that is
  // the unphased pack's own stride, LR_buf only ever holding one momentum.
  void PointRightOperand(scalar *base)
  {
    int lN_j = N_j, lnxyz = nxyz, lNsc = Nsc, lnmom = nmom;
    for (int t = 0; t < nt; t++)
      acceleratorPut(LR_mom_ptrs[t], base + (size_t)t * lnmom * lN_j * lnxyz * lNsc);
  }

  // Binds the grid and everything derived from it, so this runs before
  // AllocateLeft and before any Pack.
  //
  // LR_mom_ptrs names the GEMM's right operand: LR_buf while no phase has
  // been applied, LR_mom_buf once ApplyAllPhaseRight has written one. An
  // unphased caller therefore needs neither a second buffer nor a copy --
  // LR_buf already holds exactly what the GEMM reads -- while a single
  // momentum carrying a real phase still gets its own storage. Only nmom=1
  // can be left unphased, since LR_buf has room for one momentum.
  void AllocateRight(int _N_j, GridBase *_grid, int _nmom = 1)
  {
    grid = _grid;
    N_j  = _N_j;
    nmom = _nmom;
    Coordinate ldims = grid->LocalDimensions();
    nt   = ldims[grid->Nd() - 1];
    nxyz = grid->lSites() / nt;
    Nsc  = sizeof(sobj) / sizeof(scalar);

    wire_ring_reduce = 0.0;
    for (int d = 0; d < grid->Nd() - 1; d++) {
      int Pd = grid->ProcessorGrid()[d];
      if (Pd > 1) wire_ring_reduce += 2.0 * (Pd - 1) / (double)Pd;
    }

    size_t LRsz = (size_t)nt * N_j * nxyz * Nsc;
    if (LR_buf.size()      < LRsz)        LR_buf.resize(LRsz);
    if (LR_mom_ptrs.size() < (size_t)nt)  LR_mom_ptrs.resize(nt);

    // Only nmom=1 has a meaningful unphased default. Above that,
    // ApplyAllPhaseRight must run before the GEMM anyway, and it both sizes
    // LR_mom_buf and aims the pointers.
    if (nmom == 1) PointRightOperand(&LR_buf[0]);
  }

  // AllocateRight must be called first: N_j, nt, nxyz, Nsc and nmom come
  // from there rather than being passed again.
  void AllocateLeft(int _N_i)
  {
    N_i = _N_i;

    size_t Wsz    = (size_t)nt * N_i * nxyz * Nsc;
    size_t EMFMsz = (size_t)nt * nmom * N_j * N_i;
    if (W_buf.size()        < Wsz)        W_buf.resize(Wsz);
    if (EMF_mom_buf.size()  < EMFMsz)     EMF_mom_buf.resize(EMFMsz);
    if (W_ptrs.size()       < (size_t)nt) W_ptrs.resize(nt);
    if (EMF_mom_ptrs.size() < (size_t)nt) EMF_mom_ptrs.resize(nt);

    scalar *Wh    = &W_buf[0];
    scalar *EMFMh = &EMF_mom_buf[0];
    int lN_i = N_i, lN_j = N_j, lnxyz = nxyz, lNsc = Nsc, lnmom = nmom;
    for (int t = 0; t < nt; t++) {
      acceleratorPut(W_ptrs[t],       Wh    + (size_t)t * lN_i * lnxyz * lNsc);
      acceleratorPut(EMF_mom_ptrs[t], EMFMh + (size_t)t * lnmom * lN_j * lN_i);
    }
  }

  void PackLeft(const std::vector<Lattice<vobj>> &leftv, int start = 0, int count = -1)
  {
    if (count < 0) count = (int)leftv.size();
    GRID_ASSERT(start + count <= (int)leftv.size());
    GRID_ASSERT(count == N_i);
    PackVectors(leftv, &W_buf[0], N_i, start);
  }

  void PackRight(const std::vector<Lattice<vobj>> &loopRight, int start = 0, int count = -1)
  {
    if (count < 0) count = (int)loopRight.size();
    GRID_ASSERT(start + count <= (int)loopRight.size());
    GRID_ASSERT(count == N_j);
    PackVectors(loopRight, &LR_buf[0], N_j, start);
  }

  // Read directly from original (unconjugated) left vectors, conjugating during pack.
  void PackLeftConj(const std::vector<Lattice<vobj>> &left, int start = 0, int count = -1)
  {
    if (count < 0) count = (int)left.size();
    GRID_ASSERT(start + count <= (int)left.size());
    GRID_ASSERT(count == N_i);
    PackVectors<true>(left, &W_buf[0], N_i, start);
  }

public:
  // Pack vecs[start..start+N-1] lattice fields into buf[nt][N][nxyz*Nsc], extracting all SIMD lanes.
  // DoConj=true conjugates each element during extraction (used by PackLeftConj).
  template<bool DoConj = false>
  void PackVectors(const std::vector<Lattice<vobj>> &vecs, scalar *buf, int N, int start = 0)
  {
    int nd     = grid->_ndimension;
    int osites = grid->oSites();
    int Nsimd  = vobj::Nsimd();
    int lN     = N;
    int lNsc   = Nsc;
    int lnxyz  = nxyz;
    Coordinate rdimensions = grid->_rdimensions;
    Coordinate ldims       = grid->LocalDimensions();
    Coordinate simd        = grid->_simd_layout;

    for (int n = 0; n < N; n++) {
      autoView(src_v, vecs[start + n], AcceleratorRead);
      accelerator_for(sf, osites, Nsimd, {
#ifdef GRID_SIMT
        {
          int lane = acceleratorSIMTlane(Nsimd);
#else
          for (int lane = 0; lane < Nsimd; lane++) {
#endif
          Coordinate icoor(nd), ocoor(nd), lcoor(nd);
          Lexicographic::CoorFromIndex(icoor, lane, simd);
          Lexicographic::CoorFromIndex(ocoor, sf, rdimensions);
          for (int d = 0; d < nd; d++)
            lcoor[d] = rdimensions[d] * icoor[d] + ocoor[d];

          int     l_t = lcoor[nd - 1];
          Coordinate xyz_coor = lcoor;
          xyz_coor[nd - 1] = 0;
          int64_t l_xyz;
          Lexicographic::IndexFromCoor(xyz_coor, l_xyz, ldims);

          sobj    data   = extractLane(lane, src_v[sf]);
          if constexpr (DoConj) data = conjugate(data);
          scalar *data_s = (scalar *)&data;

          int64_t base = (int64_t)l_t * lN * lnxyz * lNsc
                       + (int64_t)n   * lnxyz * lNsc
                       + l_xyz * lNsc;
          for (int sc = 0; sc < lNsc; sc++)
            buf[base + sc] = data_s[sc];
        }
      });
    }
  }

public:

  // BLAS (column-major, OP_T on A):
  //   C[N_jxN_i] = A^T[N_ixK] * B[N_jxK]    with K=nxyz*Nsc
  //   reading A as C row-major [N_i][K] and B as C row-major [N_j][K]
  //   -> C[i,j] = sum_k W[i,k] * LR[j,k] = EMF[i,j]
  //
  // result's layout defaults to ColMajor (Eigen::Tensor's own default);
  // callers whose consumer expects j-fastest (e.g. writing into a RowMajor
  // A2AMatrixSet for HDF5) should pass a RowMajor result -- the scatter's
  // writes are then contiguous instead of striding by nt_global*N_i per j.
  //
  // bytesMoved mirrors timings (slot 0/GEMM is FLOP-bound, not bandwidth-
  // bound, so left untouched) with the bytes handled by that stage, summed
  // the same way (+=) so a caller accumulating across many calls gets a
  // matching total to divide by for an average throughput. Host-side
  // gather/scatter slots count read+write (2x element count); comms slots
  // count bytes on the wire, see wire_ring_reduce above.

  // Unpack a ComplexField phase into a flat array of one scalar per spatial site l_xyz.
  // ph is assumed time-independent; all t-layers write the same value so redundant
  // writes across timeslices are safe.  Mirrors the PackVectors SIMD/SIMT extraction.
  template<class phvobj>
  static void PackPhase(GridBase *_grid, const Lattice<phvobj> &ph,
                        deviceVector<scalar> &phase_buf)
  {
    int nd     = _grid->_ndimension;
    int lnt    = _grid->LocalDimensions()[nd - 1];
    int lnxyz  = _grid->lSites() / lnt;
    int osites = _grid->oSites();
    int lNsimd = _grid->Nsimd();

    phase_buf.resize(lnxyz);
    scalar *phase_data = &phase_buf[0];

    Coordinate rdimensions = _grid->_rdimensions;
    Coordinate ldims       = _grid->LocalDimensions();
    Coordinate simd_layout = _grid->_simd_layout;

    autoView(ph_v, ph, AcceleratorRead);

    accelerator_for(sf, osites, lNsimd, {
#ifdef GRID_SIMT
      {
        int lane = acceleratorSIMTlane(lNsimd);
#else
        for (int lane = 0; lane < lNsimd; lane++) {
#endif
        Coordinate icoor(nd), ocoor(nd), lcoor(nd);
        Lexicographic::CoorFromIndex(icoor, lane, simd_layout);
        Lexicographic::CoorFromIndex(ocoor, sf, rdimensions);
        for (int d = 0; d < nd; d++)
          lcoor[d] = rdimensions[d] * icoor[d] + ocoor[d];

        Coordinate xyz_coor = lcoor;
        xyz_coor[nd - 1]    = 0;
        int64_t l_xyz;
        Lexicographic::IndexFromCoor(xyz_coor, l_xyz, ldims);

        auto    ph_site = extractLane(lane, ph_v[sf]);
        scalar *ph_s    = (scalar *)&ph_site;
        phase_data[l_xyz] = ph_s[0];
      }
    });
  }

  // Multiply LR_buf[t][j][l_xyz*Nsc + sc] by phase_buf[l_xyz] for all (t, j, sc).
  // Nsc lanes per (j, l_xyz) pair: adjacent lanes access consecutive sc values -> stride-1 coalesced.
  void ApplyPhaseRight(const deviceVector<scalar> &phase_buf)
  {
    scalar       *LR  = &LR_buf[0];
    const scalar *ph  = &phase_buf[0];
    int lN_j = N_j, lnxyz = nxyz, lNsc = Nsc, lnt = nt;
    accelerator_for(idx, (size_t)(lN_j * lnxyz), lNsc, {
      int    j      = idx / lnxyz;
      int    l_xyz  = idx % lnxyz;
      scalar ph_val = ph[l_xyz];
#ifdef GRID_SIMT
      {
        int sc = acceleratorSIMTlane(lNsc);
#else
        for (int sc = 0; sc < lNsc; sc++) {
#endif
        for (int t = 0; t < lnt; t++) {
          int64_t base = (int64_t)t * lN_j * lnxyz * lNsc
                       + (int64_t)j * lnxyz * lNsc
                       + l_xyz * lNsc;
          LR[base + sc] *= ph_val;
        }
      }
    });
  }

  // Read the unphased pack in LR_buf (built by PackRight, untouched by this
  // path) and write nmom phase-multiplied copies into LR_mom_buf[t][m][j][
  // l_xyz*Nsc+sc]. One kernel launch, with m folded into the parallel index
  // space alongside (j, l_xyz) -- not nmom separate launches. Requires
  // AllocateRight to have been called first.
  //
  // Sole writer of LR_mom_buf, so it also aims LR_mom_ptrs there, superseding
  // the unphased default AllocateRight leaves at nmom=1. Source and
  // destination are always distinct buffers, so no thread reads a slot
  // another is phasing.
  void ApplyAllPhaseRight(const std::vector<deviceVector<scalar>> &phase_bufs)
  {
    GRID_ASSERT((int)phase_bufs.size() == nmom);

    size_t LRMsz = (size_t)nt * nmom * N_j * nxyz * Nsc;
    if (LR_mom_buf.size() < LRMsz) LR_mom_buf.resize(LRMsz);
    PointRightOperand(&LR_mom_buf[0]);

    deviceVector<scalar *> ph_ptrs(nmom);
    for (int m = 0; m < nmom; m++)
      acceleratorPut(ph_ptrs[m], const_cast<scalar *>(&phase_bufs[m][0]));

    const scalar *LR  = &LR_buf[0];
    scalar       *LRM = &LR_mom_buf[0];
    scalar      **ph  = &ph_ptrs[0];
    int lN_j = N_j, lnxyz = nxyz, lNsc = Nsc, lnt = nt, lnmom = nmom;

    accelerator_for(idx, (size_t)(lnmom * lN_j * lnxyz), lNsc, {
      int    m      = idx / (lN_j * lnxyz);
      int    rem    = idx % (lN_j * lnxyz);
      int    j      = rem / lnxyz;
      int    l_xyz  = rem % lnxyz;
      scalar ph_val = ph[m][l_xyz];
#ifdef GRID_SIMT
      {
        int sc = acceleratorSIMTlane(lNsc);
#else
        for (int sc = 0; sc < lNsc; sc++) {
#endif
        for (int t = 0; t < lnt; t++) {
          int64_t src = (int64_t)t * lN_j * lnxyz * lNsc
                      + (int64_t)j * lnxyz * lNsc
                      + l_xyz * lNsc;
          int64_t dst = (int64_t)t * lnmom * lN_j * lnxyz * lNsc
                      + (int64_t)m * lN_j * lnxyz * lNsc
                      + (int64_t)j * lnxyz * lNsc
                      + l_xyz * lNsc;
          LRM[dst + sc] = LR[src + sc] * ph_val;
        }
      }
    });
  }

  // Split-collective reduction: rather than one padded GlobalSumVector over
  // the whole communicator, the two operations that call was standing in for
  // are done separately, each with the primitive that matches it.
  //
  // Post-GEMM this rank's result is incomplete in two independent ways. It
  // is incomplete in K, because the GEMM contracted only over this rank's
  // own spatial sites -- every rank sharing this t coordinate holds a
  // partial sum of the same element, and those must genuinely be added. It
  // is incomplete in t, because the GEMM has only nt local batch elements
  // -- the other timeslices exist on other ranks and only need to be moved,
  // never summed. The padded allreduce did both at once by making addition
  // impersonate concatenation, and paid reduction cost on nt_global when
  // only nt_local carried information.
  //
  //   CartesianRingAllReduce(orthogDim = nd-1)
  //                        incomplete in K   arithmetic, P_xyz ranks,
  //                                          slab sized (nt local)
  //   CartesianRingAllGather(dim = nd-1)
  //                        incomplete in t   movement, P_t ranks,
  //                                          no arithmetic at all
  //
  // Both are Grid's own primitives, and both are handed nd-1 here for
  // opposite reasons: orthogDim names the dimension to SKIP, dim names the
  // only dimension to RING. orthogDim landed in develop as 7abc19dc for
  // exactly this case; dim is its dual, and without it the gather would ring
  // every dimension and materialise P_xyz identical copies of each slab, the
  // reduce having already made those ranks agree.
  //
  // The order is forced: the gather only relays, so it is legal only once
  // every slab is final, which is what the reduce establishes.
  //
  // Both run on point-to-point SendToRecvFrom rather than MPI collectives,
  // which also sidesteps the device-buffer MPI_Allreduce size cliff
  // documented in RingAllReduce.h. As that header assumes, an accelerator
  // build needs ACCELERATOR_AWARE_MPI here: the rings hand the working
  // buffers to both MPI and accelerator_for, so there is no host-bounce
  // path to fall back to.
  //
  // GEMM OPERAND ORDER: LR is A and W is B, so each batch element writes
  // C[row + i*Nwide] -- j fastest, i.e. the layout [i][m][j]. The other
  // choice, W as A, writes C[i + col*N_i] ([m][j][i], i fastest), which the
  // gather below would have to read as a full index reversal; this way it
  // walks contiguous runs instead. Flops, K, batch count and the numbers
  // themselves are identical either way -- only the output's storage
  // orientation differs.
  //
  // RESULT LAYOUT: result[nt_global][N_i][nmom][N_j], nmom BEFORE N_j. That
  // ordering agrees with the GEMM's [i][m][j] output and with a RowMajor
  // result at the same time, so the scatter is contiguous on both sides. It
  // also keeps the Hadrons IO fill loop contiguous, since that reads at
  // fixed m. The on-disk HDF5 layout is unaffected by the choice.
  //
  // timings[]/bytesMoved[] carry six slots here rather than five:
  //   [0] GEMM              [1] device->host
  //   [2] gather to slab    [3] spatial reduce
  //   [4] scatter           [5] temporal gather
  //
  // Slot [4] stays at zero whenever the direct path below applies, which is
  // the normal case for a caller sizing its tensor per block. It is therefore
  // the diagnostic for the fast path NOT having engaged: a nonzero scatter
  // time means some call fell back, either a CPU build tiling the block with
  // cacheBlock < N_i, or a caller whose tensor shape does not match its block.
  //
  // The two comms slots, [3] and [5], are counted as bytes ON THE WIRE, not
  // as payload: a ring moves each byte many times, so payload over elapsed
  // time measures the algorithm, while wire over elapsed time measures the
  // fabric and is the only figure comparable with a link rate or with
  // Benchmark_allreduce. The GlobalSumVector variants report payload in
  // their slot [3] because MPI's internal traffic is not knowable from
  // here, so do not read the two rates against each other.
  template <int Layout = Eigen::ColMajor>
  void SumRing(Eigen::Tensor<ComplexD, 4, Layout> &result,
               int cacheBlock,
               std::array<double, 6> *timings = nullptr,
               std::array<double, 6> *bytesMoved = nullptr)
  {
    GridBLAS BLAS;
    double dt;

    int K     = nxyz * Nsc;
    int Nwide = nmom * N_j;

    dt = -usecond();
    BLAS.gemmBatched(GridBLAS_OP_T, GridBLAS_OP_N,
                     Nwide, N_i, K,
                     scalar(1.0),
                     LR_mom_ptrs,
                     W_ptrs,
                     scalar(0.0),
                     EMF_mom_ptrs);
    BLAS.synchronise();
    dt += usecond();
    if (timings) (*timings)[0] += dt;

    int nt_global = result.dimension(0);
    int nd        = grid->Nd();
    int ct        = grid->ThisProcessorCoor()[nd - 1];
    int Pt        = grid->ProcessorGrid()[nd - 1];

    // Everything below indexes slabs by t coordinate, so the slab a rank
    // owns must sit at slot ct. A permuted rank ordering would not fail
    // here, it would silently produce a transposed time axis, so check it
    // rather than trust it.
    GRID_ASSERT(grid->LocalStarts()[nd - 1] == ct * nt);
    GRID_ASSERT(nt_global == Pt * nt);

    size_t tileMax = (size_t)nt_global * cacheBlock * cacheBlock * nmom;

    if (tile_buf.size() < tileMax) tile_buf.resize(tileMax);

    // The [4] scatter below is a pure relabelling whenever the tile spans the
    // whole block and the caller's tensor has exactly the tile's shape in
    // RowMajor: the gathered panel's [gt][iii][m][jjj] layout and a RowMajor
    // (nt_global, N_i, nmom, N_j) tensor are then the same addresses, so the
    // device->host copy can land in result.data() and the scatter is skipped.
    //
    // result.dimension(1) == N_i does double duty. It is the shape check, and
    // for a caller that allocates one buffer padded to its largest block and
    // reuses it for every block, it is also the "this is the partial tail
    // block" check - the tail's addresses genuinely differ from the padded
    // tensor's, so it falls back on its own with no extra logic. A caller that
    // sizes its tensor to each block exactly never takes the fallback.
    //
    // cacheBlock < N_i or < N_j means the caller asked for a genuine
    // sub-rectangle of the block, whose addresses differ as well, so the CPU
    // path keeps the scatter. Both are runtime conditions, so one binary
    // serves both builds.
    //
    // The scalar check is not cosmetic: result is a ComplexD tensor, and when
    // vobj carries a single-precision vector_type the scatter's elementwise
    // assignment is a widening conversion that a raw copy would silently skip.
    const bool direct = std::is_same<scalar, ComplexD>::value
                     && (Layout == Eigen::RowMajor)
                     && (cacheBlock >= N_i) && (cacheBlock >= N_j)
                     && (result.dimension(1) == N_i)
                     && (result.dimension(2) == nmom)
                     && (result.dimension(3) == N_j);

    // Only the fallback needs a host landing buffer, and the tiles that take
    // it are by construction the small ones, so size it to what this call
    // actually needs rather than to tileMax.
    if (!direct)
    {
      size_t hostWords = (size_t)nt_global * nmom
                       * std::min(cacheBlock, N_i) * std::min(cacheBlock, N_j);
      if (tile_host.size() < hostWords) tile_host.resize(hostWords);
    }

    const scalar *emf_p  = &EMF_mom_buf[0];
    scalar       *tile_p = &tile_buf[0];

    int lN_i = N_i, lN_j = N_j, lnmom = nmom;
    for (int ii = 0; ii < N_i; ii += cacheBlock)
    {
      int Niii = std::min(N_i - ii, cacheBlock);
      for (int jj = 0; jj < N_j; jj += cacheBlock)
      {
        int Njjj = std::min(N_j - jj, cacheBlock);
        int lNiii = Niii, lNjjj = Njjj, lii = ii, ljj = jj;

        // Panel of Pt slots, slot k holding the timeslices owned by t
        // coordinate k, laid out [gt][iii][m][jjj] with jjj fastest. gt is
        // the slowest index, so this rank's nt timeslices form one
        // contiguous run -- which is what lets the gather treat tile_buf as
        // a panel of equal slots with no repacking.
        uint64_t slabWords = (uint64_t)nt        * Niii * nmom * Njjj;
        uint64_t tileWords = (uint64_t)nt_global * Niii * nmom * Njjj;
        scalar  *slab_p    = tile_p + (uint64_t)ct * slabWords;

        // No zero-pad and no transpose: the swapped GEMM already emits
        // [i][m][j], so this walks contiguous runs of Njjj on both sides.
        // When cacheBlock spans the whole block it degenerates to a
        // straight copy, and EMF_mom_ptrs could then be aimed at slab_p to
        // drop it entirely -- left for later so this stays correct for the
        // partial-tile case that CPU callers still want.
        dt = -usecond();
        accelerator_for(idx, slabWords, 1, {
            uint64_t r   = idx;
            int      jjj = r % lNjjj;  r /= lNjjj;
            int      m   = r % lnmom;  r /= lnmom;
            int      iii = r % lNiii;  r /= lNiii;
            int      lt  = (int)r;
            slab_p[idx] = emf_p[(uint64_t)lt * lN_i * lnmom * lN_j
                              + (uint64_t)(lii + iii) * lnmom * lN_j
                              + (uint64_t)m * lN_j
                              + (ljj + jjj)];
        });
        dt += usecond();
        if (timings) (*timings)[2] += dt;
        if (bytesMoved) (*bytesMoved)[2] += 2.0 * slabWords * sizeof(scalar);

        // orthogDim = nd-1 skips the time axis, so this reduces over exactly
        // the P_xyz ranks sharing this rank's t coordinate -- the set holding
        // partial K sums of the same elements. No sub-communicator: the rings
        // run on ShiftedRanks neighbours of the parent Cartesian communicator,
        // one process axis at a time.
        dt = -usecond();
        CartesianRingAllReduce(grid, slab_p, slabWords, nd - 1);
        dt += usecond();
        if (timings) (*timings)[3] += dt;
        if (bytesMoved) (*bytesMoved)[3] += wire_ring_reduce * slabWords * sizeof(scalar);

        // dim = nd-1 rings the time axis only, so the panel grows from
        // this rank's slab to all Pt slabs and nothing is summed. Legal only
        // because the reduce above has already made every slab final. The
        // library allocates a second Pt*slabWords panel internally and copies
        // back into tile_p, which is the price of not carrying a private
        // in-place copy of the same loop.
        dt = -usecond();
        CartesianRingAllGather(grid, tile_p, slabWords, nd - 1);
        dt += usecond();
        if (timings) (*timings)[5] += dt;
        if (bytesMoved) (*bytesMoved)[5] += (double)(Pt - 1) * slabWords * sizeof(scalar);

        if (direct)
        {
          // Same addresses on both sides, so the gathered panel already is the
          // result: land it in the caller's tensor and leave [4] at zero. The
          // tile is the whole block here, so it is the whole tensor too.
          GRID_ASSERT((uint64_t)result.size() == tileWords);

          dt = -usecond();
          acceleratorCopyFromDevice(tile_p, result.data(), tileWords * sizeof(scalar));
          dt += usecond();
          if (timings) (*timings)[1] += dt;
          if (bytesMoved) (*bytesMoved)[1] += (double)tileWords * sizeof(scalar);
        }
        else
        {
          dt = -usecond();
          acceleratorCopyFromDevice(tile_p, tile_host.data(), tileWords * sizeof(scalar));
          dt += usecond();
          if (timings) (*timings)[1] += dt;
          if (bytesMoved) (*bytesMoved)[1] += (double)tileWords * sizeof(scalar);

          const scalar *red_p = tile_host.data();
          dt = -usecond();
          thread_for_collapse(4, gt, nt_global, {
              for (int iii = 0; iii < Niii; iii++)
              for (int m   = 0; m   < lnmom; m++)
              for (int jjj = 0; jjj < Njjj; jjj++)
                result((int)gt, ii + iii, m, jj + jjj)
                    = red_p[(((uint64_t)gt * Niii + iii) * lnmom + m) * Njjj + jjj];
          });
          dt += usecond();
          if (timings) (*timings)[4] += dt;
          if (bytesMoved) (*bytesMoved)[4] += 2.0 * tileWords * sizeof(scalar);
        }
      }
    }
  }

};

NAMESPACE_END(Grid);
