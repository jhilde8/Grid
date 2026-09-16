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

  Given:
    leftv[N_i][osite]    - conjugated left SpinColourVectors (SIMD-packed)
    loopRight[N_j][osite]- type-contracted right SpinColourVectors (SIMD-packed)

  Computes:
    EMF[i,j,t] = sum_{x,s,c} leftv[i][x,t,s,c] * loopRight[j][x,t,s,c]

  via batched GEMM over nt local time slices, then a ring reduction and a
  ring gather across MPI (see SumRing).

  Memory layout (all C row-major):
    W_buf      [nt][N_i][nxyz*Nsc]        W[t][i][x*Nsc+sc]    = leftv[i] at (x,t)
    LR_buf     [nt][nmom][N_j][nxyz*Nsc]  LR[t][m][j][x*Nsc+sc] = loopRight[j] at (x,t) * phase_m[x]
    EMF_mom_buf[nt][N_i][nmom*N_j]        GEMM output
  

  Momentum folds into the GEMM's N dimension. PackRight writes the unphased vectors 
  into the m = 0 slot of LR_buf, and ApplyPhaseRight fills the m >= 1 slots, 
  through the application of elements of a phase buffer from PackPhase on the 
  zero momentum buffer in slot m=0. GEMM reduction dimension K = nxyz * Nsc, outer 
  dimensions M = N_i (left A2A vectors), N=nmom*N_j with nmom the slower-varying 
  sub-index, so each timeslice is nmom contiguous [N_j][nxyz*Nsc] blocks 
  -- what gemmBatched wants, no repacking. 
  A caller with nmom = 1 and no phase (EMF, CMOF) never calls
  ApplyPhaseRight and the GEMM reads the pack as written.

  Usage: Allocate once, outside the caller's block loops, for the largest
  block on each side. Inside the loops only PackLeft/PackLeftConj, PackRight,
  ApplyPhaseRight and SumRing are called. The packs record the block size
  they wrote, and SumRing aims all three pointer tables at those sizes before
  the GEMM, so the calls inside the loops can come in any order. Deallocate
  undoes Allocate.

  See Grid/qcd/utils/A2Autils.h's compute methods for the call structure of these functions. 

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

  // Capacity reserved by Allocate. N_i and N_j are the sizes of the blocks
  // currently packed, set by the packs, and never exceed these.
  int maxN_i, maxN_j;

  // Block sizes the pointer tables were last aimed for. The buffers never
  // move after Allocate, so the tables only need rewriting when a size
  // changes -- the first block and the tail blocks.
  int N_i_prev, N_j_prev;

  // Bytes the reduce puts on the wire per byte of payload: 2(P_d-1)/P_d
  // summed over the spatial dimensions only, matching
  // CartesianRingAllReduce(orthogDim = nd-1). Fixed by the process grid, so
  // set once in Allocate. Exact, being a count of sends this code
  // issues itself rather than an assumption about MPI.
  double wire_ring_reduce;

  // GEMM operands and output, and one device pointer per local timeslice
  // into each -- gemmBatched reads its batch element t through these.
  deviceVector<scalar>   W_buf;
  deviceVector<scalar>   LR_buf;
  deviceVector<scalar>   EMF_mom_buf;
  deviceVector<scalar *> W_ptrs;
  deviceVector<scalar *> LR_ptrs;
  deviceVector<scalar *> EMF_mom_ptrs;

  // Staging for SumRing below: one block's output at the output's time
  // extent - Pt*nt normally, nt in localT, where the rank keeps only its own
  // slab and the panel collapses to one slot. Sized on the first SumRing
  // rather than in Allocate, because the time extent is only known there.
  deviceVector<scalar>   tile_buf;

  // Site map read by PackVectors: for each (outer site, SIMD lane), the local
  // timeslice and the spatial index within it. Fixed by the grid alone, so it
  // is built once and every pack reads it -- see BuildSiteMap.
  GridBase              *map_grid;
  deviceVector<int>      t_map;
  deviceVector<int>      xyz_map;

  A2ASpatialSum() : grid(nullptr), N_i(0), N_j(0), nt(0), nxyz(0), Nsc(0), nmom(1),
                    maxN_i(0), maxN_j(0), N_i_prev(0), N_j_prev(0),
                    wire_ring_reduce(0.0), map_grid(nullptr) {}

  // Return every buffer to the allocator, leaving the object as constructed.
  // The buffers only ever grow, so an owner that outlives its last
  // contraction -- a Hadrons module, which the VM keeps until the job ends --
  // would otherwise hold its largest block on the device for good: ~11 GiB
  // per rank for a 27-momentum A2AMesonField at block size 256.
  void Deallocate(void)
  {
    grid = nullptr;
    N_i = 0; N_j = 0;
    nt = 0; nxyz = 0; Nsc = 0;
    nmom = 1;
    maxN_i = 0; maxN_j = 0;
    N_i_prev = 0; N_j_prev = 0;
    wire_ring_reduce = 0.0;
    map_grid = nullptr;
    W_buf.resize(0);        W_buf.shrink_to_fit();
    LR_buf.resize(0);       LR_buf.shrink_to_fit();
    EMF_mom_buf.resize(0);  EMF_mom_buf.shrink_to_fit();
    W_ptrs.resize(0);       W_ptrs.shrink_to_fit();
    LR_ptrs.resize(0);      LR_ptrs.shrink_to_fit();
    EMF_mom_ptrs.resize(0); EMF_mom_ptrs.shrink_to_fit();
    tile_buf.resize(0);     tile_buf.shrink_to_fit();
    t_map.resize(0);        t_map.shrink_to_fit();
    xyz_map.resize(0);      xyz_map.shrink_to_fit();
  }

  // Aim the three pointer tables at the blocks currently packed. Timeslice t
  // of each buffer starts at t times its per-timeslice stride, which depends
  // on the block size, so a tail block moves every pointer past t = 0. The
  // tables are copied host->device in one transfer each.
  void PointOperands(void)
  {
    if (N_i == N_i_prev && N_j == N_j_prev) return;

    size_t K = (size_t)nxyz * Nsc;
    std::vector<scalar *> Wh(nt), LRh(nt), EMFh(nt);
    for (int t = 0; t < nt; t++) {
      Wh[t]   = &W_buf[0]       + (size_t)t * N_i * K;
      LRh[t]  = &LR_buf[0]      + (size_t)t * nmom * N_j * K;
      EMFh[t] = &EMF_mom_buf[0] + (size_t)t * nmom * N_j * N_i;
    }
    acceleratorCopyToDevice(&Wh[0],   &W_ptrs[0],       (size_t)nt * sizeof(scalar *));
    acceleratorCopyToDevice(&LRh[0],  &LR_ptrs[0],      (size_t)nt * sizeof(scalar *));
    acceleratorCopyToDevice(&EMFh[0], &EMF_mom_ptrs[0], (size_t)nt * sizeof(scalar *));

    N_i_prev = N_i;
    N_j_prev = N_j;
  }

  // Precompute the coordinate decode every packer needs: for each (outer site,
  // SIMD lane), the local timeslice l_t and the spatial index l_xyz within it.
  // Both are fixed by the grid geometry, so this runs once per grid and
  // PackVectors and PackPhase both read the result. Without it each launch
  // re-derived the same answer on device at eight integer divisions per thread,
  // and a trajectory issues of order a million of them.
  //
  // l_t and l_xyz are kept apart rather than fused into one offset because N
  // sits between them in the address arithmetic,
  //
  //     base = ((l_t*N + n)*nxyz + l_xyz)*Nsc
  //
  // and N differs between the W_buf pack (N_i) and the LR_buf pack (N_j).
  // Split, one map serves both and survives every change of block shape --
  // and PackPhase, which is spatial only, reads xyz_map alone.
  void BuildSiteMap(void)
  {
    int    osites = grid->oSites();
    int    Nsimd  = vobj::Nsimd();
    size_t npt    = (size_t)osites * Nsimd;

    if (map_grid == grid && t_map.size() == npt) return;

    int nd = grid->_ndimension;
    Coordinate rdimensions = grid->_rdimensions;
    Coordinate ldims       = grid->LocalDimensions();
    Coordinate simd        = grid->_simd_layout;

    std::vector<int> t_host(npt), xyz_host(npt);
    thread_for(sf, osites, {
      Coordinate ocoor(nd);
      Lexicographic::CoorFromIndex(ocoor, sf, rdimensions);
      for (int lane = 0; lane < Nsimd; lane++) {
        Coordinate icoor(nd), lcoor(nd);
        Lexicographic::CoorFromIndex(icoor, lane, simd);
        for (int d = 0; d < nd; d++)
          lcoor[d] = rdimensions[d] * icoor[d] + ocoor[d];

        int l_t = lcoor[nd - 1];
        lcoor[nd - 1] = 0;
        int64_t l_xyz;
        Lexicographic::IndexFromCoor(lcoor, l_xyz, ldims);

        t_host[sf * Nsimd + lane]   = l_t;
        xyz_host[sf * Nsimd + lane] = (int)l_xyz;
      }
    });

    t_map.resize(npt);
    xyz_map.resize(npt);
    acceleratorCopyToDevice(&t_host[0],   &t_map[0],   npt * sizeof(int));
    acceleratorCopyToDevice(&xyz_host[0], &xyz_map[0], npt * sizeof(int));
    map_grid = grid;
  }

  // Bind the grid and size every buffer for the largest block on each side,
  // once, before the caller's block loops. Nothing is allocated after this,
  // so the buffers never move and the pointer tables stay valid until a block
  // size changes (see PointOperands).
  //
  // nmom is fixed for the owner's lifetime: it sets the right buffer's
  // momentum slots and the width of the GEMM output.
  void Allocate(GridBase *_grid, int _nmom, int _maxN_i, int _maxN_j)
  {
    grid   = _grid;
    nmom   = _nmom;
    maxN_i = _maxN_i;
    maxN_j = _maxN_j;
    Coordinate ldims = grid->LocalDimensions();
    nt   = ldims[grid->Nd() - 1];
    nxyz = grid->lSites() / nt;
    Nsc  = sizeof(sobj) / sizeof(scalar);

    BuildSiteMap();

    wire_ring_reduce = 0.0;
    for (int d = 0; d < grid->Nd() - 1; d++) {
      int Pd = grid->ProcessorGrid()[d];
      if (Pd > 1) wire_ring_reduce += 2.0 * (Pd - 1) / (double)Pd;
    }

    size_t K = (size_t)nxyz * Nsc;
    W_buf.resize((size_t)nt * maxN_i * K);
    LR_buf.resize((size_t)nt * nmom * maxN_j * K);
    EMF_mom_buf.resize((size_t)nt * nmom * maxN_j * maxN_i);
    W_ptrs.resize(nt);
    LR_ptrs.resize(nt);
    EMF_mom_ptrs.resize(nt);

    N_i = 0;      N_j = 0;
    N_i_prev = 0; N_j_prev = 0;
  }

  void PackLeft(const std::vector<Lattice<vobj>> &leftv, int start = 0, int count = -1)
  {
    if (count < 0) count = (int)leftv.size();
    GRID_ASSERT(start + count <= (int)leftv.size());
    GRID_ASSERT(count > 0 && count <= maxN_i);
    N_i = count;
    PackVectors(leftv, &W_buf[0], N_i, N_i, start);
  }

  // Read directly from original (unconjugated) left vectors, conjugating during pack.
  void PackLeftConj(const std::vector<Lattice<vobj>> &left, int start = 0, int count = -1)
  {
    if (count < 0) count = (int)left.size();
    GRID_ASSERT(start + count <= (int)left.size());
    GRID_ASSERT(count > 0 && count <= maxN_i);
    N_i = count;
    PackVectors<true>(left, &W_buf[0], N_i, N_i, start);
  }

  // Writes the m = 0 slot of LR_buf only. The per-timeslice stride is the full
  // nmom*N_j, so slots m >= 1 are left for ApplyPhaseRight.
  void PackRight(const std::vector<Lattice<vobj>> &loopRight, int start = 0, int count = -1)
  {
    if (count < 0) count = (int)loopRight.size();
    GRID_ASSERT(start + count <= (int)loopRight.size());
    GRID_ASSERT(count > 0 && count <= maxN_j);
    N_j = count;
    PackVectors(loopRight, &LR_buf[0], N_j, nmom * N_j, start);
  }

public:
  // Pack vecs[start..start+N-1] lattice fields into buf[nt][stride][nxyz*Nsc],
  // extracting all SIMD lanes. stride is the per-timeslice count of blocks the
  // buffer is laid out for; it equals N except for the right buffer, whose
  // timeslices hold nmom blocks of N and whose pack fills the first.
  // DoConj=true conjugates each element during extraction (used by PackLeftConj).
  //
  // The (l_t, l_xyz) decode is read from the site map instead of being derived
  // per thread; BuildSiteMap explains why it can be. Addresses and values are
  // unchanged, so the buffer this fills is bit-identical to the open-coded
  // version -- a diff against it should be exactly zero, not machine epsilon.
  template<bool DoConj = false>
  void PackVectors(const std::vector<Lattice<vobj>> &vecs, scalar *buf, int N,
                   int stride, int start)
  {
    int osites = grid->oSites();
    int Nsimd  = vobj::Nsimd();
    int lN     = stride;
    int lNsc   = Nsc;
    int lnxyz  = nxyz;
    const int *tm = &t_map[0];
    const int *xm = &xyz_map[0];

    for (int n = 0; n < N; n++) {
      autoView(src_v, vecs[start + n], AcceleratorRead);
      accelerator_for(sf, osites, Nsimd, {
#ifdef GRID_SIMT
        {
          int lane = acceleratorSIMTlane(Nsimd);
#else
          for (int lane = 0; lane < Nsimd; lane++) {
#endif
          int64_t idx = (int64_t)sf * Nsimd + lane;

          sobj    data   = extractLane(lane, src_v[sf]);
          if constexpr (DoConj) data = conjugate(data);
          scalar *data_s = (scalar *)&data;

          int64_t base = ((int64_t)(tm[idx] * lN + n) * lnxyz + xm[idx]) * lNsc;
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
  // bytesMoved mirrors timings (slot 0/GEMM is FLOP-bound, not bandwidth-
  // bound, so left untouched) with the bytes handled by that stage, summed
  // the same way (+=) so a caller accumulating across many calls gets a
  // matching total to divide by for an average throughput. The gather slot
  // counts read+write (2x element count); comms slots count bytes on the
  // wire, see wire_ring_reduce above.

  // Unpack a ComplexField phase into a flat array of one scalar per spatial site l_xyz.
  // ph is assumed time-independent; all t-layers write the same value so redundant
  // writes across timeslices are safe.  Mirrors the PackVectors SIMD/SIMT extraction.
  // Reads the site map built by Allocate, so it runs after Allocate.
  template<class phvobj>
  void PackPhase(const Lattice<phvobj> &ph, deviceVector<scalar> &phase_buf)
  {
    GRID_ASSERT(grid != nullptr);

    int osites = grid->oSites();
    int lNsimd = grid->Nsimd();

    // The map is indexed sf*vobj::Nsimd() + lane over the bound grid, so the
    // phase field must have the same sites and SIMD width.
    GRID_ASSERT(ph.Grid()->oSites() == osites);
    GRID_ASSERT(lNsimd == vobj::Nsimd());

    phase_buf.resize(nxyz);
    scalar *phase_data = &phase_buf[0];

    const int *xm = &xyz_map[0];

    autoView(ph_v, ph, AcceleratorRead);

    accelerator_for(sf, osites, lNsimd, {
#ifdef GRID_SIMT
      {
        int lane = acceleratorSIMTlane(lNsimd);
#else
        for (int lane = 0; lane < lNsimd; lane++) {
#endif
        auto    ph_site = extractLane(lane, ph_v[sf]);
        scalar *ph_s    = (scalar *)&ph_site;
        phase_data[xm[(int64_t)sf * lNsimd + lane]] = ph_s[0];
      }
    });
  }

  // Turn the unphased pack PackRight left in slot m = 0 of LR_buf[t][m][j][
  // l_xyz*Nsc+sc] into all nmom phased slots, in two launches:
  //
  //   1. slots m >= 1 = slot 0 * phase_m
  //   2. slot 0      *= phase_0             (in place)
  //
  // The order keeps both race-free: launch 1 reads only slot 0 and writes only
  // other slots, and launch 2 reads and writes each element of slot 0 alone.
  // m is folded into launch 1's index space, so it is one launch rather than
  // nmom-1; at nmom = 1 its range is empty and it submits nothing. Must follow
  // PackRight for the same block.
  void ApplyPhaseRight(const std::vector<deviceVector<scalar>> &phase_bufs)
  {
    GRID_ASSERT((int)phase_bufs.size() == nmom);
    GRID_ASSERT(N_j > 0);

    deviceVector<scalar *> ph_ptrs(nmom);
    for (int m = 0; m < nmom; m++)
      acceleratorPut(ph_ptrs[m], const_cast<scalar *>(&phase_bufs[m][0]));

    scalar  *LR = &LR_buf[0];
    scalar **ph = &ph_ptrs[0];
    int lN_j = N_j, lnxyz = nxyz, lNsc = Nsc, lnt = nt, lnmom = nmom;

    accelerator_for(idx, (size_t)((lnmom - 1) * lN_j * lnxyz), lNsc, {
      int    m      = 1 + (int)(idx / (lN_j * lnxyz));
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
          int64_t src = (((int64_t)t * lnmom    ) * lN_j + j) * lnxyz * lNsc
                      + l_xyz * lNsc;
          int64_t dst = (((int64_t)t * lnmom + m) * lN_j + j) * lnxyz * lNsc
                      + l_xyz * lNsc;
          LR[dst + sc] = LR[src + sc] * ph_val;
        }
      }
    });

    accelerator_for(idx, (size_t)(lN_j * lnxyz), lNsc, {
      int    j      = idx / lnxyz;
      int    l_xyz  = idx % lnxyz;
      scalar ph_val = ph[0][l_xyz];
#ifdef GRID_SIMT
      {
        int sc = acceleratorSIMTlane(lNsc);
#else
        for (int sc = 0; sc < lNsc; sc++) {
#endif
        for (int t = 0; t < lnt; t++) {
          int64_t base = (((int64_t)t * lnmom) * lN_j + j) * lnxyz * lNsc
                       + l_xyz * lNsc;
          LR[base + sc] *= ph_val;
        }
      }
    });
  }

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
  // only dimension to RING. The order is forced: the gather only relays, so it is legal only once
  // every slab is final, which is what the reduce establishes.
  //
  // localT stops after the reduce. Every rank then holds its own slab and
  // nothing else, and result carries nt in its time dimension rather than
  // Pt*nt. That is the whole of the per-timeslice IO path: a caller writing
  // one file per timeslice only ever writes timeslices it owns, so the
  // gather - which exists solely to hand every rank a copy of a result that
  // one rank writes - is pure movement with no consumer. The reduce is not
  // optional in either mode; it is what makes a slab final.
  //
  // Note this is not an IO switch. It changes which collective runs and the
  // shape of result, so a caller that sets it and then indexes result as if
  // it held nt_global timeslices reads the wrong data rather than failing.
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
  // RESULT LAYOUT: result[nt_out][N_i][nmom][N_j], RowMajor, nmom BEFORE N_j,
  // where nt_out is Pt*nt normally and nt in localT -- in the latter, t index
  // k is global timeslice ct*nt + k. That ordering agrees with the GEMM's
  // [i][m][j] output, and RowMajor makes the gathered panel and the caller's
  // tensor the same addresses, so the device->host copy lands directly in
  // result.data(). The caller sizes result to exactly this shape for every
  // block, tails included. The on-disk HDF5 layout is unaffected.
  //
  // timings[]/bytesMoved[] carry five slots:
  //   [0] GEMM              [1] device->host
  //   [2] gather to slab    [3] spatial reduce
  //   [4] temporal gather
  //
  // Slot [4] stays at zero in localT, where the temporal gather is skipped
  // outright rather than merely being cheap. A nonzero [4] on a caller that
  // asked for localT means the flag did not reach here.
  //
  // The two comms slots, [3] and [4], are counted as bytes ON THE WIRE, not
  // as payload: a ring moves each byte many times, so payload over elapsed
  // time measures the algorithm, while wire over elapsed time measures the
  // fabric and is the only figure comparable with a link rate or with
  // Benchmark_allreduce.
  void SumRing(Eigen::Tensor<ComplexD, 4, Eigen::RowMajor> &result,
               std::array<double, 5> *timings = nullptr,
               std::array<double, 5> *bytesMoved = nullptr,
               bool localT = false)
  {
    // The device->host copy is a raw byte copy into a ComplexD tensor, so the
    // packed scalar must already be ComplexD -- a narrower one would need an
    // elementwise widening that this path no longer has.
    static_assert(std::is_same<scalar, ComplexD>::value,
                  "A2ASpatialSum::SumRing requires a ComplexD scalar");

    GridBLAS BLAS;
    double dt;

    int K     = nxyz * Nsc;
    int Nwide = nmom * N_j;

    GRID_ASSERT(N_i > 0 && N_j > 0);
    PointOperands();

    dt = -usecond();
    BLAS.gemmBatched(GridBLAS_OP_T, GridBLAS_OP_N,
                     Nwide, N_i, K,
                     scalar(1.0),
                     LR_ptrs,
                     W_ptrs,
                     scalar(0.0),
                     EMF_mom_ptrs);
    BLAS.synchronise();
    dt += usecond();
    if (timings) (*timings)[0] += dt;

    // The output's time extent: Pt*nt normally, nt in localT. Every sizing
    // and loop bound below is really this, so it is carried directly rather
    // than as a separate nt_global -- the one quantity that genuinely needs
    // the panel's full extent is tileWords, and that is exactly Pt*slabWords.
    int nt_out = result.dimension(0);
    int nd     = grid->Nd();
    int ct     = grid->ThisProcessorCoor()[nd - 1];
    int Pt     = grid->ProcessorGrid()[nd - 1];

    // Everything below indexes slabs by t coordinate, so the slab a rank
    // owns must sit at slot ct. A permuted rank ordering would not fail
    // here, it would silently produce a transposed time axis, so check it
    // rather than trust it. In localT it carries more weight still: it is
    // what makes result's t index k mean global timeslice ct*nt + k, which
    // is the number the caller puts in a file name.
    GRID_ASSERT(grid->LocalStarts()[nd - 1] == ct * nt);
    GRID_ASSERT(nt_out == (localT ? nt : Pt * nt));
    GRID_ASSERT(result.dimension(1) == N_i);
    GRID_ASSERT(result.dimension(2) == nmom);
    GRID_ASSERT(result.dimension(3) == N_j);

    // Panel of Pt slots, slot k holding the timeslices owned by t coordinate
    // k, laid out [gt][i][m][j] with j fastest. gt is the slowest index, so
    // this rank's nt timeslices form one contiguous run -- which is what lets
    // the gather treat tile_buf as a panel of equal slots with no repacking.
    //
    // Every difference between the two modes is in these lines and the
    // gather below. In localT there is no panel: the rank keeps its own slab
    // and nothing else, so the slab sits at slot 0 and is itself the output.
    uint64_t slabWords = (uint64_t)nt * N_i * nmom * N_j;
    uint64_t outWords  = localT ? slabWords : (uint64_t)Pt * slabWords;

    // Grows to the largest block's panel on the first call and is reused
    // after that; it is sized here rather than in Allocate because only the
    // caller's tensor says whether this is a localT panel.
    if (tile_buf.size() < outWords) tile_buf.resize(outWords);

    scalar       *tile_p = &tile_buf[0];
    scalar       *slab_p = localT ? tile_p : tile_p + (uint64_t)ct * slabWords;
    const scalar *emf_p  = &EMF_mom_buf[0];

    // The GEMM already emits [i][m][j] at this block's strides, so this is a
    // straight copy of its first slabWords into this rank's slot.
    dt = -usecond();
    accelerator_for(idx, slabWords, 1, {
        slab_p[idx] = emf_p[idx];
    });
    dt += usecond();
    if (timings) (*timings)[2] += dt;
    if (bytesMoved) (*bytesMoved)[2] += 2.0 * slabWords * sizeof(scalar);

    // orthogDim = nd-1 skips the time axis, so this reduces over exactly the
    // P_xyz ranks sharing this rank's t coordinate -- the set holding partial
    // K sums of the same elements. No sub-communicator: the rings run on
    // ShiftedRanks neighbours of the parent Cartesian communicator, one
    // process axis at a time.
    dt = -usecond();
    CartesianRingAllReduce(grid, slab_p, slabWords, nd - 1);
    dt += usecond();
    if (timings) (*timings)[3] += dt;
    if (bytesMoved) (*bytesMoved)[3] += wire_ring_reduce * slabWords * sizeof(scalar);

    // dim = nd-1 rings the time axis only, so the panel grows from this
    // rank's slab to all Pt slabs and nothing is summed. Legal only because
    // the reduce above has already made every slab final. The library
    // allocates a second Pt*slabWords panel internally and copies back into
    // tile_p, which is the price of not carrying a private in-place copy of
    // the same loop.
    if (!localT)
    {
      dt = -usecond();
      CartesianRingAllGather(grid, tile_p, slabWords, nd - 1);
      dt += usecond();
      if (timings) (*timings)[4] += dt;
      if (bytesMoved) (*bytesMoved)[4] += (double)(Pt - 1) * slabWords * sizeof(scalar);
    }

    // The panel (or, in localT, the slab) and the RowMajor result are the
    // same addresses, so the copy lands directly in the caller's tensor.
    GRID_ASSERT((uint64_t)result.size() == outWords);

    dt = -usecond();
    acceleratorCopyFromDevice(tile_p, result.data(), outWords * sizeof(scalar));
    dt += usecond();
    if (timings) (*timings)[1] += dt;
    if (bytesMoved) (*bytesMoved)[1] += (double)outWords * sizeof(scalar);
  }

};

NAMESPACE_END(Grid);
