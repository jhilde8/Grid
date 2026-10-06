
echo spack
. /autofs/nccs-svm1_home1/paboyle/spack/share/spack/setup-env.sh

export CLIME=`spack find --paths c-lime | grep ^c-lime | awk '{print $2}' `
export MPFR=`spack find --paths mpfr    | grep ^mpfr  | awk '{print $2}' `
export OPENSSL=`spack find --paths openssl | grep openssl | awk  '{print $2}' `
export GMP=`spack find --paths gmp      | grep ^gmp | awk '{print $2}' `

module load cce/20.0.0
module load cpe/25.09
module load rocm/6.4.0
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/rocm-6.4.0/lib/llvm/lib/:$LD_LIBRARY_PATH
module load emacs
export FI_MR_CACHE_MONITOR=kdreg2       # REQUIRED for device-buffer MPI on Slingshot: libfabric memhooks monitor (default) is defective, see systems/WorkArounds.txt (libfabric #11451)

# Standing avoidance of the CXI NO_TRANSLATION fault on persistent device
# buffers under hipMalloc/hipFree churn (libfabric issue #12775; ruling
# 2026-09-07). 
export FI_HMEM_ROCR_USE_DMABUF=0
