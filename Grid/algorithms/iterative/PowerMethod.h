#pragma once
namespace Grid {
template<class Field> class PowerMethod  
{ 
 public: 

  template<typename T>  static RealD normalise(T& v) 
  {
    RealD nn = norm2(v);
    nn = sqrt(nn);
    v = v * (1.0/nn);
    return nn;
  }

  RealD operator()(LinearOperatorBase<Field> &HermOp, const Field &src) 
  { 
    GridBase *grid = src.Grid(); 
    
    // quickly get an idea of the largest eigenvalue to more properly normalize the residuum 
    RealD evalMaxApprox = 0.0; 
    auto src_n = src; 
    auto tmp = src; 
    const int _MAX_ITER_EST_ = 200; 

    for (int i=0;i<_MAX_ITER_EST_;i++) { 
      
      normalise(src_n); 
      HermOp.HermOp(src_n,tmp); 
      RealD vnum = real(innerProduct(src_n,tmp)); // HermOp. 
      RealD vden = norm2(src_n); 
      RealD na = vnum/vden; 

      std::cout << GridLogMessage << "PowerMethod: Current approximation of largest eigenvalue " << na << std::endl;
      
      //      if ( (fabs(evalMaxApprox/na - 1.0) < 0.0001) || (i==_MAX_ITER_EST_-1) ) { 
	// 	evalMaxApprox = na; 
	// 	return evalMaxApprox; 
      //      } 
      evalMaxApprox = na; 
      src_n = tmp;
    }
    std::cout << GridLogMessage << " Approximation of largest eigenvalue: " << evalMaxApprox << std::endl;
    return evalMaxApprox;
  }
};

// Non-Hermitian sibling of PowerMethod.  Drives Op() -- the operator the
// caller actually applies -- rather than HermOp(), so it returns |lambda_max|
// of a NON-Hermitian operator and reports the spectral-edge diagnostics the
// Hermitian PowerMethod cannot:
//   step 0 : |A v|/|v| on the (random) start src -- a one-sample lower bound
//            on sigma_max(A).
//   step k : |A v_k| -> |lambda_max| as v_k -> the dominant eigenvector; the
//            complex Rayleigh quotient <v,Av> gives its phase (real => on the
//            axis, a non-converging oscillation => a conjugate pair of equal
//            modulus at the top).
//   return : |lambda_max|, with a SUMMARY line flagging non-normality when the
//            step-0 sigma_max lower bound sits well above the converged
//            |lambda_max| -- the case where the numerical range extends beyond
//            the spectrum and a spectrum-based smoother/Chebyshev bound is
//            unsafe (the field of values must be used instead).
template<class Field> class PowerMethodNonHermitian
{
 public:

  template<typename T> static RealD normalise(T& v)
  {
    RealD nn = sqrt(norm2(v));
    v = v * (1.0/nn);
    return nn;
  }

  RealD operator()(LinearOperatorBase<Field> &Op, const Field &src)
  {
    GridBase *grid = src.Grid();
    Field v(grid), Av(grid);
    v = src;

    RealD ratio = 0.0, ratio0 = 0.0;
    ComplexD rq(0.0);
    const int _MAX_ITER_EST_ = 200;

    for (int i=0;i<_MAX_ITER_EST_;i++) {
      normalise(v);                       // v is now unit
      Op.Op(v,Av);
      ratio = sqrt(norm2(Av));            // |A v| = |lambda_max| in the limit
      rq    = innerProduct(v,Av);         // complex Rayleigh quotient

      if ( i==0 ) {
        ratio0 = ratio;
        std::cout << GridLogMessage << "PowerMethodNonHermitian: step 0 (random v): |Av|/|v| = "
                  << ratio << "   [lower bound on sigma_max]" << std::endl;
      }
      if ( (i%10==0) || (i==_MAX_ITER_EST_-1) )
        std::cout << GridLogMessage << "PowerMethodNonHermitian: step " << i << " |Av|/|v| = " << ratio
                  << "  Rayleigh (" << real(rq) << "," << imag(rq) << ")" << std::endl;

      v = Av;
    }

    std::cout << GridLogMessage << "PowerMethodNonHermitian: |lambda_max| ~ " << ratio
              << "  Rayleigh (" << real(rq) << "," << imag(rq) << ")"
              << "  phase " << atan2(imag(rq),real(rq)) << " rad"
              << "  step-0/converged = " << ratio0/ratio
              << (ratio0/ratio > 1.2 ? "   ** non-normal: sigma_max well above |lambda_max| **"
                                     : "   (near-normal)")
              << std::endl;
    return ratio;
  }
};
}
