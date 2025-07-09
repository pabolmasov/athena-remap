//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file slowSW
//! \brief Problem generator for slow shock-wave formation and stability 
//!
//! Can only be run in 2D or 3D.  Input parameters are:
//!  -  problem/bx   = magnetic field (uniform)
//!  -  problem/rhoinj   = injection density
//!  -  problem/vxinj = injection velocity
//!  -  problem/pressinj  = injection pressure
//!  -  vamp = velocity perturbation (0 by default)
//!  -  mv   = velocity perturbation mode (integer number, 1 by default)
//!  -  xshock = shock front initial position
//!  -  dxshock = front shape perturbation (0 by default)
//!  -  bthresh = cut-off magnetic field strength for AMR (valid only if AMR is used)
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"

#if !MAGNETIC_FIELDS_ENABLED
#error "This problem generator requires magnetic fields"
#endif

int RefinementCondition(MeshBlock *pmb);

void InjectionInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		      FaceField &b, Real time, Real dt,
		      int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void DejectionOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		      FaceField &b, Real time, Real dt,
		      int il, int iu, int jl, int ju, int kl, int ku, int ngh);

Real bsqthresh, rhoinj, pressinj, vxinj, jinj, bx, vyamp, xamp, rhofac;
Real Hugoniot_sigma, Hugoniot_Pi, Mach1, Hugoniot_sigma0;

namespace{

  Real stepfun(Real x);

  Real xshock, dxshock;

  int mv;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {

  rhoinj = pin->GetReal("problem", "rhoinj");
  pressinj = pin->GetReal("problem", "pressinj");
  vxinj = pin->GetReal("problem", "vxinj");
  bx = pin->GetReal("problem", "bx");

  vyamp = pin->GetOrAddReal("problem", "vamp", 0.0);
  xamp = pin->GetOrAddReal("problem", "xamp", 0.0);
  
  //  rhofac = pin->GetReal("problem", "rhofac");

  xshock = pin->GetReal("problem", "xshock");
  dxshock = pin->GetReal("problem", "dxshock");
 
  mv = pin->GetInteger("problem", "mv");

  jinj = rhoinj * vxinj; 
  
  // AMR setup (not used so far)
  if (adaptive==true){
    EnrollUserRefinementCondition(RefinementCondition);
    bsqthresh = SQR(pin->GetReal("problem", "bthresh"));
  }
  // user-defined BC (outflow for hydro, infinite conductor for B):
  std::string inner_Xboundary = pin->GetString("mesh", "ix1_bc");
  std::string outer_Xboundary = pin->GetString("mesh", "ox1_bc");
  if (inner_Xboundary == "user")EnrollUserBoundaryFunction(BoundaryFace::inner_x1, InjectionInnerX1);
  if (outer_Xboundary == "user")EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DejectionOuterX1);    
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  Real gamma = pin->GetReal("hydro","gamma"); // adiabatic gamma

  std::cout << "Mach number = " << vxinj / std::sqrt(gamma * pressinj / rhoinj) << "\n";
  std::cout << "Alfven Mach number = " << vxinj * std::sqrt(rhoinj) / bx << "\n";

  //  getchar();

  Real x1max = pmy_mesh->mesh_size.x1max, x1min = pmy_mesh->mesh_size.x1min, x;
  Real x2max = pmy_mesh->mesh_size.x2max, x2min = pmy_mesh->mesh_size.x2min, y;

  // Hugoniot parameters:
  Hugoniot_sigma0 = (gamma+1.)/(gamma-1.); // compression for strong shock
  Mach1 = vxinj / std::sqrt(gamma*pressinj/rhoinj);

  // (Raychaudhury section 6.5)
  Hugoniot_Pi = 1. + 2.*gamma/(gamma+1.) * (SQR(Mach1)-1.);
  Hugoniot_sigma = (gamma+1.) / (gamma-1. + 2./SQR(Mach1));

  std::cout  << "Mach = " << Mach1 << "\n";
  std::cout  << "Hugoniot_sigma = " <<  Hugoniot_sigma  << "\n";
  std::cout  << "Hugoniot_Pi = " <<  Hugoniot_Pi  << "\n";
   
  // getchar();

  rhofac = Hugoniot_sigma;
  
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      y = pcoord->x2v(j);
      for (int i=is; i<=ie; i++) {
	Real xshock_mod = xshock + xamp  * std::sin((double)mv*PI*(y - x2min)/(x2max-x2min)); 
	x = pcoord->x1v(i);
	phydro->u(IDN,k,j,i) = rhoinj * (1. + (Hugoniot_sigma - 1.) * stepfun((x-xshock_mod)/dxshock));
	phydro->u(IM1,k,j,i) = jinj; // mass flux conservation
	phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i) * vyamp
	  * std::sin((double)mv*PI*(y - x2min)/(x2max-x2min))
	  * std::sin(PI*(x - x1min)/(x1max-x1min)); //
	phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*0. ; //
	if (NON_BAROTROPIC_EOS) {
	  phydro->u(IEN,k,j,i) = pressinj /(gamma-1.) * (1. + (Hugoniot_Pi - 1.) * stepfun((x-xshock_mod)/dxshock)) 
	    + (SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))+SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i) * 0.5 + SQR(bx) * 0.5;
	}
      }
    }
  }
  

  // initialize interface B
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie+1; i++) {
	pfield->b.x1f(k,j,i) = bx; // constant, nothing to worry about
      }
    }
  }
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je+1; j++) {
      for (int i=is; i<=ie; i++) {
        pfield->b.x2f(k,j,i) = 0.;                               
      }
    }
  }
  for (int k=ks; k<=ke+1; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        pfield->b.x3f(k,j,i) = 0.;
      }
    }
  }


  return;
}

// injection BC (inner x)
void InjectionInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
			 FaceField &b, Real time, Real dt,
			 int il, int iu, int jl, int ju, int kl, int ku, int ngh){
  // cell-centered:
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
	prim(IDN,k,j,il-i) = rhoinj;
	prim(IEN,k,j,il-i) = pressinj;
	prim(IVX,k,j,il-i) = vxinj;
	prim(IVY,k,j,il-i) = 0.;
	prim(IVZ,k,j,il-i) = 0.;
	//	if(NSCALARS>0){
	//  for (int n=0; n<(NSCALARS); ++n)pmb->pscalars->s(n,k,j,il-i) = 0;
	//	}
      }
    }
  }

  // fields:
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
	for (int i=1; i<=ngh; ++i) {
	  b.x1f(k,j,il-i) = bx;
	}
      }
    }
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
	for (int i=1; i<=ngh; ++i) {
	  b.x2f(k,j,il-i) = -b.x2f(k,j,il);
	}
      }
    }
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
	for (int i=1; i<=ngh; ++i) {
	  b.x3f(k,j,il-i) = -b.x3f(k,j,il);
	}
      }
    }
  }

}

// squeeze-out BC (outer x)
void DejectionOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
			 FaceField &b, Real time, Real dt,
			 int il, int iu, int jl, int ju, int kl, int ku, int ngh){
  //  Real gamma =  1.333333333333333333333 ; // TODO: read the actual gamma
  Real vxd = vxinj / Hugoniot_sigma; // expected flow speed after the shock
  // Real x0 = pco->x1v(iu), x;
  // cell-centered:
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      Real rho = prim(IDN,k,j,iu), press = prim(IPR,k,j,iu), vx = prim(IVX,k,j,iu);
      for (int i=1; i<=ngh; ++i) {
	// x = pco->x1v(iu+i);
	prim(IDN,k,j,iu+i) = rho;
	prim(IVX,k,j,iu+i) = std::max(vxd + std::min(vx-vxd, 0.), 0.); // picks up the minimal of vx and vxd, if this value is positive
	prim(IPR,k,j,iu+i) = press ; // + (gamma-1)/gamma/2. * (SQR(vx)-SQR(prim(IVX,k,j,iu+i))) * rho; // + std::max(rho * vx - jinj, 0.) * (vx - vxinj) * (x-x0);
	  // std::max(std::min(prim(IVX,k,j,iu),vxinj), 0.);
	prim(IVY,k,j,iu+i) = prim(IVY,k,j,iu);
	prim(IVZ,k,j,iu+i) = prim(IVZ,k,j,iu);
	//	if(NSCALARS>0){
	//  for (int n=0; n<(NSCALARS); ++n)pmb->pscalars->s(n,k,j,iu+i) = 0;
	//	}
      }
    }
  }
  // fields:
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
	for (int i=1; i<=ngh; ++i) {
	  b.x1f(k,j,iu+i+1) = b.x1f(k,j,iu-i+1);
	}
      }
    }
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
	for (int i=1; i<=ngh; ++i) {
	  b.x2f(k,j,iu+i) = b.x2f(k,j,iu-i+1); // 0.;
	}
      }
    }
    // what if one layer is corrupted, and we need to move the BC one cell closer?
    for (int k=kl; k<=ku+1; ++k) { 
      for (int j=jl; j<=ju; ++j) {
	for (int i=1; i<=ngh; ++i) {
	  b.x3f(k,j,iu+i) =  b.x3f(k,j,iu-i+1); 
	}
      }
    }
  }
}


int RefinementCondition(MeshBlock *pmb)
{// it makes sense to turn on the refinement condition when transverse fields are non-zero
    Real bsq, maxbsq = 0.;
    
    for(int k=pmb->ks; k<pmb->ke; k++) {
        // Real  z = pmb->pcoord->x3v(k);
        for(int j=pmb->js; j<pmb->je; j++) {
            //  Real y = pmb->pcoord->x2v(j);
            for(int i=pmb->is; i<pmb->ie; i++) {
                bsq = SQR(pmb->pfield->b.x3f(k,j,i)) + SQR(pmb->pfield->b.x2f(k,j,i)) + 0. * SQR(pmb->pfield->b.x1f(k,j,i)) ;
                maxbsq = std::max(bsq, maxbsq);
            }
        }
    }
    // std::cout << maxbsq << "\n";
    if (maxbsq > bsqthresh) return 1; // refinement
    if (maxbsq < (bsqthresh/4.)) return -1; // derefinement
    return 0.;
}
namespace{
  // step function
  // TODO: make it global, practically any simulation needs it
  Real stepfun(Real x){
    return (std::tanh(x)+1.0)/2.0; 
  }
}
