//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file jet.cpp
//! \brief Sets up a nonrelativistic jet introduced through L-x1 boundary (left edge)
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"


#ifdef MPI_PARALLEL
#include <mpi.h>
#endif


// BCs on L-x3 (lower edge) of grid with jet inflow conditions
void JetInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);

int RefinementCondition(MeshBlock *pmb);

namespace {
// Make radius of jet and jet variables global so they can be accessed by BC functions
// Real r_amb,
  Real d_amb, p_amb, vx_amb, vy_amb, vz_amb, bx_amb, by_amb, bz_amb;
  Real r_jet, d_jet, p_jet, vx_jet, vy_jet, vz_jet, bx_jet, by_jet, bz_jet;
  Real dr_jet;
  Real mang, dang;
  Real gad, gm1, x1_0, x2_0, x1min;
  Real atw_jet, atw_amb, hg_amb, hg_jet, rang_jet, rang_amb, phang_jet, phang_amb;
  Real SmoothStep(Real x);
} // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // initialize global variables
  d_amb  = pin->GetReal("problem", "d");
  p_amb  = pin->GetReal("problem", "p");
  vx_amb = pin->GetReal("problem", "vx");
  vy_amb = pin->GetReal("problem", "vy");
  vz_amb = pin->GetReal("problem", "vz");
  if (MAGNETIC_FIELDS_ENABLED) {
    bx_amb = pin->GetReal("problem", "bx");
    by_amb = pin->GetReal("problem", "by");
    bz_amb = pin->GetReal("problem", "bz");
  }
  d_jet  = pin->GetReal("problem", "djet");
  p_jet  = pin->GetReal("problem", "pjet");
  vx_jet = pin->GetReal("problem", "vxjet"); // sets the opening angle of the jet (\tg = vxjet/vyjet)
  vy_jet = pin->GetReal("problem", "vyjet"); // sets the rotation 4-velocity at the jet boundary
  vz_jet = pin->GetReal("problem", "vzjet");
  if (MAGNETIC_FIELDS_ENABLED) {
    bx_jet = pin->GetReal("problem", "bxjet");
    by_jet = pin->GetReal("problem", "byjet");
    bz_jet = pin->GetReal("problem", "bzjet");
  }
  r_jet = pin->GetReal("problem", "rjet");
  dr_jet = pin->GetReal("problem", "drjet");
  x1min = mesh_size.x1min;
  x1_0 = 0.5*(mesh_size.x1max + mesh_size.x1min);
  x2_0 = 0.5*(mesh_size.x2max + mesh_size.x2min);

  // openangle = pin->GetReal("problem", "openangle"); // opening angle of the jet, radians
    
    mang = pin->GetReal("problem", "mang");
    dang = pin->GetReal("problem", "dang");

    gad = pin->GetReal("hydro", "gamma"); // adiabatic index
    
    // parameter combinations
    Real gamma_amb = sqrt(1.+vx_amb*vx_amb+vy_amb*vy_amb+vz_amb*vz_amb);
    atw_amb = gamma_amb*gamma_amb * (d_amb + gad/(gad-1.) * p_amb) ; // Atwood parameter
    hg_amb = (1.+gad/(gad-1.)*p_amb / d_amb) * gamma_amb; // \gamma_inf
    rang_amb = vx_amb / vz_amb ;
    phang_amb = vy_amb / vz_amb ;

    Real gamma_jet = sqrt(1.+vx_jet*vx_jet+vy_jet*vy_jet+vz_jet*vz_jet);
    atw_jet = gamma_jet*gamma_jet * (d_jet + gad/(gad-1.) * p_jet) ; // Atwood parameter
    hg_jet = (1.+gad/(gad-1.)*p_jet / d_jet) * gamma_jet; // \gamma_inf
    rang_jet = vx_jet / vz_jet ;
    phang_jet = vy_jet / vz_jet ; 

  // enroll boundary value function pointers
  EnrollUserBoundaryFunction(BoundaryFace::inner_x3, JetInnerX3);
    
  if(adaptive==true)
      EnrollUserRefinementCondition(RefinementCondition);

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the Jet problem
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  gm1 = peos->GetGamma() - 1.0;
  gad = peos->GetGamma() ;

  // initialize conserved variables
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
          phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = d_amb;
          if(std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
              phydro->w(IM1,k,j,i) = phydro->w1(IM1,k,j,i) = vx_amb;
              phydro->w(IM2,k,j,i) = phydro->w1(IM2,k,j,i) = vy_amb;
              phydro->w(IM3,k,j,i) = phydro->w1(IM3,k,j,i) = vz_amb;

          }
          else{
              phydro->w(IM1,k,j,i) = phydro->w1(IM1,k,j,i) = vx_amb;
              phydro->w(IM2,k,j,i) = phydro->w1(IM2,k,j,i) = vy_amb; // perpendicular
              phydro->w(IM3,k,j,i) = phydro->w1(IM3,k,j,i) = vz_amb; // along the jet
          }
        phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = p_amb;
      }
    }
  }

    AthenaArray<Real> bb;

    
  // initialize interface B
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          pfield->b.x1f(k,j,i) = bx_amb;
            bb(IB1, k,j,i) = bx_amb;
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=is; i<=ie; ++i) {
          pfield->b.x2f(k,j,i) = by_amb;
            bb(IB2, k,j,i) = by_amb;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          pfield->b.x3f(k,j,i) = bz_amb;
            bb(IB3, k,j,i) = bz_amb;
        }
      }
    }
  }
    
  peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, is, ie, js, je, ks, ke);
    
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void JetInnerX1()
//  \brief Sets boundary condition on left X boundary (iib) for jet problem

void JetInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // set primitive variables in inlet ghost zones
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
	Real rad, pert;
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          rad = pco->x1v(i);
	  pert = (1.+dang * cos(pco->x2v(j)*mang)) ; // perturbation   
          rad *= pert ; 
        }
        else{
              rad = std::sqrt(SQR(pco->x1v(i)-x1min) + SQR(pco->x2v(j)-x2_0));
          }
          Real step = SmoothStep(-(rad - r_jet)/dr_jet);
          // Real divfactor = (rad/pert-x1min) / r_jet * openangle ;  // opening * (R/Rjet)
	  Real atw = (atw_jet-atw_amb) * step + atw_amb ;
          Real hg = (hg_jet - hg_amb) * step + hg_amb ;
          Real rang = (rang_jet - rang_amb) * step + rang_amb ;
          rang *= (rad - pco->x1v(0)) / r_jet ; // expansion rate -> physical velocity
          Real phang = (phang_jet - phang_amb) * step + phang_amb ;
          // phang *= (rad-pco->x1v(0)) / r_jet ; // omega -> physical velocity
	  Real press = (p_jet - p_amb) * step + p_amb ;
	  Real gamma = atw / (8.*press * hg) * (sqrt(1.+16.*press*hg*hg/atw)-1.);
	  prim(IDN,kl-k,j,i) = atw/gamma/gamma-4.*press ; // (d_jet-d_amb) * step + d_amb;
	  prim(IVZ,kl-k,j,i) =  sqrt((gamma*gamma-1.)/(1. + rang * rang + phang * phang)) ; // (vz_jet-vz_amb) * step + vz_amb;
	  prim(IVX,kl-k,j,i) = prim(IVZ,kl-k,j,i) * rang ; // (vz_jet*divfactor-vx_amb) * step + vx_amb; // radial velocity
	  prim(IVY,kl-k,j,i) = prim(IVZ,kl-k,j,i) * phang ;  // (vy_jet*rad-vy_amb) * step + vy_amb;
	  prim(IPR,kl-k,j,i) = (p_jet-p_amb) * step + p_amb;
	  
		    // prim(IVX,kl-k,j,i) = (vz_jet*divfactor-vx_amb) * step + vx_amb; // radial velocity
		    // prim(IVY,kl-k,j,i) = (vy_jet*rad-vy_amb) * step + vy_amb;
		    // prim(IVZ,kl-k,j,i) = (vz_jet-vz_amb) * step + vz_amb;
		    // prim(IPR,kl-k,j,i) = (p_jet-p_amb) * step + p_amb;
          if(MAGNETIC_FIELDS_ENABLED) {
              b.x1f(k,j,il-i) = (bx_jet-bx_amb) * step + bx_amb;
              b.x2f(k,j,il-i) = (by_jet-by_amb) * step + by_amb;
              b.x3f(k,j,il-i) = (bz_jet-bz_amb) * step + bz_amb;
          }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn int RefinementCondition(MeshBlock *pmb)
//! \brief refinement condition: maximum density and pressure curvature
// taken from dmr pgen

int RefinementCondition(MeshBlock *pmb) {
  AthenaArray<Real> &w = pmb->phydro->w;
  Real maxeps = 0.0;
    for (int k = pmb->ks; k<=pmb->ke; k++)
    {
        for (int j=pmb->js; j<=pmb->je; j++) {
            for (int i=pmb->is; i<=pmb->ie; i++) {
                Real epsr = (std::abs(w(IDN,k,j,i+1) - 2.0*w(IDN,k,j,i) + w(IDN,k,j,i-1)))
                  /w(IDN,k,j,i);
                Real epsphi = (std::abs(w(IDN,k,j+1,i) - 2.0*w(IDN,k,j,i) + w(IDN,k,j-1,i)))
                  /w(IDN,k,j,i);
                Real epsz = (std::abs(w(IDN,k+1,j,i) - 2.0*w(IDN,k,j,i) + w(IDN,k-1,j,i-1)))
                /w(IDN,k,j,i);
                Real eps = std::max(epsz, std::max(epsr, epsphi));
                maxeps = std::max(maxeps, eps);
            }
        }
    }
  // refine : curvature > 0.01
  if (maxeps > 0.01) return 1;
  // derefinement: curvature < 0.005
  if (maxeps < 0.005) return -1;
  // otherwise, stay
  return 0;
}


namespace {
    Real SmoothStep(Real x)
    {
        // step function approximation
        return (tanh(x)+1.)/2. ; // x/std::sqrt(x*x+1.);
    }
} // namespace
