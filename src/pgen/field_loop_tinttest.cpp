//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file field_loop_tinttest.cpp
//! \brief Field loop advection with AMR; based on the original field_loop pgen iprob=1 (XY-plane loop)
//!
//! Can only be run in 2D or 3D.  Input parameters are:
//!  -  problem/rad   = radius of field loop
//!  -  problem/amp   = amplitude of vector potential (and therefore B)
//!  -  problem/x0 = X coordinate of the loop center (=0 by default)
//!  -  problem/y0 = Y coordinate of the loop center (=0 by default)
//!  -  problem/vflowx = flow velocity X direction (=0 by default)
//!  -  problem/vflowy = flow velocity Y direction (=0 by default)
//!  -  problem/vflowz = flow velocity Z direction (=0 by default)
//!  -  problem/drat  = density ratio in loop, to test density advection and conduction (1 by defaults)
//!  -  problem/bthresh =  magnetic field cut-off for AMR
//! REFERENCE: T. Gardiner & J.M. Stone, "An unsplit Godunov method for ideal MHD via
//! constrined transport", JCP, 205, 509 (2005)
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

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief field loop advection problem generator for 2D/3D problems.
//========================================================================================

int RefinementCondition(MeshBlock *pmb);

Real bsqthresh ;

void Mesh::InitUserMeshData(ParameterInput *pin) {
    
  // AMR setup:
  if (adaptive==true){
    EnrollUserRefinementCondition(RefinementCondition);
    bsqthresh = SQR(pin->GetReal("problem", "bthresh"));
  }   
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gm1 = peos->GetGamma() - 1.0;
  Real iso_cs =peos->GetIsoSoundSpeed();

  AthenaArray<Real> ax, ay, az;
  // nxN != ncellsN, in general. Allocate to extend through 2*ghost, regardless # dim
  int nx1 = block_size.nx1 + 2*NGHOST;
  int nx2 = block_size.nx2 + 2*NGHOST;
  int nx3 = block_size.nx3 + 2*NGHOST;
  ax.NewAthenaArray(nx3, nx2, nx1);
  ay.NewAthenaArray(nx3, nx2, nx1);
  az.NewAthenaArray(nx3, nx2, nx1);

  // Read initial conditions, diffusion coefficients (if needed)
  Real rad = pin->GetReal("problem","rad");
  Real amp = pin->GetReal("problem","amp");
  Real vflow_x = pin->GetOrAddReal("problem","vflowx", 0.);
  Real vflow_y = pin->GetOrAddReal("problem","vflowy", 0.);
  Real vflow_z = pin->GetOrAddReal("problem","vflowz", 0.);
  Real drat = pin->GetOrAddReal("problem","drat",1.0);

  // Use vector potential to initialize field loop
  // the origin of the initial loop
  Real x0 = pin->GetOrAddReal("problem","x0",0.0);
  Real y0 = pin->GetOrAddReal("problem","y0",0.0);
  // Real mz = pin->GetOrAddReal("problem","mz",0.0);

  for (int k=ks; k<=ke+1; k++) {
    for (int j=js; j<=je+1; j++) {
      for (int i=is; i<=ie+1; i++) {	
          ax(k,j,i) = 0.0;
          ay(k,j,i) = 0.0;
          if ((SQR(pcoord->x1f(i)-x0) + SQR(pcoord->x2f(j)-y0)) < rad*rad) {
            az(k,j,i) = amp*(rad - std::sqrt(SQR(pcoord->x1f(i)-x0) +
                                             SQR(pcoord->x2f(j)-y0))) ;// * std::sin(2.*M_PI * mz * pcoord->x3f(j));
              // added a sinusoidal modulation factor
          } else {
            az(k,j,i) = 0.0;
          }
      }
    }
  }

  // Initialize density and momenta.  If drat != 1, then density and temperature will be
  // different inside loop than background values

  Real x1size = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
  Real x2size = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
  Real x3size = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;
  Real diag = std::sqrt(x1size*x1size + x2size*x2size + x3size*x3size);
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
          phydro->u(IDN,k,j,i) = 1.0;
          phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*vflow_x; //  *x1size/diag;
          phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*vflow_y; // *x2size/diag;
          phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*vflow_z ; //*x3size/diag;
        if ((SQR(pcoord->x1v(i)) + SQR(pcoord->x2v(j)) + SQR(pcoord->x3v(k))) < rad*rad) {
          phydro->u(IDN,k,j,i) = drat;
	  phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*vflow_x ; //*x1size/diag;
	  phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*vflow_y ;// *x2size/diag;
	  phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*vflow_z ; // *x3size/diag;
        }
      }
    }
  }

  // initialize interface B
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie+1; i++) {
	pfield->b.x1f(k,j,i) = (az(k,j+1,i) - az(k,j,i))/pcoord->dx2f(j) -
	  (ay(k+1,j,i) - ay(k,j,i))/pcoord->dx3f(k);	
      }
    }
  }
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je+1; j++) {
      for (int i=is; i<=ie; i++) {
        pfield->b.x2f(k,j,i) = (ax(k+1,j,i) - ax(k,j,i))/pcoord->dx3f(k) -
                               (az(k,j,i+1) - az(k,j,i))/pcoord->dx1f(i);
      }
    }
  }
  for (int k=ks; k<=ke+1; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        pfield->b.x3f(k,j,i) = (ay(k,j,i+1) - ay(k,j,i))/pcoord->dx1f(i) -
                               (ax(k,j+1,i) - ax(k,j,i))/pcoord->dx2f(j);
      }
    }
  }

  // initialize total energy
  if (NON_BAROTROPIC_EOS) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          phydro->u(IEN,k,j,i) =
              1.0/gm1 +
              0.5*(SQR(0.5*(pfield->b.x1f(k,j,i) + pfield->b.x1f(k,j,i+1))) +
                   SQR(0.5*(pfield->b.x2f(k,j,i) + pfield->b.x2f(k,j+1,i))) +
                   SQR(0.5*(pfield->b.x3f(k,j,i) + pfield->b.x3f(k+1,j,i)))) + (0.5)*
              (SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i))
               + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        }
      }
    }
  }


  return;
}


int RefinementCondition(MeshBlock *pmb)
{
    Real bsq, maxbsq = 0.;
    
    for(int k=pmb->ks; k<pmb->ke; k++) {
        // Real  z = pmb->pcoord->x3v(k);
        for(int j=pmb->js; j<pmb->je; j++) {
            //  Real y = pmb->pcoord->x2v(j);
            for(int i=pmb->is; i<pmb->ie; i++) {
                bsq = SQR(pmb->pfield->b.x3f(k,j,i)) + SQR(pmb->pfield->b.x2f(k,j,i)) + SQR(pmb->pfield->b.x1f(k,j,i)) ;
                maxbsq = std::max(bsq, maxbsq);
            }
        }
    }
    // std::cout << maxbsq << "\n";
    if (maxbsq > bsqthresh) return 1; // refinement
    if (maxbsq < (bsqthresh/4.)) return -1; // derefinement
    return 0.;
}
    
// bin/athena -i athinput.field_loop -d models/loop time/nlim=1
