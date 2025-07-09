//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file binary_gravity.cpp
//  \brief Problem generator to test Multigrid Poisson solver with Multipole Expansion

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstring>    // memset
#include <ctime>
#include <iomanip>
#include <iostream>
#include <list>
#include <fstream> // reading and writing files
using std::ifstream;
#include <limits>

#include <gsl/gsl_sf.h> // GNU scientific library

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../gravity/gravity.hpp"
#include "../gravity/mg_gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "../multigrid/multigrid.hpp"
#include "../scalars/scalars.hpp"
#include "../parameter_input.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../utils/utils.hpp"     // ran2()

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#if SELF_GRAVITY_ENABLED != 2
#error "This problem generator requires Multigrid gravity solver."
#endif

namespace{
  Real four_pi_G;
  Real rstar, mstar, bgdrho, temp, amp, rscale;
  Real xshift, yshift, zshift;
  int lharm, mharm;
  Real rmin, rmax, magbeta;
 bool ifflat, iftab, ifinclined, iftracer = true;
 Real rzero;
 std::string starmodel; // the name of the MESA file containing the density and pressure profiles
 // Real instar_interpolate(std::list<Real> instar_radius, std::list<Real>instar_array, std::list<Real> instar_mass, Real r2, Real rres);
 void instar_interpolate(std::list<Real> instar_radius, std::list<Real> instar_lrho, std::list<Real> instar_lpress, std::list<Real> instar_mass, Real r2, Real rres, Real *rho, Real *p);
 void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
 void GetCylCoord_f(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
 Real psi_AB(Real r, Real cth, Real phi);
 Real azfun(Real r, Real z);
 Real SmoothStep(Real x);
 
 Real BHgmax, rgrav, addmass, BHdist, Mcoeff, tper, rper, rBH;
 Real rcutoff, drcutoff;
 Real cthA = 1., sthA = 0., phiA = 0.;
 Real Rthresh, thresh;
}

int RefinementCondition(MeshBlock *pmb);

void BHcoord(Real time, Real* xBH, Real* yBH, Real* zBH);

void BHgrav(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);
void BHcooling(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);


void DumbBoundaryInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void DumbBoundaryOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void DumbBoundaryInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void DumbBoundaryOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void DumbBoundaryInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void DumbBoundaryOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);


void Mesh::InitUserMeshData(ParameterInput *pin) {
    four_pi_G = pin->GetReal("problem","four_pi_G");
    
    // boolean switches:
    ifflat = pin->GetBoolean("problem", "ifflat");
    iftab = pin->GetBoolean("problem", "iftab");
    ifinclined = pin->GetBoolean("problem", "ifinclined");
    
    if (iftab){
        starmodel = pin->GetString("problem", "starmodel");
        std::cout << "reading from " << starmodel << std::endl ;
    }
    rmin = pin->GetReal("problem","rmin");
    rmax = pin->GetReal("problem","rmax");
    mstar = pin->GetReal("problem","mstar");
    rstar = pin->GetReal("problem","rstar");
    rscale = pin->GetReal("problem","rscale");

    rcutoff = pin->GetReal("problem","rcutoff");
    drcutoff = pin->GetReal("problem","drcutoff");

    if(ifinclined){
       Real inc = pin->GetOrAddReal("problem", "inc", 0.);
       cthA = std::cos(inc); sthA = std::sin(inc);
       phiA = pin->GetOrAddReal("problem", "phiA", 0.); // magnetic axis orientation
     }
    
    addmass = pin->GetReal("problem","mBH"); // black hole mass
    rgrav = (addmass/1.e6) * 2.1218 ; // GM_{\rm BH}/c^2
    std::cout << "rgrav = " << rgrav << std::endl ;
    rBH = pin->GetReal("problem", "rBH");
    BHgmax = addmass / SQR(rgrav); // GM/R^2 at R = 3GM/c^2

    rper = pin->GetReal("problem","rper"); // pericenter distance
    Mcoeff = std::sqrt(addmass / 2.) * std::pow(rper, -1.5);
    tper = pin->GetReal("problem","tper"); // time lag
    
    bgdrho = pin->GetReal("problem","bgdrho");
    temp = pin->GetReal("problem","temp");
    amp = pin->GetReal("problem","amp");
    xshift = pin->GetReal("problem","xshift");
    yshift = pin->GetReal("problem","yshift");
    zshift = pin->GetReal("problem","zshift");
    lharm = pin->GetReal("problem","lharm");
    mharm = pin->GetReal("problem","mharm");
    magbeta = pin->GetReal("problem","magbeta"); // magnetic parameter
    Rthresh = pin->GetOrAddReal("problem","rthresh", 0.5); // R0 cut-off
    thresh = pin->GetOrAddReal("problem","thresh", 0.1); // relative one-cell variation for refinement

    // boundary conditions
    // user-defined BC (outflow for hydro, infinite conductor for B):                                                                                                                                             
    std::string inner_Xboundary = pin->GetString("mesh", "ix1_bc");
    std::string outer_Xboundary = pin->GetString("mesh", "ox1_bc");
    std::string inner_Yboundary = pin->GetString("mesh", "ix2_bc");
    std::string outer_Yboundary = pin->GetString("mesh", "ox2_bc");
    std::string inner_Zboundary = pin->GetString("mesh", "ix3_bc");                                             std::string outer_Zboundary = pin->GetString("mesh", "ox3_bc");

    if (inner_Xboundary == "user")EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DumbBoundaryInnerX1);
    if (outer_Xboundary == "user")EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DumbBoundaryOuterX1);
    if (inner_Yboundary == "user")EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DumbBoundaryInnerX2);
    if (outer_Yboundary == "user")EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DumbBoundaryOuterX2);
    if (inner_Zboundary == "user")EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DumbBoundaryInnerX3);
    if (outer_Zboundary == "user")EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DumbBoundaryOuterX3);

    // AMR setup:
    if (adaptive==true)
      EnrollUserRefinementCondition(RefinementCondition);

    EnrollUserExplicitSourceFunction(BHgrav);
    SetFourPiG(four_pi_G);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for gravity from a binary
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
 //   Real x0 = 12.0/1024.0, y0 = 0.0, z0 = 0.0, r = 6.0/1024.0;
 
    // Real starrho = mstar  / (4. * PI /3. * rstar * rstar * rstar);

    Real G = four_pi_G / (4.0 * PI), drsq = 1.;
    
    Real den1;
    Real dencurrent = 0., pcurrent = 0.;
    
    std::int64_t iseed = - 1 - gid;

    // density distribution in the star:
     std::list<Real> instar_mass = {};
     std::list<Real> instar_radius = {};
     std::list<Real> instar_lrho = {};
     std::list<Real> instar_lpress = {};

    Real tmp_m, tmp_r, tmp_lrho, tmp_lpress, instar_rmax = 0., instar_mtot = 0.;
    
    if (iftab){
        ifstream indata;
	clock_t tstart1 = clock();
        indata.open(starmodel); // opens the file
        // std::string s;
        // indata >> s; // header (skipping one line)
        while ( !indata.eof() ) { // keep reading until end-of-file
            indata >> tmp_m >> tmp_r >> tmp_lrho >> tmp_lpress; // sets EOF flag if no value found
            // std::cout << tmp_m << " "<< tmp_r << " " << tmp_lrho << " " << tmp_lpress << std::endl ;
            instar_mass.push_back(tmp_m);
            instar_radius.push_back(tmp_r);
            instar_lrho.push_back(tmp_lrho); // -0.770848
            instar_lpress.push_back(tmp_lpress/G); // -16.0512
            if (tmp_r > instar_rmax) instar_rmax = tmp_r;
            if (tmp_m > instar_mtot) instar_mtot = tmp_m;
        }
        indata.close();
	
	clock_t tend1 = clock();
	float cpu_time = (tend1>tstart1 ? static_cast<Real>(tend1-tstart1) : 1.0) /
	  static_cast<Real>(CLOCKS_PER_SEC);
	if(Globals::my_rank == 0){
	  std::cout << "instar_rmax = " << instar_rmax << std::endl ;
	  std::cout << "instar_mtot = " << instar_mtot << std::endl ;
	  std::cout << "reading time = " << cpu_time << std::endl ;
	}
    }
    
    Real starmass_empirical = 0.;
    
    Real rnorm = std::sqrt(rmin*rmax/3.);
    // interpolation for velocity normalization
    if (iftab){
        instar_interpolate(instar_radius, instar_lrho, instar_lpress, instar_mass, rnorm, 0., &dencurrent, &pcurrent) ;
    }
    
    if(ifflat){
        dencurrent = 1. ;
        pcurrent =  1.-SQR(rnorm/rscale) ; // flat star case
    }
    Real rcsnorm = std::sqrt(dencurrent * pcurrent) / fabs(rmax - rnorm) * fabs(rnorm - rmin);

    Real dnorm = mstar * 3./4./PI * std::pow(rscale, -3.); // default values (for a flat star)
    
    if(!ifflat){
        // n=5 solution
        dnorm = mstar / (4.0 * PI * std::sqrt(3.)) * std::pow(rscale, -3.0);
        pcurrent = std::pow(1.+SQR(rnorm/rscale)/3., -2.5);
    }
    Real pnorm = four_pi_G * SQR(dnorm * rscale) / 6. ;

    if (iftab){
        dnorm = mstar * std::pow(rscale/instar_rmax, -3.);
        pnorm = SQR(mstar/SQR(rscale/instar_rmax));
    }
    
    Real rvir = G*mstar / temp;
    Real rvirBH = G*addmass / temp;
    Real xBH, yBH, zBH;
    BHcoord(0., &xBH, &yBH, &zBH);
    Real reff0 = std::sqrt(SQR(xBH)+SQR(yBH)+SQR(zBH));
    
    if(Globals::my_rank == 0)std::cout << "distance toward the BH = " << reff0 << std::endl ;
    
    rzero = reff0;
    
    if (magbeta > 0.){ // magbeta <0 would turn off magnetic fields
       // interpolation for MF normalization
        rnorm = (rmax+rmin)/2.;
        rcsnorm = std::sqrt(2. * pcurrent * pnorm / magbeta);
        if(Globals::my_rank == 0)std::cout << "B norm = " << rcsnorm << std::endl;
     }
    
    AthenaArray<Real> ax, ay, az;
    //    AthenaArray<Real> ax, ay, az;
    int nx1 = block_size.nx1 + 2*NGHOST;
    int nx2 = block_size.nx2 + 2*NGHOST;
    int nx3 = block_size.nx3 + 2*NGHOST;
    ax.NewAthenaArray(nx3, nx2, nx1);
    ay.NewAthenaArray(nx3, nx2, nx1);
    az.NewAthenaArray(nx3, nx2, nx1);

    Real cthsqmax = 0.99 ; // a limit for cos^2\theta
    
    clock_t tstart = clock();

    for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
            for (int i = is; i <= ie; ++i) {
                Real z = pcoord->x3v(k)-zshift;
                Real x = pcoord->x1v(i)-xshift, y = pcoord->x2v(j)-yshift;
                Real xf = pcoord->x1f(i), yf = pcoord->x2f(j), zf = pcoord->x3f(k);
                ax(k,j,i) = ay(k,j,i) = az(k,j,i) = 0.;
                Real r1 = std::sqrt(x*x+y*y+z*z), r2f = std::sqrt(SQR(xf)+SQR(yf)+SQR(zf));;
                Real cth1 = z / r1;
                Real phi1 = std::atan2(y,x);
                Real cth2f = zf / r2f, phi2f = std::atan2(yf,xf);
                Real reff = std::sqrt(SQR(x-xBH)+SQR(y-yBH)+SQR(z-zBH));
                Real PhiBH =  - rvirBH / (reff-2.*rgrav);
                if (reff<(3.*rgrav)){
		  PhiBH = (- 1.5 + SQR(reff/rgrav)/6.) * rvirBH / rgrav;
                }
                PhiBH += rvirBH / (reff0-2.*rgrav); // zeroing at the star centre
                Real rhogas = bgdrho * std::exp(std::max(std::min(rvir * (-1./std::sqrt(rscale*rscale+drsq) + 1./std::sqrt(r1*r1+drsq)), std::log(dnorm/bgdrho)-2.),-30.));
                Real dx = pcoord->dx1f(i), dy = pcoord->dx2f(j), dz = pcoord->dx3f(k);
                Real dr = std::sqrt(SQR(dx)+SQR(dy)+SQR(dz)), dv = std::fabs(dx*dy*dz);
                // Real dr = std::sqrt(SQR(pcoord->dx1f(i))+SQR(pcoord->dx2f(j))+SQR(pcoord->dx3f(k))), dv = std::fabs(pcoord->dx1f(i) * pcoord->dx2f(j) * pcoord->dx3f(k));
                Real vr = 0.,vth = 0., vphi = 0., cs = 0.;
                if (ifflat || iftab){
		  if (r1<=std::max(rstar,dr)){
                        if (ifflat){
                            dencurrent = dnorm ;
                            pcurrent = (1.-SQR(r1/rscale)) * pnorm ;
                        }
                        else{
			  //                            if (iftab){
			  instar_interpolate(instar_radius, instar_lrho, instar_lpress, instar_mass, r1 / rscale * instar_rmax, dr / rscale * instar_rmax, &dencurrent, &pcurrent) ; // * pcoord->dx1f(i));
			  dencurrent *= dnorm ;
			  pcurrent *= pnorm ;
			      //}
                        }
		  }else{
                        dencurrent = 0. ;
                        pcurrent = 0.;
                    }
                }
                else{
                    // n=5 solution
		  dencurrent = dnorm * std::pow(1.+SQR(r1/std::max(rscale,dr))/3., -2.5) * SmoothStep(-(r1-rcutoff)/std::max(drcutoff,dr));
		  pcurrent = pnorm * std::pow(1.+SQR(r1/std::max(rscale,dr))/3., -3.0) * SmoothStep(-(r1-rcutoff)/std::max(drcutoff,dr));
                }
                // vr = vth = vphi = 0.; // maybe we will do something to the velocities
                // Real vxr = (vr * x/r1 + vth * cth1 * cos(phi1) - vphi * sin(phi1)) ;
                // Real vyr = (vr * y/r1 + vth * cth1 * sin(phi1) + vphi * cos(phi1)) ;
                // Real vzr = (vr * z/r1 - vth * std::sqrt(1.-cth1*cth1)) ;
                
		if (pcurrent >= (rhogas*temp)){
		  phydro->w(IDN,k,j,i) = dencurrent;
		  phydro->w(IPR,k,j,i) = pcurrent;
		}
		else{
                  phydro->w(IDN,k,j,i) = rhogas;
                  phydro->w(IPR,k,j,i) = rhogas * temp ;		  
		}
		//                 phydro->w(IDN,k,j,i) = std::max(dencurrent, rhogas);
		// phydro->w(IPR,k,j,i) = std::max(pcurrent, rhogas*temp);
                phydro->w(IM1,k,j,i) = 0.;
                phydro->w(IM2,k,j,i) = 0.;
                phydro->w(IM3,k,j,i) = 0.;
                
                // tracer:
                if (NSCALARS>0){
                   constexpr int scalar_norm = NSCALARS > 0 ? NSCALARS : 1.0;
                   Real d0 = phydro->w(IDN,k,j,i);
                   if (r1<std::max(rstar,dr)){
                     for (int n=0; n<NSCALARS; ++n) {
                       pscalars->s(n,k,j,i) = d0*1.0/scalar_norm;
                       starmass_empirical += dencurrent * dv ;
                       //              pscalars->r(n, k, j, i) = 1.0;
                     }
                   }
                   else{
                     for (int n=0; n<NSCALARS; ++n) {
                       pscalars->s(n,k,j,i) = d0*0.0/scalar_norm;
                       // pscalars->r(n,k,j,i) = 0.0;
                     }
                     //              pscalars->s(0,k,j,i) = 0.;
                   }
                 }

                //  magnetic fields (vector potentials)
                if((magbeta > 0.) && MAGNETIC_FIELDS_ENABLED){
                    if (lharm <= 0){ // toroidal field
                        // Real rc = std::sqrt(SQR(xf)+SQR(yf)); // cylindric radius
                        // az(k,j,i) = azfun(rc, zf);
                        // vector potential at the boundaries:
                        for(int di = -1; di<=1; di++){
                            for(int dj = -1; dj<=1; dj++){
                                for(int dk = -1; dk<=1; dk++){
                                    Real xf1, yf1, zf1;
                                    if (di == -1)xf1 = xf - dx;
                                    if (di == 0)xf1 = xf;
                                    if (di == 1)xf1 = xf + dx;
                                    if (dj == -1)yf1 = yf - dy;
                                    if (dj == 0)yf1 = yf;
                                    if (dj == 1)yf1 = yf + dy;
                                    if (dk == -1)zf1 = zf - dz;
                                    if (dk == 0)zf1 = zf;
                                    if (dk == 1)zf1 = zf + dz;
                                    //                    Real yf1 = yf +(Real)dj * dy;
                                    // Real zf1 = zf +(Real)dk * dz;
                                    // rc = std::sqrt(SQR(xf1)+SQR(yf1));
                                    
                                    if(ifinclined){
                                        Real zf1i = zf1 * cthA + (xf1 * std::cos(phiA) + yf1 * std::sin(phiA)) * sthA ; // zf1 in the inclined frame
                                        //  Real xf1i = (zf1-zf1i * cthA) / sthA ;
                                        // Real yf1i = yf1 * std::cos(phiA) - xf1 * std::sin(phiA) ;
                                        Real azd = azfun(std::sqrt(SQR(xf1)+SQR(yf1)+SQR(zf1)-SQR(zf1i)), zf1i);
                                        az(k+dk,j+dj,i+di) = azd * cthA;
                                        ax(k+dk,j+dj,i+di) = azd * sthA * std::cos(phiA);
                                        ay(k+dk,j+dj,i+di) = azd * sthA * std::sin(phiA);
                                        // if ((zf1i < 1.0) && (std::fabs(az(k+dk,j+dj,i+di)) > 0.))std::cout << "Az = " << az(k+dk,j+dj,i+di) << std::endl;
                                    }
                                    else{
                                        az(k+dk,j+dj,i+di) = azfun(std::sqrt(SQR(xf1)+SQR(yf1)), zf1);
                                        // if (az(k+dk,j+dj,i+di) > 0.)std::cout << "Az = " << az(k+dk,j+dj,i+di) << std::endl;
                                    }
                                    
                                }
                            }
                        }
                    }
                    else{
                        Real aphi = psi_AB(r2f, cth2f, phi2f);
                        for(int di = -1; di<=1; di++){
                            for(int dj = -1; dj<=1; dj++){
                                for(int dk = -1; dk<=1; dk++){
                                    // Real xf1 = xf + (Real)di * dx;
                                    // Real yf1 = yf + (Real)dj * dy;
                                    // Real zf1 = zf + (Real)dk * dz;
                                    Real xf1, yf1, zf1;
                                    if (di == -1)xf1 = xf - dx;
                                    if (di == 0)xf1 = xf;
                                    if (di == 1)xf1 = xf + dx;
                                    if (dj == -1)yf1 = yf - dy;
                                    if (dj == 0)yf1 = yf;
                                    if (dj == 1)yf1 = yf + dy;
                                    if (dk == -1)zf1 = zf - dz;
                                    if (dk == 0)zf1 = zf;
                                    if (dk == 1)zf1 = zf + dz;
                                    
                                    r2f = std::sqrt(SQR(xf1)+SQR(yf1)+SQR(zf1));
                                    
                                    Real cth1f = zf1 / r2f; // theta in lab frame
                                    Real sth1f = std::sqrt(1.-std::min(SQR(cth1f),cthsqmax)) ;
                                    
                                    if (ifinclined){
                                        cth2f = (zf1 * cthA + (xf1 * cos(phiA) + yf1 * sin(phiA)) * sthA) / r2f; // theta in inclined frame
                                        phi2f = std::atan2(yf1,xf1); // obviously, the expression should be different; do we need this angle?
                                    }
                                    else{
                                        cth2f = cth1f;
                                        phi2f = std::atan2(yf1,xf1);
                                    }
                                    
                                    aphi = psi_AB(r2f, cth2f, phi2f);
                                   //  std::cout << "Aphi = " << aphi << std::endl;
                                    if (ifinclined){
                                        Real sth2f = std::sqrt(1.-std::min(SQR(cth2f),cthsqmax)) ; // |\mu \times R|/R
                                        Real mx = sthA * std::cos(phiA), my = sthA * std::sin(phiA), mz = cthA;
                                        Real mrx = (my * cth1f - mz * std::sin(phi2f) * sth1f) / sth2f, mry = (mz * sth1f * std::cos(phi2f)- mx * cth1f) / sth2f, mrz = -sth1f * (my * std::cos(phi2f)-mx * std::sin(phi2f)) / sth2f;
                                        ax(k+dk,j+dj,i+di) = mrx * aphi ;
                                        ay(k+dk,j+dj,i+di) = mry * aphi ;
                                        az(k+dk,j+dj,i+di) = mrz * aphi ;
                                        // (sthA * std::sin(phiA) * cth2f - cthA * sth2f * std::sin(phi2f) ) / sth2f * aphi ;
                                        // ay(k+dk,j+dj,i+di) = (cthA * sth2f * std::cos(phi2f) - cth2f * sthA * std::sin(phiA) ) / sth2f * aphi ;
                                        // az(k+dk,j+dj,i+di) = sthA * (std::sin(phiA) * sth2f - sth2f * std::sin(phiA) * sth2f * std::cos(phi2f) ) / sth2f * aphi ;
                                    }
                                    else{
                                        ax(k+dk,j+dj,i+di) = -aphi * std::sin(phi2f) ;
                                        ay(k+dk,j+dj,i+di) = aphi * std::cos(phi2f) ;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // std::cout << "z = " << z << std::endl;
    }

    clock_t tend = clock();
    float cpu_time = (tend>tstart ? static_cast<Real>(tend-tstart) : 1.0) /
      static_cast<Real>(CLOCKS_PER_SEC);

    std::cout << "IC time = " << cpu_time << std::endl;
    std::cout << "starmass = " << starmass_empirical << std::endl;
    
    if(MAGNETIC_FIELDS_ENABLED){
      // initialize interface B
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie+1; i++) {
            if(magbeta > 0.){
              pfield->b.x1f(k,j,i) = ((az(k,j+1,i) - az(k,j,i))/pcoord->dx2f(j) -
                                      (ay(k+1,j,i) - ay(k,j,i))/pcoord->dx3f(k)) * rcsnorm;
            }
            else{
              pfield->b.x1f(k,j,i) = 0.;
            }
          }
        }
      }
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je+1; j++) {
          for (int i=is; i<=ie; i++) {
            if(magbeta > 0.){
              pfield->b.x2f(k,j,i) = ((ax(k+1,j,i) - ax(k,j,i))/pcoord->dx3f(k) -
                                      (az(k,j,i+1) - az(k,j,i))/pcoord->dx1f(i)) * rcsnorm;
            }
            else{
              pfield->b.x2f(k,j,i) = 0.;
            }
          }
        }
      }
      for (int k=ks; k<=ke+1; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            if(magbeta > 0.){
              pfield->b.x3f(k,j,i) = ((ay(k,j,i+1) - ay(k,j,i))/pcoord->dx1f(i) -
                                      (ax(k,j+1,i) - ax(k,j,i))/pcoord->dx2f(j)) * rcsnorm;
                // if (pfield->b.x3f(k,j,i) > 0.)std::cout << pfield->b.x3f(k,j,i) << std::endl;
            }
            else{
              pfield->b.x3f(k,j,i) = 0.;
            }
          }
        }
      }
    }
    
    if(MAGNETIC_FIELDS_ENABLED){
      pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, is-NGHOST, ie+NGHOST, js-NGHOST, je+NGHOST, ks-NGHOST, ke+NGHOST);
      peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie, js, je, ks, ke);
    }
    else{
      AthenaArray<Real> bb;
      bb.NewAthenaArray(3, ke+2*NGHOST+1, je+2*NGHOST+1, ie+2*NGHOST+1);
      peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, is, ie, js, je, ks, ke);
      // peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, is-NGHOST, ie+NGHOST, js-NGHOST, je+NGHOST, ks-NGHOST, ke+NGHOST);
    }

 //     peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie, js, je, ks, ke);
}

int RefinementCondition(MeshBlock *pmb)
{
    AthenaArray<Real> &w = pmb->phydro->w;
    AthenaArray<Real> &R = pmb->pscalars->r; // scalar (=1 inside the star, =0 outside)
    
    Real maxR0 = 0.0, maxeps = 0.;
    Real refden = 10. * bgdrho;
    
    for(int k=pmb->ks; k<=pmb->ke; k++) {
      // Real  z = pmb->pcoord->x3v(k);
      for(int j=pmb->js; j<=pmb->je; j++) {
	//  Real y = pmb->pcoord->x2v(j);
	for(int i=pmb->is; i<=pmb->ie; i++) {
	  // Real x = pmb->pcoord->x1v(i), dx = pmb->pcoord->dx1v(i); // , y = pmb->pcoord->x2v(j), z = pmb->pcoord->x3v(k);
	  //        Real xf = pmb->pcoord->x1f(i), yf = pmb->pcoord->x2f(j), zf = pmb->pcoord->x3f(k);
	  // Real r1 = std::sqrt(SQR(x)+SQR(y)+SQR(z)); // distance to the star centre
	  Real r = R(0, k, j, i);
	  Real den = w(IDN,k,j,i);
	  maxR0 = std::max(maxR0, R(0, k, j, i));
	  if ((r>Rthresh) && (den>refden)){
	    Real eps = std::abs(w(IDN,k,j,i+1)+w(IDN,k,j,i-1) - 2. * den) + std::abs(w(IDN,k,j+1,i)+w(IDN,k,j-1,i) - 2. * den) + std::abs(w(IDN,k+1,j,i)+w(IDN,k-1,j,i) - 2. * den);
	      //std::sqrt(SQR(w(IDN,k,j,i+1)-w(IDN,k,j,i-1))
	      //	    +SQR(w(IDN,k,j+1,i)-w(IDN,k,j-1,i))
	      //	    +SQR(w(IDN,k+1,j,i)-w(IDN,k-1,j,i)))/(den+refden);
	    maxeps = std::max(maxeps, eps / (den+refden));
	  }
	}
      }
    }
    
    if (maxeps > thresh) return 1; // refinement
    if ((maxR0 < (0.25*Rthresh))||(maxeps < (0.25*thresh))) return -1; // derefinement
    
}

Real BHgfun(Real x, Real y, Real z){
  // black hole gravity
  Real r = std::sqrt(SQR(x)+SQR(y)+SQR(z));

  if (r>=rBH){
    return BHgmax / SQR(r/rgrav-2.); // addmass/SQR(r-2.*rgrav);
  }
  else{
    return (r/rBH) / SQR(rBH/rgrav-2.) * BHgmax; // addmass/SQR(rgrav);
  }
}

Real true_anomaly(Real time){
    Real M = Mcoeff * (time - tper); // global Mcoeff is sqrt(GM/Rp^3), global tper is the time of pericenter passage
    return 2. * std::atan(2.*std::sinh(std::asinh(1.5*M)/3.));
}

void BHcoord(Real time, Real* xBH, Real* yBH, Real* zBH){
    // parabolic motion with pericenter distance rper, true anomaly nu
    Real nu = true_anomaly(time);
    Real rad = 2. * rper / (1.0+std::cos(nu));
    
    *xBH = rad * std::cos(nu);
    *yBH = rad * std::sin(nu);
    *zBH = 0.;
}

Real stitcher(Real x, Real delta){
    // function making a smooth transition from 0 at x=delta to 1 at x=1
    if(x<=delta) return 0;
    if (x>=1) return 1;
    
    Real xc = x*SQR(x), dc = delta * SQR(delta);
    
    return xc/3. - (delta+1.)/2.*SQR(x) + delta * x + dc / 6. - SQR(delta) ;
}

void BHgrav(MeshBlock *pmb, const Real time, const Real dt,
            const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
            const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
            AthenaArray<Real> &cons_scalar){
    Real gad = 5./3.; // adiabatic index (make sure it is consistent with the athinput)
    Real rg2 = SQR(rgrav);
    Real xBH, yBH, zBH;
    BHcoord(time+dt/2., &xBH, &yBH, &zBH);
    Real BH0 = BHgfun(-xBH,-yBH,-zBH);
    Real BH0x = -xBH/std::sqrt(SQR(xBH)+SQR(yBH)+SQR(zBH)) * BH0;
    Real BH0y = -yBH/std::sqrt(SQR(xBH)+SQR(yBH)+SQR(zBH)) * BH0;
    Real BH0z = -zBH/std::sqrt(SQR(xBH)+SQR(yBH)+SQR(zBH)) * BH0;

    Real trcut = 100.;
    Real dencut = trcut * bgdrho;
    
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
        Real z = pmb->pcoord->x3v(k);
        for (int j=pmb->js; j<=pmb->je; ++j) {
            Real y = pmb->pcoord->x2v(j);
            for (int i=pmb->is; i<=pmb->ie; ++i) {
                Real x = pmb->pcoord->x1v(i);
                
                Real rsqeff = std::max(SQR(x-xBH)+SQR(y-yBH)+SQR(z-zBH), rg2);
                Real reff = std::sqrt(rsqeff), fadd = BHgfun(x-xBH,y-yBH,z-zBH), den = prim(IDN,k,j,i);
                
                Real g1 = fadd * (x-xBH)/reff - BH0x; // (BHphifun(pmb->pcoord->x1v(i+1), y, z)-BHphifun(pmb->pcoord->x1v(i-1), y, z))/2.,
                Real g2 = fadd * (y-yBH)/reff - BH0y; // (BHphifun(x,pmb->pcoord->x2v(j+1), z)-BHphifun(x, pmb->pcoord->x2v(j-1), z))/2.,
                Real g3 = fadd * (z-zBH)/reff - BH0z; // (BHphifun(x,y,pmb->pcoord->x3v(k+1))-BHphifun(x, y, pmb->pcoord->x3v(k-1)))/2.;
                
                // density cut-off:
                Real s=1.;
		if (iftracer){
		  s = stitcher(trcut * prim_scalar(0,k,j,i), 0.1);
		}
		else{ 
	        s = stitcher(den/dencut, 0.5);		  
		}
		g1 *= s;
                g2 *= s;
                g3 *= s;

                cons(IM1,k,j,i) -= (g1 * dt) * den ; //dtodx1 * den * (BHphifun(pmb->pcoord->x1v(i+1), y, z)-BHphifun(pmb->pcoord->x1v(i-1), y, z))/2.;
                cons(IM2,k,j,i) -= (g2 * dt ) * den ; // dtodx2 * den * (BHphifun(x,pmb->pcoord->x2v(j+1), z)-BHphifun(x, pmb->pcoord->x2v(j-1), z))/2.;
                cons(IM3,k,j,i) -= ( g3 * dt ) * den ; // dtodx3 * den * (BHphifun(x,y,pmb->pcoord->x3v(k+1))-BHphifun(x, y, pmb->pcoord->x3v(k-1)))/2.;
                
                Real cursefactor;
                if (NON_BAROTROPIC_EOS) {
                    cons(IEN,k,j,i) -= (g1 * prim(IM1, k,j,i) + g2 * prim(IM2,k,j,i) +  g3 * prim(IM3, k,j,i)) * dt * den;
                }
		// all the low-density regions require cleaning:
		if(den < bgdrho) cons_scalar(0,k,j,i) = 0.0;
            }
        }
    }
}


namespace {
//----------------------------------------------------------------------------------------
 //! transform to cylindrical coordinate
 // from disk.cpp

Real SmoothStep(Real x)
{
  // step function approximation
  if (x>30.)return 1.;
  if (x<-30.)return 0.;
  return (tanh(x)+1.)/2. ; // x/std::sqrt(x*x+1.);
}


void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad=pco->x1v(i);
    phi=pco->x2v(j);
    z=pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad=std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi=pco->x3v(k);
    z=pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}


void GetCylCoord_f(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad=pco->x1f(i);
    phi=pco->x2f(j);
    z=pco->x3f(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad=std::abs(pco->x1f(i)*std::sin(pco->x2f(j)));
    phi=pco->x3f(k);
    z=pco->x1f(i)*std::cos(pco->x2f(j));
  }
  return;
}

Real psi_AB(Real r, Real cth, Real phi){
   if ((r< rmin) || (r>rmax)){
     return 0.;
   }
   Real pl = gsl_sf_legendre_Pl(lharm, cth), pl1 = 0.;
   if (lharm > 0) pl1 = gsl_sf_legendre_Pl(lharm-1, cth);

   return (cth * pl - pl1) * std::max((r/rmin-1.) * (rmax/r-1.), 0.) ;
 }

Real azfun(Real r, Real z){ // Atheta(r) for a toroidal loop
   Real rmid = (rmax+rmin)/2., rthick = (rmax-rmin)/2.;

   if(std::abs(z)>rthick){
     return 0.;
   }

   Real R1 = rmid - std::sqrt(std::max(SQR(rthick)-SQR(z), 0.)), R2 = rmid + std::sqrt(std::max(SQR(rthick)-SQR(z), 0.));

   if(r <= R1){
     return 0.;
   }

   if (r >= R2){
     return (R2-R1); // /(rmax-rmin);
   }
   else{
     return (r-R1); // /(rmax-rmin);
   }
 }

  void instar_interpolate(std::list<Real> instar_radius, std::list<Real> instar_lrho, std::list<Real> instar_lpress, std::list<Real> instar_mass, Real r2, Real rcore, Real *rho, Real *p){
    int nl =instar_radius.size();

    Real rho1, rho0=0., press1, press0 = 0., r0=0.0, r1, m1, mcore=-1., pcore = -1., rhocore = 0., m0 = mstar;

    std::list<Real> instar_radius1 = instar_radius;
    std::list<Real> instar_lrho1 = instar_lrho;
    std::list<Real> instar_lpress1 = instar_lpress;
    std::list<Real> instar_mass1 = instar_mass;
    
    //    std::cout << "lrho: " << instar_lrho1.front() << std::endl;
    
    if (r2>instar_radius1.front()){
      rho1 = instar_lrho1.front(); // no conversion so far
      press1 = instar_lpress1.front(); // no conversion so far
      
      *rho = rho1; // (rho1-rho0)/(r1-r0)*(r2-r0)+rho0;
      *p = press1; // (press1-press0)/(r1-r0)*(r2-r0)+press0;
      return;
    }
    
    for(int k = 0; k<nl; k++){
      rho1 = instar_lrho1.front(); // no conversion so far
      press1 = instar_lpress1.front(); // no conversion so far
      r1 = instar_radius1.front();
      //    r1 = std::sqrt(SQR(r1)+SQR(rcore));
      m1 = instar_mass1.front();
      
      // std::cout << instar_array.size() << std::endl;
      instar_lrho1.pop_front(); instar_lpress1.pop_front(); instar_radius1.pop_front(); instar_mass1.pop_front();
      if (r1<=r2){ // &&(r2 >= r1)
	//      std::cout << r0 << " > "<< r2 << " > " << r1 << std::endl;
	// std::cout << std::pow(10., (lrho1-lrho0)/(r1-r0)*(r2-r0)+lrho0) << std::endl;
	*rho = rho1 ; // (rho1-rho0)/(r1-r0)*(r2-r0)+rho0;
	*p = press1 ; // (press1-press0)/(r1-r0)*(r2-r0)+press0;
	return;
	// return std::pow(10., (lrho1-lrho0)/(r1-r0)*(r2-r0)+lrho0);
      }
      rho0 = rho1;
      press0 = press1 ;
      r0 = r1;
    }
    *rho = rho1;
    *p = press1;
    std::cerr << "interpolate: you should not be here! \n";
    return;
  }
}



// X direction
void DumbBoundaryInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh){
  // cell-centered:
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,il-i) = prim(IDN,k,j,il);
        prim(IEN,k,j,il-i) = prim(IEN,k,j,il);
        prim(IVX,k,j,il-i) = std::min(prim(IVX,k,j,il),0.);
        prim(IVY,k,j,il-i) = prim(IVY,k,j,il);
        prim(IVZ,k,j,il-i) = prim(IVZ,k,j,il);
        if(NSCALARS>0){
          for (int n=0; n<(NSCALARS); ++n)pmb->pscalars->s(n,k,j,il-i) = 0;
        }
      }
    }
  }
  // fields:
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          b.x1f(k,j,il-i) = b.x1f(k,j,il);
        }
      }
    }
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,il-i) = 0.; // b.x2f(k,jl,il);
        }
      }
    }
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,il-i) = 0.; // -b.x3f(k,jl+j,i);
        }
      }
    }
  }

}

void DumbBoundaryOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
			 FaceField &b, Real time, Real dt,
			 int il, int iu, int jl, int ju, int kl, int ku, int ngh){
    // cell-centered:
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
	for (int i=1; i<=ngh; ++i) {
	  prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu);
	  prim(IEN,k,j,iu+i) = prim(IEN,k,j,iu);
	  prim(IVX,k,j,iu+i) = std::max(prim(IVX,k,j,iu),0.);
	  prim(IVY,k,j,iu+i) = prim(IVY,k,j,iu);
	  prim(IVZ,k,j,iu+i) = prim(IVZ,k,j,iu);
	  if(NSCALARS>0){
	    for (int n=0; n<(NSCALARS); ++n)pmb->pscalars->s(n,k,j,iu+i) = 0;
	  }
	}
      }
    }
    // fields:
    if (MAGNETIC_FIELDS_ENABLED) {
      for (int k=kl; k<=ku; ++k) {
	for (int j=jl; j<=ju; ++j) {
	  for (int i=1; i<=ngh; ++i) {
	    b.x1f(k,j,iu+i+1) = b.x1f(i,j,iu+1);
	  }
	}
      }
      for (int k=kl; k<=ku; ++k) {
	for (int j=jl; j<=ju+1; ++j) {
	  for (int i=1; i<=ngh; ++i) {
	    b.x2f(k,j,iu+i) = 0.; // b.x2f(k,ju+1,i);
	  }
	}
      }
      // what if one layer is corrupted, and we need to move the BC one cell closer?
      for (int k=kl; k<=ku+1; ++k) {
	for (int j=jl; j<=ju; ++j) {
	  for (int i=1; i<=ngh; ++i) {
	    b.x3f(k,j,iu+i) = 0.; // -b.x3f(k,ju-j+1,i);
	  }
	}
      }
    }
							  }

// Y direction
void DumbBoundaryInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh){
  // cell-centered:
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        prim(IDN,k,jl-j,i) = prim(IDN,k,jl,i);
        prim(IEN,k,jl-j,i) = prim(IEN,k,jl,i);
        prim(IVX,k,jl-j,i) = prim(IVX,k,jl,i);
        prim(IVY,k,jl-j,i) = std::min(prim(IVY,k,jl,i),0.);
        prim(IVZ,k,jl-j,i) = prim(IVZ,k,jl,i);
        if(NSCALARS>0){
          for (int n=0; n<(NSCALARS); ++n)pmb->pscalars->s(n,k,jl-j,i) = 0;
        }
      }
    }
  }


  // fields:
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu+1; ++i) {
          b.x1f(k,jl-j,i) = 0.; // -b.x1f((kl+k-1),j,i);
        }
      }
    }
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          b.x2f(k,jl-j,i) = b.x2f(k,jl,i);
        }
      }
    }
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          b.x3f(k,jl-j,i) = 0.; // -b.x3f(k,jl+j,i);
        }
      }
    }
  }
}


void DumbBoundaryOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh){
  // cell-centered:
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        prim(IDN,k,ju+j,i) = prim(IDN,k,ju,i);
        prim(IEN,k,ju+j,i) = prim(IEN,k,ju,i);
        prim(IVX,k,ju+j,i) = prim(IVX,k,ju,i);
        prim(IVY,k,ju+j,i) = std::max(prim(IVY,k,ju,i),0.);
        prim(IVZ,k,ju+j,i) = prim(IVZ,k,ju,i);
        if(NSCALARS>0){
          for (int n=0; n<(NSCALARS); ++n)pmb->pscalars->s(n,k,ju+j,i) = 0;
        }
      }
    }
  }
  // fields:
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu+1; ++i) {
          b.x1f(k,ju+j,i) = 0.; // -b.x1f((ku-k+1),j,i);
        }
      }
    }
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          b.x2f(k,ju+j+1,i) = b.x2f(k,ju+1,i);
        }
      }
    }
    // what if one layer is corrupted, and we need to move the BC one cell closer?
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          b.x3f(k,ju+j,i) = 0.; // -b.x3f(k,ju-j+1,i);
        }
      }
    }
  }
}

// Z direction:

void DumbBoundaryInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh){

  // cell-centered:
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        prim(IDN,kl-k,j,i) = prim(IDN,kl,j,i);
        prim(IEN,kl-k,j,i) = prim(IEN,kl,j,i);
        prim(IVX,kl-k,j,i) = prim(IVX,kl,j,i);
        prim(IVY,kl-k,j,i) = prim(IVY,kl,j,i);
        prim(IVZ,kl-k,j,i) = std::min(prim(IVZ,kl,j,i),0.);
        if(NSCALARS>0){
          for (int n=0; n<(NSCALARS); ++n)pmb->pscalars->s(n,kl-k,j,i) = 0;
        }
      }
    }
  }

  // fields:
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu+1; ++i) {
          b.x1f((kl-k),j,i) = 0.; // -b.x1f((kl+k-1),j,i);
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
        for (int i=il; i<=iu; ++i) {
          b.x2f((kl-k),j,i) = 0.; // -b.x2f((kl+k-1),j,i);
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          b.x3f((kl-k),j,i) = b.x3f(kl,j,i);
        }
      }
    }
  }
}


void DumbBoundaryOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh){
  // cell-centered:
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        prim(IDN,ku+k,j,i) = prim(IDN,ku,j,i);
        prim(IEN,ku+k,j,i) = prim(IEN,ku,j,i);
        prim(IVX,ku+k,j,i) = prim(IVX,ku,j,i);
        prim(IVY,ku+k,j,i) = prim(IVY,ku,j,i);
        prim(IVZ,ku+k,j,i) = std::max(prim(IVZ,ku,j,i),0.);
        if(NSCALARS>0){
          for (int n=0; n<(NSCALARS); ++n)pmb->pscalars->s(n,ku+k,j,i) = 0;
        }
      }
    }
  }
  // fields:
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu+1; ++i) {
          b.x1f((ku+k  ),j,i) = 0.; // -b.x1f((ku-k+1),j,i);
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
        for (int i=il; i<=iu; ++i) {
          b.x2f((ku+k  ),j,i) = 0.; // -b.x2f((ku-k+1),j,i);
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          b.x3f((ku+k+1),j,i) = b.x3f((ku+1),j,i);
        }
      }
    }
  }
}
