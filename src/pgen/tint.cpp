//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file tint.cpp
//! \brief interpolates an HDF5 output (conservative + B) and creates a restarting file for a TDE simulation
//! based on from_array.cpp

// C headers

// C++ headers
#include <algorithm>  // max()
#include <string>     // c_str(), string
#include <sstream>   // string stream
#include <vector>   // vector arrays
#include <iomanip>

// Athena++ headers
#include "../athena.hpp"              // Real
#include "../athena_arrays.hpp"       // AthenaArray
#include "../eos/eos.hpp"
#ifdef FFT
#include "../fft/athena_fft.hpp"     // Fourier transforms
#endif
#include "../field/field.hpp"         // Field
#include <fstream>
using std::ifstream; // reading and writing files
#include "../globals.hpp"             // Globals

#include "../gravity/mg_gravity.hpp" // multigrid self-gravity

#include "../hydro/hydro.hpp"         // Hydro
#include "../inputs/hdf5_reader.hpp"  // HDF5ReadRealArray()
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"     // ParameterInput
#include "../scalars/scalars.hpp"

//----------------------------------------------------------------------------------------
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function for setting initial conditions
//!
//! Inputs:
//! - pin: parameters
//! Outputs: (none)
//! Notes:
//! - uses input parameters to determine which file contains array of conserved values
//!   dataset must be 5-dimensional array with the following sizes:
//!   - NHYDRO
//!   - total number of MeshBlocks
//!   - MeshBlock/nx3
//!   - MeshBlock/nx2
//!   - MeshBlock/nx1
//!

#if !MAGNETIC_FIELDS_ENABLED
#error "This problem generator requires magnetic fields"
#endif

#if NSCALARS < 1
#error "This problem generator requires at least one passive scalar variable"
#endif

int RefinementCondition(MeshBlock *pmb); // proper refinement condition used for the TDE problem
int RefinementCondition_Bonly(MeshBlock *pmb); // simple refinement condition using only a cut-off MF value

Real BHgfun(Real x, Real y, Real z); //  black hole gravity g as a function of distance

void BHgrav(MeshBlock *pmb, const Real time, const Real dt,
	    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
	    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
	    AthenaArray<Real> &cons_scalar); // BH gravity and more

/*
void Bdiff_scalar(MeshBlock *pmb, const Real time, const Real dt,
		  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
		  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
		  AthenaArray<Real> &cons_scalar);
void Bdiff_vec(MeshBlock *pmb, const Real time, const Real dt,
	       const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
	       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
	       AthenaArray<Real> &cons_scalar);
void Bclean(MeshBlock *pmb, const Real time, const Real dt,
	    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
	    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
	    AthenaArray<Real> &cons_scalar);
*/ // probably I do not use this, check!

// ideal conductor + outflow BCs (ensure divB = 0 at the boundaries)

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


Real MyTimeStep(MeshBlock *pmb); // not used??

namespace{
  Real tzero, tper, addmass, rper, Mcoeff ;
  Real bgdrho = 1e-8, bgdp = 1e-8, rotangle = 0., addx, addy, addz, addvx, addvy, addvz;
  Real bgdvx, bgdvy, bgdvz;
  Real massboost;

  void losslesscout(Real x);
  //  Real rBH, BHgmax, rgrav; // 
  Real int_linear(AthenaArray<Real>& u, int index, Real kest, Real jest, Real iest);
  Real int_linear(AthenaArray<Real>& u, int index1, int index2, Real kest, Real jest, Real iest); // one index more
  Real int_nearest(AthenaArray<Real>& u, int index1, int index2, Real kest, Real jest, Real iest); // 0-order interpolation 8o
  Real true_anomaly(Real t);
  // Real Bint_linearZ(AthenaArray<Real> &u, int index1, int index2, Real kf, Real j, Real i, int order);
  // Real Bint_linearY(AthenaArray<Real> &u, int index1, int index2, Real k, Real jf, Real i, int order);
  // Real Bint_linearX(AthenaArray<Real> &u, int index1, int index2, Real k, Real j, Real i_f, int order);
  void star_coord(Real time, Real* xstar, Real* ystar, Real* zstar, Real* vx, Real* vy, Real* vz);
  // void intcurl(AthenaArray<Real>& A1, AthenaArray<Real>& A2, AthenaArray<Real>& A3, Real kav, Real jav, Real iav, Real kavf, Real javf, Real iavf, Real* b1, Real* b2, Real* b3);

  Real splint(AthenaArray<Real>& A, Real kav, Real jav, Real iav, Real dk, Real dj, Real di, int splitlevel, bool verboseflag); // correct Avec interpolation
  
  Real Acurl(AthenaArray<Real>& A, Real k, Real j, Real i);
  Real x1min;
  
  bool levelpressure, ifcomoving, ifavec, ifnearest, ifevacuate, carved, ifbonly;

  int maxsplit;
}

Real thresh, Rthresh, bsqthresh;
Real BHgmax, rgrav, rBH;
bool ifXYZ, ifstitcher, fromXYZ;
Real refden, refB; 
Real bomega, omega, divBmaxLimit, divatol, tclean, Bdecayfactor ;

bool ifcentrelens;
Real rcentrecircle; 

void Mesh::InitUserMeshData(ParameterInput *pin) {
    
    bgdrho = pin->GetReal("problem","bgdrho");
    bgdp = pin->GetReal("problem","bgdp");
    bgdvx = pin->GetOrAddReal("problem","bgdvx", 0.0);
    bgdvy = pin->GetOrAddReal("problem","bgdvy", 0.0);
    bgdvz = pin->GetOrAddReal("problem","bgdvz", 0.0);

    tper = pin->GetReal("problem","tper");

    addmass = pin->GetReal("problem","mBH"); // black hole mass. may be 0
    rgrav = (addmass/1.e6) * 2.1218 ; // GM_{\rm BH}/c^2
    rBH = pin->GetOrAddReal("problem","rBH", rgrav); // makes sense to make it larger; different potential inside rBH

    rper = pin->GetReal("problem","rper");

    Mcoeff = std::sqrt(addmass / 2.) * std::pow(rper, -1.5);

    BHgmax = addmass / SQR(rgrav); // GM/R^2 at R = 3GM/c^2 

    //    star_coord(tzero, &addx, &addy, &addz, &addvx, &addvy, &addvz);

    massboost = pin->GetOrAddReal("problem","massboost", 1.0); 

    rotangle = pin->GetOrAddReal("problem","rotangle", 0.); // TODO: set this up!
    
    ifXYZ = pin->GetOrAddBoolean("problem", "ifXYZ", "true"); // to a BH frame box
    fromXYZ = pin->GetOrAddBoolean("problem", "fromXYZ", "false"); // from a BH frame box

    ifstitcher = pin->GetOrAddBoolean("problem", "ifstitcher", "false"); // turning off gravity with decreasing s
    // ifbclean = pin->GetOrAddBoolean("problem", "ifbclean", "false");
    // ifBcleandecay = pin->GetOrAddBoolean("problem", "ifBcleandecay", "false");
    ifevacuate = pin->GetOrAddBoolean("problem", "ifevacuate", "false"); // mass loss from the BH vicinity

    carved = pin->GetOrAddBoolean("problem", "ifcarved", "false"); // this is a memory-concerving trick: from the initial data arrays, only the parts that fit the new grid were used

    levelpressure = pin->GetOrAddBoolean("problem", "levelpressure", "false"); // if true, overwrites the pressure with the equilibrium pressure distribution from the background
    ifcomoving = pin->GetOrAddBoolean("problem", "ifcomoving", "false"); 

    refden = pin->GetReal("problem","refden");
    refB = pin->GetReal("problem","refB");

    ifavec = pin->GetOrAddBoolean("problem", "ifavec", "false"); // Avec: reading vector potential values for the magnetic field
    bomega = pin->GetOrAddReal("problem","bomega", 0.);
    omega = pin->GetOrAddReal("problem","omega", 1.);
    divBmaxLimit = pin->GetOrAddReal("problem","divBmaxLimit", 1.e-10);
    divatol = pin->GetOrAddReal("problem","divatol", 1.e-10);
    tclean = pin->GetOrAddReal("problem","tclean", 0.0); // initial time period when the MF is cleaned
    Bdecayfactor= pin->GetOrAddReal("problem","Bdecayfactor", 0.01); // initial time period when the MF is cleaned

    ifnearest = pin->GetOrAddBoolean("problem", "ifnearest", "false");

    ifbonly = pin->GetOrAddBoolean("problem", "ifbonly", "false");

    maxsplit =  pin->GetOrAddInteger("problem","maxsplit", 1);

    // if we are going to derefine the regions outside certain radius
    ifcentrelens = pin->GetOrAddBoolean("problem", "ifcentrelens", "false");
    // and this is the radius
    rcentrecircle = pin->GetOrAddReal("problem","rcentrecircle", 3000.);    

    // gravity:
    if(addmass > 0.0)  EnrollUserExplicitSourceFunction(BHgrav);

    if (SELF_GRAVITY_ENABLED) {
      SetFourPiG(4. * M_PI);
    }
    if (adaptive) {
      if (ifbonly){
	bsqthresh = SQR(pin->GetReal("problem", "bthresh")); // MF cut-off
	EnrollUserRefinementCondition(RefinementCondition_Bonly);
      }else{
	EnrollUserRefinementCondition(RefinementCondition);      
	thresh = pin->GetReal("problem", "thresh"); // cell-to-cell variation threshold for refinement
	Rthresh = pin->GetReal("problem", "Rthresh"); // tracer cut-off for refinement
      }
    }

    // user-defined BC (outflow for hydro, infinite conductor for B):
    std::string inner_Xboundary = pin->GetString("mesh", "ix1_bc");
    std::string outer_Xboundary = pin->GetString("mesh", "ox1_bc");
    std::string inner_Yboundary = pin->GetString("mesh", "ix2_bc");                                         
    std::string outer_Yboundary = pin->GetString("mesh", "ox2_bc");
    std::string inner_Zboundary = pin->GetString("mesh", "ix3_bc");
    std::string outer_Zboundary = pin->GetString("mesh", "ox3_bc");

    if (inner_Xboundary == "user")EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DumbBoundaryInnerX1);
    if (outer_Xboundary == "user")EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DumbBoundaryOuterX1);
    if (inner_Yboundary == "user")EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DumbBoundaryInnerX2);
    if (outer_Yboundary == "user")EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DumbBoundaryOuterX2);
    if (inner_Zboundary == "user")EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DumbBoundaryInnerX3);
    if (outer_Zboundary == "user")EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DumbBoundaryOuterX3);

    if (!ifXYZ){
      Real xBH, yBH, zBH, vxBH, vyBH, vzBH, BH0, BH0x, BH0y, BH0z;

      star_coord(0., &xBH, &yBH, &zBH, &vxBH, &vyBH, &vzBH);
      xBH = -xBH; yBH = -yBH; zBH = -zBH;
      vxBH = -vxBH; vyBH = -vyBH; vzBH = -vzBH; // do we need the velocity of the BH?
      
      BH0 = BHgfun(-xBH, -yBH, -zBH);
      BH0x = -xBH/std::sqrt(SQR(xBH)+SQR(yBH)+SQR(zBH)) * BH0;
      BH0y = -yBH/std::sqrt(SQR(xBH)+SQR(yBH)+SQR(zBH)) * BH0;
      BH0z = -zBH/std::sqrt(SQR(xBH)+SQR(yBH)+SQR(zBH)) * BH0;

      // output the position of the BH and the force components:                                                                       
      std::cout << "(x, y)BH = " << xBH << ", " << yBH << "\n";
      // std::cout << "F(x, y)BH = " <<BH0x << ", " << BH0y << "\n";
      //  exit(1);
    }

}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    
  int nvars = NHYDRO+NSCALARS; // the number of hydrodynamic variables (normally, 5) + the number of scalars (tracers, normally 1)

  Real gamma = pin->GetReal("hydro","gamma"); // adiabatic gamma
  Real dfloor = pin->GetReal("hydro","dfloor");

  // Determine locations of initial values
  std::string input_filename = pin->GetString("problem", "input_filename");
  std::string b1_input_filename = pin->GetString("problem", "B1_input_filename");
  std::string b2_input_filename = pin->GetString("problem", "B2_input_filename");
  std::string b3_input_filename = pin->GetString("problem", "B3_input_filename");
  std::string dataset_cons = pin->GetString("problem", "dataset_cons");

  // vector potential source file (HDF5, produced by Avec.py):

  std::string avecfile = pin->GetString("problem", "avecfile");
  
  // indices for particular variables read from the file
  int index_dens = pin->GetInteger("problem", "index_dens"); // density
  int index_mom1 = pin->GetInteger("problem", "index_mom1"); // X momentum  
  int index_mom2 = pin->GetInteger("problem", "index_mom2"); // Y momentum
  int index_mom3 = pin->GetInteger("problem", "index_mom3"); // Z momentum
  int index_etot = pin->GetInteger("problem", "index_etot"); // energy
  int index_r0 = pin->GetInteger("problem", "index_r0"); // tracer 0 (larger number of tracers is not supported)

  if(gid == 0)std::cout << "index_r0 = " << index_r0 << "\n"; // index of the tracer variable (may vary!)

  int index_b1 = pin->GetInteger("problem", "index_b1"); // indices for magnetic field components (if the MF is read directly!)
  int index_b2 = pin->GetInteger("problem", "index_b2");
  int index_b3 = pin->GetInteger("problem", "index_b3");
  
  std::string dataset_b1 = pin->GetString("problem", "dataset_b1"); // files for the magnetic field components (could be different)
  std::string dataset_b2 = pin->GetString("problem", "dataset_b2");
  std::string dataset_b3 = pin->GetString("problem", "dataset_b3");
  
  Real coord_range1[3], coord_range2[3], coord_range3[3];
  int coord_ncells[3], coord_blockcells[3];
    
  tzero = HDF5RealAttribute(input_filename.c_str(), "Time"); 

  if (ifXYZ&&!fromXYZ){
    star_coord(tzero, &addx, &addy, &addz, &addvx, &addvy, &addvz); // we are switching from the star frame to BH frame
  }
  else{
    addx = 0.; addy = 0.; addz = 0.; // star frame
    addvx = 0.; addvy =0.; addvz = 0.;
  }
  
  Real rstar = std::sqrt(SQR(addx)+SQR(addy)+SQR(addz));
  Real rvir = addmass * bgdrho / bgdp / gamma;
  
  //    addx = -addx ; addy = -addy ; addz = -addz; // we the the BH coordinates with respect to the star, not vice versa

  if (gid == 0){
    std::cout << "t0 = " << tzero << "\n";
    std::cout << "star coordinates: X = " << addx << "; Y = " << addy << "; Z = " << addz << "\n";
    std::cout << "star velocity: X = " << addvx << "; Y = " << addvy << "; Z = " << addvz << "\n";
    if(ifavec){
      std::cout << "avecfile = " << avecfile << "\n";  
    }
  }
  
    // reading old coord grid parameters (athena HDF files)
    HDF5TripleRealAttribute(input_filename.c_str(), "RootGridX1", coord_range1);
    HDF5TripleRealAttribute(input_filename.c_str(), "RootGridX2", coord_range2);
    HDF5TripleRealAttribute(input_filename.c_str(), "RootGridX3", coord_range3);
    // HDF5TripleIntAttribute(input_filename.c_str(), "RootGridSize", coord_ncells);
    HDF5TripleIntAttribute(input_filename.c_str(), "MeshBlockSize", coord_blockcells);
    int numblocks = HDF5IntAttribute(input_filename.c_str(), "NumMeshBlocks");
    
    // Set conserved array selections
    int start_cons_file[5];
    // [0] is the id of variable
    start_cons_file[1] = 0; // gid is core/thread ID? this is current core ID for the new run
    start_cons_file[2] = 0;
    start_cons_file[3] = 0;
    start_cons_file[4] = 0;
    // std::cout << "start cons = " << start_cons_file[0] << " " << start_cons_file[1] << " " << start_cons_file[2] << " " << start_cons_file[3]  << " " << start_cons_file[4] << "\n";
    // getchar();
    int start_cons_indices[nvars];
    start_cons_indices[IDN] = index_dens;
    start_cons_indices[IM1] = index_mom1;
    start_cons_indices[IM2] = index_mom2;
    start_cons_indices[IM3] = index_mom3;
    start_cons_indices[IEN] = index_etot;
    
    if (NSCALARS>0)start_cons_indices[NHYDRO] = index_r0;

    int count_cons_file[5];
    count_cons_file[0] = 1;
    count_cons_file[1] = 1; // now it is a fake dimension (numblocks)
    count_cons_file[2] = coord_blockcells[2]; // block_size.nx3;
    count_cons_file[3] = coord_blockcells[1]; // block_size.nx2;
    count_cons_file[4] = coord_blockcells[0]; // block_size.nx1;
    
    int start_cons_mem[5];
    // [0] is the id
    // [1] is block
    start_cons_mem[0] = 0;
    start_cons_mem[1] = 0;
    

    start_cons_mem[2] = 0; // should it be -NGHOST or something?
    start_cons_mem[3] = 0;
    start_cons_mem[4] = 0;

    int count_cons_mem[5];
    count_cons_mem[0] = 1;
    count_cons_mem[1] = 1; // numblocks;
    count_cons_mem[2] = coord_blockcells[2]; // block_size.nx3;
    count_cons_mem[3] = coord_blockcells[1]; // block_size.nx2;
    count_cons_mem[4] = coord_blockcells[0]; // block_size.nx1;
        
    // parameters of the coordinate arrays:
    int start_x_file[2], start_y_file[2], start_z_file[2];
    start_x_file[1] = 0; start_x_file[0] = 0;
    start_y_file[1] = 0; start_y_file[0] = 0;
    start_z_file[1] = 0; start_z_file[0] = 0;
    int count_x_file[2], count_y_file[2], count_z_file[2];
    count_x_file[0] = numblocks; count_x_file[1] = coord_blockcells[2];
    count_y_file[0] = numblocks; count_y_file[1] = coord_blockcells[1];
    count_z_file[0] = numblocks; count_z_file[1] = coord_blockcells[0];
    int start_x_mem[2], start_y_mem[2], start_z_mem[2];
    start_x_mem[0] = 0;  start_x_mem[1] = 0;
    start_y_mem[0] = 0;  start_y_mem[1] = 0;
    start_z_mem[0] = 0;  start_z_mem[1] = 0;
    int count_x_mem[2], count_y_mem[2], count_z_mem[2];
    count_x_mem[0] = numblocks; count_x_mem[1] = coord_blockcells[2];
    count_y_mem[0] = numblocks; count_y_mem[1] = coord_blockcells[1];
    count_z_mem[0] = numblocks; count_z_mem[1] = coord_blockcells[0];
    
    // coordinate arrays (old athena HDF)
    AthenaArray<Real> x1v_old;
    x1v_old.NewAthenaArray(numblocks, coord_blockcells[2]);
    AthenaArray<Real> x2v_old;
    x2v_old.NewAthenaArray(numblocks, coord_blockcells[1]);
    AthenaArray<Real> x3v_old;
    x3v_old.NewAthenaArray(numblocks, coord_blockcells[0]);
    HDF5ReadRealArray(input_filename.c_str(), "x1v", 2, start_x_file,
                      count_x_file, 2, start_x_mem,
                      count_x_mem, x1v_old, true); // this sets the old mesh
    HDF5ReadRealArray(input_filename.c_str(), "x2v", 2, start_y_file,
                      count_y_file, 2, start_y_mem,
                      count_y_mem, x2v_old, true); // this sets the old mesh
    HDF5ReadRealArray(input_filename.c_str(), "x3v", 2, start_z_file,
                      count_z_file, 2, start_z_mem,
                      count_z_mem, x3v_old, true); // this sets the old mesh
    
    // coordinate arrays: faces
    AthenaArray<Real> x1f_old;
    x1f_old.NewAthenaArray(numblocks, coord_blockcells[2]+1);
    AthenaArray<Real> x2f_old;
    x2f_old.NewAthenaArray(numblocks, coord_blockcells[1]+1);
    AthenaArray<Real> x3f_old;
    x3f_old.NewAthenaArray(numblocks, coord_blockcells[0]+1);
    count_x_file[1] = coord_blockcells[2]+1;
    count_x_mem[1] = coord_blockcells[2]+1;
    count_y_file[1] = coord_blockcells[1]+1;
    count_y_mem[1] = coord_blockcells[1]+1;
    count_z_file[1] = coord_blockcells[0]+1;
    count_z_mem[1] = coord_blockcells[0]+1;

    HDF5ReadRealArray(input_filename.c_str(), "x1f", 2, start_x_file,
                      count_x_file, 2, start_x_mem,
                      count_x_mem, x1f_old, true); // this sets the old mesh
    HDF5ReadRealArray(input_filename.c_str(), "x2f", 2, start_y_file,
                      count_y_file, 2, start_y_mem,
                      count_y_mem, x2f_old, true); // this sets the old mesh
    HDF5ReadRealArray(input_filename.c_str(), "x3f", 2, start_z_file,
                      count_z_file, 2, start_z_mem,
                      count_z_mem, x3f_old, true); // this sets the old mesh
   
    AthenaArray<int> npoints; 
    AthenaArray<Real> mindx;
    npoints.NewAthenaArray(block_size.nx3+2*NGHOST, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST); // number of points in the new block
    mindx.NewAthenaArray(block_size.nx3+2*NGHOST, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST); // maximal resolution in the particular point                                      
    
    //getchar();
    
    //    AthenaArray<Real> MF_old; // 1, MF_old2, MF_old3; // magnetic fields (face-centered)
    int start_MF_file[5];
    int count_MF_file[5];
    int start_MF_mem[5];
    int count_MF_mem[5];
        
    if(MAGNETIC_FIELDS_ENABLED){        
        start_MF_file[0] = index_b1;  start_MF_mem[0] = 0;
        start_MF_file[1] = 0;  start_MF_mem[1] = 0;
        
        count_MF_file[0] = 1;  count_MF_mem[0] = 1;
        count_MF_file[1] = 1;  count_MF_mem[1] = 1;
        
        start_MF_file[2] = 0;
        start_MF_file[3] = 0;
        start_MF_file[4] = 0;
        
	start_MF_mem[2] = 0;
	start_MF_mem[3] = 0;
	start_MF_mem[4] = 0;
                
        count_MF_file[2] = coord_blockcells[2];
        count_MF_file[3] = coord_blockcells[1];
        count_MF_file[4] = coord_blockcells[0];
        count_MF_mem[2] = coord_blockcells[2];
        count_MF_mem[3] = coord_blockcells[1];
        count_MF_mem[4] = coord_blockcells[0];
	// setting the fields to 0      
        for (int k = ks-NGHOST; k <= ke+NGHOST; k++) {
          for (int j = js-NGHOST; j <= je+NGHOST; j++) {
            for (int i = is-NGHOST; i <= ie+NGHOST; i++) {
              pfield->b.x1f(k,j,i) = pfield->b.x2f(k,j,i) = pfield->b.x3f(k,j,i) = 0. ;
	      pfield->b.x1f(k,j,i+1) = pfield->b.x2f(k,j+1,i) = pfield->b.x3f(k+1,j,i) = 0.;
            }
          }
        }
    }
    
    // setting all the HD variables to 0
    for (int k = ks-NGHOST; k < ke+NGHOST; k++) {
      for (int j = js-NGHOST; j < je+NGHOST; j++) {
	for (int i = is-NGHOST; i < ie+NGHOST; i++) {
	  phydro->u(IDN, k, j, i) = 0.;
	  phydro->u(IM1, k, j, i) = phydro->u(IM2, k, j, i) = phydro->u(IM3, k, j, i) = phydro->u(IEN, k, j, i) = 0.;
	  if(NSCALARS>0){              
	    for (int n=0; n<NSCALARS;++n)pscalars->r(n,k,j,i) = pscalars->s(n,k,j,i) = 0.;
	  }
	  npoints(k,j,i) = 0; mindx(k,j,i) = 0.;
	}
      }
    }

    AthenaArray<Real> u_old, MF_old;
    u_old.NewAthenaArray(nvars, 1, coord_blockcells[2], coord_blockcells[1], coord_blockcells[0]);
    if(MAGNETIC_FIELDS_ENABLED && !ifavec)MF_old.NewAthenaArray(3, 1, coord_blockcells[2]+1, coord_blockcells[1]+1, coord_blockcells[0]+1); // relevant if MFs are read from 
    AthenaArray<Real> x1a, x2a, x3a, avec1, avec2, avec3; // ifavec case: old face-centered grid and vector potential components
    int nxa, nya, nza; // Nx, Ny, Nz for the vector potential grid

    Real rbox = 0.; // this variable will be used to store the total size of the old mesh. Probably, there is a variable for that... 

    bool blockbint = false;  // if we have intersections between the old and the new blocks

    int startx=0, starty=0, startz=0, endx=0, endy=0, endz=0; 
    
    // reading the vector potential
    if (ifavec){
      nxa = HDF5IntAttribute(avecfile.c_str(), "nx");
      nya = HDF5IntAttribute(avecfile.c_str(), "ny");
      nza = HDF5IntAttribute(avecfile.c_str(), "nz");

      // if (gid==0){
      //	std::cout << "Avec Nx, Ny, Nz = " << nxa << ", " << nya << ", " << nza << "\n";
      //}

      x1a.NewAthenaArray(nxa);  x2a.NewAthenaArray(nya); x3a.NewAthenaArray(nza);

      int start_xa_file[1], start_ya_file[1], start_za_file[1];
      start_xa_file[0] = start_ya_file[0] = start_za_file[0] = 0;
      int start_xa_mem[1], start_ya_mem[1], start_za_mem[1];
      start_xa_mem[0] = start_ya_mem[0] = start_za_mem[0] = 0;

      int count_xa_file[1], count_ya_file[1], count_za_file[1];
      count_xa_file[0] = nxa ;    count_ya_file[0] = nya ;     count_za_file[0] = nza;
      int count_xa_mem[1], count_ya_mem[1], count_za_mem[1];
      count_xa_mem[0] = nxa ;    count_ya_mem[0] = nya ;     count_za_mem[0] = nza;

      // reading the coordinate arrays:
      HDF5ReadRealArray(avecfile.c_str(), "X", 1, start_xa_file,
			count_xa_file, 1, start_xa_mem,
			count_xa_mem, x1a, true); // this sets the old mesh                                                                                                 
      HDF5ReadRealArray(avecfile.c_str(), "Y", 1, start_ya_file,
			count_ya_file, 1, start_ya_mem,
			count_ya_mem, x2a, true); // this sets the old mesh                                                                                                 
      HDF5ReadRealArray(avecfile.c_str(), "Z", 1, start_za_file,
			count_za_file, 1, start_za_mem,
			count_za_mem, x3a, true); // this sets the old mesh                                                                                                 
      blockbint = (x1a(0) <= pcoord->x1f(ie+NGHOST)) && (x1a(nxa-1) >= pcoord->x1f(is-NGHOST)) &&
	(x2a(0) <= pcoord->x2f(je+NGHOST)) && (x2a(nya-1) >= pcoord->x2f(js-NGHOST)) &&
	(x3a(0) <= pcoord->x3f(ke+NGHOST)) && (x3a(nza-1) >= pcoord->x3f(ks-NGHOST)) ;
      
      // std::cout << "X = " << x1a(0) << ".." << x1a(nxa-1) << "\n";
      // std::cout << "Y = " << x2a(0) << ".." << x2a(nya-1) << "\n";
      // std::cout << "Z = " << x3a(0) << ".." << x3a(nza-1) << "\n";
      // we can try to read only a part of the block, based on the relative position of the grid
      if (blockbint){

	int avecbuffer = 3;
	
	startx = (pcoord->x1f(is-NGHOST) - x1a(0)) / (x1a(1)-x1a(0)) -avecbuffer;
	endx =  (pcoord->x1f(ie+NGHOST) - x1a(0)) / (x1a(1)-x1a(0)) + avecbuffer;
	starty = (pcoord->x2f(js-NGHOST) - x2a(0)) / (x2a(1)-x2a(0)) - avecbuffer ;
	endy =  (pcoord->x2f(je+NGHOST) - x2a(0)) / (x2a(1)-x2a(0)) + avecbuffer;
	startz = (pcoord->x3f(ks-NGHOST) - x3a(0)) / (x3a(1)-x3a(0)) - avecbuffer ;
	endz =  (pcoord->x3f(ke+NGHOST) - x3a(0)) / (x3a(1)-x3a(0)) + avecbuffer;
        
	startx = std::max(startx,0); 	endx = std::min(endx,nxa);
	starty = std::max(starty,0); 	endy = std::min(endy,nya);
	startz = std::max(startz,0); 	endz = std::min(endz,nza);
	
	std::cout << "startx = " << startx << "; endx = " << endx << "\n";
	std::cout << "starty = " << starty << "; endy = " << endy << "\n";
	std::cout << "startz = " << startz << "; endz = " << endz << "\n";


	// getchar();
	
	// reading the avec file
	int start_file_a1[3], count_file_a1[3];
	int start_file_a2[3], count_file_a2[3];
	int start_file_a3[3], count_file_a3[3];
	int start_mem_a1[3], count_mem_a1[3];
	int start_mem_a2[3], count_mem_a2[3];
	int start_mem_a3[3], count_mem_a3[3];

	if (!carved){
	  startx = starty = startz = 0;
	  endx = nxa ; endy = nya ; endz = nza;
	}
	
	//	if (carved){
	// because we carved a fraction of the original array
	start_file_a1[0] = start_file_a2[0] = start_file_a3[0] = startz;
	start_file_a1[1] = start_file_a2[1] = start_file_a3[1] = starty;
	start_file_a1[2] = start_file_a2[2] = start_file_a3[2] = startx;	  
	count_file_a1[0] = count_file_a2[0] = count_file_a3[0] = endz - startz; 
	count_file_a1[1] = count_file_a2[1] = count_file_a3[1] = endy - starty; 
	count_file_a1[2] = count_file_a2[2] = count_file_a3[2] = endx - startx; 
	
	start_mem_a1[0] =  start_mem_a1[1] = start_mem_a1[2] = 0;
	start_mem_a2[0] =  start_mem_a2[1] = start_mem_a2[2] = 0;
	start_mem_a3[0] =  start_mem_a3[1] = start_mem_a3[2] = 0;
	count_mem_a1[0] = count_mem_a2[0] = count_mem_a3[0] = endz - startz; 
	count_mem_a1[1] = count_mem_a2[1] = count_mem_a3[1] = endy - starty; 
	count_mem_a1[2] = count_mem_a2[2] = count_mem_a3[2] = endx - startx; 
	
	std::cout << "gid = " << gid << ": reading (" << endx - startx << " X "  << endy - starty << " X "  << endz - startz << ") instead of  ("  << nxa << " X "  << nya << " X "  << nza << ") \n" ;
		  
	avec1.NewAthenaArray(count_mem_a1[0], count_mem_a1[1], count_mem_a1[2]); // A_x 
	avec2.NewAthenaArray(count_mem_a2[0], count_mem_a2[1], count_mem_a2[2]); // A_y
	avec3.NewAthenaArray(count_mem_a3[0], count_mem_a3[1], count_mem_a3[2]); // A_z
        
	if(gid==0)std::cout << "reading Avec\n";

	// reading vector potentials
	HDF5ReadRealArray(avecfile.c_str(), "A1", 3, start_file_a1,
			  count_file_a1, 3, start_mem_a1,count_mem_a1, avec1, true); 
      
	std::cout << "set 1 read" << "\n";
	//      getchar();
        
	HDF5ReadRealArray(avecfile.c_str(), "A2", 3, start_file_a2,
			  count_file_a2, 3, start_mem_a2,
			  count_mem_a2, avec2, true); 
      
	std::cout << "set 2 read" << "\n";
	
	HDF5ReadRealArray(avecfile.c_str(), "A3", 3, start_file_a3,
			  count_file_a3, 3, start_mem_a3,
			  count_mem_a3, avec3, true);
        
	std::cout << "set 3 read" << "\n";
	
	Real amax = 0.;
	
	int ancientghosts = 2;
        
	for (int i=0; i<(endx-startx); i++){
	  for (int j=0; j<(endy-starty); j++){
	    for (int k=0; k<(endz-startz); k++){
	      if (std::abs(avec1(k,j,i))>100.){
		std::cout << "kji = " << k << ", " << j << ", " << i << ": A1 = " << avec1(k,j,i) << "\n";
		std::cout << "kji = " << k << ", " << j << ", " << i << ": A1 = " << avec2(k,j,i) << "\n";
		std::cout << "kji = " << k << ", " << j << ", " << i << ": A1 = " << avec3(k,j,i) << "\n";
		getchar();
	      }
	      amax = std::max(amax, std::abs(avec1(k,j,i)));
	      amax = std::max(amax, std::abs(avec2(k,j,i)));
	      amax = std::max(amax, std::abs(avec3(k,j,i)));
	      int corner = std::max(ancientghosts-i, i-(nxa-1-ancientghosts))+std::max(ancientghosts-j, j-(nya-1-ancientghosts))+std::max(ancientghosts-k, k-(nza-1-ancientghosts));
	      if (corner>=0) avec1(k,j,i) = avec2(k,j,i) =  avec3(k,j,i) = 0.;

	      //if (((i<ancientghosts)||(i>=(nxa-1-ancientghosts))) || ((j<ancientghosts)||(j>=(nya-1-ancientghosts))) || ((k<ancientghosts)||(k>=(nza-1-ancientghosts)))){
	      //	      avec1(k,j,i) = avec2(k,j,i) =  avec3(k,j,i) = 0.;
	      //}
	      
	      // if (i>=(nxa-2-ancientghosts)) avec2(k,j,i) =  avec3(k,j,i) = 0.;
	      // if (j>=(nya-2-ancientghosts)) avec1(k,j,i) =  avec3(k,j,i) = 0.;
	      // if (k>=(nza-2-ancientghosts)) avec2(k,j,i) =  avec1(k,j,i) = 0.;
	      
	    }
	  }
	}
	
      // avecstream1.close() ;   avecstream2.close() ;   avecstream3.close() ;
      // std::cout << "read all the vector potentials\n";
	std::cout << "gid = " << gid << "; amax = " << amax << "\n";
      }else{
	std::cout << "no intersections with avec, magnetic fields = 0\n";
      }	
    }
    
    
    // loop over all the original meshblocks
    for (int kb = 0; kb < numblocks; ++kb){
      Real xold_min = x1v_old(kb, 0), xold_max = x1v_old(kb, coord_blockcells[2]-1);
      Real yold_min = x2v_old(kb, 0), yold_max = x2v_old(kb, coord_blockcells[1]-1);
      Real zold_min = x3v_old(kb, 0), zold_max = x3v_old(kb, coord_blockcells[0]-1);

      rbox = std::max(xold_max,rbox); 
      rbox = std::max(yold_max,rbox);
      rbox = std::max(zold_max,rbox);

      if ((xold_min <= (pcoord->x1v(ie)-addx)) && (xold_max >= (pcoord->x1v(is)-addx)) &&                                                                         
          (yold_min <= (pcoord->x2v(je)-addy)) && (yold_max >= (pcoord->x2v(js)-addy)) &&                                                                       
	  (zold_min <= (pcoord->x3v(ke)-addz)) && (zold_max >= (pcoord->x3v(ks)-addz))) {
	start_cons_file[1] = kb; // start_cons_mem[1] = 0;
	//	u_old.NewAthenaArray(nvars, 1, coord_blockcells[2], coord_blockcells[1], coord_blockcells[0]);
	
	for (int n = 0; n < nvars; ++n) {
	  start_cons_file[0] = start_cons_indices[n];
	  start_cons_mem[0] = n;
	  HDF5ReadRealArray(input_filename.c_str(), dataset_cons.c_str(), 5, start_cons_file,
			    count_cons_file, 5, start_cons_mem, count_cons_mem, u_old, true); // this sets the variables on the old mesh                          
	}
	std::cout << "core " << gid << " read cons from kb = " << kb << "\n";
	if(MAGNETIC_FIELDS_ENABLED && !ifavec){
	  // MF_old.NewAthenaArray(3, 1, coord_blockcells[2]+1, coord_blockcells[1]+1, coord_blockcells[0]+1);
	  
	  start_MF_file[1] = kb; start_MF_mem[1] = 0;
	  start_MF_file[0] = index_b1;  start_MF_mem[0] = 0;
	  HDF5ReadRealArray(b1_input_filename.c_str(), dataset_b1.c_str(), 5, start_MF_file,
			    count_MF_file, 5, start_MF_mem, count_MF_mem, MF_old, true); // this sets the variables on the old mesh                                                  
	  start_MF_file[0] = index_b2;  start_MF_mem[0] = 1;
	  // std::cout << "reading MF "<< dataset_b2.c_str() << " \n";
	  HDF5ReadRealArray(b2_input_filename.c_str(), dataset_b2.c_str(), 5, start_MF_file,
			    count_MF_file, 5, start_MF_mem, count_MF_mem, MF_old, true); // this sets the variables on the old mesh                                                   
	  start_MF_file[0] = index_b3; start_MF_mem[0] = 2;
	  // std::cout << "reading MF "<< dataset_b3.c_str()<< " \n";
	  HDF5ReadRealArray(b3_input_filename.c_str(), dataset_b3.c_str(), 5, start_MF_file,
			    count_MF_file, 5, start_MF_mem, count_MF_mem, MF_old, true); // this sets the variables on the old mesh                                                         
	  std::cerr << "read MF from kb = " << kb << "\n";
	}

      //Real xold_min = x1v_old(kb, 0), xold_max = x1v_old(kb, coord_blockcells[2]-1);
      //Real yold_min = x2v_old(kb, 0), yold_max = x2v_old(kb, coord_blockcells[1]-1);
      //Real zold_min = x3v_old(kb, 0), zold_max = x3v_old(kb, coord_blockcells[0]-1);
	Real xold_fmin = x1f_old(kb, 0), xold_fmax = x1f_old(kb, coord_blockcells[2]);
	Real yold_fmin = x2f_old(kb, 0), yold_fmax = x2f_old(kb, coord_blockcells[1]);
	Real zold_fmin = x3f_old(kb, 0), zold_fmax = x3f_old(kb, coord_blockcells[0]);
	Real dx_old = x1v_old(kb,1)-x1v_old(kb,0); //(xold_max-xold_min) / ((double)coord_blockcells[2]);
	Real dy_old = x2v_old(kb,1)-x2v_old(kb,0); // (yold_max-yold_min) / ((double)coord_blockcells[1]);
	Real dz_old = x3v_old(kb,1)-x3v_old(kb,0); //(zold_max-zold_min) / ((double)coord_blockcells[0]);
	// getchar();
	for (int k = ks-NGHOST; k < ke+NGHOST; k++) {
	  for (int j = js-NGHOST; j < je+NGHOST; j++) {
	    for (int i = is-NGHOST; i < ie+NGHOST; i++) {
	      Real x = pcoord->x1v(i)-addx, y = pcoord->x2v(j)-addy, z = pcoord->x3v(k)-addz;
	      Real xf = pcoord->x1f(i)-addx, yf = pcoord->x2f(j)-addy, zf = pcoord->x3f(k)-addz;
	      Real dx = pcoord->dx1v(i), dy = pcoord->dx2v(j), dz = pcoord->dx3v(k);
              Real dxf = pcoord->dx1f(i), dyf = pcoord->dx2f(j), dzf = pcoord->dx3f(k);

	      Real x1 = x * std::cos(rotangle) + y * std::sin(rotangle), y1 = y * std::cos(rotangle) - x * std::sin(rotangle);
	      Real xf1 = xf * std::cos(rotangle) + yf * std::sin(rotangle), yf1 = yf * std::cos(rotangle) - xf * std::sin(rotangle);
	      
	      Real kold = (z-zold_min)/dz_old, jold = (y1-yold_min)/dy_old, iold = (x1-xold_min)/dx_old;
	      Real koldf = (zf-zold_fmin)/dz_old, joldf = (yf1-yold_fmin)/dy_old, ioldf = (xf1-xold_fmin)/dx_old;
	      
	      Real dd, m1, m2, m3, ee;
	      
	      int bufferold = 2;
	      
	      if ((kold >= (double)bufferold)&&(kold <= (double)(coord_blockcells[0]-bufferold))&&
		  (jold >= (double)bufferold)&&(jold <= (double)(coord_blockcells[1]-bufferold))&&
		  (iold >= (double)bufferold)&&(iold <= (double)(coord_blockcells[2]-bufferold))){ // condition that iold, jold, and kold are within the initial box

		// looks silly but maybe this would work:
		// bufferold = NGHOST; // 

		//iold = std::max(iold, (double)bufferold); jold =std::max(jold, (double)bufferold); kold = std::max(kold, (double)bufferold);
		//iold = std::min(iold, (double)(coord_blockcells[2]-bufferold)); jold =std::min(jold, (double)(coord_blockcells[1]-bufferold)); kold =std::min(kold, (double)(coord_blockcells[0]-bufferold));
		
		dd = int_linear(u_old, IDN, 0, kold, jold, iold);
		m1 = int_linear(u_old, IM1, 0, kold, jold, iold);
		m2 = int_linear(u_old, IM2, 0, kold, jold, iold);
		m3 = int_linear(u_old, IM3, 0, kold, jold, iold);
		ee = int_linear(u_old, IEN, 0, kold, jold, iold);

		//if((x1>(xold_fmin-dx)) && (x1 < (xold_fmax+dx)) && (y1>(yold_fmin-dy)) && (y1 < (yold_fmax+dy)) &&(z>(zold_fmin-dz)) && (z < (zold_fmax+dz))){
		// if(dd>dfloor){
		// std::cout << "iold = " << iold << "; jold = " << jold << "; kold = " << kold << "\n" ;
		
		phydro->u(IDN, k, j, i) += std::max(dd, 0.);
		
		// u_old(IDN, kb, kold, jold, iold);
		//if (NON_BAROTROPIC_EOS) {
		phydro->u(IEN, k, j, i) += ee + dd * (SQR(addvx) + SQR(addvy) + SQR(addvz))/2. + m1 * addvx + m2 * addvy + m3 * addvz;
                   
		// u_old(IEN, kb, kold, jold, iold)
		// + u_old(IDN, kb, kold, jold, iold) * (SQR(addvx) + SQR(addvy)+SQR(addvz))/2.;
		//}
		phydro->u(IM1, k, j, i) += m1 + dd * addvx;
		// u_old(IM1, kb, kold, jold, iold) + u_old(IDN, kb, kold, jold, iold) * addvx;
		phydro->u(IM2, k, j, i) += m2 + dd * addvy;
		// u_old(IM2, kb, kold, jold, iold) + u_old(IDN, kb, kold, jold, iold) * addvy;
		phydro->u(IM3, k, j, i) += m3 + dd * addvz;
		
		if(NSCALARS>0){
		  for (int n=0; n<NSCALARS;++n)pscalars->s(n,k,j,i) += int_linear(u_old, NHYDRO+n, 0, kold, jold, iold);
// if (pscalars->s(0,k,j,i) > 1.0) std::cout << "r = " <<  pscalars->s(0,k,j,i) << "\n";
		}

		if(MAGNETIC_FIELDS_ENABLED){
		  bufferold=0;
		  if(!ifavec){
		   // ioldf = std::max(ioldf, (double)bufferold); joldf =std::max(joldf, (double)bufferold); koldf = std::max(koldf, (double)bufferold);
		   // ioldf = std::min(ioldf, (double)(coord_blockcells[2]-bufferold)); joldf = std::min(joldf, (double)(coord_blockcells[1]-bufferold)); koldf = std::min(koldf, (double)(coord_blockcells[0]-bufferold));

		    if ((koldf >= (double)bufferold)&&(koldf <= (double)(coord_blockcells[0]-bufferold))&&
			(joldf >= (double)bufferold)&&(joldf <= (double)(coord_blockcells[1]-bufferold))&&
			(ioldf >= (double)bufferold)&&(ioldf <= (double)(coord_blockcells[2]-bufferold))&&
		   	((mindx(k,j,i)<=0.)||(dx_old < mindx(k,j,i)))){ // choosing the finest grid to interpolate

		      if (ifnearest){
			pfield->b.x3f(k,j,i) = int_nearest(MF_old, 0, 0, koldf, jold, iold);
			pfield->b.x2f(k,j,i) = int_nearest(MF_old, 1, 0, kold, joldf, iold);
			pfield->b.x1f(k,j,i) = int_nearest(MF_old, 2, 0, kold, jold, ioldf);
		      }else{
			pfield->b.x3f(k,j,i) = int_linear(MF_old, 0, 0, kold, jold, ioldf);
			pfield->b.x2f(k,j,i) = int_linear(MF_old, 1, 0, kold, joldf, iold);
			pfield->b.x1f(k,j,i) = int_linear(MF_old, 2, 0, koldf, jold, iold);
		      }
		      if(std::isnan(pfield->b.x1f(k,j,i))==true){
			std::cout << "kold = " << kold << "; jold = " << jold << "; " << ioldf << "\n";
			std::cout << "Bx = " << pfield->b.x1f(k,j,i) << "\n";
			// getchar();
		      }
		      mindx(k,j,i) = dx_old;
		    }
		  }
		}
              npoints(k,j,i) ++;
	      }
	    }
	  }
	}
      }
    }
    //	u_old.DeleteAthenaArray();
    std::cout << "read all hydro \n";
    //if(MAGNETIC_FIELDS_ENABLED) MF_old.DeleteAthenaArray();

    if (MAGNETIC_FIELDS_ENABLED&&ifavec&&blockbint){

      Real iav, jav, kav, iavf, javf, kavf, iavf1, javf1, kavf1;
      Real btmp1, btmp2, btmp3;
      
      Real dxf = pcoord->dx1f(0), dyf = pcoord->dx2f(0), dzf = pcoord->dx3f(0);
      Real dxold = (x1a(1)-x1a(0)), dyold = x2a(1)-x2a(0), dzold = x3a(1)-x3a(0);
      
      std::cout << " new dxyz = "<< dxf << ", " << dyf << ", " << dzf << "\n";
      std::cout << " old dxyz = "<< dxold << ", "<< dyold << ", " << dzold << "\n";

      int numlevel = pin->GetInteger("mesh","numlevel");

      int splitlevel = (int)std::round(std::log(dxf/dxold)/std::log(2.))+maxsplit;
      splitlevel = pmy_mesh->root_level - loc.level + numlevel;
      // splitlevel = 0; //!!! temporary
      std::cout << "refinement level = " << loc.level - pmy_mesh->root_level << "; " ; 
      std::cout << "split level = " << splitlevel << "\n";
      // (refinement level) + (split level)  = constant

      Real ax00, ax01, ax10, ay00, ay01, ay10, az00, az01, az10;

      int ghostbuffer = 1;
      for (int k = ks-ghostbuffer; k <= ke+ghostbuffer; k++) {
	for (int j = js-ghostbuffer; j <= je+ghostbuffer; j++) {
	  for (int i = is-ghostbuffer; i <= ie+ghostbuffer; i++) {
	    Real x = pcoord->x1v(i)-addx, y = pcoord->x2v(j)-addy, z = pcoord->x3v(k)-addz;
	    Real xf = pcoord->x1f(i)-addx, yf = pcoord->x2f(j)-addy, zf = pcoord->x3f(k)-addz;
	    Real x1 = x * std::cos(rotangle) + y * std::sin(rotangle), y1 = y * std::cos(rotangle) - x * std::sin(rotangle);
	    Real xf1 = xf * std::cos(rotangle) + yf * std::sin(rotangle), yf1 = yf * std::cos(rotangle) - xf * std::sin(rotangle);	    
	    Real xf1shift = (xf+dxf) * std::cos(rotangle) + (yf+dyf) * std::sin(rotangle),
	      yf1shift = (yf+dyf) * std::cos(rotangle) - (xf+dxf) * std::sin(rotangle);

	    iavf = (xf1-x1a(startx)) / dxold; javf = (yf1-x2a(starty)) / dyold ;  kavf = (zf-x3a(startz)) / dzold;
	    iavf1 = (xf1shift-x1a(startx)) / dxold ; javf1 = (yf1shift-x2a(starty)) / dyold ;  kavf1 = (zf+dzf-x3a(startz)) / dzold;

	    iavf = std::max(iavf, (double)0); iavf = std::min(iavf, (double)(endx-startx-2));
	    iavf1 = std::max(iavf1, (double)0); iavf1 = std::min(iavf1, (double)(endx-startx-2));
	    javf = std::max(javf, (double)0); javf = std::min(javf, (double)(endy-starty-2));
	    javf1 = std::max(javf1, (double)0); javf1 = std::min(javf1, (double)(endy-starty-2));
	    kavf = std::max(kavf, (double)0); kavf = std::min(kavf, (double)(endz-startz-2));
	    kavf1 = std::max(kavf1, (double)0); kavf1 = std::min(kavf1, (double)(endz-startz-2)); 	    

	    // iavf1 = iavf + dxf/dxold ; javf1 = javf + dyf/dyold ; kavf1 = kavf + dzf/dzold ;
	    
	    // bool vflag = (std::abs(xf1-xref) < (dxf*1e-5)) && (std::abs(yf1-yref)<(dyf*1e-5)) && (std::abs(zf-zref)<(dzf*1e-5)); //  (gid == 58) && (i == 2) ; // & (j == 2) & (k == 2);
	    // vflag = false;
	    // if ((std::abs(xf1-xref) < (dxf*1e-5)) && (std::abs(yf1-yref-dyf)<(dyf*1e-5)) && (std::abs(zf-zref-dzf)<(dzf*1e-5))) vflag = true;
	    //   if ((std::abs(xf1-xref) < (dxf*1e-5)) && (std::abs(yf1-yref)<(dyf*1e-5)) && (std::abs(zf-zref-dzf)<(dzf*1e-5))) vflag = true;
	      // if ((std::abs(xf1-xref) < (dxf*1e-5)) && (std::abs(yf1-yref-dyf)<(dyf*1e-5)) && (std::abs(zf-zref)<(dzf*1e-5))) vflag = true;

	    // vector potentials integrated over the edges of the cell:
	    az00 = splint(avec3, kavf, javf, iavf, kavf1-kavf, 0., 0., splitlevel, false); 
	    az10 = splint(avec3, kavf, javf1, iavf, kavf1-kavf, 0., 0., splitlevel, false); 
	    az01 = splint(avec3, kavf, javf, iavf1, kavf1-kavf, 0., 0., splitlevel, false); 
	    ay00 = splint(avec2, kavf, javf, iavf, 0., javf1-javf, 0., splitlevel, false); 
	    ay10 = splint(avec2, kavf1, javf, iavf, 0., javf1-javf, 0., splitlevel, false); 
	    ay01 = splint(avec2, kavf, javf, iavf1, 0., javf1-javf, 0., splitlevel, false); 
	    ax00 = splint(avec1, kavf, javf, iavf, 0., 0., iavf1-iavf, splitlevel, false); 
	    ax10 = splint(avec1, kavf1, javf, iavf, 0., 0., iavf1-iavf, splitlevel, false); 
	    ax01 = splint(avec1, kavf, javf1, iavf, 0., 0., iavf1-iavf, splitlevel, false);

	    /*
	    if (vflag){
	      std::cout << "vflag: gid = " << gid << "; coorf = " << xf1 << ", " << yf1 << ", " << zf << "; splitlevel = " << splitlevel << "\n";
	      std::cout << "kavf = " << kavf << "; javf = " << javf << "; iavf = " << iavf << "\n";
	      std::cout << "kavf1 = " << kavf1 << "; javf1 = " << javf1 << "; iavf1 = " << iavf1 << "\n";
	      losslesscout(az00-6.03935396758638622072799806428378133205114863812923431396484375e-07);
	      if(splitlevel == maxsplit){
		losslesscout(az01-6.03935396758638622072799806428378133205114863812923431396484375e-07);
		losslesscout((az00+az01)/2.-6.03935396758638622072799806428378133205114863812923431396484375e-07);
		getchar();
	      }

	    }
	    */
	    
	    // magnetic fields from interpolatec Avec:
	    pfield->b.x1f(k,j,i) = (az10 - az00) / dyf - (ay10 - ay00) / dzf ;
	    pfield->b.x2f(k,j,i) = (ax10 - ax00) / dzf - (az01 - az00) / dxf ;
	    pfield->b.x3f(k,j,i) = (ay01 - ay00) / dxf - (ax01 - ax00) / dyf ;
	  }
	}
      }
    }

    std::cout << "core " << gid << " interpolation finished \n";
    std::cout << "core " << gid << ": rbox = " << rbox << "\n";
    int maxnpoints = 0;
    Real entrainfactor = 1.; // entrainfactor allows for smooth velocity shift between the interpolated stellar material and the new background 
    
    // normalize by the number of points:
    
    for (int k = ks-NGHOST*1; k < ke+NGHOST*1; k++) {
      for (int j = js-NGHOST*1; j < je+NGHOST*1; j++) {
	for (int i = is-NGHOST*1; i < ie+NGHOST*1; i++) {
	  Real x = pcoord->x1v(i), y = pcoord->x2v(j), z = pcoord->x3v(k); // coords with respect to BH
	  Real r = std::sqrt(SQR(x)+SQR(y)+SQR(z)), r2star = std::sqrt(SQR(x-addx)+SQR(y-addy)+SQR(z-addz));
	  Real rhogas = bgdrho * std::exp(std::max(std::min(rvir/r-rvir/rstar, 3.), -3.)), pgas = bgdp  * std::exp(std::max(std::min(rvir/r-rvir/rstar, 3.), -3.));
	  if (ifXYZ){
	    rhogas = bgdrho  * std::exp(std::max(std::min(rvir/r-rvir/rstar, 3.), -3.));
	    pgas = bgdp  * std::exp(std::max(std::min(rvir/r-rvir/rstar, 3.), -3.));
	  }
	  else{
	    rhogas = bgdrho; 
	    pgas = bgdp;
	  }
	  
	  if (npoints(k,j,i) > 0){
	    maxnpoints = std::max(npoints(k,j,i), maxnpoints);
	    // normalizing by the number of points
	    phydro->u(IDN, k, j, i) /= (double)npoints(k,j,i) ;
	    phydro->u(IM1, k, j, i) /= (double)npoints(k,j,i) ;
	    phydro->u(IM2, k, j, i) /= (double)npoints(k,j,i) ;
	    phydro->u(IM3, k, j, i) /= (double)npoints(k,j,i) ;
	    //                    if (NON_BAROTROPIC_EOS) {
	    phydro->u(IEN, k, j, i) /= (double)npoints(k,j,i) ;
	    
	    // }
	    if(NSCALARS>0){
	      for (int n=0; n<NSCALARS;++n){
		pscalars->s(n,k,j,i) /= (double)npoints(k,j,i) ;
		pscalars->r(n,k,j,i) = std::min(std::max(pscalars->s(n,k,j,i) / phydro->u(IDN, k, j, i), 0.), 1.);
		if (phydro->u(IDN, k, j, i)<bgdrho) pscalars->s(n,k,j,i) = pscalars->r(n,k,j,i) = 0.;
	      }
	    }
	    /*
	      if(MAGNETIC_FIELDS_ENABLED){
	      pfield->b.x1f(k,j,i) /= (double)npoints(k,j,i) ;
	      pfield->b.x2f(k,j,i) /= (double)npoints(k,j,i) ;
	      pfield->b.x3f(k,j,i) /= (double)npoints(k,j,i) ;
	      }
	    */
	    //!!! temporary turned off the averaging for the MFs 
	    // ensuring there is no underpressure
	    // phydro->u(IEN, k, j, i) = std::max(phydro->u(IEN, k, j, i), pgas / (gamma-1.) + rhogas * (SQR(addvx)+SQR(addvy)+SQR(addvz))/2.);
	    //				       + (SQR(pfield->b.x1f(k,j,i))+SQR(pfield->b.x2f(k,j,i))+SQR(pfield->b.x3f(k,j,i)))/2.);
	    
	    Real scalefactor=1., dd=0.;
	    
	    if (massboost>1.0){// making the star 'massboost' times heavier (no effect on the dynamics, as we ignore self-gravity)
	      // valid only if NSCALARS > 0
	      pscalars->s(0,k,j,i) *= massboost; // tracer cons
	      scalefactor = 1.+(massboost-1.) * pscalars->r(0,k,j,i); // boosting only stellar material
	      
	      phydro->u(IDN, k, j, i) *= scalefactor ;
	      phydro->u(IEN, k, j, i) *= scalefactor ;
	      
	      phydro->u(IM1, k, j, i) *= scalefactor ;
	      phydro->u(IM2, k, j, i) *= scalefactor ;
	      phydro->u(IM3, k, j, i) *= scalefactor ;		
	    }
	    
	    // if the density is locally smaller than bgdrho
	    if (phydro->u(IDN, k, j, i) < rhogas){
	      dd = rhogas-phydro->u(IDN, k, j, i);
	      phydro->u(IDN, k, j, i) = rhogas;
	      phydro->u(IM1, k, j, i) += dd * (addvx+bgdvx);
	      phydro->u(IM2, k, j, i) += dd * (addvy+bgdvy);
	      phydro->u(IM3, k, j, i) += dd * (addvz+bgdvz);
	      phydro->u(IEN, k, j, i) += dd * (bgdp/bgdrho / (gamma-1.) + (SQR(addvx+bgdvx)+SQR(addvy+bgdvy)+SQR(addvz+bgdvz))/2.) ;
	      pscalars->s(0,k,j,i) = pscalars->r(0,k,j,i) = 0.;
	    }
	    pscalars->s(0,k,j,i) = std::min(pscalars->s(0,k,j,i), phydro->u(IDN, k, j, i));
	    pscalars->r(0,k,j,i) = std::min(std::max(pscalars->s(0,k,j,i) / phydro->u(IDN, k, j, i), 0.), 1.);
	    // ensuring there is no underpressure                                                                                                                                     
	    // 	      if (ifcomoving){// if the ambient medium is 
	    phydro->u(IEN, k, j, i) = std::max(phydro->u(IEN, k, j, i), pgas / (gamma-1.) + rhogas * (SQR(addvx+bgdvx)+SQR(addvy+bgdvy)+SQR(addvz+bgdvz))/2.);
	    //	      } else{
	    //phydro->u(IEN, k, j, i) = std::max(phydro->u(IEN, k, j, i), pgas / (gamma-1.)) + rhogas * (SQR(bgdvx)+SQR(bgdvy)+SQR(bgdvz))/2.;
	    // }
	  }
	  else{
	    //	      if (ifcomoving){
	    phydro->u(IDN, k, j, i) = bgdrho;
	    //		phydro->u(IM1, k, j, i) = bgdrho * (addvx+bgdvx);
	    // phydro->u(IM2, k, j, i) = bgdrho * (addvy+bgdvy);
	    // phydro->u(IM3, k, j, i) = bgdrho * (addvz+bgdvz);
	    //}
	    //else{
	    // within the sphere with R = half diagonal of the initial cube, the ambient matter is comoving with the star
	    entrainfactor = std::exp(std::min(1.-SQR(r2star/rbox)/3.,0.)*0.5);
	    phydro->u(IDN, k, j, i) = rhogas ; // bgdrho  * std::exp(std::max(std::min(rvir/r-rvir/rstar, 3.), 0.1));
	    phydro->u(IM1, k, j, i) = rhogas * (addvx+bgdvx) * entrainfactor;
	    phydro->u(IM2, k, j, i) = rhogas * (addvy+bgdvy) * entrainfactor;
	    phydro->u(IM3, k, j, i) = rhogas * (addvz+bgdvz) * entrainfactor;
	  }
	  /*
	    if(MAGNETIC_FIELDS_ENABLED){
	    pfield->b.x1f(k,j,i) = 0.;
	    pfield->b.x2f(k,j,i) = 0.;
	    pfield->b.x3f(k,j,i) = 0.;
	    }
	  */
	  if(NSCALARS>0){
	    
	    for (int n=0; n<NSCALARS;++n)pscalars->r(n,k,j,i) =  pscalars->s(n,k,j,i) = 0.;
	  }
	  //                    if (NON_BAROTROPIC_EOS) {
	  //  if(ifcomoving){
	  //	phydro->u(IEN, k, j, i) = pgas / (gamma-1.) + rhogas * (SQR(addvx)+SQR(addvy)+SQR(addvz))/2.;
	  //}
	  //else{
	  entrainfactor = std::exp(std::min(1.-SQR(r2star/rbox)/3.,0.)*0.5);
	  phydro->u(IEN, k, j, i) = pgas / (gamma-1.) + rhogas * (SQR(addvx)+SQR(addvy)+SQR(addvz))/2. * SQR(entrainfactor);  // * std::exp(std::max(std::min(rvir/r-rvir/rstar, 3.), 0.1));
	  // }
	  
	  if(std::isnan(phydro->u(IEN, k, j, i))){
	    std::cout << "!!! nan U, coords = " << k << ", " << j << ", " << i << "\n npoints = " << npoints(k,j,i) << "\n";
	    // getchar();
	  }
	  if(phydro->u(IEN, k, j, i) < pgas / (gamma-1.)){
	    std::cout << "!!! U = "<< phydro->u(IEN, k, j, i)<<", coords = " << k << ", " << j << ", " << i << "\n npoints = " << npoints(k,j,i) << "\n";
	    //getchar();
	  }
	  /*
	    phydro->w(IPR, k,j,i) = std::max(phydro->u(IEN, k, j, i) - 0.5*(SQR(phydro->u(IM1, k, j, i)) + SQR(phydro->u(IM1, 2, j, i)) + SQR(phydro->u(IM3, k, j, i)))/phydro->u(IDN, k,j,i), pgas);
	    phydro->w(IDN, k,j,i) = phydro->u(IDN, k,j,i) ;
	    phydro->w(IM1, k,j,i) = phydro->u(IM1, k,j,i)/phydro->u(IDN, k,j,i) ;
            phydro->w(IM2, k,j,i) = phydro->u(IM2, k,j,i)/phydro->u(IDN, k,j,i) ;
            phydro->w(IM3, k,j,i) = phydro->u(IM3, k,j,i)/phydro->u(IDN, k,j,i) ;
	  */
	}
      }
    }
    
    std::cout << "gid = " << gid << " finished; maximal points = " << maxnpoints << "\n";

    if(MAGNETIC_FIELDS_ENABLED){
      pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, is-NGHOST, ie+NGHOST, js-NGHOST, je+NGHOST, ks-NGHOST, ke+NGHOST);

      peos->ConservedToPrimitive(phydro->u, phydro->w, pfield->b,phydro->w, pfield->bcc, pcoord, is-NGHOST*1, ie+NGHOST*1, js-NGHOST*1, je+NGHOST*1, ks-NGHOST*1, ke+NGHOST*1);
        // we need this to update the energy according to the new magnetic field values
      //      pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, is-NGHOST, ie+NGHOST, js-NGHOST, je+NGHOST, ks-NGHOST, ke+NGHOST);
      // ensuring pressure is in balance (compensating the loss of self-gravity)
      
      if (levelpressure){
	for (int k = ks-NGHOST; k <= ke+NGHOST; k++) {
	  for (int j = js-NGHOST; j <= je+NGHOST; j++) {
	    for (int i = is-NGHOST; i <= ie+NGHOST; i++) {
	      Real x = pcoord->x1v(i), y = pcoord->x2v(j), z = pcoord->x3v(k); // coords with respect to BH    
	      Real r = std::sqrt(SQR(x)+SQR(y)+SQR(z));
	      Real bgdp_local = bgdp;
	      if (ifXYZ) bgdp_local = bgdp * std::exp(std::max(std::min(rvir/r-rvir/rstar, 3.), -3.));
	      phydro->w1(IPR, k,j,i) = phydro->w(IPR, k,j,i) = bgdp_local; //, phydro->w(IPR, k,j,i)); //  - (SQR(pfield->bcc(IB3,k,j,i))+SQR(pfield->bcc(IB2, k,j,i))+SQR(pfield->bcc(IB1, k,j,i)))/2.;
	      // pressure changes smooth within the star (or stellar debris cloud), it remains an entropy perturbation
	    }
	  }
	}
      }
      else{
	for (int k = ks-NGHOST*0; k <= ke+NGHOST*0; k++) {
	  for (int j = js-NGHOST*0; j <= je+NGHOST*0; j++) {
	    for (int i = is-NGHOST*0; i <= ie+NGHOST*0; i++) {
	      Real x = pcoord->x1v(i), y = pcoord->x2v(j), z = pcoord->x3v(k); // coords with respect to BH
	      Real r = std::sqrt(SQR(x)+SQR(y)+SQR(z));
	      Real bgdp_local = bgdp;
	      if(std::isnan(phydro->u(IEN, k, j, i))){
	      	std::cout << "!!! after C->P nan U, coords = " << k << ", " << j << ", " << i << "\n npoints = " << npoints(k,j,i) << "\n";
		// getchar();
	      }

              if(std::isnan(pfield->bcc(IB1, k,j,i))){
		std::cout << "Bcc1 = NaN\n";
		std::cout << "indices k= "<< k << ", j = "<< j<< ", i = " << i << ": " <<  pfield->b.x1f(k,j,i) << ".." << pfield->b.x1f(k,j,i+1) << "\n";
		pfield->b.x1f(k,j,i) = 0.;
		// getchar();
              }
	      if(std::isnan(pfield->bcc(IB2, k,j,i))){
		std::cout << "Bcc2 = NaN\n";
		std::cout << "indices k= "<< k << ", j = "<< j<< ", i = " << i << ": " <<pfield->b.x2f(k,j,i) << ".." << pfield->b.x2f(k,j+1,i) << "\n";
                pfield->b.x2f(k,j,i) = 0.;
		// getchar();
              }
	      if(std::isnan(pfield->bcc(IB3, k,j,i))){
		std::cout << "Bcc3 = NaN\n";
		std::cout << "indices k= "<< k << ", j = "<< j<< ", i = " << i << ": " << pfield->b.x3f(k,j,i) << ".." << pfield->b.x3f(k+1,j,i) << "\n";
		pfield->b.x3f(k,j,i) = 0.;
		//getchar();
	      }
	      //	if (ifXYZ)  bgdp_local = bgdp * std::exp(std::max(std::min(rvir/r-rvir/rstar, 3.), -3.));
	      // if (phydro->w(IPR, k,j,i) < 1e-2)std::cout << "P ("<< k << ", " << j << ", " << i << ") = " << phydro->w(IPR, k,j,i) << "; (U =  " << phydro->u(IEN, k,j, i)<<  " (internal)  + " << (SQR(pfield->bcc(IB1, k,j,i)) + SQR(pfield->bcc(IB2, k,j,i)) + SQR(pfield->bcc(IB3, k,j,i)))/2. << " (magnetic) +" << (SQR(phydro->u(IM1, k,j,i)) + SQR(phydro->u(IM2, k,j,i)) + SQR(phydro->u(IM3, k,j,i)) ) /2. << " (kinetic)\n"; 
	      	     
	      //	      getchar();
	      phydro->w1(IPR, k,j,i) =  phydro->w(IPR, k,j,i) = std::max(bgdp_local, phydro->w(IPR, k,j,i)); //  - (SQR(pfield->bcc(IB3,k,j,i))+SQR(pfield->bcc(IB2, k,j,i))+SQR(pfield->bcc(IB1, k,j,i)))/2.;
	      // maximal pressure (star or background)	      
	    }
	  }
	}
      }
      
      peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is-NGHOST, ie+NGHOST, js-NGHOST, je+NGHOST, ks-NGHOST, ke+NGHOST);
    }

      //      pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, is, ie, js, je, ks, ke);
      //      peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie, js, je, ks, ke);

    std::cout << "cleaning up arrays\n ";
    x1v_old.DeleteAthenaArray(); x2v_old.DeleteAthenaArray(); x3v_old.DeleteAthenaArray();
    x1f_old.DeleteAthenaArray(); x2f_old.DeleteAthenaArray(); x3f_old.DeleteAthenaArray();

    
    npoints.DeleteAthenaArray(); mindx.DeleteAthenaArray();
    u_old.DeleteAthenaArray();
    if(ifavec){
      x1a.DeleteAthenaArray();  x2a.DeleteAthenaArray(); x3a.DeleteAthenaArray();
      std::cout << "avec coordinates free\n";
    }else{
      MF_old.DeleteAthenaArray();
    }
    if (ifavec&&blockbint){
      avec1.DeleteAthenaArray();
      std::cout << "avec1 free\n";
      avec2.DeleteAthenaArray();
      std::cout << "avec2 free\n";
      avec3.DeleteAthenaArray();
      std::cout << "avec3 free\n";
    }    

  return;
}

Real stitcher(Real x, Real delta){
  // function making a smooth transition from 0 at x=delta to 1 at x=1                                                             
  if(x<=delta) return 0.0;
  if (x>=1) return 1.0;

  Real xc = x*SQR(x), dc = delta * SQR(delta);

  return xc/3. - (delta+1.)/2.*SQR(x) + delta * x + dc / 6. - SQR(delta) ;
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


void BHgrav(MeshBlock *pmb, const Real time, const Real dt,
            const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
            const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
            AthenaArray<Real> &cons_scalar){

  //  bool barycenter = false;                                                                                                      
  Real xBH = 0., yBH = 0., zBH = 0.;
  Real BH0 = 0., BH0x =0., BH0y =0., BH0z =0., vxBH = 0., vyBH = 0., vzBH = 0.;
           
  Real presscutoff = 1.*bgdp, dendot = 0., denfac = 0.;
 
  if (!ifXYZ){
    star_coord(time+dt/2., &xBH, &yBH, &zBH, &vxBH, &vyBH, &vzBH);
    xBH = -xBH; yBH = -yBH; zBH = -zBH;
    vxBH = -vxBH; vyBH = -vyBH; vzBH = -vzBH; // do we need the velocity of the BH?

    BH0 = BHgfun(-xBH, -yBH, -zBH);
    BH0x = -xBH/std::sqrt(SQR(xBH)+SQR(yBH)+SQR(zBH)) * BH0;
    BH0y = -yBH/std::sqrt(SQR(xBH)+SQR(yBH)+SQR(zBH)) * BH0;
    BH0z = -zBH/std::sqrt(SQR(xBH)+SQR(yBH)+SQR(zBH)) * BH0;
 
  }

  //   std::cout << "BH mass = " << addmass << "\n";

  Real trcut = 100.;
  Real dencut = trcut * bgdrho;

  //  Real dtodx1 = dt / pmb->pcoord->dx1v(0), dtodx2 = dt / pmb->pcoord->dx2v(0), dtodx3 = dt / pmb->pcoord->dx3v(0); // relies on the uniformity of the grid!!!                   

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real z = pmb->pcoord->x3v(k);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real y = pmb->pcoord->x2v(j);
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real x = pmb->pcoord->x1v(i);

        Real rsqeff = SQR(x-xBH)+SQR(y-yBH)+SQR(z-zBH);
        Real reff = std::sqrt(rsqeff), fadd = BHgfun(x-xBH,y-yBH,z-zBH), den = prim(IDN,k,j,i), press = prim(IEN,k,j,i);

        Real g1 = fadd * (x-xBH)/reff -BH0x; // (BHphifun(pmb->pcoord->x1v(i+1), y, z)-BHphifun(pmb->pcoord->x1v(i-1), y, z))/2.,                                                            
        Real g2 = fadd * (y-yBH)/reff -BH0y; // (BHphifun(x,pmb->pcoord->x2v(j+1), z)-BHphifun(x, pmb->pcoord->x2v(j-1), z))/2.,                                                             
        Real g3 = fadd * (z-zBH)/reff -BH0z; // (BHphifun(x,y,pmb->pcoord->x3v(k+1))-BHphifun(x, y, pmb->pcoord->x3v(k-1)))/2.;                               
	Real s;             
	if (ifstitcher){
	  // s = stitcher(den/dencut, 0.1);
	  s = stitcher(prim_scalar(0,k,j,i)/0.01, 0.1);
	  // }
	  g1 *= s; g2 *= s; g3 *= s;
	}

        cons(IM1,k,j,i) -=  ( g1 * dt ) * den ; //dtodx1 * den * (BHphifun(pmb->pcoord->x1v(i+1), y, z)-BHphifun(pmb->pcoord->x1v(i-1), y, z))/2.;                                  
        cons(IM2,k,j,i) -=  ( g2 * dt ) * den ; // dtodx2 * den * (BHphifun(x,pmb->pcoord->x2v(j+1), z)-BHphifun(x, pmb->pcoord->x2v(j-1), z))/2.;                                  
        cons(IM3,k,j,i) -=  ( g3 * dt ) * den ; // dtodx3 * den * (BHphifun(x,y,pmb->pcoord->x3v(k+1))-BHphifun(x, y, pmb->pcoord->x3v(k-1)))/2.;                                   

	//fadd * (z / std::sqrt(rsqeff)) ;                                                                                                                                        
        if (NON_BAROTROPIC_EOS) {
          cons(IEN,k,j,i) -= (g1 * prim(IM1, k,j,i) + g2 * prim(IM2,k,j,i) +  g3 * prim(IM3, k,j,i)) * dt * den;
        }
	for (int n=0; n<NSCALARS;++n){
	  cons_scalar(n,k,j,i) = std::min(cons_scalar(n,k,j,i), cons(IDN, k, j, i));
	  if(den < bgdrho){
	    cons_scalar(n,k,j,i) = cons(IDN, k, j, i) * 0.;
	  }
	}

	if (ifevacuate && (press>presscutoff) && (reff < rBH)){
	  // mass loss in the BH vicinity
	  dendot = (press/presscutoff-1.)  * (1. - SQR(reff/rBH)) * std::sqrt(addmass/rBH)/rBH * dt ;
	  denfac = std::max(1.-dendot,0.9);
	  cons(IDN, k, j, i) *= denfac; cons(IEN,k,j,i) *= denfac;
	  cons(IM1,k,j,i) *= denfac ; 
          cons(IM2,k,j,i) *= denfac;
          cons(IM3,k,j,i) *= denfac ;
	  for (int n=0; n<NSCALARS;++n){
	    cons_scalar(n,k,j,i) *= denfac;
	  }
	}
      }
    }
  }

}

int RefinementCondition_Bonly(MeshBlock *pmb)
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
    return 0;
}


int RefinementCondition(MeshBlock *pmb)
{
  AthenaArray<Real> &w = pmb->phydro->w;
  AthenaArray<Real> &R = pmb->pscalars->r; // scalar (=1 inside the star, =0 outside)                                                 

  Real maxR0 = 0.0, maxeps = 0., bsq, eps, den, r, maxbsq = 0.0;
  
  Real x, y, z, r1=0.;

  for(int k=pmb->ks; k<pmb->ke; k++) {
    if(ifcentrelens)z = pmb->pcoord->x3v(k);                                                                                                 
    for(int j=pmb->js; j<pmb->je; j++) {
      if(ifcentrelens)y = pmb->pcoord->x2v(j);                                                                                               
      for(int i=pmb->is; i<pmb->ie; i++) {
        if(ifcentrelens){
	  x = pmb->pcoord->x1v(i); //, dx = pmb->pcoord->dx1v(i); // , y = pmb->pcoord->x2v(j), z = pmb->pcoord->x3v(k);            
	  //        Real xf = pmb->pcoord->x1f(i), yf = pmb->pcoord->x2f(j), zf = pmb->pcoord->x3f(k);                                  
	  r1 = std::sqrt(SQR(x)+SQR(y)+SQR(z)); // distance to the star or BH centre, depending on the setup (ifXYZ)
	}        
	r = R(0, k, j, i);
        den = (w(IDN,k,j,i)+w(IDN,k,j,i+1)+w(IDN,k,j,i-1)+w(IDN,k,j+1,i)+w(IDN,k,j-1,i)+w(IDN,k+1,j,i)+w(IDN,k-1,j,i))/7.;
        maxR0 = std::max(maxR0, R(0, k, j, i));
	
	eps = std::abs(w(IDN,k,j,i+1)-w(IDN,k,j,i-1))+ std::abs(w(IDN,k,j+1,i)-w(IDN,k,j-1,i))+std::abs(w(IDN,k+1,j,i)-w(IDN,k-1,j,i));
	if (MAGNETIC_FIELDS_ENABLED){
	  bsq = SQR(pmb->pfield->b.x3f(k,j,i)) + SQR(pmb->pfield->b.x2f(k,j,i)) + SQR(pmb->pfield->b.x1f(k,j,i)) ;
	  maxbsq = std::max(bsq, maxbsq);
	  if ((r>Rthresh) && (bsq > SQR(refB)) && (!ifcentrelens||(r1 < rcentrecircle))){
	    maxeps = std::max(maxeps, eps / (den+refden));
	  }
	}
	else{
	  if ((r>Rthresh) && (den>refden)&& (!ifcentrelens||(r1 < rcentrecircle))){
	    maxeps = std::max(maxeps, eps / (den+refden));
	  }
	}
      }
    }
  }

  if (maxeps > thresh) return 1; // refinement                                                                                        
  if ((maxeps < (thresh/4.)) || (maxR0 < (Rthresh/4.))) return -1; // derefinement                                       
  if(MAGNETIC_FIELDS_ENABLED && (std::sqrt(maxbsq) < ( refB / 16.))) return -1; // derefinement

  return 0;
}


namespace{

  void losslesscout(Real x){
    std::streamsize ss = std::cout.precision();
    std::cout << std::setprecision( std::numeric_limits<int>::max() ) << x << "\n";
    std::cout.precision (ss);
  }
  
  Real true_anomaly(Real t){
    Real M = Mcoeff * (t - tper); // global Mcoeff is sqrt(GM/Rp^3), global tper is the time of pericenter passage
    return 2. * std::atan(2.*std::sinh(std::asinh(1.5*M)/3.));
  }

  void star_coord(Real t, Real* xBH, Real* yBH, Real* zBH, Real* vx, Real* vy, Real* vz){
    // parabolic motion with pericenter distance rper, true anomaly nu
    Real nu = true_anomaly(t);
    
    Real sinnu = std::sin(nu), cosnu = std::cos(nu);
    Real rad = 2. * rper / (1.0+cosnu);

    *xBH = -rad * cosnu;
    *yBH = -rad * sinnu;
    *zBH = 0.;

    Real vnorm = std::sqrt(addmass/rad / (1.+cosnu));

    *vx = sinnu * vnorm;
    *vy = -(1.+cosnu) * vnorm;
    *vz = 0.; 
  }

  Real splint(AthenaArray<Real>& A, Real kav, Real jav, Real iav, Real dk, Real dj, Real di, int splitlevel, bool verboseflag){
    // we are estimating the value at kav + dk/2, jav + dj/2, iav + di/2
    // only one of the steps (di, dj, dk) should be non-zero
    Real A0, A1, A2;
    if((di*dj+dj*dk+di*dk) > 0.){
      std::cout << "splint: more than one step is non-zero: " << di << ", " << dj << ", " << dk << "\n";
      exit(1);
    }
    if ((splitlevel <= 0)||((di + dj + dk) <= 0.)){
      return Acurl(A, kav+dk*0.5, jav+dj*0.5, iav+di*0.5); // Acurl(A, kav+dk, jav+dj, iav+di)-Acurl(A, kav, jav, iav);
    }else{ 
      A0 = splint(A, kav, jav, iav, dk*0.5, dj*0.5, di*0.5, splitlevel-1, false); // first half  of the edge, centred on dx/4
      A1 = splint(A, kav+dk*0.5, jav+dj*0.5, iav+di*0.5, dk*0.5, dj*0.5, di*0.5, splitlevel-1, false); // second half, centered on dx 3/4
      A2 = (A1 + A0)/2.;
      if (verboseflag){
	std::cout << A0 << " + " << A1 << " = " << A2 << "\n";
	std::cout << "differences = " << A0-A2 << ", " << A1 - A2 << "\n";
      }
      return A2; 
    }
    
  }
  
  Real Acurl(AthenaArray<Real>& A, Real kav, Real jav, Real iav){
    int k0 = (int)floor(kav), j0 = (int)floor(jav), i0 = (int)floor(iav); 

    int k1 = k0+1, j1 = j0+1, i1 = i0+1;

    Real cz = kav-(double)k0, cy = jav-(double)j0, cx = iav-(double)i0;

    Real aint_z0 = A(k0,j0,i0) + cy * (A(k0,j1,i0)-A(k0,j0,i0)) + cx * (A(k0,j0,i1)-A(k0,j0,i0))
      + cy * cx * (A(k0,j1,i1) - A(k0,j0,i1) - A(k0,j1,i0) + A(k0,j0,i0));
    
    Real aint_z1 = A(k1,j0,i0) + cy * (A(k1,j1,i0)-A(k1,j0,i0)) + cx * (A(k1,j0,i1)-A(k1,j0,i0))
      + cy * cx * (A(k1,j1,i1) - A(k1,j0,i1) - A(k1,j1,i0) + A(k1,j0,i0));

    return aint_z0 + cz * (aint_z1-aint_z0);
  }
  /*
  Real Bspline3(Real x){
    Real ax = std::fabs(x);
    if (ax <= 1.0){
      return 2./3. - SQR(x)-ax*SQR(x)/2.;
    } else if(ax < 2.0){
      return (2.-ax)*SQR(2.-ax)/6.;
    } else{
      return 0.;
    }
  }
  
Real Bspline2(Real x){
    // second-order B-spline
    Real ax = std::fabs(x);
    
    if (ax <= 0.5){
        return 0.75 - SQR(x);
    }else if (ax <= 1.5){
        return SQR(2.*ax-3.)*0.125 ;
    }else{
        return 0.;
    }
    
}

 Real Bspline1(Real x){
    // first-order B-spline
    return std::max(1.-std::fabs(x), 0.);
 }

 Real Bspline0(Real x){
   if (std::fabs(x)<=0.5) {
     return 1.;
   }else{
     return 0.;
   }
 }
 
 Real Bspline(Real x, int order){
   if(order == 0){
     return Bspline0(x);
   } else if (order == 1){
     return Bspline1(x);
   }else if (order == 2){
     return Bspline2(x);
   } else if (order == 3){
     return Bspline3(x);
   } else{
     return 0.; // not supported so far
   }
 }
 
 
 Real Bint_linearZ(AthenaArray<Real> &u, int index1, int index2, Real kf, Real j, Real i, int order){
    // two foreindices, with respect to the Z cell face
   // order is the base order of the interpolation scheme
    
    int k0 = (int)floor(kf), j0 = (int)floor(j), i0 = (int)floor(i);
    int k1 = k0+1, j1 = j0+1, i1 = i0+1, km1 = k0-1, k2 = k0+2;
    
    Real cz = -std::fmod(kf,k0), cy = -std::fmod(j,j0), cx = -std::fmod(i,i0); // fractional coordinates of the closest node (face center) to the left
    
    Real unew0 = u(index1, index2,  k0,j0,i0) * Bspline(cx, order) * Bspline(cy, order) +  u(index1, index2,  k0,j0,i1) * Bspline(cx+1., order) * Bspline(cy, order)  +  u(index1, index2,  k0,j1,i0) * Bspline(cx, order) * Bspline(cy+1., order)  +  u(index1, index2,  k0,j1,i1) * Bspline(cx+1., order) * Bspline(cy+1., order); // k = const
    Real unew1 = u(index1, index2,  k1,j0,i0) * Bspline(cx, order) * Bspline(cy, order) +  u(index1, index2,  k1,j0,i1) * Bspline(cx+1., order) * Bspline(cy, order)  +  u(index1, index2,  k1,j1,i0) * Bspline(cx, order) * Bspline(cy+1., order)  +  u(index1, index2,  k1,j1,i1) * Bspline(cx+1., order) * Bspline(cy+1., order);
    Real unewm1 = u(index1, index2,  km1,j0,i0) * Bspline(cx, order) * Bspline(cy, order) +  u(index1, index2,  km1,j0,i1) * Bspline(cx+1., order) * Bspline(cy, order)  +  u(index1, index2,  km1, j1, i0) * Bspline(cx, order) * Bspline(cy+1., order)  +  u(index1, index2,  km1,j1,i1) * Bspline(cx+1., order) * Bspline(cy+1., order);
    Real unew2 = u(index1, index2,  k2,j0,i0) * Bspline(cx, order) * Bspline(cy, order) +  u(index1, index2,  k2,j0,i1) * Bspline(cx+1., order) * Bspline(cy, order)  +  u(index1, index2,  k2,j1,i0) * Bspline(cx, order) * Bspline(cy+1., order)  +  u(index1, index2,  k2,j1,i1) * Bspline(cx+1., order) * Bspline(cy+1., order);
    
    return unewm1 * Bspline(cz-1., order+1) + unew0 * Bspline(cz, order+1) + unew1 * Bspline(cz+1., order+1) +unew2 * Bspline(cz+2., order) ;
}


 Real Bint_linearY(AthenaArray<Real> &u, int index1, int index2, Real k, Real jf, Real i, int order){
    // two foreindices, with respect to the Y cell face
    
    int k0 = (int)floor(k), j0 = (int)floor(jf), i0 = (int)floor(i);
    int k1 = k0+1, j1 = j0+1, i1 = i0+1, jm1 = j0-1, j2 = j0+2;
    
    Real cz = -std::fmod(k,k0), cy = -std::fmod(jf,j0), cx = -std::fmod(i,i0);
    
    Real unew0 = u(index1, index2,  k0,j0,i0) * Bspline(cz, order) * Bspline(cx, order) +  u(index1, index2,  k1,j0,i0) * Bspline(cz+1., order) * Bspline(cx, order)  +  u(index1, index2,  k0,j0,i1) * Bspline(cz, order) * Bspline(cx+1., order)  +  u(index1, index2,  k1,j0,i1) * Bspline(cz+1., order) * Bspline(cx+1., order); // j = const
    Real unew1 = u(index1, index2,  k0,j1,i0) * Bspline(cz, order) * Bspline(cx, order) +  u(index1, index2,  k1,j1,i0) * Bspline(cz+1., order) * Bspline(cx, order)  +  u(index1, index2,  k0,j1,i1) * Bspline(cz, order) * Bspline(cx+1., order)  +  u(index1, index2,  k1,j1,i1) * Bspline(cz+1., order) * Bspline(cx+1., order);
    Real unewm1 = u(index1, index2,  k0,jm1,i0) * Bspline(cz, order) * Bspline(cx, order) +  u(index1, index2,  k1,jm1,i0) * Bspline(cz+1., order) * Bspline(cx, order)  +  u(index1, index2,  k0,jm1,i1) * Bspline(cz, order) * Bspline(cx+1., order)  +  u(index1, index2,  k1,jm1,i1) * Bspline(cz+1., order) * Bspline(cx+1., order);
    Real unew2 = u(index1, index2,  k0,j2,i0) * Bspline(cz, order) * Bspline(cx, order) +  u(index1, index2,  k1,j2,i0) * Bspline(cz+1., order) * Bspline(cx, order)  +  u(index1, index2,  k0,j2,i1) * Bspline(cz, order) * Bspline(cx+1., order)  +  u(index1, index2,  k1,j2,i1) * Bspline(cz+1., order) * Bspline(cx+1., order);

    return unewm1 * Bspline(cy-1., order+1) + unew0 * Bspline(cy, order+1) + unew1 * Bspline(cy+1., order+1) +unew2 * Bspline(cy+2., order+1) ;
}


 Real Bint_linearX(AthenaArray<Real> &u, int index1, int index2, Real k, Real j, Real i_f, int order){
    // two foreindices, with respect to the X cell face
    
    int k0 = (int)floor(k), j0 = (int)floor(j), i0 = (int)floor(i_f);
    int k1 = k0+1, j1 = j0+1, i1 = i0+1, im1 = i0-1, i2 = i0+2;
    
    Real cz = -std::fmod(k,k0), cy = -std::fmod(j,j0), cx = -std::fmod(i_f,i0);
    
    Real unew0 = u(index1, index2, k0,j0,i0) * Bspline(cz, order) * Bspline(cy, order) +  u(index1, index2,  k1,j0,i0) * Bspline(cz+1., order) * Bspline(cy, order)  +  u(index1, index2,  k0,j1,i0) * Bspline(cz, order) * Bspline(cy+1., order)  +  u(index1, index2,  k1,j1,i0) * Bspline(cz+1., order) * Bspline(cy+1., order); // i = const
    Real unew1 = u(index1, index2,  k0,j0,i1) * Bspline(cz, order) * Bspline(cy, order) +  u(index1, index2,  k1,j0,i1) * Bspline(cz+1., order) * Bspline(cy, order)  +  u(index1, index2,  k0,j1,i1) * Bspline(cz, order) * Bspline(cy+1., order)  +  u(index1, index2,  k1,j1,i1) * Bspline(cz+1., order) * Bspline(cy+1., order); // i = const
    Real unewm1 = u(index1, index2,  k0,j0,im1) * Bspline(cz, order) * Bspline1(cy) +  u(index1, index2,  k1,j0,im1) * Bspline(cz+1., order) * Bspline(cy, order)  +  u(index1, index2,  k0,j1,im1) * Bspline(cz, order) * Bspline(cy+1., order)  +  u(index1, index2,  k1,j1,im1) * Bspline(cz+1., order) * Bspline(cy+1., order); // i = const
    Real unew2 = u(index1, index2,  k0,j0,i2) * Bspline(cz, order) * Bspline(cy, order) +  u(index1, index2,  k1,j0,i2) * Bspline(cz+1., order) * Bspline(cy, order)  +  u(index1, index2,  k0,j1,i2) * Bspline(cz, order) * Bspline(cy+1., order)  +  u(index1, index2,  k1,j1,i2) * Bspline(cz+1., order) * Bspline(cy+1., order); // i = const

    return unewm1 * Bspline(cx-1., order+1) + unew0 * Bspline(cx, order+1) + unew1 * Bspline(cx+1., order+1) +unew2 * Bspline(cx+2., order+1) ;
}
 

  void intcurl(AthenaArray<Real>& A1, AthenaArray<Real>& A2, AthenaArray<Real>& A3, Real kav, Real jav, Real iav, Real kavf, Real javf, Real iavf,Real* b1, Real* b2, Real *b3){
    // linear interpolation of the vector potential components to the new grid.
    // For each cell, each component should be calculated for 4 different locations, necessary for two different field components orthogonal to the vector potential component.
    // A1-3 are old vector potential arrays (3D), x{1-3}a are the old face-centred coordinates (I guess I do not even need them).


  int k0 = (int)floor(kav), j0 = (int)floor(jav), i0 = (int)floor(iav);
  int k0f = (int)floor(kavf), j0f = (int)floor(javf), i0f = (int)floor(iavf);
  int k1 = k0+1, j1 = j0+1, i1 = i0+1;
  int k2 = k0+2, j2 = j0+2, i2 = i0+2;
  int k1f = k0f+1, j1f = j0f+1, i1f = i0f+1;
  int k2f = k0f+2, j2f = j0f+2, i2f = i0f+2;

  //  int khalf = (int)floor(kavf),  jhalf = (int)floor(javf), ihalf = (int)floor(iavf);
  //int ihalf = i0, jhalf = j0, khalf = k0;

  // we need interpolated values of A on the edges
  // A1 at i = (kf0, kf1) (jf0, jf1) i 
  // A2 at i = (kf0, kf1) j (if0, if1)                                                                                                                              
  // A3 at i = k (jf0, jf1) (if0, if1)     
  
  // Real cz = std::fmod(kav,k0), cy = std::fmod(jav,j0), cx = std::fmod(iav,i0);
  //  Real czf = std::fmod(kavf,k0f), cyf = std::fmod(javf,j0f), cxf = std::fmod(iavf,i0f);

  Real cz = kav-(double)k0, cy = jav-(double)j0, cx = iav-(double)i0;
  Real czf = kavf-(double)k0f, cyf = javf-(double)j0f, cxf = iavf-(double)i0f;

  if((cz <0.)||(cz>1.)||(czf <0.)||(czf>1.)||
     (cy <0.)||(cy>1.)||(cyf <0.)||(cyf>1.)||
     (cx <0.)||(cx>1.)||(cxf <0.)||(cxf>1.)){
    std::cout << "fractional part calculated incorrectly\n";
    std::cout << "cz = " << cz << "\n";
    std::cout << "czf = " << czf << "\n";
    std::cout << "cy = " << cy << "\n";
    std::cout << "cyf = " << cyf << "\n";
    std::cout << "cx = " << cx << "\n";
    std::cout << "cxf = " << cxf << "\n";
    exit(1);
  }

  // Real czhalf = std::fmod(kavf,khalf), cyhalf = std::fmod(javf,jhalf), cxhalf = std::fmod(iavf,ihalf);

  // A1 (k0f, j0f, i0)
  Real a1int_z0 = A1(k0f,j0f,i0) + cyf * (A1(k0f,j1f,i0)-A1(k0f,j0f,i0)) + cx * (A1(k0f,j0f,i1)-A1(k0f,j0f,i0)) 
    + cyf * cx * (A1(k0f,j1f,i1) - A1(k0f,j0f,i1) - A1(k0f,j1f,i0) + A1(k0f,j0f,i0));
  Real a1int_z1 = A1(k1f,j0f,i0) + cyf * (A1(k1f,j1f,i0)-A1(k1f,j0f,i0)) + cx * (A1(k1f,j0f,i1)-A1(k1f,j0f,i0))
    + cy * cx * (A1(k1,j1,i1) - A1(k1,j0,i1) - A1(k1,j1,i0) + A1(k1,j0,i0));
  Real a1int00 = a1int_z0 + czf * (a1int_z1 - a1int_z0);

  // A1 (k0f, j1f, i0)
  a1int_z0 = A1(k0f,j1f,i0) + cyf * (A1(k0f,j2f,i0)-A1(k0f,j1f,i0)) + cx * (A1(k0f,j1f,i1)-A1(k0f,j1f,i0))
    + cyf * cx * (A1(k0f,j2f,i1) - A1(k0f,j1f,i1) - A1(k0f,j2f,i0) + A1(k0f,j1f,i0));
  a1int_z1 = A1(k1f,j1f,i0) + cyf * (A1(k1f,j2f,i0)-A1(k1f,j1f,i0)) + cx * (A1(k1f,j1f,i1)-A1(k1f,j1f,i0))
    + cyf * cx * (A1(k1f,j2f,i1) - A1(k1f,j1f,i1) - A1(k1f,j2f,i0) + A1(k1f,j1f,i0));
  Real a1int01 = a1int_z0 + czf * (a1int_z1 - a1int_z0);

  // A1 (k1f, j0f, i0)                                                                                                                                               
  a1int_z0 = A1(k1f,j0f,i0) + cyf * (A1(k1f,j1f,i0)-A1(k1f,j0f,i0)) + cx * (A1(k1f,j0f,i1)-A1(k1f,j0f,i0))
    + cyf * cx * (A1(k1f,j1f,i1) - A1(k1f,j0f,i1) - A1(k1f,j1f,i0) + A1(k1f,j0f,i0));
  a1int_z1 = A1(k2f,j0f,i0) + cyf * (A1(k2f,j1f,i0)-A1(k2f,j0f,i0)) + cx * (A1(k2f,j0f,i1)-A1(k2f,j0f,i0))
    + cyf * cx * (A1(k2f,j1f,i1) - A1(k2f,j0f,i1) - A1(k2f,j1f,i0) + A1(k2f,j0f,i0));
  Real a1int10 = a1int_z0 + czf * (a1int_z1 - a1int_z0);

  // A1 (k1f, j1f, i0)
  a1int_z0 = A1(k1f,j1f,i0) + cyf * (A1(k1f,j2f,i0)-A1(k1f,j1f,i0)) + cx * (A1(k1f,j1f,i1)-A1(k1f,j1f,i0)) 
    + cyf * cx * (A1(k1f,j2f,i1) - A1(k1f,j1f,i1) - A1(k1f,j2f,i0) + A1(k1f,j1f,i0));
  a1int_z1 = A1(k2f,j1f,i0) + cyf * (A1(k2f,j2f,i0)-A1(k2f,j1f,i0)) + cx * (A1(k2f,j1f,i1)-A1(k2f,j1f,i0))
    + cyf * cx * (A1(k2f,j2f,i1) - A1(k2f,j1f,i1) - A1(k2f,j2f,i0) + A1(k2f,j1f,i0));
  Real a1int11 = a1int_z0 + czf * (a1int_z1 - a1int_z0);

  // A2 (k0f, j0, i0f)
  Real a2int_z0 = A2(k0f,j0,i0f) + cy * (A2(k0f,j1,i0f)-A2(k0f,j0,i0f)) + cxf * (A2(k0f,j0,i1f)-A2(k0f,j0,i0f))
    + cy * cxf * (A2(k0f,j1,i1f) - A2(k0f,j0,i1f) - A2(k0f,j1,i0f) + A2(k0f,j0,i0f));
  Real a2int_z1 = A2(k1f,j0,i0f) + cy * (A2(k1f,j1,i0f)-A2(k1f,j0,i0f)) + cxf * (A2(k1f,j0,i1f)-A2(k1f,j0,i0f))
    + cy * cxf * (A2(k1f,j1,i1f) - A2(k1f,j0,i1f) - A2(k1f,j1,i0f) + A2(k1f,j0,i0f));
  Real a2int00 = a2int_z0 + czf * (a2int_z1 - a2int_z0);
  // A2 (k0f, j0, i1f)
  a2int_z0 = A2(k0f,j0,i1f) + cy * (A2(k0f,j1,i1f)-A2(k0f,j0,i1f)) + cxf * (A2(k0f,j0,i2f)-A2(k0f,j0,i1f)) 
    + cy * cxf * (A2(k0f,j1,i2f) - A2(k0f,j0,i2f) - A2(k0f,j1,i1f) + A2(k0f,j0,i1f));
  a2int_z1 = A2(k1f,j0,i1f) + cy * (A2(k1f,j1,i1f)-A2(k1f,j0,i1f)) + cxf * (A2(k1f,j0,i2f)-A2(k1f,j0,i1f))
    + cy * cxf * (A2(k1f,j1,i2f) - A2(k1f,j0,i2f) - A2(k1f,j1,i1f) + A2(k1f,j0,i1f));
  Real a2int01 = a2int_z0 + czf * (a2int_z1 - a2int_z0);
  // A2 (k1f, j0, i0f)
  a2int_z0 = A2(k1f,j0,i0f) + cy * (A2(k1f,j1,i0f)-A2(k1f,j0,i0f)) + cxf * (A2(k1f,j0,i1f)-A2(k1f,j0,i0f))
    + cy * cxf * (A2(k1f,j1,i1f) - A2(k1f,j0,i1f) - A2(k1f,j1,i0f) + A2(k1f,j0,i0f));
  a2int_z1 = A2(k2f,j0,i0f) + cy * (A2(k2f,j1,i0f)-A2(k2f,j0,i0f)) + cxf * (A2(k2f,j0,i1f)-A2(k2f,j0,i0f))
    + cy * cxf * (A2(k2f,j1,i1f) - A2(k2f,j0,i1f) - A2(k2f,j1,i0f) + A2(k2f,j0,i0f));
  Real a2int10 = a2int_z0 + czf * (a2int_z1 - a2int_z0);
  // A2 (k1f, j0, i1f)
  a2int_z0 = A2(k1f,j0,i1f) + cy * (A2(k1f,j1,i1f)-A2(k1f,j0,i1f)) + cxf * (A2(k1f,j0,i2f)-A2(k1f,j0,i1f))
    + cy * cxf * (A2(k1f,j1,i2f) - A2(k1f,j0,i2f) - A2(k1f,j1,i1f) + A2(k1f,j0,i1f));
  a2int_z1 = A2(k2f,j0,i1f) + cy * (A2(k2f,j1,i1f)-A2(k2f,j0,i1f)) + cxf * (A2(k2f,j0,i2f)-A2(k2f,j0,i1f))
    + cy * cxf * (A2(k2f,j1f,i2f) - A2(k2f,j0,i2f) - A2(k2f,j1,i1f) + A1(k2f,j0,i1f));
  Real a2int11 = a2int_z0 + czf * (a2int_z1 - a2int_z0);

  // A3 (k0, j0f, i0f)
  Real a3int_z0 = A3(k0,j0f,i0f) + cyf * (A3(k0,j1f,i0f)-A3(k0,j0f,i0f)) + cxf * (A3(k0,j0f,i1f)-A3(k0,j0f,i0f))
    + cyf * cxf * (A3(k0,j1f,i1f) - A3(k0,j0f,i1f) - A3(k0,j1f,i0f) + A3(k0,j0f,i0f));
  Real a3int_z1 = A3(k1,j0f,i0f) + cyf * (A3(k1,j1f,i0f)-A3(k1,j0f,i0f)) + cxf * (A3(k1,j0f,i1f)-A3(k1,j0f,i0f))
    + cyf * cxf * (A3(k1,j1f,i1f) - A3(k1,j0f,i1f) - A3(k1,j1f,i0f) + A3(k1,j0f,i0f));
  Real a3int00 = a3int_z0 + cz * (a3int_z1 - a3int_z0);
  // A3 (k0, j1f, i0f)
  a3int_z0 = A3(k0,j1f,i0f) + cyf * (A3(k0,j2f,i0f)-A3(k0,j1f,i0f)) + cxf * (A3(k0,j1f,i1f)-A3(k0,j1f,i0f))
    + cyf * cxf * (A3(k0,j2f,i1f) - A3(k0,j1f,i1f) - A3(k0,j2f,i0f) + A3(k0,j1f,i0f));
  a3int_z1 = A3(k1,j1f,i0f) + cyf * (A3(k1,j2f,i0f)-A3(k1,j1f,i0f)) + cxf * (A3(k1,j1f,i1f)-A3(k1,j1f,i0f))
    + cyf * cxf * (A3(k1,j2f,i1f) - A3(k1,j1f,i1f) - A3(k1,j2f,i0f) + A3(k1,j1f,i0f));
  Real a3int10 = a3int_z0 + cz * (a3int_z1 - a3int_z0);
  // A3 (k0, j0f, i1f)
  a3int_z0 = A3(k0,j0f,i1f) + cyf * (A3(k0,j1f,i1f)-A3(k0,j0f,i1f)) + cxf * (A3(k0,j0f,i2f)-A3(k0,j0f,i1f)) 
    + cyf * cxf * (A3(k0,j1f,i2f) - A3(k0,j0f,i2f) - A3(k0,j1f,i1f) + A3(k0,j0f,i1f));
  a3int_z1 = A3(k1,j0f,i1f) + cyf * (A3(k1,j1f,i1f)-A3(k1,j0f,i1f)) + cxf * (A3(k1,j0f,i2f)-A3(k1,j0f,i1f)) 
    + cyf * cxf * (A3(k1,j1f,i2f) - A3(k1,j0f,i2f) - A3(k1,j1f,i1f) + A3(k1,j0f,i1f));
  Real a3int01 = a3int_z0 + cz * (a3int_z1 - a3int_z0);
  // A3 (k0, j1f, i1f)
  a3int_z0 = A3(k0,j1f,i1f) + cyf * (A3(k0,j2f,i1f)-A3(k0,j1f,i1f)) + cxf * (A3(k0,j1f,i2f)-A3(k0,j1f,i1f))
    + cyf * cxf * (A3(k0,j2f,i2f) - A3(k0,j1f,i2f) - A3(k0,j2f,i1f) + A3(k0,j1f,i1f));
  a3int_z1 = A3(k1,j1f,i1f) + cyf * (A3(k1,j2f,i1f)-A3(k1,j1f,i1f)) + cxf * (A3(k1,j1f,i2f)-A3(k1,j1f,i1f))
    + cyf * cxf * (A3(k1,j2f,i2f) - A3(k1,j1f,i2f) - A3(k1,j2f,i1f) + A3(k1,j1f,i1f));
  Real a3int11 = a3int_z0 + cz * (a3int_z1 - a3int_z0);

  *b1 = (a3int10-a3int00) - (a2int10-a2int00); // Bx (iavf, jav, kav) (dx = dy = dz, we will divide outside the function) 
  *b2 = (a1int10-a1int00) - (a3int01-a3int00); // By 
  *b3 = (a2int01-a2int00) - (a1int01-a1int00); // Bz 

  if (std::isnan(*b1 + *b2 + *b3)){
    std::cout << "indices: i = " << i0 << ", " << i1 << ", " << i2 << "\n";
    std::cout << "indices: j = " << j0 << ", " << j1 << ", " << j2 << "\n";
    std::cout << "indices: k = " << k0 << ", " << k1 << ", " << k2 << "\n";
    std::cout << "indices: if = " << i0f << ", " << i1f << ", " << i2f << "\n";
    std::cout << "indices: jf = " << j0f << ", " << j1f << ", " << j2f << "\n";
    std::cout << "indices: kf = " << k0f << ", " << k1f << ", " << k2f << "\n";
    std::cout << "Bx = " << *b1 << "\n";
    std::cout << "By = " << *b2 << "\n";
    std::cout << "Bz = " << *b3 << "\n";
    *b1 = 0. ; *b2 = 0. ; *b3 = 0.; // could this produce issues? 
    //    getchar();
  }

}
 */


Real int_linear(AthenaArray<Real> &u, int index, Real kest, Real jest, Real iest){
    // version for one foreindex
    int k0 = (int)floor(kest), j0 = (int)floor(jest), i0 = (int)floor(iest);
    int k1 = k0+1, j1 = j0+1, i1 = i0+1;
    
    Real cz = std::fmod(kest,k0), cy = std::fmod(jest,j0), cx = std::fmod(iest,i0);
    
    Real unew0 = u(index, k0,j0,i0) + cy * (u(index, k0,j1,i0) - u(index, k0,j0,i0)) +  cx * (u(index, k0,j0,i1) - u(index, k0,j0,i0))
    + cy * cx * (u(index, k0,j1,i1) - u(index, k0,j0,i1) - u(index, k0,j1,i0) + u(index, k0,j0,i0)); // 2D interpolation at z = 0
    Real unew1 = u(index, k1,j0,i0) + cy * (u(index, k1,j1,i0) - u(index, k1,j0,i0)) +  cx * (u(index, k1,j0,i1) - u(index, k1,j0,i0))
    + cy * cx * (u(index, k1,j1,i1) - u(index, k1,j0,i1) - u(index, k1,j1,i0) + u(index, k1,j0,i0)); // 2D interpolation at z = 1
    
    return unew0 + cz * (unew1 - unew0);
}

Real int_linear(AthenaArray<Real> &u, int index1, int index2, Real kest, Real jest, Real iest){
    // version for two foreindices
    int k0 = (int)floor(kest), j0 = (int)floor(jest), i0 = (int)floor(iest);
    int k1 = k0+1, j1 = j0+1, i1 = i0+1;
    Real cz = std::fmod(kest,(double)k0), cy = std::fmod(jest,(double)j0), cx = std::fmod(iest,(double)i0);

    Real u00 = u(index1, index2, k0,j0,i0);
    
    Real unew0 = u00 + cy * (u(index1, index2, k0,j1,i0) - u00) +  cx * (u(index1, index2, k0,j0,i1) - u00)
    + cy * cx * (u(index1, index2, k0,j1,i1) - u(index1, index2, k0,j0,i1) - u(index1, index2, k0,j1,i0) + u00); // 2D interpolation at z = 0
    u00 = u(index1, index2, k1,j0,i0);
    Real unew1 = u00 + cy * (u(index1, index2, k1,j1,i0) - u00) +  cx * (u(index1, index2, k1,j0,i1) - u00)
    + cy * cx * (u(index1, index2, k1,j1,i1) - u(index1, index2, k1,j0,i1) - u(index1, index2, k1,j1,i0) + u00); // 2D interpolation at z = 1
    
    return unew0 + cz * (unew1 - unew0);
}

 Real int_nearest(AthenaArray<Real> &u, int index1, int index2, Real kest, Real jest, Real iest){

    int k0 = (int)rint(kest), j0 = (int)rint(jest), i0 = (int)rint(iest);
   
    return u(index1, index2, k0,j0,i0);
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
	prim(IVY,k,ju+j,i) = std::max(prim(IVY,k,ju,i), 0.);
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
	prim(IVZ,kl-k,j,i) = std::min(prim(IVZ,kl,j,i), 0.);
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
	prim(IVZ,ku+k,j,i) = std::max(prim(IVZ,ku,j,i), 0.);
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



