// Rotational update for fix brownian/sphere/geom:
//
// This implements the Gaussian geometric integrator for 3D overdamped
// rotational Brownian motion on the unit sphere, as described in:
//   F. Höfling, A. V. Straube,
//   Phys. Rev. Research 7, 043034 (2025).
//   https://doi.org/10.1103/wzdn-29p4
//
// The translational part and the generation of angular velocities ω
// follow fix brownian/sphere; only the map ω → u(t+Δt) is changed
// from projection + renormalization to an exact rotation on S^2
// (Rodrigues formula with infinitesimal random rotations).

/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Originally modified from CG-DNA/fix_nve_dotc_langevin.cpp.

   Contributing author: Sam Cameron (University of Bristol)
------------------------------------------------------------------------- */

#include "fix_brownian_sphere_geom.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "math_extra.h"
#include "random_mars.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixBrownianSphereGeom::FixBrownianSphereGeom(LAMMPS *lmp, int narg, char **arg) :
    FixBrownianBase(lmp, narg, arg)
{
  if (gamma_t_eigen_flag || gamma_r_eigen_flag) {
    error->all(FLERR, "Illegal fix brownian/sphere command.");
  }

  if (!gamma_t_flag || !gamma_r_flag) error->all(FLERR, "Illegal fix brownian/sphere command.");
  if (!atom->mu_flag) error->all(FLERR, "Fix brownian/sphere requires atom attribute mu");
}

/* ---------------------------------------------------------------------- */

void FixBrownianSphereGeom::init()
{
  FixBrownianBase::init();

  g3 = g1 / gamma_r;
  g4 = g2 * sqrt(rot_temp / gamma_r);
  g1 /= gamma_t;
  g2 *= sqrt(temp / gamma_t);
}

/* ---------------------------------------------------------------------- */

void FixBrownianSphereGeom::initial_integrate(int /*vflag */)
{
  if (domain->dimension == 2) {
    if (!noise_flag) {
      initial_integrate_templated<0, 0, 1, 0>();
    } else if (gaussian_noise_flag) {
      initial_integrate_templated<0, 1, 1, 0>();
    } else {
      initial_integrate_templated<1, 0, 1, 0>();
    }
  } else if (planar_rot_flag) {
    if (!noise_flag) {
      initial_integrate_templated<0, 0, 0, 1>();
    } else if (gaussian_noise_flag) {
      initial_integrate_templated<0, 1, 0, 1>();
    } else {
      initial_integrate_templated<1, 0, 0, 1>();
    }
  } else {
    if (!noise_flag) {
      initial_integrate_templated<0, 0, 0, 0>();
    } else if (gaussian_noise_flag) {
      initial_integrate_templated<0, 1, 0, 0>();
    } else {
      initial_integrate_templated<1, 0, 0, 0>();
    }
  }
}


/* ---------------------------------------------------------------------- */

template <int Tp_UNIFORM, int Tp_GAUSS, int Tp_2D, int Tp_2Drot>
void FixBrownianSphereGeom::initial_integrate_templated()
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double **torque = atom->torque;
  double **mu = atom->mu;

  double dx, dy, dz;
  double wx, wy, wz;
  double mux, muy, muz, mulen;

  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {

      // --- translational + angular-velocity noise ---

      if (Tp_2D) {

        // 2D: Euler + projection scheme (keep original code and style)
        dz = 0.0;
        wx = wy = 0.0;

        if (Tp_UNIFORM) {
          dx = dt * (g1 * f[i][0] + g2 * (rng->uniform() - 0.5));
          dy = dt * (g1 * f[i][1] + g2 * (rng->uniform() - 0.5));
          wz = (rng->uniform() - 0.5) * g4;
        } else if (Tp_GAUSS) {
          dx = dt * (g1 * f[i][0] + g2 * rng->gaussian());
          dy = dt * (g1 * f[i][1] + g2 * rng->gaussian());
          wz = rng->gaussian() * g4;
        } else {
          dx = dt * g1 * f[i][0];
          dy = dt * g1 * f[i][1];
          wz = 0.0;
        }

      } else if (Tp_2Drot) {

        // 3D translation, 2D (planar) rotation about z: wx=wy=0, only wz noise
        wx = wy = 0.0;

        if (Tp_UNIFORM) {
          dx = dt * (g1 * f[i][0] + g2 * (rng->uniform() - 0.5));
          dy = dt * (g1 * f[i][1] + g2 * (rng->uniform() - 0.5));
          dz = dt * (g1 * f[i][2] + g2 * (rng->uniform() - 0.5));
          wz = (rng->uniform() - 0.5) * g4;
        } else if (Tp_GAUSS) {
          dx = dt * (g1 * f[i][0] + g2 * rng->gaussian());
          dy = dt * (g1 * f[i][1] + g2 * rng->gaussian());
          dz = dt * (g1 * f[i][2] + g2 * rng->gaussian());
          wz = rng->gaussian() * g4;
        } else {
          dx = dt * g1 * f[i][0];
          dy = dt * g1 * f[i][1];
          dz = dt * g1 * f[i][2];
          wz = 0.0;
        }

      } else {

        // full 3D translation + full 3D angular noise

        if (Tp_UNIFORM) {
          dx = dt * (g1 * f[i][0] + g2 * (rng->uniform() - 0.5));
          dy = dt * (g1 * f[i][1] + g2 * (rng->uniform() - 0.5));
          dz = dt * (g1 * f[i][2] + g2 * (rng->uniform() - 0.5));
          wx = (rng->uniform() - 0.5) * g4;
          wy = (rng->uniform() - 0.5) * g4;
          wz = (rng->uniform() - 0.5) * g4;
        } else if (Tp_GAUSS) {
          dx = dt * (g1 * f[i][0] + g2 * rng->gaussian());
          dy = dt * (g1 * f[i][1] + g2 * rng->gaussian());
          dz = dt * (g1 * f[i][2] + g2 * rng->gaussian());
          wx = rng->gaussian() * g4;
          wy = rng->gaussian() * g4;
          wz = rng->gaussian() * g4;
        } else {
          dx = dt * g1 * f[i][0];
          dy = dt * g1 * f[i][1];
          dz = dt * g1 * f[i][2];
          wx = wy = wz = 0.0;
        }
      }

      // update positions & velocities as in original fix (FixBrownianSphere)
      // (velocities are irrelevant for overdamped dynamics)
      x[i][0] += dx;
      v[i][0] = dx / dt;

      x[i][1] += dy;
      v[i][1] = dy / dt;

      x[i][2] += dz;
      v[i][2] = dz / dt;

      // add deterministic torque contribution (same as FixBrownianSphere)
      wx += g3 * torque[i][0];
      wy += g3 * torque[i][1];
      wz += g3 * torque[i][2];

      // --- rotational update: geometric integrator for 3D angular motion ----

      // keep dipole length fixed (as in original fix):
      // store length of dipole as we need to convert it to a unit vector and
      // then back again

      mulen = sqrt(mu[i][0]*mu[i][0] + mu[i][1]*mu[i][1] + mu[i][2]*mu[i][2]);

      // avoid division by zero (not tested in original code)
      if (mulen == 0) continue;
      // note: mulen = 0 is pathologic (causes division by zero)
      // two possible alternatives:
      // 1. assertion (generates exception and stops in Debug mode, not seen in Release)
      // assert(mulen > 0);  // requires in the header "#include <cassert>"
      // 2. stop by error:
      // if (mulen == 0.0)
      //   error->one(FLERR,"Fix brownian/sphere/geom requires nonzero dipole moment for all atoms in group");

      // unit vector at time t
      mux = mu[i][0] / mulen;
      muy = mu[i][1] / mulen;
      muz = mu[i][2] / mulen;

      if (Tp_2D) {

        // 2D angular motion: Euler + projection scheme (keep original code and style)

        // un-normalised unit vector at time t + dt
        mu[i][0] = mux + (wy * muz - wz * muy) * dt;
        mu[i][1] = muy + (wz * mux - wx * muz) * dt;
        mu[i][2] = muz + (wx * muy - wy * mux) * dt;

        // original comment: normalisation introduces the stochastic drift term 
        // due to changing from Stratonovich to Ito interpretation;
        // normalization issue is discussed in reference by Höfling & Straube (2025)
        MathExtra::norm3(mu[i]);

        // multiply by original magnitude to restore original dipole length
        mu[i][0] *= mulen;
        mu[i][1] *= mulen;
        mu[i][2] *= mulen;

      } else if (Tp_2Drot) {

        // planar (2D) rotation: same Euler + projection scheme (wx=wy=0 already)

        // un-normalised unit vector at time t + dt
        mu[i][0] = mux + (wy * muz - wz * muy) * dt;
        mu[i][1] = muy + (wz * mux - wx * muz) * dt;
        mu[i][2] = muz + (wx * muy - wy * mux) * dt;

        // normalize and restore original dipole length
        MathExtra::norm3(mu[i]);
        mu[i][0] *= mulen;
        mu[i][1] *= mulen;
        mu[i][2] *= mulen;

      } else {

        // 3D angular motion: geometric integrator on S^2 (Hofling & Straube, PRR 7, 043034 (2025))
        //
        // Let u = mu/|mu| be the unit orientation at time t.
        // The effective angular velocity is omega = (wx,wy,wz) (noise + deterministic torque).
        // Only the component perpendicular to u changes the direction:
        //
        //   omega_perp = omega - dot(omega,u) u .
        //
        // The finite rotation increment in the paper can be identified as
        //
        //   dOmega = omega_perp * dt ,
        //
        // with rotation angle theta = |dOmega| = dt*|omega_perp| and axis n = dOmega/|dOmega|.
        // Since n is perpendicular to u by construction, the Rodrigues formula simplifies to
        //
        //   u_new = cos(theta) u - sin(theta) (u x n) .
        //
        // Finally we renormalize (roundoff) and restore the original dipole magnitude |mu|.

        // omega_perp = omega - (omega·u) u
        // unit vector (u) at time t is given by (mux, muy, muz)
        double dot_wu = wx*mux + wy*muy + wz*muz; // dot product (omega·u)
        double wxp = wx - dot_wu * mux;
        double wyp = wy - dot_wu * muy;
        double wzp = wz - dot_wu * muz;

        // wperp = |omega_perp|  =>  theta = dt*wperp = |ΔΩ|
        double wperp = sqrt(wxp*wxp + wyp*wyp + wzp*wzp);

        // note: if omega_perp = 0, exact map leaves u unchanged (do nothing)
        if (wperp > 0) {
          double theta = dt * wperp;

          // axis n = omega_perp / |omega_perp|
          double nx = wxp / wperp;
          double ny = wyp / wperp;
          double nz = wzp / wperp;

          double c = cos(theta);
          double s = sin(theta);

          // u × n
          double cx = muy * nz - muz * ny;
          double cy = muz * nx - mux * nz;
          double cz = mux * ny - muy * nx;

          // by construction, n is perpendicular to u, (n·u)=0 and
          // u_new = cos(theta) u - sin(theta) (u × n)
          mu[i][0] = c * mux - s * cx;
          mu[i][1] = c * muy - s * cy;
          mu[i][2] = c * muz - s * cz;

          // remove roundoff drift and restore original dipole magnitude
          MathExtra::norm3(mu[i]);
          mu[i][0] *= mulen;
          mu[i][1] *= mulen;
          mu[i][2] *= mulen;
        }
      }
    }
  }
}
