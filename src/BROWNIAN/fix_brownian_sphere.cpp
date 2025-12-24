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

#include "fix_brownian_sphere.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "math_extra.h"
#include "random_mars.h"

#include <cmath>
#include <cassert>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixBrownianSphere::FixBrownianSphere(LAMMPS *lmp, int narg, char **arg) :
    FixBrownianBase(lmp, narg, arg)
{
  if (gamma_t_eigen_flag || gamma_r_eigen_flag) {
    error->all(FLERR, "Illegal fix brownian/sphere command.");
  }

  if (!gamma_t_flag || !gamma_r_flag) error->all(FLERR, "Illegal fix brownian/sphere command.");
  if (!atom->mu_flag) error->all(FLERR, "Fix brownian/sphere requires atom attribute mu");
}

/* ---------------------------------------------------------------------- */

void FixBrownianSphere::init()
{
  FixBrownianBase::init();

  g3 = g1 / gamma_r;
  g4 = g2 * sqrt(rot_temp / gamma_r);
  g1 /= gamma_t;
  g2 *= sqrt(temp / gamma_t);
}

/* ---------------------------------------------------------------------- */

void FixBrownianSphere::initial_integrate(int /*vflag */)
{
  if (rot_style == ROT_GEOMETRIC) {

    if (domain->dimension == 2) {

      // 2D translation + 2D rotation:
      // Fallback to projection integrator (Tp_ROTGEOM = 0)
      if (!noise_flag) {
        initial_integrate_templated<0, 0, 0, 1, 0>();
      } else if (gaussian_noise_flag) {
        initial_integrate_templated<0, 0, 1, 1, 0>();
      } else {
        initial_integrate_templated<0, 1, 0, 1, 0>();
      }

    } else if (planar_rot_flag) {

      // 3D translation + planar_rotation:
      // Fallback to projection integrator (Tp_ROTGEOM = 0)
      if (!noise_flag) {
        initial_integrate_templated<0, 0, 0, 0, 1>();
      } else if (gaussian_noise_flag) {
        initial_integrate_templated<0, 0, 1, 0, 1>();
      } else {
        initial_integrate_templated<0, 1, 0, 0, 1>();
      }

    } else {

      // Full 3D: 3D translation + 3D rotation:
      // Geometric integrator (Tp_ROTGEOM = 1)
      if (!noise_flag) {
        initial_integrate_templated<1, 0, 0, 0, 0>();
      } else if (gaussian_noise_flag) {
        initial_integrate_templated<1, 0, 1, 0, 0>();
      } else {
        initial_integrate_templated<1, 1, 0, 0, 0>();
      }
    }

  } else {

    // Projection integrator (original behavior)
    if (domain->dimension == 2) {

      if (!noise_flag) {
        initial_integrate_templated<0, 0, 0, 1, 0>();
      } else if (gaussian_noise_flag) {
        initial_integrate_templated<0, 0, 1, 1, 0>();
      } else {
        initial_integrate_templated<0, 1, 0, 1, 0>();
      }

    } else if (planar_rot_flag) {

      if (!noise_flag) {
        initial_integrate_templated<0, 0, 0, 0, 1>();
      } else if (gaussian_noise_flag) {
        initial_integrate_templated<0, 0, 1, 0, 1>();
      } else {
        initial_integrate_templated<0, 1, 0, 0, 1>();
      }

    } else {

      if (!noise_flag) {
        initial_integrate_templated<0, 0, 0, 0, 0>();
      } else if (gaussian_noise_flag) {
        initial_integrate_templated<0, 0, 1, 0, 0>();
      } else {
        initial_integrate_templated<0, 1, 0, 0, 0>();
      }

    }

  }
}

/* ---------------------------------------------------------------------- */

template <int Tp_ROTGEOM, int Tp_UNIFORM, int Tp_GAUSS, int Tp_2D, int Tp_2Drot>
void FixBrownianSphere::initial_integrate_templated()
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double wx, wy, wz;
  double **torque = atom->torque;
  double **mu = atom->mu;
  double mux, muy, muz, mulen;

  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double dx, dy, dz;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (Tp_2D) {
        dz = 0;
        wx = wy = 0;
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
          wz = 0;
        }
      } else if (Tp_2Drot) {
        wx = wy = 0;
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
          wz = 0;
        }
      } else {
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
          wx = wy = wz = 0;
        }
      }

      x[i][0] += dx;
      v[i][0] = dx / dt;

      x[i][1] += dy;
      v[i][1] = dy / dt;

      x[i][2] += dz;
      v[i][2] = dz / dt;

      wx += g3 * torque[i][0];
      wy += g3 * torque[i][1];
      wz += g3 * torque[i][2];

      // store length of dipole as we need to convert it to a unit vector and
      // then back again

      mulen = sqrt(mu[i][0] * mu[i][0] + mu[i][1] * mu[i][1] + mu[i][2] * mu[i][2]);

      // assertion generates exception and stops in Debug mode, not seen in Release
      assert(mulen > 0);

      // unit vector at time t
      mux = mu[i][0] / mulen;
      muy = mu[i][1] / mulen;
      muz = mu[i][2] / mulen;

      if (Tp_ROTGEOM) {

        // --- rotational update: geometric integrator for 3D angular motion ----
        // For Tp_2D and Tp_2Drot, we currently use two separate projection placeholders

        if (Tp_2D) {

          // 2D case: projection integrator placeholder

          // un-normalised unit vector at time t + dt
          mu[i][0] = mux + (wy * muz - wz * muy) * dt;
          mu[i][1] = muy + (wz * mux - wx * muz) * dt;
          mu[i][2] = muz + (wx * muy - wy * mux) * dt;

          // normalize unit vector
          MathExtra::norm3(mu[i]);

          // multiply by original magnitude to obtain dipole of same length
          mu[i][0] = mu[i][0] * mulen;
          mu[i][1] = mu[i][1] * mulen;
          mu[i][2] = mu[i][2] * mulen;

        } else if (Tp_2Drot) {

          // planar (2D) rotation case: projection integrator placeholder

          // un-normalised unit vector at time t + dt
          mu[i][0] = mux + (wy * muz - wz * muy) * dt;
          mu[i][1] = muy + (wz * mux - wx * muz) * dt;
          mu[i][2] = muz + (wx * muy - wy * mux) * dt;

          // normalize unit vector
          MathExtra::norm3(mu[i]);

          // multiply by original magnitude to obtain dipole of same length
          mu[i][0] = mu[i][0] * mulen;
          mu[i][1] = mu[i][1] * mulen;
          mu[i][2] = mu[i][2] * mulen;

        } else {

          // full 3D: geometric integrator on S^2 [Hoefling & Straube, PRR 7, 043034 (2025)]

          // Let u = mu/|mu| be the unit orientation at time t.
          // The effective angular velocity is omega = (wx, wy, wz) (noise + deterministic torque).
          // Only the component perpendicular to u changes the direction:
          //
          //   omega_perp = omega - dot(omega, u) u.
          //
          // The finite rotation increment in the paper can be identified as
          //
          //   dOmega = omega_perp * dt,
          //
          // with rotation angle theta = |dOmega| = dt*|omega_perp| and axis n = dOmega/|dOmega|.
          // Since n is perpendicular to u by construction, the Rodrigues formula simplifies to
          //
          //   u_new = cos(theta) u - sin(theta) (u x n).
          //
          // Finally we renormalize (roundoff) and restore the original dipole magnitude |mu|.

          // dot product dot(omega, u), unit vector (u) at time t is given by (mux, muy, muz)
          double dot_wu = wx*mux + wy*muy + wz*muz; 

          // omega_perp is given by (wxp, wyp, wzp)
          double wxp = wx - dot_wu * mux;
          double wyp = wy - dot_wu * muy;
          double wzp = wz - dot_wu * muz;

          // wperp = |omega_perp| and hence theta = dt * wperp = |dOmega|
          double wperp = sqrt(wxp*wxp + wyp*wyp + wzp*wzp);

          // note: if omega_perp = 0, exact map leaves u unchanged (do nothing)
          if (wperp > 0) {

            // rotation angle theta = |omega_perp| * dt = |dOmega|
            double theta = dt * wperp;

            // axis n = omega_perp / |omega_perp|
            double nx = wxp / wperp;
            double ny = wyp / wperp;
            double nz = wzp / wperp;

            double c = cos(theta);
            double s = sin(theta);

            // cross-product u × n = (cx, cy, cz)
            double cx = muy * nz - muz * ny;
            double cy = muz * nx - mux * nz;
            double cz = mux * ny - muy * nx;

            // by construction, n is perpendicular to u, dot(n, u) = 0 and
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

      } else {

        // --- rotational update: projection integrator (original) ----

        // un-normalised unit vector at time t + dt
        mu[i][0] = mux + (wy * muz - wz * muy) * dt;
        mu[i][1] = muy + (wz * mux - wx * muz) * dt;
        mu[i][2] = muz + (wx * muy - wy * mux) * dt;

        // normalisation introduces the stochastic drift term due to changing from
        // Stratonovich to Ito interpretation
        MathExtra::norm3(mu[i]);

        // multiply by original magnitude to obtain dipole of same length
        mu[i][0] = mu[i][0] * mulen;
        mu[i][1] = mu[i][1] * mulen;
        mu[i][2] = mu[i][2] * mulen;

      }

    }
  }
}
