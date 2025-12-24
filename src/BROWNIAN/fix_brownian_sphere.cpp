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

      mulen = MathExtra::len3(mu[i]);

      // assertion generates exception and stops in Debug mode, not seen in Release
      assert(mulen > 0);

      // unit vector u = mu / |mu| at time t
      double u[3];
      MathExtra::scale3(1.0 / mulen, mu[i], u);

      // effective angular velocity omega is denoted as w = (wx, wy, wz) (deterministic torque + noise)
      double w[3] = {wx, wy, wz};

      if (Tp_ROTGEOM) {

        // --- rotational update: geometric integrator for 3D angular motion ----
        // For Tp_2D and Tp_2Drot, we currently use two separate projection placeholders

        if (Tp_2D) {

          // 2D case: projection integrator placeholder

          // cross-product wxu = w × u
          double wxu[3];
          MathExtra::cross3(w, u, wxu);

          // un-normalized unit vector at time t + dt: u <- u + dt (w × u)
          MathExtra::scaleadd3(dt, wxu, u, u);

          // normalize to |u| = 1 and rescale to |mu| = mulen
          MathExtra::snormalize3(mulen, u, mu[i]);

        } else if (Tp_2Drot) {

          // planar (2D) rotation case: projection integrator placeholder

          // cross-product wxu = w × u
          double wxu[3];
          MathExtra::cross3(w, u, wxu);

          // un-normalized unit vector at time t + dt: u <- u + dt (w × u)
          MathExtra::scaleadd3(dt, wxu, u, u);

          // normalize to |u| = 1 and rescale to |mu| = mulen
          MathExtra::snormalize3(mulen, u, mu[i]);

        } else {

          // ----------------------------------------------------------------------
          // 3D case, rotational update: geometric integrator for overdamped rotational Brownian motion
          // Reference: F. Hoefling & A. V. Straube, Phys. Rev. Research 7, 043034 (2025); DOI: 10.1103/wzdn-29p4
          //
          // Paper -> code notation:
          //   u            : unit orientation, u = mu / |mu|
          //   omega        : effective angular velocity (noise + deterministic torque) -> w
          //   omega_perp   : omega - dot(omega, u) u  (tangent component)              -> wperp
          //   dOmega       : omega_perp * dt                                           -> (dt*wperp)
          //   theta        : |dOmega| = dt * |omega_perp|                              -> theta
          //   n            : dOmega / |dOmega| = omega_perp / |omega_perp|             -> n
          //
          // Update (Rodrigues; simplified since dot(n, u) = 0 by construction):
          //   u_new = cos(theta) u - sin(theta) (u x n)
          // Renormalize u_new to suppress roundoff drift, then restore |mu|.
          // ----------------------------------------------------------------------

          // dot product dot(omega, u) = dot(w, u)
          double dot_wu = MathExtra::dot3(w, u);

          // omega_perp = omega - dot(omega, u) u
          double wperp[3];
          MathExtra::scaleadd3(-dot_wu, u, w, wperp);

          // wperplen = |omega_perp| = |wperp|
          double wperplen = MathExtra::len3(wperp);

          // note: if omega_perp = 0, exact map leaves u unchanged (do nothing)
          if (wperplen > 0) {

            // rotation angle theta = |omega_perp| * dt = |dOmega|
            double theta = dt * wperplen;

            double c = cos(theta);
            double s = sin(theta);

            // axis n = omega_perp / |omega_perp|
            double n[3];
            MathExtra::scale3(1.0 / wperplen, wperp, n);

            // cross-product u × n 
            double uxn[3];
            MathExtra::cross3(u, n, uxn);

            // new rotated orientation u <- cos(theta) u - sin(theta) (u × n)
            MathExtra::scaleadd3(c, u, -s, uxn, u);

            // remove roundoff drift and restore original dipole magnitude
            MathExtra::snormalize3(mulen, u, mu[i]);
          }

        }

      } else {

        // --- rotational update: projection integrator (original) ----

        // cross-product wxu = w × u
        double wxu[3];
        MathExtra::cross3(w, u, wxu);

        // un-normalized unit vector at time t + dt: u <- u + dt (w × u)
        MathExtra::scaleadd3(dt, wxu, u, u);

        // normalize to |u| = 1 and rescale to |mu| = mulen
        //   original comment: "normalisation introduces the stochastic drift term due to changing from
        //   Stratonovich to Ito interpretation" -- the issue is clarified in the reference
        MathExtra::snormalize3(mulen, u, mu[i]);

      }

    }
  }
}
