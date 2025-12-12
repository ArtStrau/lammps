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
  double wx, wy, wz;
  double **torque = atom->torque;
  double **mu = atom->mu;
  double mux, muy, muz, mulen;

  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double dx, dy, dz;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    // ---------------- translational + angular-velocity noise (unchanged) ----------------

    if (Tp_2D) {
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

    } else {
      // 3D translation, 3D angular noise (Tp_2Drot not treated specially yet)
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

    // update positions & "velocities" as in original fix
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

    // ---------------- rotational update ----------------

    // dipole length (we keep this fixed)
    mulen = sqrt(mu[i][0]*mu[i][0] +
                 mu[i][1]*mu[i][1] +
                 mu[i][2]*mu[i][2]);

    if (mulen == 0.0) continue;  // undefined orientation

    // unit orientation u at time t
    mux = mu[i][0] / mulen;
    muy = mu[i][1] / mulen;
    muz = mu[i][2] / mulen;

    if (Tp_2D) {
      // 2D case: keep the original projection scheme for now

      // un-normalized Euler step u + dt (ω×u)
      double dux = (wy * muz - wz * muy) * dt;
      double duy = (wz * mux - wx * muz) * dt;
      double duz = (wx * muy - wy * mux) * dt;

      mu[i][0] = mux + dux;
      mu[i][1] = muy + duy;
      mu[i][2] = muz + duz;

      MathExtra::norm3(mu[i]);
      mu[i][0] *= mulen;
      mu[i][1] *= mulen;
      mu[i][2] *= mulen;

    } else {
      // 3D case: geometric integrator on S^2

      // angular velocity omega = (wx, wy, wz)
      // project onto tangent plane: omega_perp = omega - (omega·u) u
      double dot_ou = wx*mux + wy*muy + wz*muz;
      double wxp = wx - dot_ou * mux;
      double wyp = wy - dot_ou * muy;
      double wzp = wz - dot_ou * muz;

      double wperp2 = wxp*wxp + wyp*wyp + wzp*wzp;

      if (wperp2 == 0.0) {
        // omega is parallel to u -> no rotation of direction
        mu[i][0] = mulen * mux;
        mu[i][1] = mulen * muy;
        mu[i][2] = mulen * muz;

      } else {
        // rotation angle theta = |omega_perp| * dt
        double wperp = sqrt(wperp2);
        double theta = dt * wperp;

        // full Rodrigues rotation with axis n = omega_perp / |omega_perp|
        double nx = wxp / wperp;
        double ny = wyp / wperp;
        double nz = wzp / wperp;

        double c = cos(theta);
        double s = sin(theta);

        // u × n
        double cx = muy * nz - muz * ny;
        double cy = muz * nx - mux * nz;
        double cz = mux * ny - muy * nx;

        // Eq. (25): u' = cos(theta) u - sin(theta) (u × n)
        mu[i][0] = c * mux - s * cx;
        mu[i][1] = c * muy - s * cy;
        mu[i][2] = c * muz - s * cz;

        // renormalize via MathExtra and restore original magnitude
        MathExtra::norm3(mu[i]);
        mu[i][0] *= mulen;
        mu[i][1] *= mulen;
        mu[i][2] *= mulen;
      }
    }
  }
}
