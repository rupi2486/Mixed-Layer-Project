import numpy as np
import gsw

# -----------------------------
# 1. Create profiles
# -----------------------------
def create_profiles():
    z = np.linspace(0, 1000, 200)

    # mixed layer + thermocline (better test case)
    T = np.piecewise(
        z,
        [z < 100, z >= 100],
        [lambda z: 20, lambda z: 20 - 0.02*(z-100)]
    )

    S = np.piecewise(
        z,
        [z < 100, z >= 100],
        [lambda z: 34.5, lambda z: 34.5 + 0.002*(z-100)]
    )

    return z, T, S


# -----------------------------
# 2. Density
# -----------------------------
def density_profile(T, S, z, lat=40, lon=0):
    p = gsw.p_from_z(-z, lat)
    SA = gsw.SA_from_SP(S, p, lon, lat)
    CT = gsw.CT_from_t(SA, T, p)
    return gsw.rho(SA, CT, p)


# -----------------------------
# 3. Gradient
# -----------------------------
def density_gradient(z, rho):
    return np.gradient(rho, z)


# -----------------------------
# 4. Mixed Layer Depth
# -----------------------------
def find_mld(z, drho_dz, threshold=0.01):
    for zi, grad in zip(z, drho_dz):
        if abs(grad) > threshold:
            return zi
    return None
