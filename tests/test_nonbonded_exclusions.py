"""Excluded-pair singularities must not contaminate real interactions."""

from dataclasses import replace
import math

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmm.energy import nonbonded_energy
from jaxmm.extract import NonbondedParams


def parameters():
    return NonbondedParams(
        charges=jnp.array([.4, -.7, .3]), sigmas=jnp.array([.25, .31, .27]),
        epsilons=jnp.array([.2, .4, .3]), n_atoms=3,
        exclusion_pairs=jnp.array([[0, 1]], dtype=jnp.int32),
        exception_pairs=jnp.array([[1, 2]], dtype=jnp.int32),
        exception_chargeprod=jnp.array([-.08]), exception_sigma=jnp.array([.19]),
        exception_epsilon=jnp.array([.11]))


def scalar_reference(x, cutoff=None, switch=None):
    # Only ordinary pair (0,2) and exception (1,2) exist. Explicit constants
    # avoid importing production pair formulas or parsing their parameters.
    def pair(i, j, sigma, epsilon, chargeprod, ordinary):
        r = float(np.linalg.norm(x[i]-x[j]))
        if ordinary and cutoff is not None and r >= cutoff:
            return 0.
        lj = 4*epsilon*((sigma/r)**12-(sigma/r)**6)
        if ordinary and switch is not None and r > switch:
            a = (r-switch)/(cutoff-switch)
            lj *= 1-10*a**3+15*a**4-6*a**5
        return lj+138.93545764438198*chargeprod/r
    return pair(0, 2, .26, math.sqrt(.06), .12, True)+pair(1, 2, .19, .11, -.08, False)


@pytest.mark.parametrize('distance,cutoff,switch', [(1.e-9, None, None), (0., None, None),
                                                   (2., .7, None), (1.e-9, 1.4, .8)])
def test_excluded_pair_never_enters_sum_or_gradient(distance, cutoff, switch):
    p = replace(parameters(), cutoff=cutoff, switch_distance=switch)
    x = np.array([[0., 0., 0.], [distance, 0., 0.], [.9, .3, -.2]])
    evaluate = jax.jit(jax.value_and_grad(lambda y: nonbonded_energy(y, p)))
    value, grad = evaluate(jnp.asarray(x))
    np.testing.assert_allclose(value, scalar_reference(x, cutoff, switch), rtol=1e-13, atol=1e-12)
    # Central differences of the independent, nonsingular two-pair expression.
    expected = np.empty_like(x)
    h = 1.e-5
    for i in range(3):
        for j in range(3):
            plus, minus = x.copy(), x.copy()
            plus[i, j] += h
            minus[i, j] -= h
            expected[i, j] = (scalar_reference(plus, cutoff, switch)-scalar_reference(minus, cutoff, switch))/(2*h)
    np.testing.assert_allclose(grad, expected, rtol=1e-8, atol=1e-7)
