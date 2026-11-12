"""Independent change-of-variables and numerical-domain checks; no OpenMM."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmm import SamplingCoordinates


def mapping():
    return SamplingCoordinates(
        reference=jnp.array([.12, .4, -np.pi]),
        scale=jnp.array([.006, .12, 1.]),
        lower=jnp.array([.02, -1., -np.pi]),
        upper=jnp.array([np.inf, 2., np.pi]), periodic_indices=(2,))


def test_center_scale_roundtrip_and_absolute_jacobian():
    """AD of the actual map independently anchors the analytic log determinant."""
    transform = mapping()
    origin = jnp.array([0., 0., -np.pi])
    np.testing.assert_allclose(transform.to_physical(origin), transform.reference, atol=1e-15)
    np.testing.assert_allclose(jax.jacfwd(transform.to_physical)(origin),
                               np.diag([.006, .12, 1.]), atol=1e-15)
    z = jnp.array([.7, -.8, 3.05])
    physical = transform.to_physical(z)
    np.testing.assert_allclose(physical[0], .02 + .1*np.exp(.06*.7), atol=1e-15)
    np.testing.assert_allclose(transform.to_sampling(physical), z, atol=1e-13)
    jacobian = jax.jacfwd(transform.to_physical)(z)
    np.testing.assert_allclose(transform.log_abs_det_jacobian(z),
                               np.linalg.slogdet(jacobian)[1], atol=1e-13)


def test_exponential_density_stays_normalized():
    """Independent quadrature catches an omitted, reversed or doubled determinant."""
    transform = SamplingCoordinates(jnp.array([.2]), jnp.array([.08]),
                                    jnp.array([0.]), jnp.array([np.inf]))
    z = jnp.linspace(-40., 40., 4097)[:, None]
    values = jax.jit(lambda t, v: t.pullback(lambda q: -q[..., 0], v))(transform, z)
    density, grid = np.exp(values), np.asarray(z[:, 0])
    integral = np.sum((density[1:]+density[:-1])/2*np.diff(grid))
    # Missing physical mass at the lower endpoint is .2*exp(-16) < 2.3e-8;
    # the smooth trapezoid error is much smaller on this grid.
    np.testing.assert_allclose(integral, 1., atol=3e-8, rtol=0)


def test_batches_placeholders_and_parameter_derivatives():
    transform = mapping()
    points = jnp.array([[.7, -.8, 3.05], [-.4, .2, -2.7]])
    fn = jax.jit(lambda t, x: t.to_physical(x))
    np.testing.assert_allclose(fn(transform, points),
                               jax.vmap(lambda t, x: t.to_physical(x), in_axes=(None, 0))(transform, points))
    assert fn(transform, jnp.empty((2, 0, 3))).shape == (2, 0, 3)
    derivative = jax.grad(lambda s: replace(transform, scale=s).log_abs_det_jacobian(points[0]))(transform.scale)
    np.testing.assert_allclose(derivative[0], 1/.006 + .7/.1, rtol=1e-12)
    assert np.all(np.isfinite(derivative))


def test_rejected_map_events_have_zero_gradients_without_poisoning_valid_rows():
    """Overflow, saturation, nonfinite input and noncanonical circles are rejected."""
    transform = mapping()
    valid = jnp.array([.2, -.4, 2.9])
    points = jnp.stack([valid, valid.at[0].set(1e6), valid.at[0].set(-1e6),
                        valid.at[1].set(1e6), valid.at[1].set(-1e6),
                        valid.at[2].set(np.pi), valid.at[0].set(np.nan),
                        valid.at[1].set(np.inf)])
    density = lambda q: -.7*jnp.sum(q*q, axis=-1)
    evaluate = jax.jit(jax.vmap(jax.value_and_grad(lambda z: transform.pullback(density, z))))
    values, gradients = evaluate(points)
    assert np.isfinite(values[0]) and np.all(np.isfinite(gradients[0]))
    np.testing.assert_array_equal(values[1:], -np.inf)
    np.testing.assert_array_equal(gradients[1:], 0.)
    for point in points[1:]:
        with pytest.raises(ValueError, match="representable physical domain"):
            transform.to_physical(point)


def test_callback_support_suppresses_map_gradient_and_bad_results_raise():
    transform = mapping()
    point = jnp.array([.2, -.4, 2.9])
    call = lambda z: transform.pullback(lambda q: jnp.full(q.shape[:-1], -jnp.inf), z)
    value, gradient = jax.jit(jax.value_and_grad(call))(point)
    assert value == -np.inf
    np.testing.assert_array_equal(gradient, 0.)
    with pytest.raises(ValueError, match="must have shape"):
        transform.pullback(lambda q: q, point)
    with pytest.raises(ValueError, match="NaN or"):
        transform.pullback(lambda q: jnp.array(np.nan), point)
    with pytest.raises(TypeError, match="callable"):
        transform.pullback(None, point)


@pytest.mark.parametrize("changes,message", [
    ({"scale": jnp.array([.006, 0., 1.])}, "positive and finite"),
    ({"scale": jnp.array([.006, .12, 2.])}, "unit scale"),
    ({"scale": jnp.ones(1)}, "expected shape"),
    ({"reference": jnp.array([.02, .4, 0.])}, "inside the physical domain"),
    ({"lower": jnp.array([0., -np.inf, -np.pi])}, "finite lower"),
    ({"upper": jnp.array([np.inf, -2., np.pi])}, "upper > lower"),
    ({"periodic_indices": (2, 2)}, "distinct"),
    ({"periodic_indices": (3,)}, "must be in"),
])
def test_constructor_rejects_misconfiguration(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(mapping(), **changes)


def test_boundary_inverse_shape_and_dtype_errors():
    transform = mapping()
    for bad in [jnp.array([.02, .4, 0.]), jnp.array([.12, 2., 0.]),
                jnp.array([.12, .4, np.pi])]:
        with pytest.raises(ValueError, match="Physical event"):
            transform.to_sampling(bad)
    with pytest.raises(ValueError, match="trailing shape"):
        jax.jit(lambda x: transform.to_physical(x))(jnp.zeros(2))
    with pytest.raises(TypeError, match="float64"):
        transform.to_physical(jnp.zeros(3, dtype=jnp.float32))
    with pytest.raises(TypeError, match="tuple of integer"):
        replace(transform, periodic_indices=(True,))
