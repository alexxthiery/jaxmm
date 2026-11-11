"""Physical volume, support and packed-coordinate contracts without OpenMM."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxmm
from test_coordinates import ALDP_XYZ, _tiny_params, _tiny_zmatrix


def geometry():
    transform = jaxmm.InternalCoordinates(_tiny_zmatrix())
    q = jnp.array([0.13, 0.17, 0.16, 1.7, 2.1, -0.8])
    return transform, q


@pytest.mark.parametrize("molecule", ["chain", "alanine"])
def test_physical_volume_against_full_cartesian_jacobian(molecule):
    """Dropping the frame factor fails a full 12D change of variables.

    Add three infinitesimal rotation and three translation coordinates at the
    identity. Haar density in these rotation coordinates is one there, so the
    square determinant independently gives the physical shape-volume factor.
    """
    transform, q = geometry()
    if molecule == "alanine":
        transform = jaxmm.InternalCoordinates(jaxmm.aldp_zmatrix())
        q = transform.to_internal(jnp.asarray(ALDP_XYZ))
    d = transform.dim
    def full(v):
        x = transform.to_cartesian(v[:d])
        return (x + jnp.cross(v[d:d+3], x) + v[d+3:d+6]).ravel()
    jac = jax.jacfwd(full)(jnp.concatenate((q, jnp.zeros(6))))
    expected = jnp.linalg.slogdet(jac)[1]
    np.testing.assert_allclose(transform.log_volume(q), expected, atol=1e-12)
    assert abs(float(transform.fixed_frame_log_det(q) - expected)) > 1
    b, a, _ = transform.unpack(q)
    correction = 2*jnp.log(b[0]) + jnp.log(b[1]) + jnp.log(jnp.sin(a[0]))
    np.testing.assert_allclose(transform.log_volume(q)-transform.fixed_frame_log_det(q),
                               correction, atol=1e-12)


def test_physical_density_and_parameter_derivative():
    """The Boltzmann evaluator must use physical volume and the supplied T."""
    transform, q = geometry()
    ff = _tiny_params()
    temperature = 427.0
    expected = (-jaxmm.total_energy(transform.to_cartesian(q), ff) /
                (jaxmm.KB*temperature) + transform.log_volume(q))
    evaluate = lambda v, t: jaxmm.log_boltzmann_internal(
        transform.z_matrix, *transform.unpack(v), ff, t)
    np.testing.assert_allclose(jax.jit(evaluate)(q, temperature), expected, atol=1e-10)
    derivative = jax.grad(lambda t: evaluate(q, t))(temperature)
    np.testing.assert_allclose(derivative,
        jaxmm.total_energy(transform.to_cartesian(q), ff)/(jaxmm.KB*temperature**2), rtol=1e-12)


@pytest.mark.parametrize("index,value", [
    (0, 0.), (1, -0.1), (2, np.inf), (0, np.nan),
    (3, 0.), (4, np.pi), (4, -1.), (5, np.pi), (5, np.inf), (5, np.nan),
])
def test_off_support_is_zero_probability_without_nan_gradients(index, value):
    """Reject invalid events before energy/log evaluation, including torsions."""
    transform, q = geometry()
    bad = q.at[index].set(value)
    ff = _tiny_params()
    call = lambda v: jaxmm.log_boltzmann_internal(
        transform.z_matrix, *transform.unpack(v), ff, 300.)
    actual, gradient = jax.jit(jax.value_and_grad(call))(bad)
    assert actual == -jnp.inf
    np.testing.assert_array_equal(gradient, jnp.zeros_like(q))
    with pytest.raises(ValueError, match="Expected finite bonds"):
        transform.to_cartesian(bad)


def test_pack_roundtrip_batches_and_pytree_reconstruction():
    """Both coordinate maps preserve batches and work with a traced transform."""
    transform, q = geometry()
    for batch in [(), (2, 3), (0,), (2, 0)]:
        points = jnp.broadcast_to(q, batch + q.shape)
        actual = jax.jit(lambda t, v: t.to_internal(t.to_cartesian(v)))(transform, points)
        np.testing.assert_allclose(actual, points, atol=1e-12)
    mapped = jax.jit(jax.vmap(lambda t, v: t.log_volume(v), in_axes=(None, 0)))
    np.testing.assert_allclose(mapped(transform, jnp.stack((q, q*1.01))),
                               transform.log_volume(jnp.stack((q, q*1.01))))
    low, high = transform.bounds
    assert transform.periodic_indices == (5,)
    assert bool(jnp.all(q > low) & jnp.all(q < high))


def test_shape_dtype_and_broadcast_misuse():
    transform, q = geometry()
    for call, error, match in [
        (lambda: transform.unpack(q[:-1]), ValueError, "expected trailing shape"),
        (lambda: transform.to_internal(jnp.zeros((5, 3))), ValueError, "expected trailing shape"),
        (lambda: transform.unpack(q.astype(jnp.float32)), TypeError, "float64"),
        (lambda: transform.unpack(jnp.ones(6, dtype=int)), TypeError, "real floating"),
        (lambda: transform.pack(jnp.ones((2, 3)), jnp.ones(2), jnp.ones(1)), ValueError, "identical batch"),
        (lambda: transform.to_internal(jnp.zeros((4, 3))), ValueError, "Expected finite bonds"),
        (lambda: jaxmm.InternalCoordinates(None), TypeError, "ZMatrix"),
    ]:
        with pytest.raises(error, match=match):
            call()
    with pytest.raises(ValueError, match="expected trailing shape"):
        jax.jit(transform.unpack)(q[:-1])


def test_captured_coordinates_with_traced_construction_metadata():
    """A concrete captured array can still produce a traced shape predicate."""
    transform, q = geometry()
    x = transform.to_cartesian(q)
    actual = jax.jit(lambda: jaxmm.InternalCoordinates(_tiny_zmatrix()).to_internal(x))()
    np.testing.assert_allclose(actual, q, atol=1e-12)


@pytest.mark.parametrize("bad,match", [
    ("lists", "integer arrays"), ("float_refs", "integer shape"),
    ("rank", "vector"), ("small", "at least three"),
    ("float_order", "integer shape"), ("duplicate_order", "permutation"),
])
def test_metadata_misuse_is_diagnostic(bad, match):
    from dataclasses import replace
    z = _tiny_zmatrix()
    if bad == "lists":
        z = replace(z, bond_ref=[-1, 0, 1, 2])
    elif bad == "float_refs":
        z = replace(z, bond_ref=z.bond_ref.astype(float))
    elif bad == "rank":
        z = replace(z, bond_ref=z.bond_ref[None, :])
    elif bad == "small":
        z = replace(z, bond_ref=z.bond_ref[:2])
    elif bad == "float_order":
        z = replace(z, atom_order=jnp.arange(4, dtype=float))
    else:
        z = replace(z, atom_order=jnp.array([0, 1, 1, 3]))
    with pytest.raises((TypeError, ValueError), match=match):
        jaxmm.InternalCoordinates(z)


def test_alanine_chirality_sector_is_rigid_motion_invariant():
    x = jnp.asarray(ALDP_XYZ)
    assert bool(jaxmm.aldp_in_l_sector(x))
    assert not bool(jaxmm.aldp_in_l_sector(x*jnp.array([1., 1., -1.])))
    assert bool(jaxmm.aldp_in_l_sector(x[:, jnp.array([1, 2, 0])]+.4))


def test_pullback_reuses_the_supplied_cartesian_density():
    """A generic invariant Gaussian energy works without a force-field object."""
    transform, q = geometry()
    def target(x):
        return -0.7*jnp.sum((x-x.mean(axis=-2, keepdims=True))**2, axis=(-2, -1))
    points = jnp.stack((q, q*1.03))
    expected = target(transform.to_cartesian(points))+transform.log_volume(points)
    np.testing.assert_allclose(jax.jit(lambda v: transform.pullback(target, v))(points),
                               expected, atol=1e-12)
    for batch in [(0,), (2, 0)]:
        assert transform.pullback(target, jnp.broadcast_to(q, batch+q.shape)).shape == batch


def test_cartesian_support_rejection_also_masks_volume_gradient():
    """-inf plus a finite volume must not leave a spurious volume derivative."""
    transform, q = geometry()
    def target(x):
        # A rotation-invariant support condition on the first bond.
        keep = jnp.sum((x[..., 1, :]-x[..., 0, :])**2, axis=-1) > 0.04
        centered = x-x.mean(axis=-2, keepdims=True)
        return jnp.where(keep, -jnp.sum(centered**2, axis=(-2, -1)), -jnp.inf)
    value, gradient = jax.jit(jax.value_and_grad(
        lambda v: transform.pullback(target, v, reference=q)))(q)
    assert value == -jnp.inf
    np.testing.assert_array_equal(gradient, jnp.zeros_like(q))


def test_pullback_contract_misuse():
    transform, q = geometry()
    for call, error, match in [
        (lambda: transform.pullback(None, q), TypeError, "callable"),
        (lambda: transform.pullback(lambda x: 0., q, measure="slice"), ValueError, "measure"),
        (lambda: transform.pullback(lambda x: jnp.zeros(1), q), ValueError, "must have shape"),
        (lambda: transform.pullback(lambda x: jnp.nan, q), ValueError, "NaN or \\+inf"),
        (lambda: transform.pullback(lambda x: jnp.inf, q), ValueError, "NaN or \\+inf"),
        (lambda: transform.pullback(lambda x: 1j, q), TypeError, "real valued"),
        (lambda: transform.pullback(lambda x: 0., q, reference=q[None, :]), ValueError, "exactly shape"),
        (lambda: transform.pullback(lambda x: 0., q, reference=q.at[0].set(-1)), ValueError, "valid internal"),
    ]:
        with pytest.raises(error, match=match):
            call()


@pytest.mark.parametrize("bad", [True, 300.+1j, "300", None])
def test_temperature_validation_rejects_nonphysical_scalar_types(bad):
    """Both regularized and ordinary densities share the scalar T contract."""
    transform, q = geometry()
    x = transform.to_cartesian(q)
    for call in [jaxmm.log_boltzmann, jaxmm.log_boltzmann_regularized]:
        kwargs = {"energy_cut": 100., "energy_max": 1000.} if call is jaxmm.log_boltzmann_regularized else {}
        with pytest.raises(ValueError, match="temperature must be"):
            call(x, _tiny_params(), bad, **kwargs)


@pytest.mark.parametrize("temperature", [0., -1., np.nan, np.inf, jnp.ones(1)])
def test_invalid_temperature_fails_loudly(temperature):
    transform, q = geometry()
    with pytest.raises(ValueError, match="temperature must be"):
        jaxmm.log_boltzmann_internal(transform.z_matrix, *transform.unpack(q),
                                     _tiny_params(), temperature)


def test_wrong_atom_count_and_noninvariant_energy_are_rejected():
    """A rigid-motion marginal is invalid for external restraints or a box."""
    from dataclasses import replace
    transform, q = geometry()
    ff = _tiny_params()
    for bad, match in [(replace(ff, n_atoms=5), "force-field atoms"),
                       (replace(ff, box=jnp.ones(3)), "no box")]:
        with pytest.raises(ValueError, match=match):
            jaxmm.log_boltzmann_internal(transform.z_matrix, *transform.unpack(q), bad, 300.)
