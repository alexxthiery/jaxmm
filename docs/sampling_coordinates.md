# Sampling coordinates

`SamplingCoordinates` supplies a pure coordinate map for molecular samplers and
flows. It is separate from the physical target and from the Z-matrix geometry.
It stores only the reference, local scales, bounds and static circular indices;
it has no fitted statistics, cached derived state or force-field dependency.

For a half-line with lower bound $l$, reference $r$ and scale $s$,

$$
q(z)=l+(r-l)\exp\left(\frac{s}{r-l}z\right).
$$

For a finite open interval $(l,h)$, define $f=(r-l)/(h-l)$ and
$a=s/((h-l)f(1-f))$. Then

$$
q(z)=l+(h-l)\operatorname{sigmoid}(\operatorname{logit}(f)+az).
$$

Both maps satisfy $q(0)=r$ and $q'(0)=s$. Circular entries are unchanged, have
unit scale and use the supplied half-open principal interval. The map does not
identify repeated real copies of a circle: the sampler must wrap its proposals.

```python
coordinates = jaxmm.SamplingCoordinates(
    reference=q0, scale=scales, lower=lower, upper=upper,
    periodic_indices=periodic_indices,
)
z = coordinates.to_sampling(q0)
q = coordinates.to_physical(z)
log_p = coordinates.pullback(internal_target, z)
```

All array fields have the same nonempty shape `(dim,)`; no broadcasting is
inferred. Lower bounds are finite. Upper bounds are finite or positive infinity.
Ordinary references lie strictly inside their bounds; circular references may
equal their lower bound. Float64 is required. All methods accept leading batch
axes; numeric fields are pytree children, and circular indices are static.

`log_abs_det_jacobian(z)` is in the sampling-to-physical direction. `pullback`
adds it to the callback density, which must already contain any physical
conformational volume. For the physical `InternalCoordinates` target, both
Jacobians appear exactly once. A temperature path between two such transformed
targets preserves the geometric factors with coefficient one at every level.

Coordinate conversions reject invalid concrete events. Shape and dtype checks
remain active under tracing. `pullback` safely rejects nonfinite sampling events,
noncanonical circular values, exponential overflow/underflow and finite-precision
logit saturation. It probes representability without differentiation and substitutes
valid inputs before evaluating the callback, giving -inf and zero rejection
gradients. It never clips physical bonds into validity. Callback NaN/+inf remain
arithmetic errors; eager checks raise and compiled errors remain visible.

The map does not enforce molecular chirality or regular construction geometry;
the callback retains responsibility for that support. For the shipped L-alanine
chart, its first torsion can be supplied as the open interval (0, pi) rather than
as a full circle. All bonds and angles remain flexible.
