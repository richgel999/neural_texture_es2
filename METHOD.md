# Training a neural texture with Evolution Strategies: notes for implementers

This document explains the mathematics behind `main.cpp`, why the method
works, how it compares to backpropagation, the pitfalls encountered while
building it, and where to take it next. It assumes you have read the README
and can follow the code.

**Experimental baseline.** Unless a paragraph says otherwise, every number
below is from a 512×512 center crop of kodim23, 3000 iterations, the default
hyperparameters in the table in §1.3, a 24,24 leaky-ReLU MLP with sigmoid
output, and positional input `uv` only. PSNR figures are for the latent
quantized post-hoc to 8 bits ("q8" in the logs) unless marked fp32. Some
earlier experiments used the original positional input `uv,fourier:1`
(14 MLP inputs for an 8-channel latent instead of 10); those are flagged
where they appear. The README's headline table and the checked-in
`out_128c8` model use `uv,fourier:1`; in the two configurations where both
were run, the `uv`-only version is 0.26–0.32 dB better. The logs are `out_*.log` in the repository root, and the
bit-depth sweep is `out_qbits.log`.

Notation: the image has `W×H` pixels (a material is `T` such images of equal
size, `T ≤ 4`, sharing everything below), the latent texture `Z` has `LW×LH`
texels with `C` channels, the decoder `f_w` is an MLP with weight vector `w`,
and `r = W/LW = H/LH` is the pixel-to-texel ratio (8 for a 64×64 latent on
a 512² image, 4 for 128×128; the code does not require the two ratios to be
equal, but every footprint formula below assumes they are). `K` is the
number of antithetic pairs per step. An optional second latent level
(`--latent2`, §1.4) has its own `LW_1×LH_1×C_1` and ratio `r_1`; unless a
passage says otherwise, everything below is written for a single level and
applies to each level separately.

---

## 1. The model

### 1.1 Decoder

Every output pixel `p = (px, py)` is reconstructed as

```
Î(p) = f_w( bilinear(Z, u, v), phi(u, v) ),      u = (px + 0.5)/W,  v = (py + 0.5)/H
```

The bilinear tap, exactly as in `bilinear_tap` and `sample_latent`:

```
x  = u·LW − 0.5            y  = v·LH − 0.5
i  = floor(x)              j  = floor(y)               (unclamped integer parts)
fx = x − i                 fy = y − j                  (from the UNclamped values)
x0 = clamp(i,     0, LW−1) y0 = clamp(j,     0, LH−1)
x1 = clamp(i + 1, 0, LW−1) y1 = clamp(j + 1, 0, LH−1)  (both from the unclamped i, j)
sample = (1−fx)(1−fy)·Z[y0][x0] + fx(1−fy)·Z[y0][x1] + (1−fx)fy·Z[y1][x0] + fx·fy·Z[y1][x1]
```

Texel centers therefore sit at UV `(i + 0.5)/LW`. For even `r` no pixel
center lands exactly on a texel center; the two pixels straddling one read it
with `fx = 0.5/r` and `1 − 0.5/r`. Only the indices are clamped; `fx, fy` are
not. Consequently, in the outermost half-texel of
the image `x0 == x1` (or `y0 == y1`) and the two weights land on the same
texel. This matters for the scatter in §3.4.

The output layer has `3T` units, three per texture, each mapped with
`out = 1 / (1 + exp(−s))` (sigmoid, the default) or `out = clamp(s + 0.5, 0, 1)`
with `--clamp`; note the `+0.5`.

`phi` is a small positional encoding. The default, and the best of the
variants tested at 64×64, is `(2u−1, 2v−1)`; at 128×128×8 the cell-periodic
variant `--pos ldct:2`, used *instead of* `uv`, is 0.25 dB better on the
parrots (§7) but worse on cartoon content, where every cell-periodic input
tested lost to plain `uv`. With a second latent level, `phi` and its
cell-relative kinds are computed from the level-0 tap.

### 1.2 Objective and support structure

The training objective is per-pixel mean squared error:

```
L(Z, w) = (1 / 3WH) · Σ_p  ℓ_p(Z, w),      ℓ_p = ‖Î(p) − I(p)‖²  (summed over R,G,B)
```

For a material of `T` textures with loss weights `w_t` (`--weights`, default
all 1) this becomes `ℓ_p = Σ_t w_t ‖Î_t(p) − I_t(p)‖²` and
`L = Σ_p ℓ_p / (3 · Σ_t w_t · W · H)`. Dividing by the weight sum makes the
weights purely relative and reduces exactly to the single-texture form when
`T = 1, w = 1`. The loss is still a sum over pixels, so nothing in §3 changes;
the per-pixel term just has `3T` channels. Reported per-texture PSNRs are the
unweighted `SSE_t / (3WH)`.

Two properties of this objective are load-bearing for everything below:

1. **It is a sum over pixels.** Any loss of this form works. A loss that
   couples distant pixels (a large-window SSIM, for instance) enlarges every
   footprint below and weakens the method; see §6.
2. **Each `ℓ_p` depends on `Z` only through the 4 texels `S(p)` that pixel
   `p`'s tap reads**, and on all of `w`. With two levels it is 4 texels per
   level, `S(p) = S_0(p) ∪ S_1(p)`.

The set of pixels that read texel `t = (tx, ty)` is its *footprint* `F(t)`.
From the tap formula, texel `tx` is read by pixels whose latent coordinate
`x` lies in `[tx − 1, tx + 1)`, i.e. pixels with

```
px ∈ [ r·(tx − 0.5) − 0.5,  r·(tx + 1.5) − 0.5 )        (clamped to the image)
```

a span of `2r` pixels for interior texels: 16×16 at `r = 8`, 8×8 at `r = 4`.
The block is the
texel's own `r×r` cell extended by `r/2` pixels on every side, so for even
`r` its first pixel is `r·tx − r/2`. For `r = 8`, texel `tx` is read by
`px = 8tx − 4 … 8tx + 11`. Border texels have footprints truncated by the
clamp, `1.5r` wide instead of `2r`.

Property 2 is the *support structure* of the decoder. Everything that makes
this method practical follows from it.

### 1.3 Defaults

All from `Options` in `main.cpp`. None of these were tuned beyond a first
guess that worked; treat them as a starting point, not a measured optimum.

| Parameter | Default | Flag |
|---|---|---|
| MLP ES pairs per step | 32 | `--mlp-pairs` |
| MLP minibatch, pixels (sampled with replacement) | 4096 | `--mlp-batch` |
| MLP sigma | 0.02 | `--mlp-sigma` |
| MLP Adam learning rate | 0.005 | `--mlp-lr` |
| MLP step every N iterations | 1 | `--mlp-every` |
| Latent ES pairs per step | 4 | `--lat-pairs` |
| Latent sigma | 0.05 | `--lat-sigma` |
| Latent Adam learning rate | 0.02 | `--lat-lr` |
| Latent init, standard deviation | 0.1 | `--lat-init` |
| Second latent level | off | `--latent2 W H C` |
| Second-level sigma | = `--lat-sigma` | `--lat2-sigma` |
| Learning-rate anneal (start fraction, final multiplier) | off | `--lr-anneal` |
| MLP finite-difference phase (start fraction), step | off, 1e−3 | `--mlp-fd`, `--mlp-fd-h` |
| MLP full-image ES phase (start fraction), pairs | off, = `--mlp-pairs` | `--mlp-full`, `--mlp-full-pairs` |
| MLP frozen from (start fraction) | off | `--mlp-freeze` |
| Alternate latent levels per pair | off | `--lat-alt` |
| Sigma anneal | off | `--sigma-anneal` |
| Adam `β1, β2, eps` | 0.9, 0.999, 1e−8 | fixed |
| Leaky ReLU negative slope | 0.01 | fixed |
| Hidden widths, activation | 24,24, leaky | `--mlp`, `--act` |
| Output mapping | sigmoid | `--clamp` for hard clamp |
| Seed | 1 | `--seed` |
| Center crop | 512 | `--crop` |
| Max units in any layer, incl. input (`MAXH`) | 128 | fixed |
| Max hidden layers (`MAXL`) | 8 | fixed |
| Stats print / PNG save interval (RNG-neutral) | 10 / 100 | `--print-every`, `--save-every` |

RNG draw order, for anyone reproducing a run: one `mt19937` seeded with
`--seed` draws the latent init, then the MLP init (weights only; biases are
zero and consume no draws); each MLP step draws all `K·P` perturbations
(pair-major), then the minibatch indices; each latent pair draws `ε` in
texel-major, channel-minor order, level 0 first and then level 1 if present
(the latent init draws follow the same level order, so a second level shifts
the MLP init draws relative to a single-level run). A fresh `std::normal_distribution` object
is constructed per init and per step, so any cached second sample is
discarded between them. `--load` does not change the draw order: the init
draws still happen and the file then overwrites their result, so a loaded
model continues from the same RNG state as a fresh run with the same seed.

Each iteration runs one MLP step on the current latent, then one latent step
on the just-updated MLP. The two Adam states are independent. `--mlp-every N`
skips the MLP step on iterations not divisible by `N`; it is untested beyond
`N = 1`.

### 1.4 Optional second latent level

`--latent2 W H C` adds a second latent texture, typically coarser (32×32×4
next to 128×128×4). It is sampled bilinearly at the same UV with its own tap,
and its `C_1` channels are appended after level 0's before `phi`, so the MLP
input is `C_0 + C_1 + pos.count()`. All levels live in one flat parameter
vector, level 0 first, so one Adam state and one ES perturbation cover both
(with `--lat-alt`, each pair perturbs only one level; see §3.3). With
`--latent2` absent, the code path and all numbers are exactly the
single-level ones.

**Late-training phases for the decoder.** Three optional switches, each a
fraction of `--iters` at which it begins (`>= 1` means off), gated by
`--mlp-every` like the ES step, with precedence freeze > fd > full:
`--mlp-full` evaluates the MLP ES step on every pixel with `--mlp-full-pairs`
pairs; `--mlp-fd` replaces ES with central finite differences per weight
(§6) and resets the MLP's Adam state at the switch; `--mlp-freeze` stops
updating the MLP so only the latent trains. Under `--mlp-fd` the stats line
prints `fd-rms` (root-mean-square per-weight loss difference at step `h`)
in place of `dstd`, and `frozen` when frozen. Measured results are in §7.

---

## 2. Evolution Strategies in one page

We never compute `∇L`. Instead we use the Gaussian-smoothed objective

```
J_σ(θ) = E_{ε ~ N(0, I)} [ L(θ + σε) ]
```

whose gradient has the score-function form

```
∇J_σ(θ) = (1/σ) · E[ L(θ + σε) · ε ]
```

With antithetic pairs `±ε` the constant part of `L` cancels, and with `K`
pairs the estimator is

```
ĝ = (1 / 2Kσ) · Σ_{i=1..K} [ L(θ + σε_i) − L(θ − σε_i) ] · ε_i
```

This is unbiased for `∇J_σ`, and for small `σ`, `J_σ ≈ L`, so it is a
slightly smoothed gradient of the real loss. It needs only loss evaluations:
`L` can contain anything, including quantizers, clamps, table lookups, or a
block texture codec.

**Its variance grows with dimension.** Linearize the loss difference,
`L(θ+σε) − L(θ−σε) ≈ 2σ · ε·∇L`. Then

```
ĝ ≈ (1/K) · Σ_i ε_i (ε_iᵀ ∇L)
```

which has expectation `∇L`, and each component `ĝ_j` has variance
approximately

```
Var[ĝ_j] ≈ ( ‖∇L‖² + (∂_j L)² ) / K
```

The `‖∇L‖²` term is the point: the noise on *every* component is set by the
gradient norm over *all* `d` dimensions. If the components are of comparable
size, the relative error of one component is roughly `sqrt(d / K)`. With 4
pairs that is about 64 for the default 64×64×4 latent (16384 values) and
about 180 for 128×128×8 (131072 values): the estimate is noise. This is why plain ES is considered hopeless for large
parameter vectors, and why RL-style ES uses thousands of evaluations per
step. Note that the `1/(2Kσ)` normalization makes the expectation and the
linearized noise independent of `σ` to first order; `K` enters only through
the `1/K` in the variance.

Footprint attribution is a structured-perturbation variant of this
estimator: it uses known sparsity in the loss's dependence on the parameters
to restrict which loss terms each parameter is credited with. The estimator
itself is the one popularized by Salimans et al., "Evolution Strategies as a
Scalable Alternative to Reinforcement Learning" (2017), and is closely
related to SPSA (Spall, 1992). Structured-perturbation variants that exploit
known sparsity appear in later work; what is specific here is deriving the
structure from a texture filter's sampling footprint.

---

## 3. Footprint attribution: the mechanism that makes the latent tractable

### 3.1 Derivation

Consider one latent step. We perturb every latent value at once,
`Z ± σε`, with `w` held fixed, and decode the full image twice. This gives
per-pixel loss differences

```
Δℓ_p = ℓ_p(Z + σε) − ℓ_p(Z − σε)
```

For a single pair, the plain ES estimate for texel `t` (all its channels at
once, since they share a footprint; `ε_t` is the `C`-vector of that texel's
perturbations) is

```
ĝ_t = (1 / 2σ) · (1 / 3WH) · Σ_p  Δℓ_p · ε_t          (sum over ALL pixels)
```

and the `K`-pair estimate averages this over pairs.

Split the sum into pixels inside and outside `F(t)`. For a pixel `p ∉ F(t)`,
`ℓ_p` depends on `ε` only through `ε_{S(p)}`, and `t ∉ S(p)`. The components
of `ε` are independent, so `Δℓ_p` is independent of `ε_t` and

```
E[ Δℓ_p · ε_t ] = E[Δℓ_p] · E[ε_t] = 0        for p ∉ F(t)
```

Those terms contribute nothing in expectation and only add variance.
Dropping them gives the **footprint-attributed estimator**

```
ĝ_t = (1 / 2σ) · (1 / 3WH) · Σ_{p ∈ F(t)}  Δℓ_p · ε_t       (per pair, averaged over K)
```

It has the **same expectation** as the plain estimator. This part is exact
and does not rely on any linearization. The variance statement is a little
weaker than "strictly less": the dropped term's variance scales with the
far-field gradient norm `Σ_{q ∉ B(t)} (∂_q L)²`, essentially all of `‖∇L‖²`,
where `B(t)` is the texel neighborhood defined next, while dropping it can
also change the residual coupling to the 8 neighbors by a small amount of
either sign. In every
realistic regime the far-field term dominates by orders of magnitude, so the
attributed estimator has far less variance. What matters for an implementer:
this is an unbiased variance reduction that follows directly from the
decoder's dependency graph, not a heuristic that happens to work.

### 3.2 How much variance is removed

Let `L_F(t) = (1/3WH) · Σ_{p ∈ F(t)} ℓ_p` (with `3WH` read as `3·Σw·WH` for a
material) be the loss restricted to `t`'s footprint, with the same
normalization as `L`,
and `B(t) = ∪_{p ∈ F(t)} S(p)` the set of texels any of those pixels read:
the 3×3 texel neighborhood of `t` (distance-1 neighbors share pixels with
`t`; distance-2 texels do not). Applying the linearization of §2 to the
restricted sum, the effective dimension seen by texel `t` is `9C` rather than
`LW·LH·C` (all latent values, over all levels):

```
Var[ĝ_t] ≈ ‖∇_{B(t)} L_F(t)‖² / K        instead of        ‖∇ L‖² / K
```

For comparable per-texel gradient magnitudes the variance reduction factor is
of order `LW·LH / 9`: roughly 450× for a 64×64 latent and 1800× for 128×128.
This is a conservative figure. The neighbors' contributions in
`∇_{B(t)} L_F(t)` are partial-footprint gradients (about half of a
neighbor's own gradient for edge neighbors, a quarter for corners), so the
true factor lies between `LW·LH/9` and about `LW·LH/2.25`, matching the
`d_eff` range made explicit in §3.3. Either way, four pairs with
footprint attribution are worth on the order of thousands of pairs of plain
ES, at the cost of two full decodes per pair.

The per-step relative error that remains is about `sqrt(9C / K)`: roughly 3
for `C = 4` and 4 for `C = 8` with `K = 4`. That is still noisy per step;
it is Adam's momentum (§5) that averages it down. The MLP's corresponding
figure is about `sqrt(P / 32) ≈ 5–6` for `P` = 843 (`uv`, C = 4) to 1035
(`uv,fourier:1`, C = 8) weights (§4).

### 3.3 What remains: crosstalk

The 8 neighbors of `t` still share pixels with it and are perturbed in the
same pass, so their contributions to `L_F(t)`'s difference do not vanish for
a single pair. They are zero-mean with respect to `ε_t`, so they average out
over pairs and over Adam's momentum, but they are the residual noise source.
The per-texel estimates are therefore **simultaneous and local, but not
statistically independent**: neighbors share pixels and the same `ε` draw.
Do not describe them as independent in anything formal.

**With two levels** the same argument holds per level, since the two levels'
perturbations are independent, but the crosstalk is larger for the coarse
level. A coarse texel's footprint of `2r_1` pixels also reads every fine texel
inside it, all perturbed in the same pass, so its residual noise has roughly
`9C_1 + (2r_1/r_0 + 1)²·C_0` dimensions instead of `9C_1`: about 36 + 324
for 32×32×4 over 128×128×4 on a 512² image. The coarse level is therefore
noisier per step than the fine one, which is why `--lat2-sigma` exists as a
separate knob. Plain 2×2 phase cycling would no longer remove crosstalk
unless the levels were also perturbed in alternate evaluations.
`--lat-alt` does exactly that: pair `k` of step `s` perturbs only level
`(k + s) mod nlevels`, the other level's `ε` is zero, and each level's
gradient is the standard `1/(2K_l σ_l)` estimator over the `K_l` pairs that
perturbed it. It removes the `(2r_1/r_0 + 1)²·C_0` cross-level term at the
cost of giving each level only `K/nlevels` pairs per step. Measured on the
128×128×4 + 64×64×4 configuration it *lost* 0.17 dB at 3000 iterations
(§7): with a 64×64 coarse level the cross-level term is only about 36 extra
dimensions, so halving the pair count costs more than the crosstalk did. It
may still pay for much coarser second levels.

Crosstalk can be eliminated with **2×2 phase cycling**. Here and in §8 one
*evaluation* means one antithetic pair, i.e. two full-image decodes; the
current latent step is 4 evaluations. Perturb only texels with
`(tx mod 2, ty mod 2) = (a, b)` in one evaluation and cycle through the 4
phases. Texels two apart have disjoint footprints (`[tx−1, tx+1)` versus
`[tx+1, tx+3)`), so each texel's `Δ L_F(t)` becomes an exact function of its
own `C`-vector perturbation, and the perturbed footprints tile the image up
to the clamped half-texel border strips. Be precise about what this buys.
The per-channel estimate is still a random projection within the texel's `C`
channels, and each texel is now perturbed in one of the 4 evaluations
instead of all 4, so its relative error per update (one latent step, 4
evaluations either way) is about `sqrt(C)`. The current scheme's error is
`sqrt(d_eff / K)` with `K = 4` and `d_eff` somewhere between about `2.25C`
and `9C`. The
lower figure is the §3.2 estimate made concrete: `C` for the texel itself,
plus 4 edge neighbors at half a footprint gradient, `4·(1/2)²·C`, plus 4
corners at a quarter, `4·(1/4)²·C`, which holds when the residual is smooth
enough that partial-footprint gradients scale with footprint fraction. The
upper figure is neighbors as strong as the texel itself. That puts the
current error between `0.75·sqrt(C)` and `1.5·sqrt(C)`, so phase cycling
changes the per-texel variance by a factor between roughly 0.45× (strong
neighbors, `d_eff ≈ 9C`) and 1.8× (weak neighbors, `d_eff ≈ 2.25C`). The ratio is `4C / d_eff` for any `K` that is a
multiple of 4, since phase cycling gives each texel `K/4` evaluations per
update. It removes
crosstalk exactly, but it is not guaranteed to reduce variance; measure it. Exact per-channel finite differences need `4C` evaluations per
update (cycle channels as well as phases), which for `C = 1` costs the same
as today. See §8.

With **nearest-neighbor sampling** each pixel reads exactly one texel, so
footprints never overlap and the plain latent step is already crosstalk-free
with no phase cycling at all. Among common filters, bilinear is the narrowest
whose footprints overlap, and they overlap only within the 3×3 neighborhood;
wider filters such as bicubic couple a larger neighborhood and would need a
larger phase grid, but are unlikely choices for a latent texture.

### 3.4 Implementation

The latent step in `LatentTrainer::step`:

```
grad[] = 0
for k in 1..K:
    draw ε ~ N(0, I) over all latent values of all levels (level 0 first;
                        with --lat-alt only one level per pair, the rest ε = 0)
    err_plus  = per-pixel squared error decoding with Z + σ_l·ε   (full image; σ_l per level)
    err_minus = per-pixel squared error decoding with Z − σ_l·ε
    for each level l:
        dtex[] = 0
        for each pixel p:
            Δ_p = (err_plus[p] − err_minus[p]) / (3·Σw·W·H)      // Σw = 1 for a single texture
            for each DISTINCT texel in S_l(p):  dtex[texel] += Δ_p    // scatter, level-l tap
        for each texel t of level l, channel c:
            grad[l][t][c] += dtex[t] · ε[l][t][c] / (2·K_l·σ_l)     // K_l = pairs that perturbed level l (K unless --lat-alt)
Z ← Z − lr · Adam(grad)          // grad estimates ∇L, so this is descent; one lr for all levels
```

`dtex[t]` is computed once per texel and reused for all `C` channels, since
the channels share a footprint; only `ε` differs per channel. "DISTINCT"
matters: at the border the clamp makes `x0 == x1` or `y0 == y1`, and the code
guards with `if (t.x1 != t.x0)` / `if (t.y1 != t.y0)` so a pixel is not
credited twice to the same texel.

**Gather form.** Because `F(t)` is the closed-form rectangle in §1.2, the
scatter can be replaced by a gather: for each texel, sum `Δ_p` over
`px` from `ceil(r(tx−0.5)−0.5)` to `ceil(r(tx+1.5)−0.5) − 1` (for `r = 8`,
`8tx−4 … 8tx+11`) and the same in `y`, intersected with the image. This has no atomics and is the natural GPU formulation. With the
clamp applied to the rectangle, border pixels are credited exactly once
automatically.

---

## 4. The MLP step

The decoder weights are global: every weight affects every pixel, so no
support restriction is available. They get ordinary antithetic ES on a
minibatch:

* `K = 32` pairs per step by default (`--mlp-full-pairs` during a full-image phase).
* One minibatch `M` of 4096 pixel indices is drawn per step, uniformly with
  replacement (so it contains a few dozen duplicates; harmless). During a
  `--mlp-full` phase `M` is every pixel in order and no index draws are made.
* **Within a pair, `+ε` and `−ε` must be evaluated on the same minibatch.**
  Otherwise the difference contains a batch-sampling term
  `L_{M+}(θ) − L_{M−}(θ)` of the order of the loss's own fluctuation,
  independent of `σ`, which swamps the `O(σ)` signal.
* The code also shares `M` **across** all 32 pairs. That is a convenience,
  not a variance-reduction step: it means all 32 differences measure the same
  batch objective `L_M`, so the step is a clean ES gradient of one function
  (exactly what an SGD step on batch `M` would target), and the printed
  `dstd` statistic reflects pure ES spread. Under the linearized analysis,
  drawing an independent batch per pair gives the same or slightly lower
  variance. A GPU implementer who wants per-pair batches for parallelism may
  use them, provided each pair stays internally consistent.
* The estimate goes through Adam.

`L_M` is the weighted mean squared error over the batch's `3T·|M|` channel
values, divided by `3·Σw·|M|` (plain MSE over `3·|M|` for one texture),
which sets the absolute scale of the printed `dstd`. At `P` = 843 (`uv`,
C = 4) to 1035 (`uv,fourier:1`, C = 8) weights the per-step relative error
is around `sqrt(P/32) ≈ 5–6`.
The MLP is the part of the system that pays the ES tax. Adam's momentum
averages roughly the last 10 steps, which brings the effective pair count to
a few hundred. This is why the decoder is deliberately tiny, and why nearly
doubling its weight count gained only 0.16 dB while training slower (§7).

---

## 5. Adam, and why it matters here

Both gradient estimates are passed to bias-corrected Adam
(`β1 = 0.9, β2 = 0.999, eps = 1e−8`) rather than used directly. Two of its
properties matter for ES specifically:

* **Momentum averages noisy estimates for free.** With `β1 = 0.9`, each update
  effectively sees ~10 steps' worth of pairs.
* **Per-parameter normalization makes the learning rate a step-size bound in
  parameter units.** The raw ES estimate's magnitude depends on the loss
  scale, the `1/3WH` factor, and the noise level. Adam divides that out, so
  `--lat-lr 0.02` means latent values move *at most* about 0.02 per step,
  reaching that bound only when the gradient sign is consistent across steps.
  The bound is a steady-state property, not a theorem: a parameter whose
  gradient was near zero for a long time and then holds a consistent value
  can transiently step by several times `lr`, peaking near `6.5·lr` after
  about a dozen such steps, because `m` responds faster than `sqrt(v)`
  (first-step gains `1−β1 = 0.1` versus `sqrt(1−β2) ≈ 0.032`, and `v` has a
  1000-step time constant). The bias correction on `v` damps this early in
  training: the peak is about `2·lr` for a parameter that becomes active
  near iteration 100, `5·lr` near 1000, and `6.5·lr` only well beyond that.
  With the noisy estimates here, `|m̂| / sqrt(v̂)` is usually well below 1
  because `m` averages the noise down while `v` retains its full power. The
  useful consequence is that a background texel with a tiny but consistent
  gradient moves at the same rate as an edge texel.

Adam is optional to the method. Plain SGD with momentum would work with more
careful tuning. It is not optional in practice if you want the hyperparameters
to survive changing `σ`, the latent size, or the loss scale.

---

## 6. Contrast with backpropagation

**No backprop baseline has been run on this model.** Everything in this
section is an estimate from operation counts and from the observed
convergence behavior, not a measurement.

**Cost per iteration.** One default ES iteration is about 9 full-image
decodes (4 latent pairs × 2, plus the MLP minibatch work of 32 × 2 × 4096
pixel evaluations, which at 512² is exactly 1 decode; the periodic stats add
about 0.2 decodes per iteration at the default print interval). The late
phases change this: a full-image MLP step costs `2·pairs` decodes, and a
finite-difference step costs `2P·4096/WH` decode equivalents, about 57 for
the 1839-weight decoder. A full-image backprop step is
forward + backward, about 3 decode equivalents, and returns exact gradients
for both `Z` and `w`. Backprop's latent gradient also needs a scatter through
the bilinear weights, so 3× is a fair floor for the ES overhead per iteration.

**Iterations to converge.** The latent side should be reasonably close to
backprop: with footprint attribution each texel is a small local problem
whose per-step relative error (~3–4) is averaged down by momentum. The MLP
side is not: a ~1000-weight ES gradient is far noisier than an exact one, and
it produces the slow plateau seen after ~2000 iterations. My expectation is
that ES needs several times more iterations overall, so 10–30× more total
compute than a well-implemented backprop trainer for the same PSNR. At this
model size that is minutes, not hours. Treat these factors as unverified.

**What ES gives in exchange:**

* *Any decoder.* Quantizers, hard clamps, lookup tables, integer arithmetic,
  block texture codecs, non-differentiable positional features: all work
  unchanged. Backprop needs straight-through estimators or differentiable
  surrogates for each, and those are exactly where backprop-based texture
  compression pipelines get complicated.
* *Any loss* that decomposes over pixels, or over regions with a
  correspondingly larger footprint.
* *No framework.* The optimizer is ~40 lines. The encoder and the runtime
  decoder are the same forward function, so a GPU implementation is a compute
  shader, not a training framework.
* *Transparency.* Every gradient estimate is a sum of loss values you can
  print and inspect.

**When to prefer backprop:** if the decoder must grow well beyond ~2000
weights, or if training throughput is the actual constraint. The MLP is the
bottleneck under ES; its estimate variance grows linearly with weight count
(relative error as the square root).

**A derivative-free alternative for the MLP, implemented as `--mlp-fd`:**
the decoder is small enough that central finite differences per weight give a
second-order-accurate minibatch gradient with no analytic derivatives, i.e.
numerical differentiation, neither ES nor backprop. Cost: 2 × `P`
evaluations of 4096 pixels ≈ 26–32 decode equivalents for `P` = 843–1035 and
57 for 1839, versus 1 for the ES step. Two details mattered in practice: the
step uses the float-realized `w ± h` rather than nominal `h`, and the MLP's
Adam state is reset at the switch, because the ES phase's second-moment
estimate holds the ES noise power and would otherwise throttle FD steps for
about a thousand iterations (the first attempt without the reset gained
0.12 dB; with it, 0.27 dB, §7). Forward differences would halve the cost.

---

## 7. Gotchas, measured and learned

Each of these cost time or changed the design. Baseline as stated at the top
of the document; deviations are flagged.

**Every extra MLP input hurts unless it earns its keep.** The initial design
fed `sin/cos(2πu), sin/cos(2πv)` alongside `u, v`. Removing them *gained*
0.26 dB at 64×64×4 (26.90 → 27.16) and 0.32 dB at 128×128×8 (32.16 → 32.48),
with fewer weights and faster training. Four inputs add 96 first-layer
weights to the ES problem, and the information they carry is something the
latent represents trivially. Under ES, input dimensions are not free.

**Positional features periodic in the latent grid cause grid artifacts.**
Feeding the fractional cell offset `(fx, fy)` as an input (`--pos uv,local`)
produced visible texel-periodic speckle (see the `recon_003000.png` in
`out_128c8_pos_uv_local`) and cost 0.83 dB at 128×128×8
(31.65) and 0.28 dB at 64×64×4 (26.88; `local` alone is 26.89). Any feature that lets the MLP vary within a cell lets ES noise
express itself as a texel-aligned pattern the latent then has to cancel.
The measured results for cell-periodic cosines, `--pos ldct:N`, which emits
`cos(π·k·fx), cos(π·k·fy)` for `k = 1..N`:

| Input (`ldct` rows replace `uv`; no global position input) | 64×64×4 | 128×128×8 |
|---|---|---|
| `uv` (baseline) | 27.16 | 32.48 |
| `ldct:1`, half wave `cos(π·fx)` only | 26.92 | 32.23 |
| `ldct:2`, half wave + full wave `cos(2π·fx)` | 26.55 | **32.73** |

The half-wave term alone hurts at both sizes. Adding the full-wave term
helps at 128×128×8 and hurts more at 64×64×4. A plausible reading is that
`cos(2π·fx)` is continuous across cell edges (it is 1 at both), while
`cos(π·fx)` and `fx` itself jump at every edge, and that only at `r = 4` is
the cell small enough for a smooth within-cell correction to sharpen rather
than smear. That is a hypothesis; the half-wave term is still present in the
winning variant, so continuity is not established as the criterion. A learned
reconstruction kernel is more safely expressed as a few global kernel
parameters than as an MLP input (§8). For completeness, replacing `uv` with
the global half-cosine pair `dct:1` (`cos(πu), cos(πv)`) gave 27.01 and
32.48, no better than `uv`.

**The latent range grows without bound.** Nothing regularizes `Z`. With Adam,
any value whose gradient sign is consistent drifts by up to `lat_lr` per
step, so over 3000 steps a value can move by tens of units. The latent's
standard deviation rose from 0.1 at init to 1.3 at 3000 iterations and 2.4 at
6000, with maxima above 12 (`out_6k.log`, `uv,fourier:1` run). This is
harmless for fp32 and for 8-bit post-hoc quantization, but a min/max
quantizer stretched by a few outlier texels is what breaks coarse
quantization. A range penalty, a clipped quantizer, or quantization-aware
training would fix it. One related trap: `lat_sigma = 0.05` becomes small
relative to a latent with standard deviation 1–2. That is fine, since what
matters is the MLP's sensitivity to the perturbation and not the latent's
own scale, but it is easy to misread when tuning.

**Post-hoc quantization: 8 bits is nearly free, then it degrades fast, and
faster for bigger latents.** From `out_qbits.log` (evaluation of the saved
models, PSNR in dB):

| Latent bits | 64×64×4 (`uv`) | 64×64×8 (`fourier:1`) | 128×128×8 (`uv`) |
|---|---|---|---|
| fp32 | 27.18 | 28.23 | 32.52 |
| 8 | 27.16 | 28.21 | 32.48 |
| 6 | 26.95 | 27.96 | 31.87 |
| 5 | 26.29 | 27.23 | 30.28 |
| 4 | 24.19 | 25.03 | 26.37 |

Across all training and evaluation logs the 8-bit cost is 0.01–0.06 dB
(0.02–0.04 within `out_qbits.log` itself). Relative to fp32, six bits
costs 0.2–0.3 dB at 64×64 but 0.5–0.7 dB at 128×128×8, and four bits costs
2.6–3.2 dB at 64×64 versus 5.4–6.2 dB at 128×128×8. These are empirical observations about latents
trained in float with these hyperparameters, not properties of the method.

**The MLP, not the latent, is the ES bottleneck.** At 128×128×8 with
`uv,fourier:1` (14 inputs), going from hidden widths 24,24 (1035 weights) to
36,36 (1983 weights, nearly double) gained 0.16 dB (32.16 → 32.32) and was
behind the smaller net at every checkpoint through 2500 iterations,
overtaking only at 3000. By comparison, in the same `uv,fourier:1` set of runs, doubling latent
resolution gave 1.7–2.0 dB per doubling of bitrate and doubling channels
1.3–1.9 dB. Spend bits on the latent.

**Watch `dstd`.** The stats line prints the standard deviation of the 32 MLP
loss differences. It drops about 5× between the 100- and 200-iteration
prints and is then roughly flat (`out_train.log`, a `uv,fourier:1` run; the
ratio is what matters). That early drop is the MLP signal shrinking relative to `σ` as
the loss falls; after that the estimate's signal-to-noise stays where it
lands. `--mlp-full` evaluates the tail on every pixel, and `--mlp-fd`
replaces the ES estimate entirely; see the late-phase results below.
Reducing `σ` also helps, up to the point where `dstd` approaches the float
noise floor of the loss differences. Under `--mlp-fd` the printed statistic
is `fd-rms`, on a different scale, so do not compare it to `dstd`.

**Sigmoid output, not clamp.** A hard clamp has zero response to perturbation
wherever the pre-clamp value is outside `[0,1]`, so those pixels contribute
nothing to any loss difference and the corresponding weights get no signal.
Sigmoid keeps every pixel responsive. `--clamp` exists for comparison.

**Leaky ReLU, not ReLU.** A dead unit under ES looks like noise, not like
zero: perturbing its weights changes nothing, so its loss differences are pure
crosstalk from other weights and Adam happily follows them. The 1% leak keeps
a small signal alive. With only 24 units per layer, losing a few is
expensive.

**Initialization.** He init for hidden layers, `sqrt(1/n)` for the output
layer so the sigmoid starts near mid-gray, biases zero, latent Gaussian with
standard deviation 0.1.
The `sine` activation uses SIREN-style init (first layer `30/n_in`, later
hidden layers `sqrt(6/n)`; the output layer keeps `sqrt(1/n)`); it compiles
and runs but has not been evaluated.
Starting the latent near zero is expected to matter, since the MLP has to
learn a mapping from latent to color before the latent can learn anything,
but `--lat-init` was never swept; treat this as a reasoned default, not a
measurement.

**Adam hides sigma.** Because Adam normalizes the update, a badly chosen `σ`
does not show up as a wrong step size; it shows up as a wrong *direction*
(too large: the estimate is of a heavily smoothed loss; too small: the loss
differences fall into float noise). Tune `σ` by watching `dstd` and PSNR, not
by watching how fast parameters move.

**Border texels.** With clamped indices, pixels in the outermost half-texel
read the same border texel through both taps on that axis. The scatter must
not credit that pixel twice (§3.4). Border texels also have `1.5r`-wide
footprints instead of `2r`, so their estimates are somewhat noisier; nothing
compensates for this.

**The model file does not record the output mapping.** The header stores
latent size, MLP input count, layer widths, activation, and positional spec,
but not `--clamp`. A model trained with `--clamp` loads as sigmoid without
complaint. The same goes for `--weights`: they affect training and the
reported weighted PSNR only, so pass them again with `--load`. Store the
output mapping if you add one. For reference, the current layout
(`save_model`, version 5) is: 10 native-endian `int32` values
(little-endian on all mainstream targets)
(`magic = 0x4E544335, LW_0, LH_0, LC_0, nin, poslen, act` with
`0 leaky, 1 relu, 2 tanh, 3 sine`, `nlayers, nlevels, textures`), then `(W, H, C)` as
three `int32` for each level after the first, then `nlayers` `int32` hidden
widths, then `poslen` bytes of positional spec text, then the flat fp32
latent (level 0 first, each level `LH·LW·LC` values in `(y, x, c)` order with
`c` fastest), then the MLP parameters layer by layer as row-major
`W[out][in]` followed by `b[out]` (the output layer has `3·textures` rows).
The loader also accepts four older header versions from earlier commits: v4
(magic `0x4E544334`, no `textures` int, always one texture), v3 (magic
`0x4E544333`, the same first 8 ints with no `nlevels`, always one level), v2
(magic `0x4E544332`, which stores an `nfreq` count in place of the spec), and
a six-int header with no magic that stores one hidden width used for both
layers; new files always use the v5 layout. The MLP input vector, i.e. the `in` axis of the first layer,
is level 0's `C_0` bilinear samples, then level 1's `C_1` if present, then the
positional features in the order the spec lists them (for `uv`:
`2u−1, 2v−1`), so `nin = Σ C_l +` the spec's feature count.

**Two-level latent, first results.** With `--latent2`, footprint attribution
applied per level, one sigma and one learning rate for both levels, 3000
iterations, 8-bit latent (`out_k23_64c4_l2_16c4.log`,
`out_mario_128c4_l2_32c4.log`):

| Image | Levels | PSNR | bpp raw | Single-level reference |
|---|---|---|---|---|
| kodim23 | 64×64×4 + 16×16×4 | 27.47 | 0.59 | 27.16 at 0.55 (64×64×4) |
| mario | 128×128×4 + 32×32×4 | 29.08 | 2.18 | 28.82 at 2.05 (128×128×4) |

A gain of about 0.3 dB for 6 percent more bits in both cases, which is a real
but smaller improvement than the literature's pyramids get with backprop. The
coarse level's estimate is noisier than the fine level's (§3.3), and nothing
has been tuned for it; `--lat2-sigma` and learning-rate annealing are the
obvious levers. Alternating levels per pair (`--lat-alt`) was tried and lost
0.17 dB with a 64×64 second level (below). A 64×64×4 second level on
128×128×4 (mario, `out_mario_128c4_l2_64c4.log`) gained 1.28 dB for 0.5 bpp,
about 0.6 dB better than spending the same bits on channels; a 36,36 decoder
with 64 ES pairs added another 0.48 dB (`out_mario_128c4_l2_64c4_mlp36.log`).

**Late-phase options, first results.** mario, 128×128×4 + 64×64×4, 36,36
MLP, 64 pairs, 3000 iterations, `--lr-anneal 0.5 0.05`, each option applied
from iteration 2250, 8-bit latent PSNR (`out_x_*.log`):

| Option | PSNR | vs baseline |
|---|---|---|
| baseline | 30.64 | |
| `--mlp-fd 0.75` (Adam reset at switch) | **30.91** | +0.27 |
| `--mlp-fd 0.75`, without the Adam reset | 30.76 | +0.12 |
| `--mlp-freeze 0.75` | 30.65 | +0.01 |
| `--mlp-full 0.75 --mlp-full-pairs 16` | 30.63 | −0.01 |
| `--lat-alt` | 30.47 | −0.17 |

Finite differences are the only clear win, and only with the optimizer
reset. Freezing the decoder changes nothing, which says the latent was not
being held back by decoder noise in the last quarter. The full-image step
with 16 pairs neither helped nor hurt: what limits the ES decoder gradient
here is the random-direction noise (`sqrt(P/K)`), not minibatch sampling.
For scale, the same configuration reaches 31.38 dB at 6000 annealed
iterations and 31.78 at 12000, so a 0.27 dB gain in the last 750 iterations
is roughly what doubling the run length buys.

**Untuned knobs.** `lat_pairs = 4`, `lat_sigma = 0.05`, `mlp_sigma = 0.02`
and both learning rates were chosen once and never swept. Do not read 4 pairs
as a measured optimum; it is the smallest count that trained cleanly on the
first try.

**Platform RNG.** `std::normal_distribution` is implementation-defined.
Both libstdc++ and the MSVC STL implement it with the Marsaglia polar
method and cache the second sample of each pair, but they return the two
samples in opposite order, use slightly different rejection tests, and
convert the 32-bit engine output to a float differently in
`generate_canonical` (libstdc++ rounds `u / 2^32`, MSVC keeps the top 24
bits), so even corresponding samples can differ in the last bits, and the
occasional divergent accept/reject decision then desynchronizes the two
streams completely. `std::uniform_int_distribution`, by contrast, gives the
same minibatch indices on both: current versions of both libraries (MSVC from
VS 2022 on) use Lemire's method, and for a `512²` range, a power of two, it
never rejects. The same seed therefore gives different
perturbations on MSVC and libstdc++ and
different final PSNRs: on the default configuration, 27.18 dB fp32 (MSVC
2022 and 2026, bit-identical) versus 27.08 dB fp32 (gcc 13, WSL2). This was observed on
fresh-clone runs and is not captured in a checked-in log. Results are
reproducible per platform, not across platforms. Use your own Gaussian
generator if cross-platform determinism matters.

**Portability trivia.** `system("mkdir ... 2>nul")` creates a file named
`nul` on Linux. Default input paths relative to the working directory fail
when running from `build/Release`. Both are fixed in the code, both wasted
time.

---

## 8. Future directions

Roughly in order of expected value per line of code.

**Quantization-aware training.** Move the quantize–dequantize step inside the
decode used for ES evaluations. The float latent remains the parameter; the
optimizer sees the quantized loss. No straight-through estimator is needed.
`σ` must be at least a fraction of one quantization step or perturbations will
not cross grid boundaries. This is the most likely fix for the 4–5 bit cliff.

**2×2 phase cycling for the latent.** Perturb only texels with
`(tx mod 2, ty mod 2) = (a, b)` per evaluation and cycle the 4 phases. No two
perturbed footprints overlap, so neighbor crosstalk vanishes and each texel's
`Δ L_F(t)` is an exact function of its own `C`-vector perturbation. The
per-channel estimate remains a random projection within those `C` channels,
and each texel is sampled once per update instead of 4 times, so the
per-texel relative error becomes about `sqrt(C)` versus the current
`sqrt(d_eff/K)` with `d_eff` between about `2.25C` and `9C` (§3.3). That is a
net variance change between roughly 0.45× and 1.8×, so this is an experiment,
not a guaranteed win; cost is the same 4 evaluations (antithetic pairs) per
update as now. Exact per-channel differences, and therefore true parallel
coordinate descent with accept/reject, need `4C` evaluations per update by
cycling channels too; for single-channel latents that is the same cost as
today.

**Latent range control.** A small L2 penalty on `Z`, or a per-channel learned
scale with the latent clipped to `[−1, 1]`, would keep the quantizer's range
tight. Cheap, and it directly attacks the low-bit cliff.

**Block-compressed latents (BC1/BC4/BC7/ASTC) in the loop.** Same hook as
quantization-aware training, with the codec's encode–decode round trip as the
quantizer. Attribution should be per 4×4 block rather than per texel, since
an endpoint change moves all 16 texels. A block's footprint is the union of
its texels' footprints, `(4 + 1)·r = 5r` pixels square (40 at `r = 8`);
adjacent blocks' footprints overlap by one texel width, so 2×2 *block* phase
cycling removes inter-block crosstalk (with the same caveat as above: the
estimate within a block's own parameters is still an ES projection). BC4 is the natural first target:
one channel per texture, a ~100-line encoder, and 8 latent channels become
eight 4-bpp textures the GPU decodes natively.

**Search in the encoded domain.** Drop the float latent entirely and perturb
block endpoints and indices directly. There is then no encoder in the loop at
all, only a decoder, and the trainer is literally a block texture compressor
whose distortion metric runs through the MLP. Stochastic coordinate descent
over the encoded representation, as used in search-based texture compressors,
maps onto the phase-cycling scheme above.

**Tuning the two-level latent.** The pyramid is implemented (§1.4, §7) but
untuned. Candidates: a smaller `--lat2-sigma`; a third level; and putting
most of the channels on the coarse level (e.g. 128×128×2 + 32×32×8) to see
whether the fine level can be starved of color information cheaply.
`--lat-alt` exists but lost with a 64×64 second level (§7); it is worth
retrying only with a much coarser one.

**Learned interpolation kernels as global parameters.** Rather than feeding
cell position to the MLP (which invites grid artifacts), give the sampler a
handful of trainable coefficients, for example a per-channel blend between
bilinear and a sharper cubic tap. A dozen extra global ES dimensions, no new
decoder inputs.

**Materials, next steps.** Up to four RGB textures sharing one latent are
implemented (§1.2, `--weights`). First result: kodim23 + mario sharing
128×128×8 + 64×64×4 with a 36,36 decoder, 3000 annealed iterations
(`out_mat2.log`): 30.37 and 29.62 dB at 2.32 bpp per texture, versus roughly
30.3 (kodim23, 128×128×4 at 2.06 bpp) and 30.6 (mario, 128×128×4 + 64×64×4 at
2.61 bpp) when trained alone. Two unrelated photographs share almost nothing,
so this is the expected floor; the interesting case is a real material whose
textures are spatially correlated. What remains: non-RGB channel counts
(single-channel roughness and AO), a normal-map-aware loss, and sweeping the
latent channel budget, which is the main tuning axis for materials.

**Training schedule.** `--lr-anneal` exists and is worth 0.85 dB on mario at
6000 iterations with a single 128×128×8 latent and `ldct:2` (31.36 → 32.21 dB,
`out_mario_6k.log` vs `out_mario_anneal.log`) and 0.56 dB on the two-level
128×128×4 + 64×64×4 configuration with a 36,36 decoder (30.82 → 31.38 dB); `--sigma-anneal` exists but is untested. Candidates:
decay `σ` for the MLP as `dstd` falls; update the MLP less often once the
latent has converged (`--mlp-every` exists but is untested); start the
finite-difference phase earlier or run it on the full image
(`--mlp-fd` with `--mlp-full`, untested). `--mlp-full` and `--mlp-freeze`
exist and did nothing measurable at 3000 iterations (§7).

**Throughput.** The forward pass is scalar. Vectorizing across 8 pixels with
AVX2, parallelizing the RNG and scatter loops, and generating `ε` per thread
would plausibly give 5–10× on CPU. On a GPU the whole ES step is a compute
shader using the gather form of §3.4: every evaluation is a full decode,
which is the operation the GPU performs at runtime anyway.

---

## 9. Checklist for a reimplementation

1. Per-pixel loss that sums over pixels. Confirm each pixel's dependency set
   `S(p)` on the latent is small and known, and write down `F(t)` in closed
   form.
2. Reproduce the tap formula of §1.1 exactly, including the half-texel offset
   and clamping indices but not `fx, fy`.
3. Antithetic ES for the global decoder weights: same minibatch within a
   pair (mandatory), Adam. Optionally finish with per-weight central finite
   differences and a fresh Adam state.
4. Latent step: perturb everything, decode `±`, per-pixel loss differences,
   scatter into the distinct texels of `S(p)` (or gather over `F(t)`),
   multiply by `ε`, accumulate over pairs, Adam. Independent Adam state from
   the MLP's.
5. Guard double-counting at clamped borders if you scatter.
6. Print `dstd`, latent mean/sd/max, and quantized PSNR every few
   iterations. You cannot tune this blind.
7. Start with no positional input beyond `u, v`. Add features only if they
   measurably help at your latent resolution.
8. Leaky ReLU, sigmoid output, small latent init.
9. Save the model with its full configuration in the header, including the
   output mapping.
