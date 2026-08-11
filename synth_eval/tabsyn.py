"""synth_eval.tabsyn — mixed-type tabular VAE + latent diffusion synthesizer.

Reimplements the TabSyn architecture (Zhang et al., "Mixed-Type Tabular Data
Synthesis with Score-based Diffusion in Latent Space", ICLR 2024): a small
Transformer VAE tokenizes each row's numeric + categorical columns into a
per-column continuous latent, then a score-based (EDM / Karras et al. 2022)
diffusion model is trained on those latents so sampling draws from a learned
latent distribution instead of a plain Gaussian prior.

This is NOT a vendored copy of the paper's own reference implementation --
that's script/config-driven research code tied to a fixed, pre-described
dataset (no pip package, no fit()/sample() API), which doesn't fit a tool
where the schema is never known ahead of time. Rebuilt against this
pipeline's own mixed-type column handling (see synth_eval.columns) instead,
with a small default model sized for the row/column counts this dashboard
actually sees on CPU or a single GPU, not the paper's benchmark-scale
compute budget -- see TabSynSynthesizer's docstring for what that trades
away.

Only numeric and categorical columns are modeled (id/datetime/free-text
columns never reach a synthesizer in this pipeline -- see
backend.dashboard_core's `keep`/`reduced_train` split -- so this only needs
to handle the two roles classify_columns actually hands to a synthesizer).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_EPS = 1e-6


def _sdtypes_from_single_meta(single_meta) -> Dict[str, str]:
    """Extract {column: sdtype} from an SDV single-table metadata object.

    Confirmed by direct inspection: even a single-table view obtained via
    ``Metadata.get_table_metadata(name)`` still wraps its columns in
    ``{"tables": {name: {"columns": {...}}}}`` -- ``to_dict()`` never
    flattens it. ``synth_eval.columns._metadata_sdtypes`` handles that shape
    too, but needs the caller to already know ``name``; this constructor
    doesn't (``build_single_table_synthesizer`` never passes one), so this
    reads whichever single table is present instead of guessing one. Silently
    falling back to a dtype-only guess when this comes back empty (see
    TabSynSynthesizer.fit) would mean every *_TP_CD-style numeric-coded
    categorical gets modeled as a continuous quantity instead -- exactly the
    kind of wrong-mode-for-the-data mismatch that destabilizes training on
    real production-shaped data, not just a lost nicety.
    """
    if single_meta is None:
        return {}
    try:
        d = single_meta.to_dict() if hasattr(single_meta, "to_dict") else single_meta
    except Exception:
        return {}
    if not isinstance(d, dict):
        return {}
    tables = d.get("tables")
    tbl = next(iter(tables.values())) if isinstance(tables, dict) and tables else d
    cols = tbl.get("columns", {}) if isinstance(tbl, dict) else {}
    return {c: spec.get("sdtype") for c, spec in cols.items() if isinstance(spec, dict) and "sdtype" in spec}


# ---------------------------------------------------------------------------
# column encoding: numeric -> z-scored float, categorical -> integer index
# ---------------------------------------------------------------------------

class _ColumnSpec:
    __slots__ = ("name", "kind", "mean", "std", "categories", "is_integer", "missing_rate")

    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind            # "numerical" | "categorical" | "passthrough"
        self.mean = 0.0
        self.std = 1.0
        self.categories: list = []  # categorical: fitted vocabulary, real dtype preserved
        self.is_integer = False     # numerical: round + cast back to int at sample time
        self.missing_rate = 0.0


def _build_column_specs(df: pd.DataFrame, sdtypes: Dict[str, str]) -> List[_ColumnSpec]:
    specs = []
    for col in df.columns:
        sdt = sdtypes.get(col)
        if sdt == "numerical":
            specs.append(_ColumnSpec(col, "numerical"))
        elif sdt in ("categorical", "boolean"):
            specs.append(_ColumnSpec(col, "categorical"))
        else:
            # id / datetime / unknown -- rare here (this pipeline only ever
            # hands a synthesizer numeric+categorical columns plus, at most,
            # a relationship key). Not modeled; see _passthrough_sample.
            specs.append(_ColumnSpec(col, "passthrough"))
    return specs


def _fit_column_stats(df: pd.DataFrame, specs: List[_ColumnSpec]) -> None:
    for spec in specs:
        s = df[spec.name]
        spec.missing_rate = float(s.isna().mean())
        if spec.kind == "numerical":
            clean = pd.to_numeric(s, errors="coerce")
            spec.is_integer = bool((clean.dropna() % 1 == 0).all()) if clean.notna().any() else False
            fill = clean.median()
            fill = 0.0 if pd.isna(fill) else float(fill)
            clean = clean.fillna(fill)
            spec.mean = float(clean.mean())
            spec.std = float(clean.std()) or 1.0
        elif spec.kind == "categorical":
            mode = s.mode(dropna=True)
            fill = mode.iat[0] if len(mode) else "__missing__"
            clean = s.fillna(fill)
            spec.categories = sorted(clean.unique().tolist(), key=str)


def _encode_numeric(df: pd.DataFrame, spec: _ColumnSpec) -> np.ndarray:
    clean = pd.to_numeric(df[spec.name], errors="coerce")
    clean = clean.fillna(spec.mean * spec.std + spec.mean)  # already-fit mean as a safe fallback
    return ((clean.to_numpy(dtype=float) - spec.mean) / spec.std).astype(np.float32)


def _encode_categorical(df: pd.DataFrame, spec: _ColumnSpec) -> np.ndarray:
    idx = {v: i for i, v in enumerate(spec.categories)}
    fallback = spec.categories[0] if spec.categories else None
    clean = df[spec.name].fillna(fallback)
    return np.array([idx.get(v, 0) for v in clean.tolist()], dtype=np.int64)


# ---------------------------------------------------------------------------
# model: tokenizer + Transformer VAE
# ---------------------------------------------------------------------------

def _build_vae(specs: List[_ColumnSpec], d_token: int, d_latent: int, nhead: int, layers: int):
    import torch
    from torch import nn

    class _VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.specs = specs
            self.d_token, self.d_latent = d_token, d_latent
            self.num_tok = nn.ModuleDict()
            self.cat_tok = nn.ModuleDict()
            self.num_head = nn.ModuleDict()
            self.cat_head = nn.ModuleDict()
            for spec in specs:
                if spec.kind == "numerical":
                    self.num_tok[spec.name] = nn.Linear(1, d_token)
                    self.num_head[spec.name] = nn.Linear(d_token, 1)
                elif spec.kind == "categorical":
                    n = max(1, len(spec.categories))
                    self.cat_tok[spec.name] = nn.Embedding(n, d_token)
                    self.cat_head[spec.name] = nn.Linear(d_token, n)
            self.n_cols = len(self.num_tok) + len(self.cat_tok)
            self.pos_emb = nn.Parameter(torch.randn(1, self.n_cols, d_token) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_token, nhead=nhead, dim_feedforward=d_token * 2,
                batch_first=True, dropout=0.0)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            dec_layer = nn.TransformerEncoderLayer(
                d_model=d_token, nhead=nhead, dim_feedforward=d_token * 2,
                batch_first=True, dropout=0.0)
            self.decoder = nn.TransformerEncoder(dec_layer, num_layers=layers)
            self.to_mu = nn.Linear(d_token, d_latent)
            self.to_logvar = nn.Linear(d_token, d_latent)
            self.from_latent = nn.Linear(d_latent, d_token)
            # fixed column order -- every tensor below is indexed positionally
            self._modeled = [s for s in specs if s.kind in ("numerical", "categorical")]

        def _tokenize(self, batch: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
            toks = []
            for spec in self._modeled:
                x = batch[spec.name]
                if spec.kind == "numerical":
                    toks.append(self.num_tok[spec.name](x.unsqueeze(-1)))
                else:
                    toks.append(self.cat_tok[spec.name](x))
            return torch.stack(toks, dim=1) + self.pos_emb

        def encode(self, batch, sample: bool = True):
            h = self.encoder(self._tokenize(batch))
            mu, logvar = self.to_mu(h), self.to_logvar(h)
            # unclamped logvar is a classic VAE instability: a large positive
            # value makes exp(logvar) overflow to inf in both the
            # reparameterization trick below and the KL term in _train_vae,
            # and one inf loss corrupts every weight in the model via the
            # very next gradient step (confirmed: the whole model went NaN
            # training on real 16-column data, grad clipping alone didn't
            # help since the gradient was already non-finite by the time
            # it's computed, not just large)
            logvar = logvar.clamp(-10.0, 10.0)
            if sample:
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            else:
                z = mu
            return z, mu, logvar

        def decode(self, z: "torch.Tensor") -> "torch.Tensor":
            h = self.from_latent(z) + self.pos_emb
            return self.decoder(h)

        def reconstruct(self, h):
            out = {}
            for i, spec in enumerate(self._modeled):
                col_h = h[:, i, :]
                if spec.kind == "numerical":
                    out[spec.name] = self.num_head[spec.name](col_h).squeeze(-1)
                else:
                    out[spec.name] = self.cat_head[spec.name](col_h)
            return out

        def forward(self, batch):
            z, mu, logvar = self.encode(batch)
            recon = self.reconstruct(self.decode(z))
            return recon, mu, logvar, z

    return _VAE()


def _to_batch(df: pd.DataFrame, specs: List[_ColumnSpec], device) -> Dict[str, "torch.Tensor"]:
    import torch

    batch = {}
    for spec in specs:
        if spec.kind == "numerical":
            batch[spec.name] = torch.as_tensor(_encode_numeric(df, spec), device=device)
        elif spec.kind == "categorical":
            batch[spec.name] = torch.as_tensor(_encode_categorical(df, spec), device=device)
    return batch


def _train_vae(vae, df: pd.DataFrame, specs: List[_ColumnSpec], epochs: int, device,
                batch_size: int = 256, lr: float = 1e-3, beta: float = 1e-2) -> None:
    import torch
    from torch import nn
    from tqdm import tqdm

    vae.to(device).train()
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    n = len(df)
    batch_size = max(1, min(batch_size, n))
    full = _to_batch(df, specs, device)
    modeled = vae._modeled
    # a real tqdm bar (stderr, same as CTGAN/TVAE's own verbose=True output) --
    # not a bespoke callback: backend.dashboard_core's _TqdmTee already
    # intercepts stderr progress bars and forwards a compact live line to the
    # job console, this rides that same mechanism for free instead of
    # plumbing a separate progress callback through suite.py just for this
    for _ in tqdm(range(max(1, epochs)), desc="TabSyn·VAE"):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = {k: v[idx] for k, v in full.items()}
            recon, mu, logvar, _ = vae(batch)
            loss = torch.zeros((), device=device)
            for spec in modeled:
                if spec.kind == "numerical":
                    loss = loss + nn.functional.mse_loss(recon[spec.name], batch[spec.name])
                else:
                    loss = loss + nn.functional.cross_entropy(recon[spec.name], batch[spec.name])
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            total = loss + beta * kl
            opt.zero_grad()
            if not torch.isfinite(total):
                # defensive second layer behind logvar clamping above: skip
                # rather than let one bad batch's non-finite gradient corrupt
                # every weight in the model (observed: silent full-model NaN,
                # no exception, until sampling's softmax surfaced it as
                # "probabilities contain NaN" several steps downstream)
                continue
            total.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            opt.step()
    vae.eval()


# ---------------------------------------------------------------------------
# EDM-style score-based diffusion over the VAE's latent space
# (Karras, Aittala, Aila, Laine, "Elucidating the Design Space of
# Diffusion-Based Generative Models", NeurIPS 2022 -- the formulation
# TabSyn's own diffusion stage adopts, applied here to flat latent vectors
# instead of images: no U-Net/conv structure needed, a small residual MLP
# with a sinusoidal noise-level embedding is the whole denoiser)
# ---------------------------------------------------------------------------

def _build_denoiser(dim: int, hidden: int = 256, depth: int = 4):
    from torch import nn

    class _SinusoidalEmb(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.emb_dim = emb_dim

        def forward(self, x):
            import torch
            half = self.emb_dim // 2
            freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device) / max(half - 1, 1))
            args = x[:, None] * freqs[None, :]
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if self.emb_dim % 2:
                emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
            return emb

    class _ResBlock(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.norm = nn.LayerNorm(h)
            self.net = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))

        def forward(self, x, t):
            return x + self.net(self.norm(x) + t)

    class _Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.sigma_emb = _SinusoidalEmb(hidden)
            self.time_proj = nn.Linear(hidden, hidden)
            self.in_proj = nn.Linear(dim, hidden)
            self.blocks = nn.ModuleList([_ResBlock(hidden) for _ in range(depth)])
            self.out_proj = nn.Linear(hidden, dim)

        def forward(self, x, c_noise):
            t = self.time_proj(self.sigma_emb(c_noise))
            h = self.in_proj(x)
            for blk in self.blocks:
                h = blk(h, t)
            return self.out_proj(h)

    return _Denoiser()


def _edm_precond(sigma, sigma_data: float = 1.0):
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
    c_in = 1.0 / (sigma ** 2 + sigma_data ** 2).sqrt()
    c_noise = sigma.log() / 4.0
    return c_skip, c_out, c_in, c_noise


def _train_denoiser(denoiser, z: "torch.Tensor", epochs: int, device,
                     batch_size: int = 256, lr: float = 1e-3,
                     p_mean: float = -1.2, p_std: float = 1.2, sigma_data: float = 1.0) -> None:
    import torch
    from tqdm import tqdm

    denoiser.to(device).train()
    opt = torch.optim.Adam(denoiser.parameters(), lr=lr)
    n = z.shape[0]
    batch_size = max(1, min(batch_size, n))
    for _ in tqdm(range(max(1, epochs)), desc="TabSyn·diffusion"):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            x0 = z[perm[start:start + batch_size]]
            b = x0.shape[0]
            sigma = (torch.randn(b, device=device) * p_std + p_mean).exp()
            weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
            noise = torch.randn_like(x0) * sigma[:, None]
            c_skip, c_out, c_in, c_noise = _edm_precond(sigma, sigma_data)
            x_noisy = x0 + noise
            f_x = denoiser(c_in[:, None] * x_noisy, c_noise)
            d_x = c_skip[:, None] * x_noisy + c_out[:, None] * f_x
            loss = (weight[:, None] * (d_x - x0) ** 2).mean()
            opt.zero_grad()
            if not torch.isfinite(loss):
                continue  # same defensive skip as _train_vae, see its comment
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            opt.step()
    denoiser.eval()


def _edm_sample(denoiser, n: int, dim: int, device, num_steps: int = 30,
                 sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0,
                 sigma_data: float = 1.0):
    """Deterministic 2nd-order Heun ODE sampler (Karras et al. Algorithm 1)."""
    import torch

    with torch.no_grad():
        step = torch.arange(num_steps, dtype=torch.float64, device=device)
        sigmas = (sigma_max ** (1 / rho) + step / max(num_steps - 1, 1)
                  * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigmas = torch.cat([sigmas, torch.zeros(1, dtype=torch.float64, device=device)])

        def denoise(x, sigma_scalar):
            sigma_b = torch.full((x.shape[0],), float(sigma_scalar), dtype=torch.float64, device=device)
            c_skip, c_out, c_in, c_noise = _edm_precond(sigma_b, sigma_data)
            f_x = denoiser((c_in[:, None] * x).float(), c_noise.float()).double()
            return c_skip[:, None] * x + c_out[:, None] * f_x

        x = torch.randn(n, dim, dtype=torch.float64, device=device) * sigmas[0]
        for i in range(num_steps):
            s_cur, s_next = sigmas[i], sigmas[i + 1]
            d_cur = (x - denoise(x, s_cur)) / s_cur
            x_next = x + (s_next - s_cur) * d_cur
            if s_next > 0:
                d_next = (x_next - denoise(x_next, s_next)) / s_next
                x_next = x + (s_next - s_cur) * 0.5 * (d_cur + d_next)
            x = x_next
        return x.float()


# ---------------------------------------------------------------------------
# public synthesizer -- matches the SDV single-table interface this
# pipeline's other synthesizers implement (see synth_eval.suite)
# ---------------------------------------------------------------------------

class TabSynSynthesizer:
    """Fit a VAE + latent-diffusion model on one table, then sample from it.

    Same interface SDV's own single-table synthesizers expose --
    ``fit(df)``, ``sample(num_rows=n)``, ``_set_random_state(seed)`` -- so it
    drops into ``synth_eval.suite.build_single_table_synthesizer`` and every
    downstream step (post-hoc FK relinking, the reject-and-resample privacy
    filter, refill/PII) alongside GaussianCopula/CTGAN/TVAE/CopulaGAN without
    special-casing.

    ``epochs`` is used IN FULL by each of the two training stages (VAE, then
    the diffusion model on the VAE's frozen latents) -- epochs=350 means 350
    passes of VAE training followed by 350 passes of diffusion training, not
    350 total split between them, so a comparable epochs value gives each
    stage roughly the same training depth as a single-stage synth like TVAE
    gets. The published TabSyn numbers used a much larger compute budget
    (thousands of epochs/steps) than this dashboard's epochs slider
    (10-1000) is meant for; the model size below (small Transformer
    tokenizer, small MLP denoiser) is picked to actually finish training in
    that budget on the row/column counts this tool sees, not to reproduce
    the paper's benchmark-scale results exactly. Total wall-clock roughly
    doubles versus splitting the same epochs value across both stages.
    """

    def __init__(self, single_meta=None, epochs: int = 300,
                 d_token: int = 32, d_latent: int = 4, nhead: int = 4,
                 vae_layers: int = 2, denoiser_hidden: int = 256, denoiser_depth: int = 4,
                 sample_steps: int = 30):
        self._sdtypes = _sdtypes_from_single_meta(single_meta)
        self.epochs = epochs
        self.d_token, self.d_latent, self.nhead = d_token, d_latent, nhead
        self.vae_layers = vae_layers
        self.denoiser_hidden, self.denoiser_depth = denoiser_hidden, denoiser_depth
        self.sample_steps = sample_steps
        self._seed = 0
        self._fitted = False

    def _set_random_state(self, seed: int) -> None:
        self._seed = int(seed)

    def add_constraints(self, constraints):  # pragma: no cover - unsupported, see _apply's try/except
        raise NotImplementedError("TabSynSynthesizer does not support SDV constraints yet")

    def fit(self, df: pd.DataFrame) -> None:
        import torch

        # deliberately NOT torch.manual_seed(self._seed) here: _set_random_state
        # (which sets self._seed) is called by the suite AFTER fit(), same timing
        # SDV's own synthesizers use it at -- fit-time reproducibility instead
        # comes from _seed_global(random_state), already applied by the caller
        # before this constructor even ran. Seeding again here with the
        # not-yet-updated self._seed default would clobber that with the wrong
        # value instead of leaving it alone.
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._columns = list(df.columns)
        sdtypes = self._sdtypes or {c: ("numerical" if pd.api.types.is_numeric_dtype(df[c]) else "categorical")
                                     for c in df.columns}
        self._specs = _build_column_specs(df, sdtypes)
        _fit_column_stats(df, self._specs)
        self._n_real = len(df)
        modeled = [s for s in self._specs if s.kind in ("numerical", "categorical")]
        if not modeled:
            self._fitted = True  # nothing to model -- every column is passthrough
            return

        stage_epochs = max(1, self.epochs)
        self._vae = _build_vae(self._specs, self.d_token, self.d_latent, self.nhead, self.vae_layers)
        _train_vae(self._vae, df, self._specs, stage_epochs, self._device)

        import torch as _t
        with _t.no_grad():
            full = _to_batch(df, self._specs, self._device)
            z, mu, _ = self._vae.encode(full, sample=False)
            z_flat = mu.reshape(mu.shape[0], -1)
            self._z_mean = z_flat.mean(0, keepdim=True)
            self._z_std = z_flat.std(0, keepdim=True).clamp_min(_EPS)
            z_norm = (z_flat - self._z_mean) / self._z_std

        self._latent_dim = z_norm.shape[1]
        self._denoiser = _build_denoiser(self._latent_dim, self.denoiser_hidden, self.denoiser_depth)
        _train_denoiser(self._denoiser, z_norm, stage_epochs, self._device)
        self._fitted = True

    def sample(self, num_rows: int) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("TabSynSynthesizer.sample() called before fit()")
        n = max(0, int(num_rows))
        out = {}
        modeled = [s for s in self._specs if s.kind in ("numerical", "categorical")]
        if n == 0:
            for spec in self._specs:
                out[spec.name] = pd.Series([], dtype=object)
            return pd.DataFrame(out)[self._columns]

        if modeled:
            import torch

            torch.manual_seed(self._seed)
            z_norm = _edm_sample(self._denoiser, n, self._latent_dim, self._device,
                                 num_steps=self.sample_steps)
            z_flat = z_norm * self._z_std + self._z_mean
            z = z_flat.reshape(n, len(modeled), self.d_latent)
            with torch.no_grad():
                h = self._vae.decode(z)
                recon = self._vae.reconstruct(h)
            for i, spec in enumerate(modeled):
                if spec.kind == "numerical":
                    vals = recon[spec.name].detach().cpu().numpy() * spec.std + spec.mean
                    if spec.is_integer:
                        vals = np.rint(vals)
                    out[spec.name] = vals
                else:
                    probs = torch.softmax(recon[spec.name], dim=-1).detach().cpu().numpy()
                    cat_rng = np.random.RandomState(self._seed)
                    idx = np.array([cat_rng.choice(len(p), p=p / p.sum()) for p in probs])
                    out[spec.name] = [spec.categories[i] for i in idx]

        rng = np.random.RandomState(self._seed)
        for spec in self._specs:
            if spec.kind == "passthrough":
                out[spec.name] = self._passthrough_sample(spec, n)
            elif spec.missing_rate > 0:
                miss_n = int(round(n * spec.missing_rate))
                if not miss_n:
                    continue
                miss_idx = rng.choice(n, size=miss_n, replace=False)
                if spec.kind == "numerical":
                    arr = np.asarray(out[spec.name], dtype=float)
                    arr[miss_idx] = np.nan
                else:
                    arr = np.asarray(out[spec.name], dtype=object)
                    arr[miss_idx] = None
                out[spec.name] = arr

        return pd.DataFrame(out)[self._columns]

    @staticmethod
    def _passthrough_sample(spec: _ColumnSpec, n: int):
        # id/datetime/unknown columns aren't modeled -- a unique-per-row
        # placeholder is all that's needed here; single-table synths that
        # participate in a relationship get this column relinked to real
        # parent keys afterward (see synth_eval.link), everything else just
        # needs a stable, non-colliding value.
        return [f"{spec.name}_{i}" for i in range(n)]
