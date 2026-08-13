"""synth_eval — evaluation toolkit for SDV synthetic data.

Split into submodules (columns / viz / privacy / efficacy / suite / compare)
but re-exported here so ``import synth_eval as se`` keeps exposing every helper
exactly as before.
"""
from __future__ import annotations

from ._common import (
    plt, sns, _HAS_SNS, _fig_to_base64, _save_fig, _single_table_metadata,
    SYNTH_PALETTE, _color_for,
)
from .columns import *  # noqa: F401,F403
from .columns import ColumnRoles, _coerce_frame, _fit_mixed_encoder, _encode
from .viz import *  # noqa: F401,F403
from .privacy import *  # noqa: F401,F403
from .efficacy import *  # noqa: F401,F403
from .suite import *  # noqa: F401,F403
from .compare import *  # noqa: F401,F403
from .entity import (  # noqa: F401
    build_entity_hub,
    derive_synthetic_hub_pool,
    entity_key_tables,
    _resolve_key_column,
)
from .link import link_relationships, link_table  # noqa: F401
from .cross_table import entity_cross_table_trends  # noqa: F401
from .scd import (  # noqa: F401
    detect_ordered_date_pairs, detect_scd_window_pair, find_mirror_pair,
    repair_scd_timeline, scd_duration_fidelity,
)
from .pii import detect_pii, fake_series, apply_pii_plan  # noqa: F401
