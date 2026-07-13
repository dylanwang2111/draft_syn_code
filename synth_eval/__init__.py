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
from .entity import build_entity_hub, entity_key_tables  # noqa: F401
from .scd import repair_scd_timeline  # noqa: F401
