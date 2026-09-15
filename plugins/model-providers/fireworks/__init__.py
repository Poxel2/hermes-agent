"""Fireworks AI provider profile. Models are addressed by full catalog ID
(``accounts/fireworks/models/<slug>``), tracking fw-ai/fireconnect ``setup-cli``."""

from typing import Any

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile


class FireworksProfile(ProviderProfile):
    """Fireworks — strict OpenAI-compatible schema (no ``extra_body.reasoning``)."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, supports_reasoning: bool = False,
        model: str | None = None, **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Fireworks validates the request body with a strict schema: any ``reasoning``
        # field yields HTTP 400 "Extra inputs are not permitted, field: 'reasoning'"
        # (live-probed on glm-5p3-flash / glm-5p2 / kimi-k2p6). Current GLM-5.x
        # deployments are thinking-only — ``chat_template_kwargs: {thinking: false}``
        # is refused with "GLM-5.3 is a thinking-only model" — so a disable cannot be
        # expressed on this endpoint at all. Omit the field and let the model think
        # (same pattern as OpenRouter's reasoning-mandatory routes); callers that need
        # the full budget for JSON output must raise ``max_tokens`` instead.
        return {}, {}


fireworks = FireworksProfile(
    name="fireworks", aliases=("fireworks-ai", "fw"), display_name="Fireworks AI",
    description="Fireworks AI — OpenAI-compatible direct model API",
    signup_url="https://app.fireworks.ai/settings/users/api-keys", env_vars=("FIREWORKS_API_KEY",),
    base_url="https://api.fireworks.ai/inference/v1", auth_type="api_key",
    # Attribution headers (canonical Hermes set); via default_headers so they
    # survive switch_model and credential rotation.
    default_headers={
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-Title": "Hermes Agent",
        "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
    },
    default_aux_model="accounts/fireworks/models/glm-5p2",
    # Picker safety net when the live catalog fetch fails.
    fallback_models=(
        "accounts/fireworks/models/kimi-k2p6", "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/kimi-k2p7-code",
    ),
)

register_provider(fireworks)
