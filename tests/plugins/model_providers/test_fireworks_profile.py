"""Unit tests for the Fireworks AI provider profile.

Pins the profile's contract without going live: identity, alias registration,
and the pay-as-you-go model defaults (direct catalog ``/models/``
IDs, not the router-only tier).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def fireworks_profile():
    """Resolve the registered Fireworks profile through the real discovery path."""
    # Importing model_tools triggers plugin discovery, registering the profile.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("fireworks")
    assert profile is not None, "fireworks provider profile must be registered"
    return profile


class TestFireworksIdentity:
    def test_core_fields(self, fireworks_profile):
        p = fireworks_profile
        assert p.name == "fireworks"
        assert p.auth_type == "api_key"
        assert p.base_url == "https://api.fireworks.ai/inference/v1"
        assert "FIREWORKS_API_KEY" in p.env_vars
        assert "FIREWORKS_BASE_URL" not in p.env_vars

    def test_display_metadata_present(self, fireworks_profile):
        # Prominence copy is surfaced in the picker; keep it non-empty rather
        # than pinning exact marketing wording (that's expected to change).
        assert fireworks_profile.display_name
        assert fireworks_profile.description
        assert fireworks_profile.signup_url.startswith("https://")


class TestFireworksHeaders:
    def test_attribution_matches_canonical_hermes_values(self, fireworks_profile):
        """Fireworks requests carry the same attribution identity Hermes sends
        everywhere else.

        Asserted against the shared constant rather than the literals so a
        rebrand can't leave one provider on a stale referer/title.
        """
        from agent.auxiliary_client import _OR_HEADERS_BASE

        headers = fireworks_profile.default_headers
        assert headers["HTTP-Referer"] == _OR_HEADERS_BASE["HTTP-Referer"]
        assert headers["X-Title"] == _OR_HEADERS_BASE["X-Title"]

    def test_user_agent_identifies_hermes(self, fireworks_profile):
        # Prefix, not the full string — the version moves every release.
        assert fireworks_profile.default_headers["User-Agent"].startswith("HermesAgent/")


class TestFireworksAliases:
    @pytest.mark.parametrize("alias", ["fireworks-ai", "fw"])
    def test_alias_resolves_via_registry(self, fireworks_profile, alias):
        import providers

        resolved = providers.get_provider_profile(alias)
        assert resolved is not None
        assert resolved.name == "fireworks"

    def test_aliases_declared_on_profile(self, fireworks_profile):
        assert "fireworks-ai" in fireworks_profile.aliases
        assert "fw" in fireworks_profile.aliases


class TestFireworksModelDefaults:
    """Defaults must be usable with a standard pay-as-you-go key.

    PAYG keys address ``accounts/fireworks/models/...`` directly; the bundled
    defaults target that (the BYOK motion) so a fresh key works out of the box,
    and use the standard tier rather than turbo as the out-of-box default.
    """

    def test_aux_model_is_payg_model_not_router(self, fireworks_profile):
        aux = fireworks_profile.default_aux_model
        assert aux.startswith("accounts/fireworks/models/"), aux
        assert "/routers/" not in aux
        assert "turbo" not in aux.lower()

    def test_fallback_models_are_payg_models_not_routers(self, fireworks_profile):
        assert fireworks_profile.fallback_models, "expected curated fallbacks"
        for model in fireworks_profile.fallback_models:
            assert model.startswith("accounts/fireworks/models/"), model
            assert "/routers/" not in model
            assert "turbo" not in model.lower(), model


class TestFireworksReasoningDisable:
    """The endpoint's strict schema rejects any ``reasoning`` body field with HTTP 400
    ("Extra inputs are not permitted, field: 'reasoning'"), and GLM-5.x is thinking-only
    (``chat_template_kwargs: {thinking: false}`` is refused too) — a disable is
    unexpressible on the wire. The profile must therefore own the reasoning projection
    (omitting the field), never falling through to the generic ``extra_body.reasoning``
    fallback that 400s every auxiliary disable request."""

    def test_disable_omits_reasoning_field(self, fireworks_profile):
        extra, top = fireworks_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, supports_reasoning=True, model=None)
        assert "reasoning" not in extra and "reasoning" not in top

    def test_profile_handles_reasoning_projection(self, fireworks_profile):
        # _project_provider_profile treats a profile overriding build_api_kwargs_extras
        # as reasoning-aware, so the generic extra_body.reasoning fallback never fires.
        from providers.base import ProviderProfile

        assert type(fireworks_profile).build_api_kwargs_extras is not ProviderProfile.build_api_kwargs_extras

    def test_disable_reaches_the_wire_without_reasoning(self, fireworks_profile):
        """Full kwargs projection: the wire request for a disable carries no reasoning field."""
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            "fireworks", "accounts/fireworks/models/glm-5p3-flash",
            [{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=16,
            reasoning_config={"enabled": False}, base_url="https://api.fireworks.ai/inference/v1",
        )
        assert "reasoning" not in (kwargs.get("extra_body") or {})
