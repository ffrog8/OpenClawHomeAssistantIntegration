"""The OpenClaw integration.

Sets up the OpenClaw integration: API client, coordinator, platforms, and services.
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from datetime import datetime, timezone
import logging
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

import voluptuous as vol
from homeassistant.components import websocket_api

try:
    from homeassistant.components.lovelace.const import LOVELACE_DATA
except ImportError:  # pragma: no cover
    LOVELACE_DATA = "lovelace"

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse, callback
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .api import OpenClawApiClient, OpenClawApiError
from .const import (
    ATTR_ANALYSIS,
    ATTR_AGENT_ID,
    ATTR_FILE_COUNT,
    ATTR_FILE_PATHS,
    ATTR_IMAGE_COUNT,
    ATTR_IMAGE_PATHS,
    ATTR_INPUT_COUNT,
    ATTR_INSTRUCTIONS,
    ATTR_MESSAGE,
    ATTR_PROMPT,
    ATTR_MODEL,
    ATTR_OK,
    ATTR_RESULT,
    ATTR_ERROR,
    ATTR_DURATION_MS,
    ATTR_SESSION_ID,
    ATTR_SESSION_KEY,
    ATTR_TOOL,
    ATTR_ACTION,
    ATTR_ARGS,
    ATTR_DRY_RUN,
    ATTR_MESSAGE_CHANNEL,
    ATTR_ACCOUNT_ID,
    ATTR_SOURCE,
    ATTR_TIMESTAMP,
    CONF_ADDON_CONFIG_PATH,
    CONF_AGENT_ID,
    CONF_VOICE_AGENT_ID,
    CONF_GATEWAY_HOST,
    CONF_GATEWAY_PORT,
    CONF_GATEWAY_TOKEN,
    CONF_USE_SSL,
    CONF_VERIFY_SSL,
    CONF_CONTEXT_MAX_CHARS,
    CONF_CONTEXT_STRATEGY,
    CONF_ENABLE_TOOL_CALLS,
    CONF_INCLUDE_EXPOSED_CONTEXT,
    CONF_WAKE_WORD,
    CONF_WAKE_WORD_ENABLED,
    CONF_ALLOW_BRAVE_WEBSPEECH,
    CONF_BROWSER_VOICE_LANGUAGE,
    CONF_VOICE_PROVIDER,
    CONF_THINKING_TIMEOUT,
    CONTEXT_STRATEGY_TRUNCATE,
    DEFAULT_AGENT_ID,
    DEFAULT_VOICE_AGENT_ID,
    DEFAULT_CONTEXT_MAX_CHARS,
    DEFAULT_CONTEXT_STRATEGY,
    DEFAULT_ENABLE_TOOL_CALLS,
    DEFAULT_INCLUDE_EXPOSED_CONTEXT,
    DEFAULT_WAKE_WORD,
    DEFAULT_WAKE_WORD_ENABLED,
    DEFAULT_ALLOW_BRAVE_WEBSPEECH,
    DEFAULT_BROWSER_VOICE_LANGUAGE,
    DEFAULT_VOICE_PROVIDER,
    DEFAULT_THINKING_TIMEOUT,
    DOMAIN,
    EVENT_INPUT_ANALYSIS_RECEIVED,
    EVENT_MESSAGE_RECEIVED,
    EVENT_TOOL_INVOKED,
    OPENCLAW_CONFIG_REL_PATH,
    PLATFORMS,
    SERVICE_ANALYZE_INPUTS,
    SERVICE_CLEAR_HISTORY,
    SERVICE_INVOKE_TOOL,
    SERVICE_SEND_MESSAGE,
)
from .coordinator import OpenClawCoordinator
from .exposure import apply_context_policy, build_exposed_entities_context
from .helpers import extract_text_recursive

_LOGGER = logging.getLogger(__name__)

_MAX_CHAT_HISTORY = 200
_MAX_ANALYZE_INPUTS = 8
_MAX_ANALYZE_IMAGES = 8
_ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/heic",
    "image/heif",
}
_ALLOWED_FILE_MIME_TYPES = {
    "text/plain",
    "text/markdown",
    "text/html",
    "text/csv",
    "application/json",
    "application/pdf",
}

_VOICE_REQUEST_HEADERS = {
    "x-openclaw-source": "voice",
    "x-ha-voice": "true",
    "x-openclaw-message-channel": "voice",
}

# Path to the chat card JS inside the integration package (custom_components/openclaw/www/)
_CARD_FILENAME = "openclaw-chat-card.js"
_CARD_PATH = Path(__file__).parent / "www" / _CARD_FILENAME
# URL at which the card JS is served (registered via register_static_path)
_CARD_STATIC_URL = f"/openclaw/{_CARD_FILENAME}"
# Versioned URL used for Lovelace resource registration to avoid stale browser cache
_CARD_URL = f"{_CARD_STATIC_URL}?v=0.1.63"

OpenClawConfigEntry = ConfigEntry


# Service call schemas
SEND_MESSAGE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_MESSAGE): cv.string,
        vol.Optional(ATTR_SOURCE): cv.string,
        vol.Optional(ATTR_SESSION_ID): cv.string,
        vol.Optional(ATTR_AGENT_ID): cv.string,
    }
)


def _validate_analyze_inputs_schema(data: dict[str, Any]) -> dict[str, Any]:
    """Validate that analyze_inputs has at least one image or file path."""
    image_paths = data.get(ATTR_IMAGE_PATHS) or []
    file_paths = data.get(ATTR_FILE_PATHS) or []
    if not image_paths and not file_paths:
        raise vol.Invalid("At least one image_paths or file_paths entry is required")
    if len(image_paths) + len(file_paths) > _MAX_ANALYZE_INPUTS:
        raise vol.Invalid(f"A maximum of {_MAX_ANALYZE_INPUTS} inputs is supported")
    return data


ANALYZE_INPUTS_SCHEMA = vol.All(
    vol.Schema(
        {
            vol.Required(ATTR_PROMPT): cv.string,
            vol.Optional(ATTR_IMAGE_PATHS, default=[]): vol.All(
                cv.ensure_list, [cv.string], vol.Length(max=_MAX_ANALYZE_IMAGES)
            ),
            vol.Optional(ATTR_FILE_PATHS, default=[]): vol.All(
                cv.ensure_list, [cv.string], vol.Length(max=_MAX_ANALYZE_INPUTS)
            ),
            vol.Optional(ATTR_SESSION_ID, default="input-analysis"): cv.string,
            vol.Optional(ATTR_AGENT_ID): cv.string,
            vol.Optional(ATTR_MODEL): cv.string,
            vol.Optional(ATTR_INSTRUCTIONS): cv.string,
            vol.Optional(ATTR_SOURCE, default="automation"): cv.string,
        }
    ),
    _validate_analyze_inputs_schema,
)

CLEAR_HISTORY_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_SESSION_ID): cv.string,
    }
)

INVOKE_TOOL_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_TOOL): cv.string,
        vol.Optional(ATTR_ACTION): cv.string,
        vol.Optional(ATTR_ARGS, default={}): dict,
        vol.Optional(ATTR_SESSION_KEY): cv.string,
        vol.Optional(ATTR_DRY_RUN, default=False): cv.boolean,
        vol.Optional(ATTR_MESSAGE_CHANNEL): cv.string,
        vol.Optional(ATTR_ACCOUNT_ID): cv.string,
    }
)


async def async_setup_entry(hass: HomeAssistant, entry: OpenClawConfigEntry) -> bool:
    """Set up OpenClaw from a config entry.

    Creates the API client, coordinator, and forwards setup to platforms.
    """
    use_ssl = entry.data.get(CONF_USE_SSL, False)
    verify_ssl = entry.data.get(CONF_VERIFY_SSL, True)
    session = async_get_clientsession(hass, verify_ssl=verify_ssl)
    agent_id: str = entry.options.get(
        CONF_AGENT_ID,
        entry.data.get(CONF_AGENT_ID, DEFAULT_AGENT_ID),
    )

    client = OpenClawApiClient(
        host=entry.data[CONF_GATEWAY_HOST],
        port=entry.data[CONF_GATEWAY_PORT],
        token=entry.data[CONF_GATEWAY_TOKEN],
        use_ssl=use_ssl,
        verify_ssl=verify_ssl,
        session=session,
        agent_id=agent_id,
    )

    coordinator = OpenClawCoordinator(hass, client)

    # Store the addon config path for token re-reads
    addon_config_path = entry.data.get(CONF_ADDON_CONFIG_PATH)

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {
        "client": client,
        "coordinator": coordinator,
        "addon_config_path": addon_config_path,
        "entry": entry,
        "entry_id": entry.entry_id,
    }

    # First data fetch — if it fails the coordinator marks entities unavailable
    await coordinator.async_config_entry_first_refresh()

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register services (once, idempotent)
    _async_register_services(hass)
    _async_register_websocket_api(hass)

    # Register the frontend card resource
    hass.async_create_task(_async_register_frontend(hass))

    # Listen for addon restart events to re-read token
    if addon_config_path:
        _async_setup_token_refresh(hass, entry, client, addon_config_path)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: OpenClawConfigEntry) -> bool:
    """Unload an OpenClaw config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)
        if not hass.data[DOMAIN]:
            hass.data.pop(DOMAIN, None)

    return unload_ok


# ── Token refresh on reconnect ────────────────────────────────────────────────

def _async_setup_token_refresh(
    hass: HomeAssistant,
    entry: ConfigEntry,
    client: OpenClawApiClient,
    addon_config_path: str,
) -> None:
    """Set up a listener that re-reads the gateway token if auth fails.

    The addon may regenerate the token on restart. When we detect an auth
    failure during polling, we re-read the token from the shared filesystem
    and update the config entry + client transparently.
    """
    import json as _json

    async def _try_refresh_token() -> bool:
        """Re-read token from filesystem and update client if changed."""
        config_file = Path(addon_config_path) / OPENCLAW_CONFIG_REL_PATH

        def _read() -> str | None:
            if not config_file.exists():
                return None
            try:
                cfg = _json.loads(config_file.read_text(encoding="utf-8"))
                return cfg.get("gateway", {}).get("auth", {}).get("token")
            except Exception:  # noqa: BLE001
                return None

        new_token = await hass.async_add_executor_job(_read)
        if new_token and new_token != entry.data.get(CONF_GATEWAY_TOKEN):
            _LOGGER.info("Gateway token changed — updating config entry")
            new_data = {**entry.data, CONF_GATEWAY_TOKEN: new_token}
            hass.config_entries.async_update_entry(entry, data=new_data)
            # Update the live client in-place
            client.update_token(new_token)
            return True
        return False

    # Expose refresh function for the coordinator to call on auth errors
    hass.data[DOMAIN][entry.entry_id]["refresh_token"] = _try_refresh_token


# ── Frontend registration ─────────────────────────────────────────────────────

async def _async_register_frontend(hass: HomeAssistant) -> None:
    """Register static path + Lovelace resource for the chat card.

    Called as a fire-and-forget task from async_setup_entry.
    Wrapped entirely in try/except so it can NEVER crash the integration.
    """
    frontend_done_key = f"{DOMAIN}_frontend_registered"
    frontend_task_key = f"{DOMAIN}_frontend_registration_task"

    if hass.data.get(frontend_done_key):
        return

    existing_task = hass.data.get(frontend_task_key)
    if existing_task and not existing_task.done():
        return

    async def _register_with_retries() -> None:
        for _ in range(60):
            url = await _async_register_static_path(hass)
            if url and await _async_add_lovelace_resource(hass, url):
                hass.data[frontend_done_key] = True
                return
            await asyncio.sleep(5)

        _LOGGER.warning(
            "Could not auto-register OpenClaw chat card resource after retries. "
            "Add it manually in Dashboard resources: %s",
            _CARD_URL,
        )

    task = hass.async_create_task(_register_with_retries())
    hass.data[frontend_task_key] = task

    def _clear_task(_fut: asyncio.Future) -> None:
        hass.data.pop(frontend_task_key, None)

    task.add_done_callback(_clear_task)

    async def _on_ha_started(_event) -> None:
        if hass.data.get(frontend_done_key):
            return
        await _register_with_retries()

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _on_ha_started)


async def _async_register_static_path(hass: HomeAssistant) -> str | None:
    """Register the packaged chat-card JS as a static path when HTTP is ready."""
    static_key = f"{DOMAIN}_static_registered"
    if hass.data.get(static_key):
        return _CARD_URL

    if not _CARD_PATH.exists():
        _LOGGER.warning("Chat card JS not found at %s", _CARD_PATH)
        return None

    if hass.http is None:
        return None

    try:
        from homeassistant.components.http import StaticPathConfig  # noqa: PLC0415

        await hass.http.async_register_static_paths(
            [StaticPathConfig(_CARD_STATIC_URL, str(_CARD_PATH), cache_headers=True)]
        )
    except (ImportError, AttributeError):
        hass.http.register_static_path(_CARD_STATIC_URL, str(_CARD_PATH), True)

    hass.data[static_key] = True
    _LOGGER.debug("Registered static path: %s", _CARD_STATIC_URL)
    return _CARD_URL


async def _async_add_lovelace_resource(hass: HomeAssistant, url: str) -> bool:
    """Add the card URL to Lovelace's resource store if not already present."""
    # Lovelace stores resources in hass.data[LOVELACE_DATA].resources on
    # modern Home Assistant versions (fallback to legacy dict key below).
    # Resource collection supports:
    #   .async_items()          → list of {"id", "res_type", "url"} dicts
    #   .async_create_item(data) → persists a new resource
    lovelace_data = hass.data.get(LOVELACE_DATA) or hass.data.get("lovelace")
    if not lovelace_data:
        return False

    if isinstance(lovelace_data, dict):
        resource_collection = lovelace_data.get("resources")
    else:
        resource_collection = getattr(lovelace_data, "resources", None)

    if resource_collection is None:
        return False

    try:
        existing_items = list(resource_collection.async_items())

        desired_path = urlsplit(url).path
        legacy_paths = {
            "/openclaw/openclaw-chat-card.js",
            "/local/openclaw-chat-card.js",
            "/hacsfiles/openclaw/openclaw-chat-card.js",
        }

        to_remove: list[str] = []
        for item in existing_items:
            item_id = item.get("id")
            item_url = item.get("url")
            if not item_id or not item_url:
                continue
            item_path = urlsplit(item_url).path
            if item_path in legacy_paths and item_url != url:
                to_remove.append(item_id)

        for item_id in to_remove:
            await resource_collection.async_delete_item(item_id)
            _LOGGER.info("Removed legacy/duplicate OpenClaw Lovelace resource id=%s", item_id)

        existing_urls = {item["url"] for item in resource_collection.async_items()}
        if url in existing_urls:
            _LOGGER.debug("Lovelace resource already registered: %s", url)
            return True

        await resource_collection.async_create_item(
            {"res_type": "module", "url": url}
        )
        _LOGGER.info(
            "Auto-registered Lovelace resource: %s — the chat card is ready to use.",
            url,
        )
        return True
    except Exception as err:  # noqa: BLE001
        _LOGGER.warning(
            "Could not auto-register Lovelace resource '%s': %s. "
            "Add it manually: Settings → Dashboards → Resources.",
            url,
            err,
        )
        return False


# ── Service registration ──────────────────────────────────────────────────────

@callback
def _async_register_services(hass: HomeAssistant) -> None:
    """Register openclaw.send_message and openclaw.clear_history services."""

    def _normalize_optional_text(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        return cleaned or None

    async def handle_send_message(call: ServiceCall) -> None:
        """Handle the openclaw.send_message service call."""
        message: str = call.data[ATTR_MESSAGE]
        source: str | None = call.data.get(ATTR_SOURCE)
        session_id: str = call.data.get(ATTR_SESSION_ID) or "default"
        call_agent_id = _normalize_optional_text(call.data.get(ATTR_AGENT_ID))
        extra_headers = _VOICE_REQUEST_HEADERS if source == "voice" else None

        entry_data = _get_first_entry_data(hass)
        if not entry_data:
            _LOGGER.error("No OpenClaw integration configured")
            return

        client: OpenClawApiClient = entry_data["client"]
        coordinator: OpenClawCoordinator = entry_data["coordinator"]
        options = _get_entry_options(hass, entry_data)
        voice_agent_id = _normalize_optional_text(
            options.get(CONF_VOICE_AGENT_ID, DEFAULT_VOICE_AGENT_ID)
        )
        resolved_agent_id = call_agent_id
        if resolved_agent_id is None and source == "voice":
            resolved_agent_id = voice_agent_id

        try:
            include_context = options.get(
                CONF_INCLUDE_EXPOSED_CONTEXT,
                DEFAULT_INCLUDE_EXPOSED_CONTEXT,
            )
            max_chars = int(
                options.get(CONF_CONTEXT_MAX_CHARS, DEFAULT_CONTEXT_MAX_CHARS)
            )
            strategy = options.get(CONF_CONTEXT_STRATEGY, DEFAULT_CONTEXT_STRATEGY)
            if strategy not in {"clear", CONTEXT_STRATEGY_TRUNCATE}:
                strategy = DEFAULT_CONTEXT_STRATEGY

            raw_context = (
                build_exposed_entities_context(hass, assistant="conversation")
                if include_context
                else None
            )
            system_prompt = apply_context_policy(raw_context, max_chars, strategy)

            active_model = _normalize_optional_text(options.get("active_model"))

            _append_chat_history(hass, session_id, "user", message)

            response = await client.async_send_message(
                message=message,
                session_id=session_id,
                system_prompt=system_prompt,
                agent_id=resolved_agent_id,
                model=active_model,
                extra_headers=extra_headers,
            )

            if options.get(CONF_ENABLE_TOOL_CALLS, DEFAULT_ENABLE_TOOL_CALLS):
                tool_results = await _async_execute_tool_calls(hass, response)
                if tool_results:
                    response = await client.async_send_message(
                        message=(
                            "Tool execution results:\n"
                            + "\n".join(f"- {line}" for line in tool_results)
                            + "\nRespond to the user based on these results."
                        ),
                        session_id=session_id,
                        system_prompt=system_prompt,
                        agent_id=resolved_agent_id,
                        model=active_model,
                        extra_headers=extra_headers,
                    )

            assistant_message = _extract_assistant_message(response)
            model_used = response.get("model", "unknown")

            if not assistant_message:
                assistant_message = "Response received, but no readable message content was found."
                _LOGGER.warning(
                    "OpenClaw response had no parseable assistant message. Keys: %s",
                    list(response.keys()),
                )

            _append_chat_history(hass, session_id, "assistant", assistant_message)
            hass.bus.async_fire(
                EVENT_MESSAGE_RECEIVED,
                {
                    ATTR_MESSAGE: assistant_message,
                    ATTR_SESSION_ID: session_id,
                    ATTR_MODEL: model_used,
                    ATTR_TIMESTAMP: datetime.now(timezone.utc).isoformat(),
                },
            )
            coordinator.update_last_activity()

        except OpenClawApiError as err:
            _LOGGER.error("Failed to send message to OpenClaw: %s", err)
            _append_chat_history(hass, session_id, "assistant", f"OpenClaw error: {err}")
            hass.bus.async_fire(
                EVENT_MESSAGE_RECEIVED,
                {
                    ATTR_MESSAGE: f"OpenClaw error: {err}",
                    ATTR_SESSION_ID: session_id,
                    ATTR_MODEL: "unknown",
                    ATTR_TIMESTAMP: datetime.now(timezone.utc).isoformat(),
                },
            )

    async def handle_analyze_inputs(call: ServiceCall) -> dict[str, Any] | None:
        """Handle OpenClaw input analysis service calls."""
        prompt: str = call.data[ATTR_PROMPT]
        image_paths: list[str] = call.data.get(ATTR_IMAGE_PATHS) or []
        file_paths: list[str] = call.data.get(ATTR_FILE_PATHS) or []
        session_id: str = call.data.get(ATTR_SESSION_ID) or "input-analysis"
        call_agent_id = _normalize_optional_text(call.data.get(ATTR_AGENT_ID))
        call_model = _normalize_optional_text(call.data.get(ATTR_MODEL))
        instructions = _normalize_optional_text(call.data.get(ATTR_INSTRUCTIONS))
        source: str = call.data.get(ATTR_SOURCE) or "automation"

        entry_data = _get_first_entry_data(hass)
        if not entry_data:
            raise HomeAssistantError("No OpenClaw integration configured")

        client: OpenClawApiClient = entry_data["client"]
        coordinator: OpenClawCoordinator = entry_data["coordinator"]
        options = _get_entry_options(hass, entry_data)
        active_model = _normalize_optional_text(options.get("active_model"))
        model = call_model or active_model

        images = await _async_read_analysis_images(hass, image_paths)
        files = await _async_read_analysis_files(hass, file_paths)

        try:
            response = await client.async_create_response(
                prompt=prompt,
                images=images,
                files=files,
                session_id=session_id,
                model=model,
                instructions=instructions,
                agent_id=call_agent_id,
            )
        except OpenClawApiError as err:
            raise HomeAssistantError(f"OpenClaw input analysis failed: {err}") from err

        _raise_for_failed_openresponses_response(response)

        analysis = _extract_response_analysis(response)
        timestamp = datetime.now(timezone.utc).isoformat()
        model_used = response.get("model") if isinstance(response, dict) else None
        result = {
            ATTR_ANALYSIS: analysis,
            "response": response,
            ATTR_SESSION_ID: session_id,
            ATTR_AGENT_ID: call_agent_id,
            ATTR_MODEL: model_used or model,
            ATTR_IMAGE_COUNT: len(images),
            ATTR_FILE_COUNT: len(files),
            ATTR_INPUT_COUNT: len(images) + len(files),
            ATTR_SOURCE: source,
            ATTR_TIMESTAMP: timestamp,
        }
        hass.bus.async_fire(EVENT_INPUT_ANALYSIS_RECEIVED, result)
        coordinator.update_last_activity()
        if getattr(call, "return_response", False):
            return result
        return None

    async def handle_clear_history(call: ServiceCall) -> None:
        """Handle the openclaw.clear_history service call."""
        session_id: str | None = call.data.get(ATTR_SESSION_ID)
        _LOGGER.info("Clear history requested (session=%s)", session_id or "all")
        store = _get_chat_history_store(hass)
        if session_id:
            store.pop(session_id, None)
        else:
            store.clear()

    async def handle_invoke_tool(call: ServiceCall) -> None:
        """Handle the openclaw.invoke_tool service call."""
        tool_name: str = call.data[ATTR_TOOL]
        action: str | None = call.data.get(ATTR_ACTION)
        args: dict[str, Any] = call.data.get(ATTR_ARGS) or {}
        session_key: str | None = call.data.get(ATTR_SESSION_KEY)
        dry_run: bool = bool(call.data.get(ATTR_DRY_RUN, False))
        message_channel: str | None = call.data.get(ATTR_MESSAGE_CHANNEL)
        account_id: str | None = call.data.get(ATTR_ACCOUNT_ID)

        entry_data = _get_first_entry_data(hass)
        if not entry_data:
            _LOGGER.error("No OpenClaw integration configured")
            return

        client: OpenClawApiClient = entry_data["client"]
        coordinator: OpenClawCoordinator = entry_data["coordinator"]

        started = perf_counter()
        ok = False
        result: Any = None
        error_message: str | None = None

        try:
            response = await client.async_invoke_tool(
                tool=tool_name,
                action=action,
                args=args,
                session_key=session_key,
                dry_run=dry_run,
                message_channel=message_channel,
                account_id=account_id,
            )
            ok = bool(response.get("ok", True)) if isinstance(response, dict) else True
            result = response.get("result") if isinstance(response, dict) else response
            if isinstance(response, dict) and response.get("error"):
                error_message = str(response.get("error"))
        except OpenClawApiError as err:
            ok = False
            error_message = str(err)

        duration_ms = int((perf_counter() - started) * 1000)
        result_preview = _summarize_tool_result(result)
        coordinator.record_tool_invocation(
            tool_name=tool_name,
            ok=ok,
            duration_ms=duration_ms,
            error_message=error_message,
            result_preview=result_preview,
        )

        event_payload = {
            ATTR_TOOL: tool_name,
            ATTR_ACTION: action,
            ATTR_SESSION_KEY: session_key,
            ATTR_DRY_RUN: dry_run,
            ATTR_OK: ok,
            ATTR_RESULT: result,
            ATTR_ERROR: error_message,
            ATTR_DURATION_MS: duration_ms,
            ATTR_TIMESTAMP: datetime.now(timezone.utc).isoformat(),
        }
        hass.bus.async_fire(EVENT_TOOL_INVOKED, event_payload)

        if not ok:
            raise OpenClawApiError(error_message or "Tool invocation failed")

    if not hass.services.has_service(DOMAIN, SERVICE_SEND_MESSAGE):
        hass.services.async_register(
            DOMAIN,
            SERVICE_SEND_MESSAGE,
            handle_send_message,
            schema=SEND_MESSAGE_SCHEMA,
        )
    if not hass.services.has_service(DOMAIN, SERVICE_ANALYZE_INPUTS):
        hass.services.async_register(
            DOMAIN,
            SERVICE_ANALYZE_INPUTS,
            handle_analyze_inputs,
            schema=ANALYZE_INPUTS_SCHEMA,
            supports_response=SupportsResponse.OPTIONAL,
        )
    if not hass.services.has_service(DOMAIN, SERVICE_CLEAR_HISTORY):
        hass.services.async_register(
            DOMAIN,
            SERVICE_CLEAR_HISTORY,
            handle_clear_history,
            schema=CLEAR_HISTORY_SCHEMA,
        )
    if not hass.services.has_service(DOMAIN, SERVICE_INVOKE_TOOL):
        hass.services.async_register(
            DOMAIN,
            SERVICE_INVOKE_TOOL,
            handle_invoke_tool,
            schema=INVOKE_TOOL_SCHEMA,
        )


def _get_first_entry_data(hass: HomeAssistant) -> dict[str, Any] | None:
    """Return entry data dict for the first configured OpenClaw entry."""
    domain_data: dict = hass.data.get(DOMAIN, {})
    for entry_id, entry_data in domain_data.items():
        if isinstance(entry_data, dict) and "client" in entry_data:
            return entry_data
    return None


def _get_entry_options(hass: HomeAssistant, entry_data: dict[str, Any]) -> dict[str, Any]:
    """Return latest config entry options for an integration entry data payload."""
    latest_entry: ConfigEntry | None = None

    entry_id = entry_data.get("entry_id")
    if isinstance(entry_id, str):
        latest_entry = hass.config_entries.async_get_entry(entry_id)

    if latest_entry is None:
        cached_entry = entry_data.get("entry")
        cached_entry_id = getattr(cached_entry, "entry_id", None)
        if isinstance(cached_entry_id, str):
            latest_entry = hass.config_entries.async_get_entry(cached_entry_id) or cached_entry
        elif isinstance(cached_entry, ConfigEntry):
            latest_entry = cached_entry

    return latest_entry.options if latest_entry else {}


def _resolve_analysis_input_path(hass: HomeAssistant, raw_path: str) -> Path:
    """Resolve an input path against HA config and enforce HA path allow rules."""
    if not raw_path.strip():
        raise ServiceValidationError("Input paths must not be empty")

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(hass.config.path(str(path)))
    path = path.resolve()

    is_allowed_path = getattr(hass.config, "is_allowed_path", None)
    if callable(is_allowed_path) and not is_allowed_path(str(path)):
        raise ServiceValidationError(
            "Input path is not allowed by Home Assistant path configuration"
        )
    return path


def _guess_image_media_type(path: Path) -> str:
    """Return a supported MIME type for an image path."""
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".heic":
        return "image/heic"
    if suffix == ".heif":
        return "image/heif"
    media_type = mimetypes.guess_type(path.name)[0]
    if media_type == "image/jpg":
        media_type = "image/jpeg"
    if media_type not in _ALLOWED_IMAGE_MIME_TYPES:
        raise ServiceValidationError(
            f"Unsupported image type for '{path.name}'. Supported types: "
            f"{', '.join(sorted(_ALLOWED_IMAGE_MIME_TYPES))}"
        )
    return media_type


def _read_analysis_image(path: Path) -> bytes:
    """Read an image from disk after validating it is a regular file."""
    if not path.exists():
        raise ServiceValidationError(f"Image file does not exist: {path.name}")
    if not path.is_file():
        raise ServiceValidationError(f"Image path is not a file: {path.name}")
    try:
        return path.read_bytes()
    except OSError as err:
        raise HomeAssistantError(f"Unable to read image '{path.name}': {err}") from err


async def _async_read_analysis_images(
    hass: HomeAssistant, image_paths: list[str]
) -> list[dict[str, str]]:
    """Read, validate, and base64-encode analysis images."""
    images: list[dict[str, str]] = []
    for raw_path in image_paths:
        path = _resolve_analysis_input_path(hass, raw_path)
        media_type = _guess_image_media_type(path)
        data = await hass.async_add_executor_job(_read_analysis_image, path)
        images.append(
            {
                "media_type": media_type,
                "data": base64.b64encode(data).decode("ascii"),
            }
        )
    return images


def _guess_file_media_type(path: Path) -> str:
    """Return a supported MIME type for a generic input file path."""
    suffix = path.suffix.lower()
    suffix_media_types = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
        ".csv": "text/csv",
        ".json": "application/json",
        ".pdf": "application/pdf",
    }
    media_type = suffix_media_types.get(suffix) or mimetypes.guess_type(path.name)[0]
    if media_type not in _ALLOWED_FILE_MIME_TYPES:
        raise ServiceValidationError(
            f"Unsupported file type for '{path.name}'. Supported types: "
            f"{', '.join(sorted(_ALLOWED_FILE_MIME_TYPES))}"
        )
    return media_type


def _read_analysis_file(path: Path) -> bytes:
    """Read a generic input file after validating it is a regular file."""
    if not path.exists():
        raise ServiceValidationError(f"Input file does not exist: {path.name}")
    if not path.is_file():
        raise ServiceValidationError(f"Input path is not a file: {path.name}")
    try:
        return path.read_bytes()
    except OSError as err:
        raise HomeAssistantError(f"Unable to read input file '{path.name}': {err}") from err


async def _async_read_analysis_files(
    hass: HomeAssistant, file_paths: list[str]
) -> list[dict[str, str]]:
    """Read, validate, and base64-encode generic analysis files."""
    files: list[dict[str, str]] = []
    for raw_path in file_paths:
        path = _resolve_analysis_input_path(hass, raw_path)
        media_type = _guess_file_media_type(path)
        data = await hass.async_add_executor_job(_read_analysis_file, path)
        files.append(
            {
                "filename": path.name,
                "media_type": media_type,
                "data": base64.b64encode(data).decode("ascii"),
            }
        )
    return files


def _raise_for_failed_openresponses_response(response: dict[str, Any]) -> None:
    """Raise HomeAssistantError for failed OpenResponses statuses."""
    if not isinstance(response, dict):
        return

    status = response.get("status")
    error = response.get("error")
    error_status = error.get("status") if isinstance(error, dict) else None
    failed = status in {"failed", "cancelled"}
    failed = failed or (status == "incomplete" and isinstance(error, dict))
    failed = failed or error_status in {"failed", "cancelled", "error"}
    if not failed:
        return

    error_message: str | None = None
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            error_message = message.strip()
    elif isinstance(error, str) and error.strip():
        error_message = error.strip()

    if error_message:
        raise HomeAssistantError(
            f"OpenClaw input analysis failed ({status}): {error_message}"
        )
    raise HomeAssistantError(f"OpenClaw input analysis failed with status: {status}")


def _extract_response_analysis(response: dict[str, Any]) -> str:
    """Extract readable text from an OpenResponses response without image data."""
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    for key in ("output", "choices"):
        extracted = extract_text_recursive(response.get(key))
        if extracted:
            return extracted

    texts: list[str] = []

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") in {"input_image", "image"}:
                return
            for key in ("text", "content", "output_text"):
                text_value = value.get(key)
                if isinstance(text_value, str) and text_value.strip():
                    texts.append(text_value.strip())
            for child in value.values():
                if not isinstance(child, str):
                    _walk(child)
        elif isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(response.get("output"))
    if texts:
        return "\n".join(texts)
    return "Response received, but no readable analysis content was found."

def _summarize_tool_result(value: Any, max_len: int = 240) -> str | None:
    """Return compact string preview of tool result payload."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    text = text.strip()
    if not text:
        return None
    if len(text) > max_len:
        return f"{text[:max_len]}…"
    return text


def _extract_assistant_message(response: dict[str, Any]) -> str | None:
    """Extract assistant text from modern/legacy OpenAI-compatible responses."""
    return extract_text_recursive(response)


def _extract_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract OpenAI-style tool calls from a response payload."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return []

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return []

    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []

    valid_calls: list[dict[str, Any]] = []
    for call in tool_calls:
        if isinstance(call, dict):
            valid_calls.append(call)
    return valid_calls


async def _async_execute_tool_calls(
    hass: HomeAssistant,
    response: dict[str, Any],
) -> list[str]:
    """Execute supported tool calls and return result lines."""
    results: list[str] = []
    tool_calls = _extract_tool_calls(response)

    for call in tool_calls:
        function_data = call.get("function")
        if not isinstance(function_data, dict):
            continue

        function_name = function_data.get("name")
        arguments = function_data.get("arguments")

        if function_name not in {"execute_service", "execute_services"}:
            results.append(f"Skipped unsupported tool '{function_name}'")
            continue

        if not isinstance(arguments, str):
            results.append("Skipped tool call with invalid arguments format")
            continue

        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            results.append("Skipped tool call due to invalid JSON arguments")
            continue

        services_list = parsed.get("list") if isinstance(parsed, dict) else None
        if not isinstance(services_list, list):
            results.append("Skipped tool call without 'list' payload")
            continue

        for item in services_list:
            if not isinstance(item, dict):
                continue
            domain = item.get("domain")
            service = item.get("service")
            service_data = item.get("service_data", {})

            if not isinstance(domain, str) or not isinstance(service, str):
                results.append("Skipped invalid service item (missing domain/service)")
                continue

            if not isinstance(service_data, dict):
                service_data = {}

            try:
                await hass.services.async_call(
                    domain,
                    service,
                    service_data,
                    blocking=True,
                )
                results.append(f"Executed {domain}.{service}")
            except Exception as err:  # noqa: BLE001
                results.append(f"Failed {domain}.{service}: {err}")

    return results


def _get_chat_history_store(hass: HomeAssistant) -> dict[str, list[dict[str, str]]]:
    """Return in-memory per-session chat history store."""
    store_key = f"{DOMAIN}_chat_history"
    store = hass.data.get(store_key)
    if store is None:
        store = {}
        hass.data[store_key] = store
    return store


def _append_chat_history(hass: HomeAssistant, session_id: str, role: str, content: str) -> None:
    """Append a message to in-memory chat history."""
    store = _get_chat_history_store(hass)
    history = store.setdefault(session_id, [])
    history.append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    if len(history) > _MAX_CHAT_HISTORY:
        del history[:-_MAX_CHAT_HISTORY]


@callback
def _async_register_websocket_api(hass: HomeAssistant) -> None:
    """Register websocket API for chat history retrieval."""
    key = f"{DOMAIN}_ws_registered"
    if hass.data.get(key):
        return
    hass.data[key] = True

    @websocket_api.websocket_command(
        {
            vol.Required("type"): f"{DOMAIN}/get_history",
            vol.Optional("session_id"): cv.string,
        }
    )
    @callback
    def websocket_get_history(
        hass: HomeAssistant,
        connection: websocket_api.ActiveConnection,
        msg: dict[str, Any],
    ) -> None:
        """Return chat history for a session."""
        session_id = msg.get("session_id") or "default"
        history = _get_chat_history_store(hass).get(session_id, [])
        connection.send_result(msg["id"], {"session_id": session_id, "messages": history})

    websocket_api.async_register_command(hass, websocket_get_history)

    @websocket_api.websocket_command(
        {
            vol.Required("type"): f"{DOMAIN}/get_settings",
        }
    )
    @callback
    def websocket_get_settings(
        hass: HomeAssistant,
        connection: websocket_api.ActiveConnection,
        msg: dict[str, Any],
    ) -> None:
        """Return frontend-related integration settings."""
        entry_data = _get_first_entry_data(hass)
        options = _get_entry_options(hass, entry_data) if entry_data else {}
        connection.send_result(
            msg["id"],
            {
                CONF_WAKE_WORD_ENABLED: options.get(
                    CONF_WAKE_WORD_ENABLED,
                    DEFAULT_WAKE_WORD_ENABLED,
                ),
                CONF_WAKE_WORD: options.get(CONF_WAKE_WORD, DEFAULT_WAKE_WORD),
                CONF_ALLOW_BRAVE_WEBSPEECH: options.get(
                    CONF_ALLOW_BRAVE_WEBSPEECH,
                    DEFAULT_ALLOW_BRAVE_WEBSPEECH,
                ),
                CONF_VOICE_PROVIDER: options.get(
                    CONF_VOICE_PROVIDER,
                    DEFAULT_VOICE_PROVIDER,
                ),
                CONF_BROWSER_VOICE_LANGUAGE: options.get(
                    CONF_BROWSER_VOICE_LANGUAGE,
                    DEFAULT_BROWSER_VOICE_LANGUAGE,
                ),
                CONF_THINKING_TIMEOUT: options.get(
                    CONF_THINKING_TIMEOUT,
                    DEFAULT_THINKING_TIMEOUT,
                ),
                "language": hass.config.language,
            },
        )

    websocket_api.async_register_command(hass, websocket_get_settings)
