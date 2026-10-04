import os
import uuid
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, object_value, string_value
from tests.integration._support.database import read_rows, scratch_database
from tests.integration._support.process import owned_proxy


def _db_params(identity: str, database_url: str) -> dict[str, JsonValue]:
    rows: Final = read_rows(
        'SELECT litellm_params FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s',
        (identity,),
        database_url=database_url,
    )
    assert len(rows) == 1
    value: Final = rows[0]["litellm_params"]
    return JSON_OBJECT.validate_json(value) if isinstance(value, str) else object_value(value)


def _observed(upstream_url: str, prompt: str) -> dict[str, JsonValue]:
    with httpx.Client(base_url=upstream_url, timeout=15, trust_env=False) as upstream:
        response: Final = upstream.get("/__observations")
    assert response.status_code == 200, response.text
    observations: Final = JSON_OBJECT.validate_json(response.content)["requests"]
    assert isinstance(observations, list)
    bodies: Final = tuple(object_value(object_value(item)["body"]) for item in observations)
    matches: Final = tuple(body for body in bodies if body.get("messages") == [{"role": "user", "content": prompt}])
    assert len(matches) == 1, f"Expected exactly one upstream request for {prompt!r}"
    return matches[0]


@pytest.mark.timeout(180)
def test_null_deletion_survives_sql_readback_reload_restart_and_upstream(gateway: Gateway, tmp_path: Path) -> None:
    name: Final = f"integration-null-{uuid.uuid4().hex}"
    upstream_base: Final = f"{gateway.upstream_url}/v1"
    with scratch_database() as database_url:
        overrides: Final = {"DATABASE_URL": database_url, "OPENAI_API_BASE": upstream_base}
        with owned_proxy(gateway, tmp_path, overrides) as first:
            created: Final = first.post(
                "/model/new",
                {
                    "model_name": name,
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "synthetic-provider-secret",
                        "api_base": upstream_base,
                        "temperature": 0.7,
                        "extra_body": {"provider_option": "saved"},
                        "rpm": 100,
                    },
                },
            )
            identity: Final = string_value(object_value(created["model_info"])["id"])
            before: Final = _db_params(identity, database_url)
            first.chat(name, text=f"{name}-before")
            observed_before: Final = _observed(gateway.upstream_url, f"{name}-before")
            assert observed_before["temperature"] == 0.7
            assert observed_before["provider_option"] == "saved"
            patched: Final = first.request(
                "PATCH",
                f"/model/{identity}/update",
                {"litellm_params": {"api_base": None, "temperature": None, "extra_body": None}},
            )
            assert patched.status_code == 200, patched.text
            assert "synthetic-provider-secret" not in patched.text
            expected: Final = {
                key: value for key, value in before.items() if key not in {"api_base", "temperature", "extra_body"}
            }
            assert _db_params(identity, database_url) == expected
            first.chat(name, text=f"{name}-reloaded")
            observed_reloaded: Final = _observed(gateway.upstream_url, f"{name}-reloaded")
            assert "temperature" not in observed_reloaded
            assert "provider_option" not in observed_reloaded
        with owned_proxy(gateway, tmp_path, overrides) as restarted:
            assert _db_params(identity, database_url) == expected
            entries: Final = restarted.get("/model/info", {"litellm_model_id": identity})["data"]
            assert isinstance(entries, list) and len(entries) == 1
            params: Final = object_value(object_value(entries[0])["litellm_params"])
            assert not {"api_base", "temperature", "extra_body"}.intersection(params)
            restarted.chat(name, text=f"{name}-restarted")
            observed_restarted: Final = _observed(gateway.upstream_url, f"{name}-restarted")
            assert "temperature" not in observed_restarted
            assert "provider_option" not in observed_restarted
            assert observed_restarted["model"] == "gpt-4o-mini"
            restarted.post("/model/delete", {"id": identity})
            assert (
                read_rows(
                    'SELECT model_id FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s',
                    (identity,),
                    database_url=database_url,
                )
                == []
            )


def test_partial_patch_preserves_omitted_flags_and_supported_empty_values(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        name: Final = scenario.model(use_in_pass_through=True, drop_params=True, max_retries=3, tags=["saved"])
        rows: Final = read_rows('SELECT model_id FROM "LiteLLM_ProxyModelTable" WHERE model_name = %s', (name,))
        identity: Final = string_value(rows[0]["model_id"])
        database_url: Final = os.environ["DATABASE_URL"]
        before: Final = _db_params(identity, database_url)
        patched: Final = gateway.request("PATCH", f"/model/{identity}/update", {"litellm_params": {"rpm": 100}})
        assert patched.status_code == 200, patched.text
        assert _db_params(identity, database_url) == {**before, "rpm": 100}
        emptied: Final = gateway.request(
            "PATCH",
            f"/model/{identity}/update",
            {
                "litellm_params": {
                    "drop_params": False,
                    "max_retries": 0,
                    "tags": [],
                    "extra_body": {},
                    "organization": "",
                }
            },
        )
        assert emptied.status_code == 200, emptied.text
        after: Final = _db_params(identity, database_url)
        assert (after["drop_params"], after["max_retries"], after["tags"], after["extra_body"]) == (False, 0, [], {})
        absent: Final = gateway.request("PATCH", f"/model/{identity}/update", {"litellm_params": {"temperature": None}})
        assert absent.status_code == 200, absent.text
        assert _db_params(identity, database_url) == after
        assert object_value(gateway.chat(name)["usage"])["total_tokens"] == 40


def test_required_model_null_is_rejected_atomically_without_credentials(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        name: Final = scenario.model()
        rows: Final = read_rows(
            'SELECT model_id, litellm_params FROM "LiteLLM_ProxyModelTable" WHERE model_name = %s', (name,)
        )
        identity: Final = string_value(rows[0]["model_id"])
        rejected: Final = gateway.request(
            "PATCH",
            f"/model/{identity}/update",
            {"litellm_params": {"model": None, "api_key": "synthetic-must-not-persist"}},
        )
        assert rejected.status_code == 400, rejected.text
        assert "synthetic-must-not-persist" not in rejected.text
        assert object_value(JSON_OBJECT.validate_json(rejected.content)["error"])["param"] == "litellm_params.model"
        assert (
            read_rows(
                'SELECT model_id, litellm_params FROM "LiteLLM_ProxyModelTable" WHERE model_name = %s',
                (name,),
            )
            == rows
        )
        assert object_value(gateway.chat(name)["usage"])["total_tokens"] == 40
