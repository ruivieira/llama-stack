# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.providers.utils.kvstore.config import (
    KVStoreConfig,
    SqliteKVStoreConfig,
)
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class WeaviateVectorIOConfig(BaseModel):
    weaviate_api_key: str | None = Field(description="The API key for the Weaviate instance", default=None)
    weaviate_cluster_url: str | None = Field(description="The URL of the Weaviate cluster", default="localhost:8080")
    kvstore: KVStoreConfig | None = Field(description="Config for KV store backend (SQLite only for now)", default=None)

    @classmethod
    def sample_run_config(
        cls,
        __distro_dir__: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "weaviate_api_key": None,
            "weaviate_cluster_url": "${env.WEAVIATE_CLUSTER_URL:=localhost:8080}",
            "kvstore": SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=__distro_dir__,
                db_name="weaviate_registry.db",
            ),
        }
