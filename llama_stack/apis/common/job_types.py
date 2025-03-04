# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from enum import Enum
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


class JobStatus(Enum):
    completed = "completed"
    in_progress = "in_progress"
    failed = "failed"
    scheduled = "scheduled"
    unknown = "unknown"


@json_schema_type
class Job(BaseModel):
    job_id: str
    status: JobStatus
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
