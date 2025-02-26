# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin

from .config import AnthropicConfig
from .models import MODEL_ENTRIES


class AnthropicInferenceAdapter(LiteLLMOpenAIMixin):
    def __init__(self, config: AnthropicConfig) -> None:
        LiteLLMOpenAIMixin.__init__(self, MODEL_ENTRIES)
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass
