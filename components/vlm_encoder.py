# ------------------------------------------------------------------------------
# Copyright (c) 2022–∞, 2toINF
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class VLMEncoder(nn.Module):
    """
    Thin wrapper around a Florence-2 style VLM to produce token-level features.

    This class loads the underlying model, removes the language decoder heads,
    and exposes a single forward method that:
      1) Encodes valid images.
      2) Merges image features with text token embeddings.
      3) Runs the encoder to obtain token-level VLM features.
      4) Returns both `vlm_features` (encoder sequence) and `aux_visual_inputs`
         (flattened features from remaining views).

    Parameters
    ----------
    encoder_name : str
        Hugging Face model id or local path, e.g. "microsoft/Florence-2-base".
    torch_dtype : str | torch.dtype, default='auto'
        dtype policy for loading (passed through to HF).
    trust_remote_code : bool, default=True
        Enable custom model code.
    local_files_only : bool, default=True
        Load only from local cache.

    Notes
    -----
    - This class uses Florence-2 private helpers:
        * `_encode_image(images)`
        * `_merge_input_ids_with_image_features(image_features, inputs_embeds)`
      and accesses `language_model.model.encoder`.
      Keep `trust_remote_code=True` to ensure these are available.
    """

    def __init__(
        self,
        encoder_name: str = "microsoft/Florence-2-base"
    ):
        super().__init__()
        assert encoder_name in {
            "microsoft/Florence-2-base",
            "microsoft/Florence-2-large",
        }, "Only microsoft/Florence-2-base and microsoft/Florence-2-large are supported."

        self.model = AutoModelForCausalLM.from_pretrained(
            encoder_name,
            low_cpu_mem_usage=True,
            torch_dtype="auto",
            trust_remote_code=True
        )

        # Remove decoder-specific components to reduce memory and ensure we only
        # use encoder pathways. Guard these in case internals change.
        if hasattr(self.model, "language_model"):
            lm = self.model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            if hasattr(lm, "lm_head"):
                del lm.lm_head

        # Expose projection dimension for downstream modules
        self.num_features = self.model.config.projection_dim

    def forward(
        self,
        input_ids: torch.LongTensor,        # [B, L]
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W]
        image_mask: torch.Tensor,           # [B, V], bool or 0/1
    ) -> Dict[str, torch.Tensor]:
        """
        Produce VLM token features and auxiliary visual tokens from multi-view inputs.

        Parameters
        ----------
        input_ids : LongTensor, shape [B, L]
            Token ids for the text prompt/instruction.
        pixel_values : FloatTensor, shape [B, V, C, H, W]
            Image batch with V views per sample.
        image_mask : Tensor, shape [B, V]
            Mask indicating which views are valid (True/1) vs padded (False/0).

        Returns
        -------
        Dict[str, Tensor]
            {
              "vlm_features": FloatTensor [B, T_enc, D],  # encoder token sequence
              "aux_visual_inputs": FloatTensor [B, (V-1)*N, D]  # flattened features for views 1..V-1
            }
        """
        B, V = pixel_values.shape[:2]

        # Flatten views, select valid images, encode
        flat_mask = image_mask.view(-1).to(torch.bool)
        flat_images = pixel_values.flatten(0, 1)                    # [B*V, C, H, W]

        num_valid = int(flat_mask.sum().item())
        assert num_valid > 0, "At least one image must be valid."

        valid_images = flat_images[flat_mask]                   # [#valid, C, H, W]
        valid_feats = self.model._encode_image(valid_images)    # [#valid, N, D]
        N, D = valid_feats.shape[1:]

        # Reconstruct dense [B, V, N, D] tensor
        image_features = valid_feats.new_zeros((B * V, N, D))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(B, V, N, D)        # [B, V, N, D]

        # Text embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids) # [B, L, D]

        # Merge first-view visual tokens with text embeddings
        merged_embeds, attention_mask = self.model._merge_input_ids_with_image_features(
            image_features[:, 0],  # [B, N, D]
            inputs_embeds,         # [B, L, D]
        )

        # Run encoder to get token-level features
        enc_out = self.model.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]  # [B, T_enc, D]

        # Remaining views (1..V-1) flattened as auxiliary inputs
        aux_visual_inputs = image_features[:, 1:].reshape(B, -1, D)  # [B, (V-1)*N, D]

        return {
            "vlm_features": enc_out,
            "aux_visual_inputs": aux_visual_inputs,
        }
