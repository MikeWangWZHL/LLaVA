#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM, LlavaGeoMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava"

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)


#########################
#### custom code ####
#########################

# TODO:
from transformers.models.vit_mae.modeling_vit_mae import *
from transformers.models.vit_mae.configuration_vit_mae import *
from transformers.modeling_outputs import CausalLMOutputWithPast

class LlavaGeoOutput(CausalLMOutputWithPast):
    def __init__(self, 
                 lm_loss: Optional[torch.FloatTensor] = None, 
                 reconstruction_loss: Optional[torch.FloatTensor] = None, 
                 reconstruction_logits: torch.FloatTensor = None, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.lm_loss = lm_loss
        self.reconstruction_loss = reconstruction_loss
        self.reconstruction_logits = reconstruction_logits

### adding ViT MAE Decoder: do MAE loss on CLIP-projected patch embeddings

# 1. take the encoded full sequence of image patch embeddings
# 2. randomly mask out 75% (pick the 25%) and feed to the decoder: shuffle, then pick the first 25%, log the ids_restore => reference: https://github.com/huggingface/transformers/blob/acc394c4f5e1283c19783581790b3dc3105a3697/src/transformers/models/vit_mae/modeling_vit_mae.py#L232C14-L232C14
# 3. => decoder: append the mask tokens, then unshuffle; then compute reconstruction loss on the masked patches


class LlavaGeoConfig(LlavaConfig):
    model_type = "llava_geo"
    mae_config = {
        "base_config": "facebook/vit-mae-base",
        "norm_pix_loss": True
    }
    losses = ["lm", "mae"]
    loss_weights = {
        "lm": 1.0,
        "mae": 1.0
    }
    do_reconstruction_only = False
    def __repr__(self):
        parent_repr = super().__repr__()  # Gets the representation from the parent class
        # all attribute values
        return (f"{parent_repr}, "  
                f"\"mae_config\": {self.mae_config}, "
                f"\"losses\": {self.losses}, "
                f"\"loss_weights\": {self.loss_weights}, "
                f"\"do_reconstruction_only\": {self.do_reconstruction_only}"
        )

import wandb
from transformers import AutoImageProcessor, ViTMAEForPreTraining
class LlavaGeoLlamaForCausalLM(LlamaForCausalLM, LlavaGeoMetaForCausalLM):
    config_class = LlavaGeoConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if getattr(self.config, "mae_config", None):
            self._build_mae_encoder_decoder()
        else:
            self.mae_config = None

        # Initialize weights and apply final processing
        self.post_init()

    def _build_mae_encoder_decoder(self):
        mae_args = self.config.mae_config

        self.geo_image_processor = AutoImageProcessor.from_pretrained(mae_args['base_config'])
        self.mae_model = ViTMAEForPreTraining.from_pretrained(mae_args['base_config'])
        
        self.mae_config = self.mae_model.config

        # mae encoder - projection layer adaptor
        self.mae_enc_to_projection = nn.Linear(self.mae_config.hidden_size, self.config.mm_hidden_size)
        self.projection_to_mae_dec = nn.Linear(self.config.hidden_size, self.mae_config.hidden_size)

        # specify image size and patch size
        for key in mae_args:
            if key not in ['base_config']:
                setattr(self.mae_config, key, mae_args[key])

    def get_model(self):
        return self.model

    def mae_forward(self, 
                    pixel_values: Optional[torch.FloatTensor] = None,
                    noise: Optional[torch.FloatTensor] = None,
                    head_mask: Optional[torch.FloatTensor] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mae_model.vit(
            pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        # mae enc to projection layers
        latent = self.mae_enc_to_projection(latent)
        # go through projection layers
        latent = self.get_model().mm_projector(latent)
        # projection layers to mae dec
        latent = self.projection_to_mae_dec(latent)

        decoder_outputs = self.mae_model.decoder(latent, ids_restore)
        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        loss = self.mae_model.forward_loss(pixel_values, logits, mask)

        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ViTMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None, # (batch_size, num_channels, height, width)
        images_for_geo: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        logits = None
        lm_hidden_states = None
        lm_past_key_values = None
        lm_attentions = None
        
        if not self.config.do_reconstruction_only:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, image_features_with_cls \
                = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            lm_hidden_states = outputs.hidden_states
            lm_past_key_values = outputs.past_key_values
            lm_attentions = outputs.attentions

        loss = None
        lm_loss = None
        reconstruction_loss = None
        reconstruction_logits = None
        losses = {}
        if labels is not None or self.config.do_reconstruction_only:
            if "lm" in self.config.losses:
                assert self.config.do_reconstruction_only == False
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                lm_loss = loss_fct(shift_logits, shift_labels)
                print("lm loss:", lm_loss.item())
                # import wandb; wandb.log({"lm_loss":lm_loss.item()})
                losses['lm'] = lm_loss
            
            if "mae" in self.config.losses:

                if type(images_for_geo) is list or images_for_geo.ndim == 5:
                    images_for_geo = torch.cat([im for im in images_for_geo], dim=0)
                # check if the input tensor is zero tensor
                if images_for_geo.sum() != 0:
                    # compute MAE decoder loss
                    mae_outputs = self.mae_forward(images_for_geo)
                    reconstruction_loss = mae_outputs.loss

                    print("mae loss:", reconstruction_loss.item())
                    # wandb.log({"mae_reconstruction_loss":reconstruction_loss.item()})
                    losses['mae'] = reconstruction_loss
                else:
                    print("ignore dummy image input for MAE loss")

            for key in losses:
                if loss is None:
                    loss = losses[key] * self.config.loss_weights[key]
                else:
                    loss += losses[key] * self.config.loss_weights[key]

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaGeoOutput(
            loss=loss,
            lm_loss=lm_loss,
            reconstruction_loss=reconstruction_loss,
            reconstruction_logits=reconstruction_logits,
            logits=logits,
            past_key_values=lm_past_key_values,
            hidden_states=lm_hidden_states,
            attentions=lm_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

### Early Fusion ###

class LlavaGeoConfigEarlyFusion(LlavaConfig):
    model_type = "llava_geo"
    sam_config = {
        "base_config": "facebook/sam-vit-base",
    }
    losses = [
        "lm", 
        # "sam"
    ]
    loss_weights = {
        "lm": 1.0,
        # "sam": 1.0
    }
    def __repr__(self):
        parent_repr = super().__repr__()  # Gets the representation from the parent class
        # all attribute values
        return (f"{parent_repr}, "  
                f"\"sam_config\": {self.sam_config}, "
                f"\"losses\": {self.losses}, "
                f"\"loss_weights\": {self.loss_weights}, "
        )

import wandb
from transformers import SamModel, SamProcessor
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class LlavaGeoLlamaForCausalLMEarlyFusion(LlamaForCausalLM, LlavaGeoMetaForCausalLM):
    config_class = LlavaGeoConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if getattr(self.config, "sam_config", None):
            self._build_sam_model()
        else:
            self.geo_config = None

        # Initialize weights and apply final processing
        self.post_init()

    def _build_sam_model(self):
        sam_args = self.config.sam_config

        self.geo_image_processor = SamProcessor.from_pretrained(sam_args['base_config'])
        self.geo_encoder = SamModel.from_pretrained(sam_args['base_config'])
        self.geo_config = self.geo_encoder.config
        
        # projection layer
        self.geo_to_llm_projector = nn.Linear(
            (self.geo_config.image_size // self.geo_config.patch_size) ** 2, self.config.hidden_size
        ) # 64 * 64 = 4096 => 4096

        # specify image size and patch size
        for key in sam_args:
            if key not in ['base_config']:
                setattr(self.geo_config, key, sam_args[key])

    def encode_images_geo(self, pixel_values):
        image_features_geo = self.geo_encoder.get_image_embeddings(pixel_values) # (batch_size, chanel_size, width, height)
        batch_size, num_channels, num_patch_width, num_patch_height = image_features_geo.shape # width = height = image_size // patch_size = 64 
        image_features_geo = image_features_geo.reshape(batch_size, num_channels, num_patch_width * num_patch_height)
        image_features_geo = self.geo_to_llm_projector(image_features_geo)
        return image_features_geo


    def get_model(self):
        return self.model

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, images_for_geo
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, None
        
        # clip encode
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)
        
        # geo encode (e.g. SAM)
        if images_for_geo is not None:
            if type(images_for_geo) is list or images_for_geo.ndim == 5:
                concat_images_geo = torch.cat([image for image in images_for_geo], dim=0)
                geo_image_features = self.encode_images_geo(concat_images_geo)
                split_sizes = [image.shape[0] for image in images_for_geo]
                geo_image_features = torch.split(geo_image_features, split_sizes, dim=0)
                geo_image_features = [x.flatten(0, 1) for x in geo_image_features]
            else:
                geo_image_features = self.encode_images_geo(images_for_geo)
            # concat image features
            image_features = torch.cat([image_features, geo_image_features], dim=1)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, image_features


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None, # (batch_size, num_channels, height, width)
        images_for_geo: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if images is not None and images_for_geo is not None:
            all_images = [images, images_for_geo]

        input_ids, attention_mask, past_key_values, inputs_embeds, labels, image_features \
            = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, all_images)

        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        lm_hidden_states = outputs.hidden_states
        lm_past_key_values = outputs.past_key_values
        lm_attentions = outputs.attentions

        loss = None
        lm_loss = None
        reconstruction_loss = None
        reconstruction_logits = None
        losses = {}
        if labels is not None:
            if "lm" in self.config.losses:
                assert self.config.do_reconstruction_only == False
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                lm_loss = loss_fct(shift_logits, shift_labels)
                print("lm loss:", lm_loss.item())
                # import wandb; wandb.log({"lm_loss":lm_loss.item()})
                losses['lm'] = lm_loss
            
            if "sam" in self.config.losses:
                raise NotImplementedError("SAM loss is not implemented yet")

            for key in losses:
                if loss is None:
                    loss = losses[key] * self.config.loss_weights[key]
                else:
                    loss += losses[key] * self.config.loss_weights[key]

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaGeoOutput(
            loss=loss,
            lm_loss=lm_loss,
            reconstruction_loss=reconstruction_loss,
            reconstruction_logits=reconstruction_logits,
            logits=logits,
            past_key_values=lm_past_key_values,
            hidden_states=lm_hidden_states,
            attentions=lm_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs





















# FIXME: only for debugging; remove later
#### for debugging ####
class LlavaGeoLlamaForCausalLM_ReconstructOnly_NoProjection(LlamaForCausalLM, LlavaGeoMetaForCausalLM):
    config_class = LlavaGeoConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if getattr(self.config, "mae_decoder_config", None):
            self._build_mae_decoder()
        else:
            self.mae_decoder_config = None

        # Initialize weights and apply final processing
        self.post_init()

    def _build_mae_decoder(self):
        mae_args = self.config.mae_decoder_config
        mae_decoder_config = ViTMAEConfig.from_pretrained(mae_args['base_config'])
        vision_tower_config = self.model.get_vision_tower().config
        # specify image size and patch size
        mae_decoder_config.image_size = vision_tower_config.image_size
        mae_decoder_config.patch_size = vision_tower_config.patch_size
        mae_decoder_config.num_channels = vision_tower_config.num_channels
        for key in mae_args:
            if key not in ['base_config']:
                setattr(mae_decoder_config, key, mae_args[key])
        
        # fit hidden dimension
        # mae_decoder_config.hidden_size = self.config.hidden_size
        # mae_decoder_config.intermediate_size = self.config.intermediate_size
        mae_decoder_config.hidden_size = vision_tower_config.hidden_size
        mae_decoder_config.intermediate_size = vision_tower_config.intermediate_size
        mae_decoder_config.torch_dtype = vision_tower_config.torch_dtype

        self.mae_decoder_config = mae_decoder_config

        num_patches = (mae_decoder_config.image_size // mae_decoder_config.patch_size) ** 2
        self.mae_decoder = ViTMAEDecoder(mae_decoder_config, num_patches)

    def get_model(self):
        return self.model

    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        patch_size, num_channels = self.mae_decoder_config.patch_size, self.mae_decoder_config.num_channels
        # sanity checks
        if (pixel_values.shape[2] != pixel_values.shape[3]) or (pixel_values.shape[2] % patch_size != 0):
            raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
        if pixel_values.shape[1] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        # patchify
        batch_size = pixel_values.shape[0]
        num_patches_one_direction = pixel_values.shape[2] // patch_size
        patchified_pixel_values = pixel_values.reshape(
            batch_size, num_channels, num_patches_one_direction, patch_size, num_patches_one_direction, patch_size
        )
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_one_direction * num_patches_one_direction, patch_size**2 * num_channels
        )
        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        patch_size, num_channels = self.mae_decoder_config.patch_size, self.mae_decoder_config.num_channels
        num_patches_one_direction = int(patchified_pixel_values.shape[1] ** 0.5)
        # sanity check
        if num_patches_one_direction**2 != patchified_pixel_values.shape[1]:
            raise ValueError("Make sure that the number of patches can be squared")

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_one_direction,
            num_patches_one_direction,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_one_direction * patch_size,
            num_patches_one_direction * patch_size,
        )
        return pixel_values

    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.mae_decoder_config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def mae_loss(self, pixel_values, pred, mask):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        target = self.patchify(pixel_values)
        if self.mae_decoder_config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None, # (batch_size, num_channels, height, width)
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        logits = None
        lm_hidden_states = None
        lm_past_key_values = None
        lm_attentions = None

        
        if not self.config.do_reconstruction_only:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, image_features_with_cls \
                = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            lm_hidden_states = outputs.hidden_states
            lm_past_key_values = outputs.past_key_values
            lm_attentions = outputs.attentions
        else:
            _select_feature_arg = self.get_model().config.mm_vision_select_feature
            self.get_model().get_vision_tower().select_feature = "cls_patch"
            image_features_with_cls = self.encode_images_vision_tower(images)
            self.get_model().get_vision_tower().select_feature = _select_feature_arg


        loss = None
        lm_loss = None
        reconstruction_loss = None
        reconstruction_logits = None
        losses = {}
        if labels is not None or self.config.do_reconstruction_only:
            if "lm" in self.config.losses:
                assert self.config.do_reconstruction_only == False
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                lm_loss = loss_fct(shift_logits, shift_labels)
                # print("lm loss:", lm_loss)
                losses['lm'] = lm_loss
            
            if self.mae_decoder_config is not None and "mae" in self.config.losses and image_features_with_cls is not None:
                # compute MAE decoder loss
                sequence_unmasked, mask, ids_restore = self.random_masking(image_features_with_cls[:, 1:, :])
                sequence_unmasked_with_cls = torch.cat((image_features_with_cls[:, :1, :], sequence_unmasked), dim=1)
                mae_decoder_outputs = self.mae_decoder(sequence_unmasked_with_cls, ids_restore)
                reconstruction_logits = mae_decoder_outputs.logits
                reconstruction_loss = self.mae_loss(images, reconstruction_logits, mask)
                # print("reconstruction_loss", reconstruction_loss)
                losses['mae'] = reconstruction_loss

            for key in losses:
                if loss is None:
                    loss = losses[key] * self.config.loss_weights[key]
                else:
                    loss += losses[key] * self.config.loss_weights[key]

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaGeoOutput(
            loss=loss,
            lm_loss=lm_loss,
            reconstruction_loss=reconstruction_loss,
            reconstruction_logits=reconstruction_logits,
            logits=logits,
            past_key_values=lm_past_key_values,
            hidden_states=lm_hidden_states,
            attentions=lm_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs
