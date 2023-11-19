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
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)


#########################
#### custom code ####
#########################

# TODO:
from transformers.models.vit_mae.modeling_vit_mae import *
from transformers.models.vit_mae.configuration_vit_mae import *
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

class LlavaGeoOutput(CausalLMOutputWithPast):
    def __init__(self, 
                 lm_loss: Optional[torch.FloatTensor] = None, 
                 reconstruction_loss: Optional[torch.FloatTensor] = None, 
                 reconstruction_logits: torch.FloatTensor = None, 
                 image_start_end_indices: Optional[List] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.lm_loss = lm_loss
        self.reconstruction_loss = reconstruction_loss
        self.reconstruction_logits = reconstruction_logits
        self.image_start_end_indices = image_start_end_indices

### adding ViT MAE Decoder: do MAE loss on CLIP-projected patch embeddings

# 1. take the encoded full sequence of image patch embeddings
# 2. randomly mask out 75% (pick the 25%) and feed to the decoder: shuffle, then pick the first 25%, log the ids_restore => reference: https://github.com/huggingface/transformers/blob/acc394c4f5e1283c19783581790b3dc3105a3697/src/transformers/models/vit_mae/modeling_vit_mae.py#L232C14-L232C14
# 3. => decoder: append the mask tokens, then unshuffle; then compute reconstruction loss on the masked patches

class LlavaGeoConfigMAE(LlavaConfig):
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
class LlavaGeoLlamaForCausalLMMAE(LlamaForCausalLM, LlavaGeoMetaForCausalLM):
    config_class = LlavaGeoConfigMAE

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
        position_ids: Optional[torch.LongTensor] = None,
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
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, image_features_with_cls \
                = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images)
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        images_for_geo = kwargs.pop("images_for_geo", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if images_for_geo is not None:
            _inputs['images_for_geo'] = images_for_geo
        return _inputs

### Early Fusion ###

class LlavaGeoConfigEarlyFusion(LlavaConfig):
    model_type = "llava_geo_early_fusion"
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
    config_class = LlavaGeoConfigEarlyFusion

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.load_geo_model_()

        # Initialize weights and apply final processing
        self.post_init()

    def load_geo_model_(self):
        if getattr(self.config, "sam_config", None):
            self._build_sam_model()
        else:
            self.geo_config = None

    def _build_sam_model(self):
        sam_args = self.config.sam_config

        print("Loading SAM encoder from:", sam_args['base_config'])
        self.geo_image_processor = SamProcessor.from_pretrained(sam_args['base_config'])
        self.geo_encoder = SamModel.from_pretrained(sam_args['base_config']).vision_encoder
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
        vision_output = self.geo_encoder(pixel_values) # (batch_size, chanel_size, width, height)
        image_features_geo = vision_output[0]     
        batch_size, num_channels, num_patch_width, num_patch_height = image_features_geo.shape # width = height = image_size // patch_size = 64 
        image_features_geo = image_features_geo.reshape(batch_size, num_channels, num_patch_width * num_patch_height)
        image_features_geo = self.geo_to_llm_projector(image_features_geo)
        return image_features_geo

    def get_model(self):
        return self.model

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, images_for_geo
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        # clip encode
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_features = self.encode_images(images).to(self.device)
        
        # geo encode (e.g. SAM)
        if images_for_geo is not None:
            if type(images_for_geo) is list or images_for_geo.ndim == 5:
                concat_images_geo = torch.cat([image for image in images_for_geo], dim=0)
                geo_image_features = self.encode_images_geo(concat_images_geo)
                split_sizes = [image.shape[0] for image in images_for_geo]
                geo_image_features = torch.split(geo_image_features, split_sizes, dim=0)
                geo_image_features = [x.flatten(0, 1).to(self.device) for x in geo_image_features]
            else:
                geo_image_features = self.encode_images_geo(images_for_geo).to(self.device)
            # concat image features
            image_features = torch.cat([image_features, geo_image_features], dim=1)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)


        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        
        # if new_labels is not None:
        #     for label in new_labels:
        #         if (label == IGNORE_INDEX).all():
        #             print("ERROR: all label is IGNORE_INDEX, something is wrong")

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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

        input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, image_features \
            = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, images_for_geo)
        
        # import pdb; pdb.set_trace()
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
                
                # if loss is nan
                if torch.isnan(lm_loss):
                    print("lm loss:", lm_loss.item(), "shift_labels:", shift_labels, "shift_logits:", shift_logits)
                    for l in shift_labels:
                        print(l.item())
                    # import pdb; pdb.set_trace()
                else:
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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        images_for_geo = kwargs.pop("images_for_geo", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if images_for_geo is not None:
            _inputs['images_for_geo'] = images_for_geo
        return _inputs


class LlavaGeoConfigKD(LlavaGeoConfigEarlyFusion):
    model_type = "llava_geo_kd"
    losses = [
        "lm", 
        "kd"
    ]
    loss_weights = {
        "lm": 1.0,
        "kd": 1.0
    }


import torch.nn.functional as F
class CosineSimilarityDistillationLoss(nn.Module):
    # # Example usage:
    # teacher_logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # Hypothetical logits from teacher
    # student_logits = torch.tensor([[1.5, 2.5, 3.5], [3.0, 5.0, 7.0]]) # Hypothetical logits from student

    # # Initialize the cosine similarity distillation loss
    # temperature = 5.0 # Adjust the temperature to control the smoothing
    # distillation_loss_fn = CosineSimilarityDistillationLoss(temperature=temperature)

    # # Compute the loss
    # distillation_loss = distillation_loss_fn(student_logits, teacher_logits)
    # print(distillation_loss)

    def __init__(self, temperature=1.0):

        super(CosineSimilarityDistillationLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
    
    def forward(self, student_logits, teacher_logits):
        # # Normalize teacher and student logits
        # teacher_normalized = F.normalize(teacher_logits, p=2, dim=1)
        # student_normalized = F.normalize(student_logits, p=2, dim=1)
        
        # Compute the cosine similarity between the normalized logits
        # cosine_sim = self.cosine_similarity(teacher_normalized / self.temperature, student_normalized / self.temperature)
        cosine_sim = self.cosine_similarity(teacher_logits / self.temperature, student_logits / self.temperature)
        
        # Since we want to minimize the loss, we need to subtract the cosine similarity from 1
        loss = 1 - cosine_sim.mean()
        return loss


class LlavaGeoLlamaForCausalLMKD(LlavaGeoLlamaForCausalLMEarlyFusion):
    config_class = LlavaGeoConfigKD

    def __init__(self, config):
        super().__init__(config)

        cos_kd_loss_temp = getattr(self.config, 'cos_kd_loss_temp', 1.0)
        self.kd_loss_fn = CosineSimilarityDistillationLoss(temperature=cos_kd_loss_temp)

    def _build_sam_model(self):
        sam_args = self.config.sam_config

        print("Loading SAM encoder from:", sam_args['base_config'])
        self.geo_image_processor = SamProcessor.from_pretrained(sam_args['base_config'])
        self.geo_encoder = SamModel.from_pretrained(sam_args['base_config']).vision_encoder
        self.geo_config = self.geo_encoder.config
        
        # projection layer
        # self.geo_to_llm_projector = nn.Linear(
        #     (self.geo_config.image_size // self.geo_config.patch_size) ** 2, self.config.hidden_size
        # ) # 64 * 64 = 4096 => 4096
        self.llm_to_geo_projector = nn.Linear(
            self.config.hidden_size, self.geo_config.output_channels
        ) # 4096 -> 256

        # specify image size and patch size
        for key in sam_args:
            if key not in ['base_config']:
                setattr(self.geo_config, key, sam_args[key])

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None

        # clip encode
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features] 
                # [m*patch_num_per_image, hidden_size], where m is the number of images for instance; len(image_features) = batch_size
        else:
            image_features = self.encode_images(images).to(self.device)
            

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        image_start_end_indices = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                image_start_end_indices.append([])
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]]) # e.g., 0:5; 6:10
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0) # no image tokens embedded pieces
            cur_new_input_embeds = []
            cur_new_labels = []

            cur_image_start_end_indices = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    
                    # keep track of image token indices
                    im_start_idx = sum([len(seg) for seg in cur_new_input_embeds])
                    im_end_idx = im_start_idx + cur_image_features.shape[0]
                    cur_image_start_end_indices.append([im_start_idx, im_end_idx])

                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

            image_start_end_indices.append(cur_image_start_end_indices)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)


        new_input_embeds_padded = []
        new_image_start_end_indices_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                
                if image_start_end_indices[i] == []:
                    new_image_start_end_indices_padded.append([])
                else:
                    # pad left
                    for l,r in image_start_end_indices[i]:
                        new_image_start_end_indices_padded.append([l+max_len-cur_len, r+max_len-cur_len])
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                # no need to pad image_start_end_indices if padding is on the right
                new_image_start_end_indices_padded.append(image_start_end_indices[i])

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        assert len(new_image_start_end_indices_padded) == batch_size
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, image_features, new_image_start_end_indices_padded 

    def encode_images_geo_raw(self, pixel_values):
        vision_output = self.geo_encoder(pixel_values) 
        image_features_geo = vision_output[0]     
        # batch_size, num_channels, num_patch_width, num_patch_height = image_features_geo.shape # width = height = image_size // patch_size = 64 
        # image_features_geo = image_features_geo.reshape(batch_size, num_channels, num_patch_width * num_patch_height)
        # image_features_geo = self.geo_to_llm_projector(image_features_geo)
        return image_features_geo

    def get_kd_teacher(self, images_for_geo):
        # assert one image for each instance
        assert images_for_geo.ndim == 4
        # if type(images_for_geo) is list or images_for_geo.ndim == 5:
        #     concat_images_geo = torch.cat([image for image in images_for_geo], dim=0)
        #     geo_image_features = self.encode_images_geo_raw(concat_images_geo)
        #     split_sizes = [image.shape[0] for image in images_for_geo]
        #     geo_image_features = torch.split(geo_image_features, split_sizes, dim=0)
        #     geo_image_features = [x.to(self.device) for x in geo_image_features] # [(m, num_channels, width, height), ...]
        # else:

        geo_image_features = self.encode_images_geo_raw(images_for_geo).to(self.device) # (batch_size, num_channels, width, height)
        
        # # reshape to (batch_size, width * height, num_channels)
        # if type(geo_image_features) is list:
        #     batch_size = len(geo_image_features)
        #     _, num_channels, width, height = geo_image_features[0].shape
        #     geo_image_features = [x.reshape(x.shape[0], -1, num_channels) for x in geo_image_features]
        # else:
        batch_size, num_channels, width, height = geo_image_features.shape
        geo_image_features = geo_image_features.reshape(batch_size, -1, num_channels) # (b, 4096, 256)

        # global max pooling
        # if type(geo_image_features) is list:
        #     teacher = [torch.max(x, dim=1)[0] for x in geo_image_features] # [(m, geo_hidden_size), ...]
        # else:
        teacher = torch.max(geo_image_features, dim=1)[0] # (batch_size, geo_hidden_size)    
        
        return teacher
    
    def get_kd_student(self, llm_hidden_states, image_start_end_indices, image_features):
        assert llm_hidden_states.ndim == 3
        assert image_features.ndim == 3
        num_patches = image_features.shape[1]
        students = []
        has_image_indices = []
        for i, cur_hidden_states in enumerate(llm_hidden_states):
            cur_image_start_end_indices = image_start_end_indices[i]
            if cur_image_start_end_indices == []:
                # dummy_student = torch.zeros_like(cur_hidden_states[0:num_patches])
                # dummy_student = torch.zeros((num_patches, self.geo_config.output_channels), device=self.device, dtype=cur_hidden_states.dtype)
                # students.append(dummy_student)
                continue
            else:
                assert len(cur_image_start_end_indices) == 1
                l, r = cur_image_start_end_indices[0]
                cur_student = cur_hidden_states[l:r]
                cur_student = self.llm_to_geo_projector(cur_student)
                students.append(cur_student)
                has_image_indices.append(i)
        
        student = torch.stack(students, dim=0) # (batch_size, num_patches, geo_hidden_size)
        # max pooling
        student = torch.max(student, dim=1)[0]
        return student, has_image_indices

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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

        input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, image_features, image_start_end_indices  \
            = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images)
        # image_start_end_indices: a list of batch_size containing a list [start, end] indices of the images: e.g., 
        #   [ 
        #       [ [0,576] ], # one image in this instance
        #       [ [0,576], [580,1156] ], # two images in this instance
        #   ]
        # NOTE: training data only contain single image instances

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
        lm_loss = None
        reconstruction_loss = None
        reconstruction_logits = None
        losses = {}
        if labels is not None:
            if "lm" in self.config.losses:
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
                
                print("lm loss (reweighted) | weighting factor:", lm_loss.item() * self.config.loss_weights['lm'], self.config.loss_weights['lm'])
                losses['lm'] = lm_loss
            
            if "kd" in self.config.losses:
                teacher = self.get_kd_teacher(images_for_geo)
                student, has_image_indice = self.get_kd_student(hidden_states, image_start_end_indices, image_features)

                # filter out instances without image
                teacher = teacher[has_image_indice]
                assert len(teacher) == len(student)

                kd_loss = self.kd_loss_fn(student, teacher)

                print("kd loss (reweighted) | weighting factor:", kd_loss.item() * self.config.loss_weights['kd'], self.config.loss_weights['kd'])
                losses['kd'] = kd_loss

            # import pdb; pdb.set_trace()

            for key in losses:
                if loss is None:
                    loss = losses[key] * self.config.loss_weights[key]
                else:
                    loss += losses[key] * self.config.loss_weights[key]

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # import pdb; pdb.set_trace()

        return LlavaGeoOutput(
            loss=loss,
            lm_loss=lm_loss,
            reconstruction_loss=reconstruction_loss,
            reconstruction_logits=reconstruction_logits,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_start_end_indices=image_start_end_indices
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        images_for_geo = kwargs.pop("images_for_geo", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if images_for_geo is not None:
            _inputs['images_for_geo'] = images_for_geo
        return _inputs
