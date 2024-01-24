from llava.model import *

MODEL_TYPE_TO_MODEL_CLASS = {
    "llava": LlavaLlamaForCausalLM,
    "llava_geo_mae": LlavaGeoLlamaForCausalLMMAE,
    "llava_geo_early_fusion": LlavaGeoLlamaForCausalLMEarlyFusion,
    "llava_geo_early_fusion_kd": LlavaGeoLlamaForCausalLMEarlyFusionKD,
    "llava_geo_kd": LlavaGeoLlamaForCausalLMKD,
}

MODEL_TYPE_TO_CONFIG_CLASS = {
    "llava": LlavaConfig,
    "llava_geo_mae": LlavaGeoConfigMAE,
    "llava_geo_early_fusion": LlavaGeoConfigEarlyFusion,
    "llava_geo_early_fusion_kd": LlavaGeoConfigEarlyFusion,
    "llava_geo_kd": LlavaGeoConfigKD,
}