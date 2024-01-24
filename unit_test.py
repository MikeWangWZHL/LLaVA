import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


# # write a simple test function doing some tensor manipulation using gpu
# def test_gpu():
#     x = torch.rand(5, 3).to('cuda')
#     y = torch.rand(5, 3).to('cuda')
#     z = x + y
#     print(z)


# def test_llava():
#     print(IMAGE_TOKEN_INDEX)

# if __name__ == '__main__':
#     print('testing...')
#     test_gpu()
#     test_llava()



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CosineSimilarityDistillationLoss(nn.Module):
#     def __init__(self, temperature=1.0):
#         super(CosineSimilarityDistillationLoss, self).__init__()
#         self.temperature = temperature
#         self.cosine_similarity = nn.CosineSimilarity(dim=1)
    
#     def forward(self, student_logits, teacher_logits):
#         # # Normalize teacher and student logits
#         # teacher_normalized = F.normalize(teacher_logits, p=2, dim=1)
#         # student_normalized = F.normalize(student_logits, p=2, dim=1)
        
#         # Compute the cosine similarity between the normalized logits
#         # cosine_sim = self.cosine_similarity(teacher_normalized / self.temperature, student_normalized / self.temperature)
        
#         cosine_sim = self.cosine_similarity(student_logits / self.temperature, teacher_logits / self.temperature)
        
#         print(cosine_sim.shape, cosine_sim)
#         # Since we want to minimize the loss, we need to subtract the cosine similarity from 1
#         loss = 1 - cosine_sim.mean()
#         return loss

# # Example usage:
# teacher_logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # Hypothetical logits from teacher
# student_logits = torch.tensor([[1.5, 2.5, 3.5], [3.0, 5.0, 7.0]]) # Hypothetical logits from student

# # Initialize the cosine similarity distillation loss
# temperature = 1.0 # Adjust the temperature to control the smoothing
# distillation_loss_fn = CosineSimilarityDistillationLoss(temperature=temperature)

# # Compute the loss
# distillation_loss = distillation_loss_fn(student_logits, teacher_logits)
# print(distillation_loss)

# cur_image_features = torch.rand(576, 768)
# foo = cur_image_features[0:0]

# import pdb; pdb.set_trace()



# import json
# # check instruction tuning dataset, if there are multi image instance?
# data = json.load(open('/scratch/bcdq/wangz3/llava_data/llava_v1_5_mix665k.json'))
# print(len(data))

# for item in data:
#     for message in item['conversations']:
#         if message["from"] == "human":
#             # if <image> appears more than once, then it is multi image instance
#             if message["value"].count("<image>") > 1:
#                 print(item)


from transformers import SamModel
model = SamModel.from_pretrained("facebook/sam-vit-base").vision_encoder

import pdb; pdb.set_trace()