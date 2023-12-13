import torch.nn as nn
import torch.nn.functional as F
import torch

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

    def __init__(self, dim=1, temperature=1.0):

        super(CosineSimilarityDistillationLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=dim)
    
    def forward(self, student_logits, teacher_logits):
        # # Normalize teacher and student logits
        # teacher_normalized = F.normalize(teacher_logits, p=2, dim=1)
        # student_normalized = F.normalize(student_logits, p=2, dim=1)
        
        # Compute the cosine similarity between the normalized logits
        # cosine_sim = self.cosine_similarity(teacher_normalized / self.temperature, student_normalized / self.temperature)
        cosine_sim = self.cosine_similarity(teacher_logits / self.temperature, student_logits / self.temperature)
        
        # Since we want to minimize the loss, we need to subtract the cosine similarity from 1
        loss = 1 - cosine_sim.mean()
        import pdb; pdb.set_trace()
        return loss

loss_fc = CosineSimilarityDistillationLoss(dim=2, temperature=1.0)

input1 = torch.randn(2, 4, 8)
input2 = torch.randn(2, 4, 8)

loss = loss_fc(input1, input2)