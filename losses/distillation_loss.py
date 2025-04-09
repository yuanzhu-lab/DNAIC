import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5,
                 teacher_vocab_size=None, student_vocab_size=None,
                 mapping_hidden_size=None, mapping_dropout=0.1,
                 ignore_index=-1):
        """
        Parameters:
          temperature: Temperature factor
          alpha: Weight ratio between hard labels and soft labels loss
          teacher_vocab_size: Teacher vocabulary size (actual teacher output dimension is teacher_vocab_size - 1)
          student_vocab_size: Student vocabulary size
          mapping_hidden_size: Hidden layer size of the mapping network, default value is (input_dim + student_vocab_size)//2
          mapping_dropout: Dropout probability for the mapping network
          ignore_index: Label value to ignore in CrossEntropyLoss (should be consistent with data preprocessing)
        """
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='mean')
        
        assert teacher_vocab_size is not None and student_vocab_size is not None, "Teacher and student vocabulary sizes must be provided"
        input_dim = teacher_vocab_size - 1  # Actual teacher output dimension
        hidden_dim = mapping_hidden_size if mapping_hidden_size is not None else ((input_dim + student_vocab_size) // 2)
        
        # Two-layer feedforward mapping with ReLU activation and dropout
        self.vocab_mapping = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=mapping_dropout),
            nn.Linear(hidden_dim, student_vocab_size)
        )
    
    def forward(self, student_logits, teacher_logits, target):
        """
        Calculate knowledge distillation loss, including hard labels (cross entropy) and soft labels (KL divergence).
        Parameters:
          student_logits: Student model output, shape [B, T, student_vocab_size] or [B, student_vocab_size]
          teacher_logits: Teacher model output, shape [B, T, teacher_vocab_size] or [B, teacher_vocab_size]
          target: Target labels, shape [B, T] or [B]
        """
        # ----------------- Hard Label Loss -----------------
        # Same as the example, first flatten the logits and target
        if student_logits.dim() == 3:
            B, T, V = student_logits.size()
            student_logits_flat = student_logits.view(-1, V)
            target_flat = target.view(-1)
        else:
            student_logits_flat = student_logits
            target_flat = target
        
        hard_loss = self.ce_criterion(student_logits_flat, target_flat)
        
        # ----------------- Teacher Output Mapping -----------------
        # Map teacher output to student vocabulary space
        if teacher_logits.dim() == 3:
            B, T, _ = teacher_logits.size()
            teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))
            teacher_logits_mapped = self.vocab_mapping(teacher_logits_flat)
            teacher_logits = teacher_logits_mapped.view(B, T, -1)
        else:
            teacher_logits = self.vocab_mapping(teacher_logits)
        
        # ----------------- Align Teacher and Student Output -----------------
        if teacher_logits.dim() != student_logits.dim():
            if teacher_logits.dim() == 2 and student_logits.dim() == 3:
                teacher_logits = teacher_logits.unsqueeze(1).expand(-1, student_logits.size(1), -1)
            elif teacher_logits.dim() == 3 and student_logits.dim() == 2:
                teacher_logits = teacher_logits.mean(dim=1)
        
        # ----------------- Soft Label Loss (KL Divergence) -----------------
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        if student_logits.dim() == 3:
            student_log_probs_flat = student_log_probs.view(-1, student_log_probs.size(-1))
            teacher_probs_flat = teacher_probs.view(-1, teacher_probs.size(-1))
            soft_loss = F.kl_div(student_log_probs_flat, teacher_probs_flat, reduction='batchmean') * (self.temperature ** 2)
        else:
            soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
        
        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss
