import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5,
                 teacher_vocab_size=None, student_vocab_size=None,
                 mapping_hidden_size=None, mapping_dropout=0.1,
                 ignore_index=-1):
        """
        参数说明:
          temperature: 温度因子
          alpha: 硬标签与软标签损失的权重比
          teacher_vocab_size: 教师词汇表尺寸（实际教师输出维度为 teacher_vocab_size - 1）
          student_vocab_size: 学生词汇表尺寸
          mapping_hidden_size: 映射网络隐藏层尺寸，默认值为 (input_dim + student_vocab_size)//2
          mapping_dropout: 映射网络 dropout 概率
          ignore_index: CrossEntropyLoss 忽略的标签值（应与数据预处理保持一致）
        """
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='mean')
        
        assert teacher_vocab_size is not None and student_vocab_size is not None, "必须提供教师和学生的词汇表尺寸"
        input_dim = teacher_vocab_size - 1  # 实际教师输出维度
        hidden_dim = mapping_hidden_size if mapping_hidden_size is not None else ((input_dim + student_vocab_size) // 2)
        
        # 两层前馈映射，含 ReLU 激活及 dropout
        self.vocab_mapping = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=mapping_dropout),
            nn.Linear(hidden_dim, student_vocab_size)
        )
    
    def forward(self, student_logits, teacher_logits, target):
        """
        计算知识蒸馏损失，包含硬标签（交叉熵）和软标签（KL散度）两部分。
        参数:
          student_logits: 学生模型输出，形状 [B, T, student_vocab_size] 或 [B, student_vocab_size]
          teacher_logits: 教师模型输出，形状 [B, T, teacher_vocab_size] 或 [B, teacher_vocab_size]
          target: 目标标签，形状 [B, T] 或 [B]
        """
        # ----------------- 硬标签损失 -----------------
        # 与示例相同，先将 logits 和 target 展平
        if student_logits.dim() == 3:
            B, T, V = student_logits.size()
            student_logits_flat = student_logits.view(-1, V)
            target_flat = target.view(-1)
        else:
            student_logits_flat = student_logits
            target_flat = target
        
        hard_loss = self.ce_criterion(student_logits_flat, target_flat)
        
        # ----------------- 教师输出映射 -----------------
        # 将教师输出映射至学生词汇空间
        if teacher_logits.dim() == 3:
            B, T, _ = teacher_logits.size()
            teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))
            teacher_logits_mapped = self.vocab_mapping(teacher_logits_flat)
            teacher_logits = teacher_logits_mapped.view(B, T, -1)
        else:
            teacher_logits = self.vocab_mapping(teacher_logits)
        
        # ----------------- 对齐教师和学生输出 -----------------
        if teacher_logits.dim() != student_logits.dim():
            if teacher_logits.dim() == 2 and student_logits.dim() == 3:
                teacher_logits = teacher_logits.unsqueeze(1).expand(-1, student_logits.size(1), -1)
            elif teacher_logits.dim() == 3 and student_logits.dim() == 2:
                teacher_logits = teacher_logits.mean(dim=1)
        
        # ----------------- 软标签损失（KL散度） -----------------
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        if student_logits.dim() == 3:
            student_log_probs_flat = student_log_probs.view(-1, student_log_probs.size(-1))
            teacher_probs_flat = teacher_probs.view(-1, teacher_probs.size(-1))
            soft_loss = F.kl_div(student_log_probs_flat, teacher_probs_flat, reduction='batchmean') * (self.temperature ** 2)
        else:
            soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
        
        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss
