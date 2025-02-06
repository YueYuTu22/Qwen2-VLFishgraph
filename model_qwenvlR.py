#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多模态 + 知识图谱 + RotatE + Qwen2-VL
-------------------------------------------------
改进点：
1. 使用Qwen2-VL作为多模态编码器（可做文本+图像多模态特征抽取）。
2. 替换TransE为RotatE，用来建模实体-关系三元组。
3. 可选地实现“硬负采样”策略。
4. 用CrossAttention(如MultiheadAttention)替代CBAM。
5. 将Head embedding与多模态embedding融合改为可学习的权重(投影等)。
6. 修复了两个常见错误：
   - “embedding dimension mismatch” => 通过投影 Qwen2VL输出 => 512，再CrossAttention => emb_dim
   - “IndexError: list index out of range” => 通过补齐neg_tail_texts外层/内层长度
   - “TypeError: can only concatenate tuple (not 'list') ...” => 在拼接前转list
"""

from typing import List, Dict

import os
import sys
import csv
import uuid
import random
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.transforms as T
from PIL import Image
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# 使用 torchmetrics 进行多分类评估
from torchmetrics.classification import Accuracy
from transformers import PreTrainedTokenizer

from difflib import SequenceMatcher

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

# TorchMetrics
from torchmetrics.retrieval import RetrievalMRR, RetrievalRecall

# Transformers
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)
# LR 调度器
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

##############################################################################
# Default Hyperparams
##############################################################################
DEFAULT_MAX_LENGTH   = 128
DEFAULT_BATCH_SIZE   = 32
DEFAULT_MAX_EPOCHS   = 100
DEFAULT_LR           = 1e-4
DEFAULT_EMB_DIM      = 256
DEFAULT_NEG_SAMPLES  = 2
DEFAULT_ALPHA_MM     = 1.0

FREEZE_TEXT  = True
FREEZE_IMAGE = True

LR_SCHEDULER_CHOICES = ["none", "cosine", "plateau"]
DEFAULT_DEBUG_FLAG   = False

##############################################################################
# Paths
##############################################################################
#QWEN2_VL_MODEL_PATH = (
#    "/home/shengguang/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/47592516d3e709cd9c194715bc76902241c5edea"
#)

QWEN2_VL_MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"

CSV_FILE = r"C:\Users\yuyue\OneDrive\Documents\WeChat Files\wxid_0xz9g64tsb3b22\FileStorage\File\2025-01\FishGraph(1)\FishGraph\FishGraph\annotation.csv"   # or your actual CSV path
IMAGE_ROOT_DIR = r"C:\Users\yuyue\OneDrive\Documents\WeChat Files\wxid_0xz9g64tsb3b22\FileStorage\File\2025-01\FishGraph(1)\FishGraph\FishGraph\标注后的图像数据改名后的"

##############################################################################
# 1) 读取CSV => (病名, 关系, 对象)
##############################################################################
REL_COLS = {
    "感染对象":  "感染",
    "症状":     "症状",
    "流行季节": "流行季节",
    "流行地区": "流行地区",
    "地区":     "地区",
    "季节":     "季节",
    "温度":     "温度",
    "时间":     "时间",
}

def load_dynamic_csv(csv_file: str, image_root: str):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    df = pd.read_csv(csv_file, encoding="utf-8", dtype=str).fillna("")
    must_cols = ["病名", "image_path", "text"]
    for c in must_cols:
        if c not in df.columns:
            raise ValueError(f"Missing col={c} in CSV")

    df["image_path"] = df["image_path"].apply(lambda x: str(x).replace("\\", "/"))
    out_data = []
    for _, row in df.iterrows():
        disease_str = row["病名"].strip()
        if not disease_str:
            continue
        img_path = row["image_path"].strip()
        full_img_path = os.path.join(image_root, img_path)
        text_str = row["text"].strip()

        # 生成双通道 Prompt
        prompt_text = f"""
        请从以下文本或图像中提取相关疾病信息：
        "{text_str}"

        [LABEL]  
        [EXPLANATION]  

        请在 [LABEL] 处填写疾病关系类别，在 [EXPLANATION] 处填写解释。
        """

        for colName, relName in REL_COLS.items():
            if colName in df.columns:
                val = str(row[colName]).strip()
                if val:
                    out_data.append({
                        "disease":   disease_str,
                        "relation":  relName,
                        "object":    val,
                        "image_path": full_img_path,
                        "text":      text_str
                    })
    return out_data

def oversample_data(records, target_column="object", rare_classes=None):
    """
    对少量类样本进行过采样。
    """
    rare_classes = rare_classes or []
    oversampled_records = []

    for record in records:
        if record[target_column] in rare_classes:
            # 根据稀有类的比例复制样本
            oversampled_records.extend([record] * 5)  # 5 是过采样倍数
        else:
            oversampled_records.append(record)

    random.shuffle(oversampled_records)
    return oversampled_records

def create_prompt_examples():
    """
    生成插桩式 Prompt 数据，包括正例和负例。
    """
    prompt_examples = [
        {"text": "[LABEL]感染 [EXPLANATION]它可能是由病原体传播", "label": "感染"},
        {"text": "[LABEL]症状 [EXPLANATION]常见症状包括发烧和咳嗽", "label": "症状"},
        {"text": "[LABEL]流行季节 [EXPLANATION]这种疾病通常发生在夏季", "label": "流行季节"},
        {"text": "[LABEL]感染病 [EXPLANATION]这是不准确的写法", "label": None},
    ]
    return prompt_examples

def restrict_tokenizer_to_labels(tokenizer: PreTrainedTokenizer, label_words: list[str]):
    """
    限制 Tokenizer 的词表，仅包含指定的标签词。
    Args:
        tokenizer (PreTrainedTokenizer): 原始 Tokenizer。
        label_words (List[str]): 允许解码的标签词列表。
    Returns:
        PreTrainedTokenizer: 限制后的 Tokenizer。
    """
    vocab = tokenizer.get_vocab()
    allowed_tokens = {word: idx for word, idx in vocab.items() if word in label_words}

    # 保留指定的标签词，其他词用特殊标记替换（如 <unk>）
    tokenizer.vocab = allowed_tokens
    tokenizer.add_special_tokens({"additional_special_tokens": ["<unk>"]})
    return tokenizer

def augment_prompt_with_rare_classes(records, rare_classes=None):
    rare_classes = rare_classes or []
    augmented_records = []

    for record in records:
        augmented_records.append(record)
        # 针对稀有类增加示例
        if record["object"] in rare_classes:
            extra_prompt = {
                "disease_text": f"[LABEL]{record['relation']} [EXPLANATION]这是稀有类'{record['object']}'的示例。",
                "image_path": record["image_path"],
                "obj_id": record["obj_id"]
            }
            augmented_records.append(extra_prompt)
    return augmented_records

def label_discriminator(output_label, valid_labels, max_length=10):
    """
    判别器逻辑：
    1. 检查输出标签是否超出最大长度。
    2. 检查标签是否在合法词表中。
    3. 如果非法，基于相似度匹配最近的合法标签。
    """
    if len(output_label) > max_length:
        return None  # 非法输出

    if output_label not in valid_labels:
        # 相似度匹配纠正
        corrected_label = max(valid_labels, key=lambda label: cosine_similarity(output_label, label))
        return corrected_label

    return output_label

def cosine_similarity(str1, str2):
    """简单的余弦相似度实现"""
    set1 = set(str1)
    set2 = set(str2)
    intersection = set1.intersection(set2)
    if not set1 or not set2:
        return 0.0
    return len(intersection) / (len(set1) * len(set2)) ** 0.5

def truncate_output(generated_text, label_token="[LABEL]", max_tokens=3):
    """
    对生成的文本进行截断，限制 [LABEL] 后的 Token 数量。
    Args:
        generated_text (str): 模型生成的文本。
        label_token (str): 特殊标记，用于指示截断位置。
        max_tokens (int): [LABEL] 后允许的最大 Token 数。
    Returns:
        str: 截断后的文本。
    """
    if label_token not in generated_text:
        return generated_text  # 如果没有 [LABEL]，直接返回

    # 找到 [LABEL] 的位置并截取之后的文本
    label_index = generated_text.index(label_token)
    after_label = generated_text[label_index + len(label_token):].strip()

    # 按空格分割 Token
    tokens = after_label.split()
    if len(tokens) > max_tokens:
        truncated = " ".join(tokens[:max_tokens])
        return f"{generated_text[:label_index + len(label_token)]} {truncated}"
    
    return generated_text


def constrained_decoding_with_truncation(tokenizer, input_ids, model, label_words, max_tokens=3):
    """
    带约束的自适应截断解码，确保 [LABEL] 后只生成指定标签。
    
    Args:
        tokenizer: 用于分词和解码的 tokenizer 对象。
        input_ids: 输入的 token ids（PyTorch 张量）。
        model: Transformer 模型。
        label_words: 合法标签列表，例如 ['感染', '症状', '流行季节', ...]。
        max_tokens: [LABEL] 后允许生成的最大 token 数。
    
    Returns:
        Decoded string: 最终解码后的字符串。
    """
    device = input_ids.device
    valid_token_ids = tokenizer.convert_tokens_to_ids(label_words)  # 合法标签转为 token ids
    output_ids = input_ids.clone()  # 初始化解码输出，复制 input_ids

    # 初始模型状态
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            # 模型前向传播，获取 logits
            outputs = model(input_ids=output_ids)
            logits = outputs.last_hidden_state[:, -1, :]  # 获取最后一个时间步的 logits

            # 设置非法 token 的概率为 -inf，确保模型只生成合法标签
            mask = torch.full_like(logits, fill_value=-float('inf'), device=device)
            mask[:, valid_token_ids] = logits[:, valid_token_ids]
            filtered_logits = mask

            # 选择概率最高的 token 作为下一个生成的 token
            next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)

            # 如果生成了 <eos> 或无效 token，则停止解码
            if next_token.item() == tokenizer.eos_token_id or next_token.item() not in valid_token_ids:
                break

            # 将生成的 token 拼接到输出序列中
            output_ids = torch.cat([output_ids, next_token], dim=-1)

    # 解码最终生成的序列
    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded_output



def compute_loss_with_label_weighting(logits, labels, mask, weight_factor=2.0):
    """
    计算加权交叉熵损失，对 [LABEL] 部分提升权重。
    
    Args:
        logits: 模型输出的 logits，形状为 [B, num_classes]
        labels: 标签张量，形状为 [B]
        mask: 权重掩码，形状为 [B]
        weight_factor: 权重因子，用于控制 [LABEL] 的权重（默认 2.0）
    
    Returns:
        加权后的平均损失值
    """
    ce_loss = nn.CrossEntropyLoss(reduction='none')  # 不做平均，逐样本计算
    losses = ce_loss(logits, labels)  # [B]
    
    # 对 mask 中为 1 的样本提升权重
    weighted_loss = losses * (mask * (weight_factor - 1) + 1)
    
    # 返回加权后的平均损失
    return weighted_loss.mean()

def decode_label(self, label_idx):
    """
    将预测的索引转换为标签。
    """
    if hasattr(self, "id_to_label"):
        return self.id_to_label.get(label_idx, "UNKNOWN")
    else:
        raise ValueError("id_to_label mapping is missing.")
    
##############################################################################
# 2) Dataset => 保证 neg_tail_texts 与 neg_samples匹配
##############################################################################
class MultiModalTripleDataset(Dataset):
    def __init__(
        self,
        records,
        ent2id,
        rel2id,
        neg_samples=2,
        obj_name_map=None,
        hard_negatives: Dict[str, List[str]] = None
    ):
        super().__init__()
        self.records       = records
        self.ent2id        = ent2id
        self.rel2id        = rel2id
        self.neg_samples   = neg_samples
        self.all_entities  = list(ent2id.keys())
        self.obj_name_map  = obj_name_map or {}
        self.hard_negatives= hard_negatives or {}

        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        disease_str   = rec["disease"]
        rel_str       = rec["relation"]
        obj_str       = rec["object"]
        img_path      = rec["image_path"]
        head_text_str = rec["text"]
        tail_text_str = self.obj_name_map.get(obj_str, obj_str)

        head_id = self.ent2id.get(disease_str, 0)
        rel_id  = self.rel2id.get(rel_str, 0)
        tail_id = self.ent2id.get(obj_str, 0)

        if not os.path.exists(img_path):
            pil_img = Image.new("RGB",(224,224), color="black")
        else:
            pil_img = Image.open(img_path).convert("RGB")
        img_tsr = self.img_transform(pil_img)

            # 生成 prompt
        prompt_text = f"""
        请从以下文本或图像中提取相关疾病信息：
        "{rec['text']}"

        [LABEL]  
        [EXPLANATION]  

        请在 [LABEL] 处填写疾病关系类别，在 [EXPLANATION] 处填写解释。
        """

        # 负采样 => 返回 neg_samples 个
        neg_ids   = []
        neg_texts = []
        for _ in range(self.neg_samples):
            hard_neg_list = self.hard_negatives.get(obj_str)
            if hard_neg_list:
                cand = random.choice(hard_neg_list)
                while (cand == obj_str) or (cand not in self.ent2id):
                    cand = random.choice(self.all_entities)
            else:
                cand = random.choice(self.all_entities)
                while cand == obj_str:
                    cand = random.choice(self.all_entities)

            neg_ids.append(self.ent2id[cand])
            neg_texts.append(self.obj_name_map.get(cand, cand))

        neg_tails = torch.tensor(neg_ids, dtype=torch.long)

        return {
            "head_id":       head_id,
            "rel_id":        rel_id,
            "tail_id":       tail_id,
            "image":         img_tsr,
            "head_text":     head_text_str,
            "tail_text":     tail_text_str,
            "neg_tails":     neg_tails,
            "disease_text": rec["text"],
            "relation": rec["relation"],
            "neg_tail_texts": neg_texts,  # len=neg_samples
            "prompt": prompt_text 
        }

##############################################################################
# 3) CrossAttention => embed_dim=512
##############################################################################
class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm(attn_output + x)
        ffn_output = self.ffn(x)
        x = self.layer_norm2(ffn_output + x)
        return x

##############################################################################
# 4) RotatE
##############################################################################
class RotatE(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.emb_dim = emb_dim
        assert emb_dim % 2 == 0, "Embedding dimension must be even for RotatE."

    def forward(self, h, r, t):
        h_real, h_imag = torch.chunk(h, 2, dim=-1)
        r_real, r_imag = torch.chunk(r, 2, dim=-1)
        t_real, t_imag = torch.chunk(t, 2, dim=-1)

        t_pred_real = h_real * r_real - h_imag * r_imag
        t_pred_imag = h_real * r_imag + h_imag * r_real

        real_diff = t_pred_real - t_real
        imag_diff = t_pred_imag - t_imag
        score = torch.norm(torch.cat([real_diff, imag_diff], dim=-1), dim=-1)
        return score

##############################################################################
# 5) LightningModule
##############################################################################
class KGMulModalLightningModule(pl.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        ent_total=100,
        rel_total=5,
        emb_dim=256,
        freeze_text=True,
        freeze_image=True,
        neg_samples=2,
        alpha_mm=1.0,
        margin=1.0,
        lr_scheduler="none",
        debug=False,
        valid_labels=["感染", "症状", "流行季节", "流行地区"],
        qwen2_vl_model_name=QWEN2_VL_MODEL_PATH,
        use_image_prompt=False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.margin = margin  # 新增 margin 超参数
        self.valid_labels = valid_labels or []
        self.debug= debug
        self.use_image_prompt = use_image_prompt

        # RotatE
        self.ent_embed= nn.Embedding(ent_total, emb_dim)
        self.rel_embed= nn.Embedding(rel_total, emb_dim)
        self.rotate = RotatE(emb_dim=emb_dim)

        # Qwen2-VL
        self.processor = Qwen2VLProcessor.from_pretrained(qwen2_vl_model_name)
        self.qwen2_vl = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen2_vl_model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        if freeze_text:
            for param in self.qwen2_vl.parameters():
                param.requires_grad=False

        # 维度投影
        self.qwen_hidden_size = self.qwen2_vl.config.hidden_size  # e.g.1536
        self.proj_dim = 512
        self.qwen_proj = nn.Linear(self.qwen_hidden_size, self.proj_dim)

        self.cross_attention = CrossAttentionModule(embed_dim=self.proj_dim)
        self.fusion_proj = nn.Linear(self.proj_dim, emb_dim)

        # 损失函数改为 Margin Ranking Loss
        self.margin_loss = nn.MarginRankingLoss(margin=self.margin)

        # 仅保留排名指标
        self.train_mrr= RetrievalMRR()
        self.train_r1 = RetrievalRecall(top_k=1)
        self.val_mrr  = RetrievalMRR()
        self.val_r1   = RetrievalRecall(top_k=1)

    # =========== utils ===========
    def get_ent_embed(self, ent_ids):
        emb= self.ent_embed(ent_ids)
        emb= emb/(emb.norm(dim=1, keepdim=True)+1e-9)
        return emb

    def get_rel_embed(self, rel_ids):
        emb= self.rel_embed(rel_ids)
        emb= emb/(emb.norm(dim=1, keepdim=True)+1e-9)
        return emb

    def triple_score(self, h, r, t):
        return self.rotate(h, r, t)

    # =========== encode multimodal ===========
    def encode_multimodal(self, text_list, image_tensor=None):
        if self.use_image_prompt and (image_tensor is not None):
            text_list = [f"<|image|> {t}" for t in text_list]
            pil_img = T.ToPILImage()(image_tensor)
            images_for_proc = [pil_img]
        else:
            images_for_proc = None

        inputs = self.processor(
            text=text_list,
            images=images_for_proc,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            out= self.qwen2_vl(**inputs, output_hidden_states=True)
            hidden_states= out.hidden_states[-1]
            cls_emb= hidden_states[:,0,:]

        cls_proj= self.qwen_proj(cls_emb)       # [B, 512]
        x= cls_proj.unsqueeze(1)               # [B,1,512]
        x= self.cross_attention(x)             # [B,1,512]
        x= self.fusion_proj(x).squeeze(1)      # [B, emb_dim=256]
        return x

    # =========== forward ===========
    def forward(self,
                heads, rels, tails,
                head_text_list, head_imgs,
                tail_text_list,
                neg_tail_texts, neg_tails):
        B= heads.size(0)
        K= neg_tails.size(1)

        # head mm
        head_mm_list= []
        for i in range(B):
            txt_i= [head_text_list[i]]
            img_i= head_imgs[i]
            mm_i= self.encode_multimodal(txt_i, img_i)
            head_mm_list.append(mm_i)
        head_mm= torch.cat(head_mm_list, dim=0)  # [B, emb_dim]

        head_e_base= self.get_ent_embed(heads)   # [B, emb_dim]
        head_e= head_e_base + self.hparams.alpha_mm * head_mm
        head_e= head_e / (head_e.norm(dim=1, keepdim=True) + 1e-9)

        # rel
        rel_e= self.get_rel_embed(rels)         # [B, emb_dim]

        # tail mm
        tail_mm_list= []
        for i in range(B):
            txt_i= [tail_text_list[i]]
            mm_i= self.encode_multimodal(txt_i, None)
            tail_mm_list.append(mm_i)
        tail_mm= torch.cat(tail_mm_list, dim=0)  # [B, emb_dim]
        tail_e_base= self.get_ent_embed(tails)   # [B, emb_dim]
        tail_e= tail_e_base + self.hparams.alpha_mm * tail_mm
        tail_e= tail_e / (tail_e.norm(dim=1, keepdim=True) + 1e-9)

        pos_sc= self.triple_score(head_e, rel_e, tail_e)  # [B]

        # negative
        neg_ids_flat= neg_tails.reshape(-1)              # [B*K]
        neg_e_base= self.get_ent_embed(neg_ids_flat)     # [B*K, emb_dim]

        # flatten neg_texts
        neg_texts_flat=[]
        for i in range(B):
            # 如果 i >= len(neg_tail_texts)，则补一个空
            if i >= len(neg_tail_texts):
                neg_tail_texts.append(["Unknown"]*K)

            sample_list= neg_tail_texts[i]

            # tuple => 需转成 list
            if isinstance(sample_list, tuple):
                sample_list = list(sample_list)

            # 补齐 or 截断
            if len(sample_list) < K:
                last_txt= sample_list[-1] if sample_list else "Unknown"
                sample_list = sample_list + [last_txt]*(K - len(sample_list))
            elif len(sample_list) > K:
                sample_list= sample_list[:K]

            neg_texts_flat.extend(sample_list)

        neg_mm_list= []
        for idx, neg_str in enumerate(neg_texts_flat):
            mm_ij= self.encode_multimodal([neg_str], None)
            neg_mm_list.append(mm_ij)
        neg_mm= torch.cat(neg_mm_list, dim=0)          # [B*K, emb_dim]
        neg_e= neg_e_base + self.hparams.alpha_mm * neg_mm
        neg_e= neg_e / (neg_e.norm(dim=1, keepdim=True) + 1e-9)  # [B*K, emb_dim]

        h_e_exp= head_e.unsqueeze(1).expand(B, K, head_e.size(1)).reshape(B*K, head_e.size(1))  # [B*K, emb_dim]
        r_e_exp= rel_e.unsqueeze(1).expand(B, K, rel_e.size(1)).reshape(B*K, rel_e.size(1))    # [B*K, emb_dim]
        neg_sc= self.triple_score(h_e_exp, r_e_exp, neg_e).view(B, K)                         # [B, K]

        return pos_sc, neg_sc  # [B], [B, K]

    # =========== training_step ===========
    def training_step(self, batch, batch_idx):
        device = self.device   
        heads= batch["head_id"].to(self.device)
        rels=  batch["rel_id"].to(self.device)
        tails= batch["tail_id"].to(self.device)
        head_text_list= batch["head_text"]
        imgs= batch["image"].to(self.device)
        tail_text_list= batch["tail_text"]
        negs= batch["neg_tails"].to(self.device)
        neg_texts= batch["neg_tail_texts"]
        disease_texts = [b["disease_text"] for b in batch]
        labels = [b["obj_id"] for b in batch]


        pos_sc, neg_sc = self(
            heads, rels, tails,
            head_text_list, imgs,
            tail_text_list,
            neg_texts, negs
        )  # pos_sc => [B], neg_sc => [B, K]

        B, K = neg_sc.size()

        # 准备 preds、target 和 indexes
        preds = torch.cat([pos_sc.unsqueeze(1), neg_sc], dim=1).view(-1)  # [B*(K+1)]
        target = torch.cat([torch.ones(B, 1, device=self.device), torch.zeros(B, K, device=self.device)], dim=1).view(-1)  # [B*(K+1)]
        indexes = torch.arange(B, device=self.device).unsqueeze(1).repeat(1, K+1).view(-1)  # [B*(K+1)]

        # 定义 margin loss
        # 假设 score 越小越好（TransE/RotatE）
        # 通过取负值使得更高的分数表示更好的预测
        preds_neg = -preds  # [B*(K+1)]
        loss = self.margin_loss(preds_neg, torch.ones_like(preds_neg, device=self.device), target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=B)

        # 更新排名指标
        self.train_mrr.update(preds_neg, target, indexes)
        self.train_r1.update(preds_neg, target, indexes)

        # 1. 格式化输入 Prompt
        # 使用 [LABEL] 和 [EXPLANATION] 格式的 Prompt
        #print(f"Batch sample: {batch[0]}") 

         # 2. 将标签转为 Tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)  # [B]

         # 3. 前向传播
        logits = self.forward(disease_texts, imgs)  # => [B, num_objects]

        # 4. 定义 mask，用于提高 [LABEL] 部分的权重
        label_mask = torch.tensor(
          [1 if "[LABEL]" in txt else 0 for txt in disease_texts], dtype=torch.float32, device=device
        )

         # 5. 自定义 Loss 函数，提升 [LABEL] 后的 Token 权重
        loss = compute_loss_with_label_weighting(logits, labels_tensor, label_mask)

        # 6. 更新准确率
        preds = logits.argmax(dim=1)
        self.train_acc.update(preds, labels_tensor)

        # 7. 记录训练损失
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=len(batch))
       
        return loss

    def on_train_epoch_end(self):
        mrr= self.train_mrr.compute()
        r1= self.train_r1.compute()
        self.log("train_mrr", mrr, prog_bar=True)
        self.log("train_r1", r1)
        self.train_mrr.reset()
        self.train_r1.reset()

    def validation_step(self, batch, batch_idx):
        device = self.device
        heads= batch["head_id"].to(self.device)
        rels=  batch["rel_id"].to(self.device)
        tails= batch["tail_id"].to(self.device)
        head_text_list= batch["head_text"]
        imgs= batch["image"].to(self.device)
        tail_text_list= batch["tail_text"]
        negs= batch["neg_tails"].to(self.device)
        neg_texts= batch["neg_tail_texts"]
        labels = batch["obj_id"].tolist()
        disease_texts = batch["disease_text"]
        
        pos_sc, neg_sc = self(
            heads, rels, tails,
            head_text_list, imgs,
            tail_text_list,
            neg_texts, negs
        )  # pos_sc => [B], neg_sc => [B, K]

        B, K = neg_sc.size()

        # 准备 preds、target 和 indexes
        preds = torch.cat([pos_sc.unsqueeze(1), neg_sc], dim=1).view(-1)  # [B*(K+1)]
        target = torch.cat([torch.ones(B, 1, device=self.device), torch.zeros(B, K, device=self.device)], dim=1).view(-1)  # [B*(K+1)]
        indexes = torch.arange(B, device=self.device).unsqueeze(1).repeat(1, K+1).view(-1)  # [B*(K+1)]

        # 定义 margin loss
        preds_neg = -preds  # [B*(K+1)]
        loss = self.margin_loss(preds_neg, torch.ones_like(preds_neg, device=self.device), target)

        self.log("val_loss", loss, prog_bar=True, batch_size=B)
        # 转换标签为张量
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        logits = self.forward(disease_texts, imgs)
        loss = self.loss_fn(logits, labels_tensor)
        
        # 获取预测结果及其置信度
        probs = torch.softmax(logits, dim=1)  # 计算每个类别的概率
        preds = logits.argmax(dim=1)  # 获取预测标签索引
        confidences = probs.max(dim=1).values  # 获取每个预测的最高置信度
        # 添加 [LABEL] 通道受限解码逻辑
        valid_labels = ["感染", "症状", "流行季节", "流行地区"]  # 定义合法标签集合
        
        object2id = {label: idx for idx, label in enumerate(valid_labels)}
        print(f"[Info] object2id mapping: {object2id}")
        decoded_labels = [
           valid_labels[pred] if pred < len(valid_labels) else "未知"
           for pred in preds.cpu().numpy()
        ]

        print(decoded_labels)

         # 结合置信度和标签
        results_with_confidence = [
            {"decoded_label": decoded_labels[i], "confidence": confidences[i].item()}
             for i in range(len(decoded_labels))
           ]
        if batch_idx == 0:  # 仅记录第一批次
          print(f"Sample decoded outputs with confidence: {results_with_confidence[:3]}")

        # 更新排名指标
        self.val_mrr.update(preds_neg, target, indexes)
        self.val_r1.update(preds_neg, target, indexes)

        if batch_idx == 0 and self.debug:
            # 仅在调试模式下记录一些样本
            sample_num= min(3, heads.size(0))
            for i in range(sample_num):
                print(f"Sample {i}:")
                print(f"Head: {head_text_list[i]}")
                print(f"Relation: {rels[i].item()}")
                print(f"True Tail: {tail_text_list[i]}")
                print(f"Pos Score: {pos_sc[i].item()}")
                print(f"Neg Scores: {neg_sc[i].tolist()}")
                print("-" * 20)

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        mrr= self.val_mrr.compute()
        r1= self.val_r1.compute()
        self.log("val_mrr", mrr, prog_bar=True)
        self.log("val_r1", r1)
        self.val_mrr.reset()
        self.val_r1.reset()

    def configure_optimizers(self):
        opt= torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sch_type= self.hparams.lr_scheduler
        if sch_type=="cosine":
            sched= CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
            return [opt],[sched]
        elif sch_type=="plateau":
            sched= ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=DEFAULT_MAX_EPOCHS*0.2, min_lr=1e-6)
            return {
                "optimizer": opt,
                "lr_scheduler":{
                    "scheduler": sched,
                    "monitor": "val_loss",
                    "interval":"epoch",
                    "frequency":1
                }
            }
        else:
            return opt

##############################################################################
# main
##############################################################################
def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("--csv_file",     type=str, default=CSV_FILE)
    parser.add_argument("--image_dir",    type=str, default=IMAGE_ROOT_DIR)
    parser.add_argument("--run_id",       type=str, default=None)
    parser.add_argument("--project_name", type=str, default="mmKGE_with_Qwen2VL")

    parser.add_argument("--batch_size",   type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_epochs",   type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--lr",           type=float, default=DEFAULT_LR)
    parser.add_argument("--freeze_text",  action="store_true", default=FREEZE_TEXT)
    parser.add_argument("--freeze_image", action="store_true", default=FREEZE_IMAGE)
    parser.add_argument("--neg_samples",  type=int, default=DEFAULT_NEG_SAMPLES)
    parser.add_argument("--emb_dim",      type=int, default=DEFAULT_EMB_DIM)
    parser.add_argument("--alpha_mm",     type=float, default=DEFAULT_ALPHA_MM)
    parser.add_argument("--margin",       type=float, default=1.0, help="Margin for ranking loss")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=LR_SCHEDULER_CHOICES)
    parser.add_argument("--debug", action="store_true", default=DEFAULT_DEBUG_FLAG)
    parser.add_argument("--qwen2_vl_model_name", type=str, default=QWEN2_VL_MODEL_PATH)

    parser.add_argument("--use_image_prompt", action="store_true",
                        help="If true, insert <|image|> into text prompt so Qwen2-VL can see the image tokens.")

    args= parser.parse_args()
    if args.run_id is None:
        args.run_id= str(uuid.uuid4())[:8]

    seed_everything(42)

    # 1) load
    triple_data= load_dynamic_csv(args.csv_file, args.image_dir)
    if not triple_data:
        print("[Error] CSV empty or no triple!")
        sys.exit(1)
    random.shuffle(triple_data)

    # 2) ent2id, rel2id
    all_d= {r["disease"] for r in triple_data}
    all_o= {r["object"]  for r in triple_data}
    all_e= sorted(list(all_d.union(all_o)))
    ent2id= { e:i for i,e in enumerate(all_e) }

    all_r= sorted({ r["relation"] for r in triple_data })
    rel2id= { rr:i for i,rr in enumerate(all_r) }

    # 定义稀有类标签
    rare_classes = ["温度", "时间"]
    recs = load_dynamic_csv(args.csv_file, args.image_dir)
    # 2) 添加插桩式 Prompt 数据
    prompt_examples = create_prompt_examples()  # 调用插桩函数
    recs = augment_prompt_with_rare_classes(recs, prompt_examples)  # 数据增强


    # 3) Split the dataset: 80% for training + validation, 20% discarded (not used)
    train_val_ratio = 0.8  # 80% of the data will be used for training + validation
    val_ratio_from_train = 0.3  # 30% of training data goes to validation

    # Calculate split sizes
    train_val_size = int(train_val_ratio * len(triple_data))
    train_val_recs = triple_data[:train_val_size]
    random.shuffle(train_val_recs)
    val_size = int(val_ratio_from_train * len(train_val_recs))
    val_recs = train_val_recs[:val_size] 
    train_recs = train_val_recs[val_size:] 

     # 过采样少量类
    train_recs = oversample_data(train_recs, target_column="object", rare_classes=rare_classes)

    obj_name_map= {}
    hard_negatives= {}

    train_ds= MultiModalTripleDataset(
        train_recs, ent2id, rel2id,
        neg_samples=args.neg_samples,
        obj_name_map=obj_name_map,
        hard_negatives=hard_negatives
    )
    val_ds= MultiModalTripleDataset(
        val_recs, ent2id, rel2id,
        neg_samples=args.neg_samples,
        obj_name_map=obj_name_map,
        hard_negatives=hard_negatives
    )

    train_loader= DataLoader(train_ds, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, persistent_workers=True)
    val_loader=   DataLoader(val_ds,   batch_size=args.batch_size,
                             shuffle=False, num_workers=4, persistent_workers=True)

    valid_labels = ["感染", "症状", "流行季节", "流行地区"]
    model= KGMulModalLightningModule(
        lr=args.lr,
        ent_total=len(ent2id),
        rel_total=len(rel2id),
        emb_dim=args.emb_dim,
        freeze_text=args.freeze_text,
        freeze_image=args.freeze_image,
        neg_samples=args.neg_samples,
        alpha_mm=args.alpha_mm,
        margin=args.margin,
        lr_scheduler=args.lr_scheduler,
        debug=args.debug,
        valid_labels=valid_labels,  # 将标签传入模型
        qwen2_vl_model_name=args.qwen2_vl_model_name,
        use_image_prompt=args.use_image_prompt
    )

    run_dir= f"runs/run_{args.run_id}"
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir= os.path.join(run_dir,"ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb_logger= WandbLogger(
        project=args.project_name,
        name=f"mmKGE_Qwen2VL_{args.run_id}",
        save_dir=run_dir
    )
    ckpt_callback= ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_mrr",
        mode="max",
        filename="best",
        save_top_k=1,
        save_last=True
    )
    lr_monitor= LearningRateMonitor(logging_interval='step')

    trainer= pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=run_dir,
        logger=wandb_logger,
        callbacks=[ckpt_callback, lr_monitor],
        precision=16,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10
    )
    trainer.fit(model, train_loader, val_loader)

    print(f"[Done] best ckpt= {ckpt_dir}")


if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    main()
