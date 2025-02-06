import os
import sys
import csv
import pandas as pd
import uuid
import random
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.transforms as T
from PIL import Image
import pandas as pd
from transformers import AutoModelForSeq2SeqLM
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)

# Hyperparameters
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_EPOCHS = 10

####################################################
# Step 1: read CSV & parse partial triplets
####################################################
def load_triplets(csv_file: str, image_root_dir: str):
    """
    读取annotation.csv，返回list[{...}]结构:
    {
        "image_path": ...,
        "text": ...,
        "disease": ...,
        "relation": ...,
        "object": ...
    }
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    # 加载 CSV 文件
    df = pd.read_csv(csv_file, encoding="utf-8")

    # 检查必需列是否存在
    required_cols = ["id", "image_path", "text", "病名", "关系", "感染对象"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"[Error] Missing column {c} in CSV!")

    # 转换为 triplets 格式
    triplets = []
    for _, row in df.iterrows():
        disease_name = str(row.get("病名", ""))
        relation = str(row.get("关系", ""))
        obj = row.get("感染对象", None)

        # 跳过没有 "感染对象" 的行
        if pd.isna(obj) or str(obj).strip() == "":
            continue

        rec = {
            "image_path": os.path.join(image_root_dir, str(row["image_path"]).replace("\\", "/")),
            "text": str(row["text"]),
            "disease": disease_name,
            "relation": relation,
            "object": str(obj).strip()
        }
        triplets.append(rec)

    return triplets

# Tokenizer
TOKENIZER = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")


# Dataset class
class MultiModalDataset(Dataset):
    """
    给定三元组 (disease, relation, object) + text + image
    这里我们做 "给定(disease+relation+文本+图)，输出object" => 多分类
    因此 object要离散ID
    增加双通道⽂本⽣成函数
    """

    def __init__(self, records, object2id, transform=None):
        self.records   = records
        self.object2id = object2id
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        disease_text = f"{rec['disease']} {rec['relation']} {rec['text']}"

        img_path = rec["image_path"]
        if not os.path.exists(img_path):
            image = Image.new("RGB", (224,224), color="black")
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            # 这里先不transform，因为后面模型里会自己transform
            # 也可以选择在这里transform
            pass

        obj_text = rec["object"]
        obj_id = self.object2id[obj_text]  # 作为 label

        return {
            "disease_text": disease_text,
            "relation": rec["relation"],
            "text": rec['text'],
            "image": image,
            "obj_id": obj_id
        }

# Model with LoRA
import torch
import pytorch_lightning as pl

class LoRALightningModule(pl.LightningModule):
    def __init__(self, peft_model, lr=5e-5):
        super().__init__()
        self.model = peft_model
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["input_ids"], batch["labels"]
        attention_mask = batch["attention_mask"]

        outputs = self(input_ids=inputs, attention_mask=attention_mask)
        logits = outputs.logits

        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["input_ids"], batch["labels"]
        attention_mask = batch["attention_mask"]

        outputs = self(input_ids=inputs, attention_mask=attention_mask)
        logits = outputs.logits

        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


# 参数
CSV_FILE = "annotation.csv"
IMAGE_ROOT_DIR = "标注后的图像数据改名后的"
XLMR_MODEL_PATH = "xlm-roberta-base"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_EPOCHS = 50
DEFAULT_LR = 1e-4
FREEZE_TEXT = True
FREEZE_IMAGE = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default=CSV_FILE)
    parser.add_argument("--image_dir", type=str, default=IMAGE_ROOT_DIR)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--freeze_text", action="store_true", default=FREEZE_TEXT)
    parser.add_argument("--freeze_image", action="store_true", default=FREEZE_IMAGE)
    return parser.parse_args()

def load_triplets(csv_file: str, image_root_dir: str):
    """
    读取annotation.csv，返回list[{...}]结构:
    {
        "image_path": ...,
        "text": ...,
        "disease": ...,
        "relation": ...,
        "object": ...
    }
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    # 加载 CSV 文件
    df = pd.read_csv(csv_file, encoding="utf-8")

    # 检查必需列是否存在
    required_cols = ["id", "image_path", "text", "病名", "关系", "感染对象"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"[Error] Missing column {c} in CSV!")

    # 转换为 triplets 格式
    triplets = []
    for _, row in df.iterrows():
        disease_name = str(row.get("病名", ""))
        relation = str(row.get("关系", ""))
        obj = row.get("感染对象", None)

        # 跳过没有 "感染对象" 的行
        if pd.isna(obj) or str(obj).strip() == "":
            continue

        rec = {
            "image_path": os.path.join(image_root_dir, str(row["image_path"]).replace("\\", "/")),
            "text": str(row["text"]),
            "disease": disease_name,
            "relation": relation,
            "object": str(obj).strip()
        }
        triplets.append(rec)

    return triplets


def main():
    args = parse_args()

    if args.run_id is None:
        args.run_id = str(uuid.uuid4())[:8]
    run_dir = f"runs/run_{args.run_id}"
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("==== Configuration ====")
    print(f"CSV file:   {args.csv_file}")
    print(f"Image Dir:  {args.image_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Epochs: {args.max_epochs}")
    print(f"LR:         {args.lr}")
    print("=======================")

    # 加载数据
    recs = load_triplets(args.csv_file, args.image_dir)

    # 数据集划分：测试集10%，验证集30%
    random.shuffle(recs)
    test_ratio = 0.1
    val_ratio = 0.3
    test_size = int(test_ratio * len(recs))
    val_size = int(val_ratio * (len(recs) - test_size))

    test_recs = recs[:test_size]
    val_recs = recs[test_size:test_size + val_size]
    train_recs = recs[test_size + val_size:]

    # 构建 object2id 映射
    all_objects = list({r["object"] for r in recs})
    object2id = {obj: i for i, obj in enumerate(all_objects)}
    num_objs = len(object2id)
    print(f"[Info] total distinct objects: {num_objs}")

    # 数据集处理
    train_ds = MultiModalDataset(train_recs, object2id=object2id)
    val_ds   = MultiModalDataset(val_recs, object2id=object2id)
    test_ds  = MultiModalDataset(test_recs, object2id=object2id)

    # LoRA 微调配置
    lora_config = LoraConfig(
           r=8,  # 低秩适配维度
           lora_alpha=16,
           lora_dropout=0.1,
           target_modules=["q_proj", "k_proj", "v_proj"],  # 确保这些名称与模型中的模块匹配
           bias="none"
    )
    base_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    peft_model = get_peft_model(base_model, lora_config)
    
    model = LoRALightningModule(peft_model, lr=args.lr)

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Logger
    wandb_logger = WandbLogger(project="fish_xlm_clip_vitmae_cbam_multiclass", name=f"run_{args.run_id}", save_dir=run_dir)
    ckpt_callback = ModelCheckpoint(dirpath=ckpt_dir, save_top_k=1, monitor="val_loss", mode="min", filename="best")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=run_dir,
        logger=wandb_logger,
        callbacks=[ckpt_callback, lr_monitor],
        precision=16 if torch.cuda.is_available() else 32,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10
    )

    print(f"Trainer device: {trainer.accelerator}, #devices={trainer.num_devices}")

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")

    print("Done. Best checkpoint saved in:", ckpt_dir)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()