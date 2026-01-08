import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import class Dataset vá»«a táº¡o
from data_utils_vietnamese import VietnameseABSADataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams, model, tokenizer):
        super(T5FineTuner, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        # Forward pass qua model
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch["target_ids"]
        )
        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        
        # Setup Scheduler (Warmup)
        try:
            # Cho phien ban Pytorch Lightning moi
            total_steps = self.trainer.estimated_stepping_batches
        except:
            # Fallback neu phien ban cu hoac khong xac dinh duoc
            total_steps = 10000 

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

def init_args():
    parser = argparse.ArgumentParser()
    
    # FILE PATHS
    parser.add_argument("--train_file", default="train.txt", type=str, help="Path to training file")
    parser.add_argument("--val_file", default="validation.txt", type=str, help="Path to validation file")
    parser.add_argument("--model_name_or_path", default="VietAI/vit5-base", type=str, help="Pretrained model name")
    parser.add_argument("--output_dir", default="./outputs", type=str, help="Output directory")
    
    # HYPERPARAMETERS
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training") # Bat dau voi 8 de tranh OOM
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=100, type=int)
    
    return parser.parse_args()

def main():
    args = init_args()
    set_seed(args.seed)

    # 1. Táº¡o thÆ° má»¥c output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 2. Load Model & Tokenizer (ViT5)
    print(f"--- Loading Model: {args.model_name_or_path} ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    # 3. Load Datasets
    print("--- Loading Datasets ---")
    train_dataset = VietnameseABSADataset(
        tokenizer=tokenizer,
        data_path=args.train_file,
        max_len=args.max_seq_length
    )
    
    val_dataset = VietnameseABSADataset(
        tokenizer=tokenizer,
        data_path=args.val_file,
        max_len=args.max_seq_length
    )

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, num_workers=2)

    # 4. Setup Training
    t5_finetuner = T5FineTuner(args, model, tokenizer)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best_vit5",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=True,
        mode="min"
    )

    logger = CSVLogger(args.output_dir, name="training_logs")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=args.num_train_epochs,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        # precision=16, # Uncomment dong nay neu muon train nhanh hon (Mixed Precision)
    )

    # 5. Start Training
    print("--- Starting Training ---")
    trainer.fit(t5_finetuner, train_loader, val_loader)

    # 6. Save Final Model (HuggingFace format)
    print("--- Saving Final Model ---")
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best checkpoint from: {best_model_path}")
    
    # Load láº¡i weights tá»‘t nháº¥t
    trained_model = T5FineTuner.load_from_checkpoint(best_model_path, model=model, tokenizer=tokenizer)
    
    final_save_path = os.path.join(args.output_dir, "vit5_final_model")
    trained_model.model.save_pretrained(final_save_path)
    trained_model.tokenizer.save_pretrained(final_save_path)
    
    print(f"Model Ä‘Ã£ Ä‘Æ°á»£c lÆ°u táº¡i: {final_save_path}")

if __name__ == "__main__":
    main()