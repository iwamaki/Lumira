"""Training loop for Lumira Transformer."""

import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from ..model import LumiraTransformer, ModelConfig, TINY_CONFIG, SMALL_CONFIG, BASE_CONFIG
from ..data import LumiraTokenizer, TranslationDataset
from ..data.dataset import collate_fn
from .config import TrainingConfig


class Trainer:
    """Trainer for Lumira Transformer model."""

    MODEL_CONFIGS = {
        'tiny': TINY_CONFIG,
        'small': SMALL_CONFIG,
        'base': BASE_CONFIG,
    }

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # Set seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # Load tokenizer
        self.tokenizer = LumiraTokenizer(config.tokenizer_model)

        # Initialize model
        model_config = self.MODEL_CONFIGS[config.model_config]
        model_config.vocab_size = self.tokenizer.vocab_size
        model_config.dropout = config.dropout
        self.model = LumiraTransformer(model_config).to(self.device)

        print(f"Model parameters: {self.model.count_parameters():,}")

        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=config.label_smoothing,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Load checkpoint if resuming
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        total_steps = self.config.epochs * 10000  # Approximate
        warmup_steps = self.config.warmup_steps

        if self.config.scheduler == 'linear':
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
        elif self.config.scheduler == 'cosine':
            import math
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        else:  # constant
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """Main training loop."""
        # Load data
        train_dataset = TranslationDataset(
            self.config.train_data,
            self.tokenizer,
            self.config.max_seq_len,
        )
        val_dataset = TranslationDataset(
            self.config.val_data,
            self.tokenizer,
            self.config.max_seq_len,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, self.tokenizer.pad_id),
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, self.tokenizer.pad_id),
            num_workers=2,
            pin_memory=True,
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Training on: {self.device}")

        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)

            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best.pt')
                print(f"  New best model saved!")

            # Save epoch checkpoint
            self.save_checkpoint(f'epoch_{epoch + 1}.pt')

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}")
        for i, batch in enumerate(pbar):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)

            # Teacher forcing: input is tgt[:-1], target is tgt[1:]
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            # Forward pass
            with autocast(enabled=self.config.use_amp):
                logits = self.model(src, tgt_input)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_target.reshape(-1),
                )
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                        'lr': f'{lr:.2e}',
                    })

                # Evaluation
                if self.global_step % self.config.eval_every == 0:
                    pass  # Could add mid-epoch evaluation

                # Save checkpoint
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(f'step_{self.global_step}.pt')

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Validating"):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)

            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            with autocast(enabled=self.config.use_amp):
                logits = self.model(src, tgt_input)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_target.reshape(-1),
                )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = Path(self.config.output_dir) / filename
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
        }
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Checkpoint loaded: {path}")
        print(f"Resuming from epoch {self.epoch}, step {self.global_step}")
