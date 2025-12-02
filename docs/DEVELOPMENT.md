# Lumira Tiny LLM 開発記録

## 概要

Lumira言語を話せる小型LLM（Tiny Transformer）を開発するプロジェクト。
日本語 ↔ Lumira の翻訳を行うEncoder-Decoder Transformerモデルを
Google Colab（T4 GPU）で訓練可能な形で実装した。

## 開発期間

2025年12月

## 目標

- **タスク**: 日本語 → Lumira 翻訳（将来的に会話型へ拡張）
- **モデルサイズ**: 50-100Mパラメータ（Tiny/Small）
- **訓練環境**: Google Colab無料版（T4 GPU）

## アーキテクチャ

### モデル構成（Small Config: 約60Mパラメータ）

| パラメータ | 値 |
|-----------|-----|
| Encoder層数 | 6 |
| Decoder層数 | 6 |
| Hidden size | 512 |
| Attention heads | 8 |
| FFN size | 2048 |
| Vocab size | 8000 (BPE) |
| Max sequence length | 128 |

### 主要コンポーネント

1. **Multi-Head Attention** (`src/model/attention.py`)
   - Scaled dot-product attention
   - Sinusoidal positional encoding

2. **Encoder** (`src/model/encoder.py`)
   - Pre-norm architecture
   - GELU activation
   - Self-attention + FFN

3. **Decoder** (`src/model/decoder.py`)
   - Masked self-attention
   - Cross-attention
   - FFN

4. **LumiraTransformer** (`src/model/transformer.py`)
   - 完全なEncoder-Decoderモデル
   - Autoregressive生成（Top-k, Top-p sampling対応）

## データパイプライン

### 語彙拡充

既存の22語から500語以上に拡充するシステムを実装。

- **音韻ルール**: Lumiraの音韻体系（柔らかい子音優先、母音で終わる等）に従って新語を自動生成
- **カテゴリ**: 挨拶、感情、自然、時間、動詞、形容詞、名詞、代名詞、数詞、ポジティブ変換

### 対訳データ生成

ハイブリッド方式を採用：

1. **ルールベース生成**: テンプレートから基本パターンを大量生成
2. **LLM多様化**: （オプション）LLM APIで自然な表現に拡張

テンプレート例：
- `私は{emotion}です` → `Mi senti {emotion_l}`
- `{noun}を{verb}` → `Mi {verb_l} {noun_l}`

## ファイル構成

```
Lumira/
├── docs/                    # ドキュメント
│   ├── README.md           # プロジェクト概要
│   ├── CONCEPT.md          # 言語コンセプト
│   ├── GRAMMAR.md          # 文法ルール
│   ├── PHONOLOGY.md        # 音韻体系
│   ├── examples.md         # 例文集
│   ├── vocabulary.json     # オリジナル語彙（22語）
│   └── DEVELOPMENT.md      # 本ファイル
├── src/
│   ├── model/              # Transformerモデル
│   │   ├── transformer.py  # メインモデル
│   │   ├── encoder.py      # Encoder
│   │   ├── decoder.py      # Decoder
│   │   ├── attention.py    # Attention & Positional Encoding
│   │   └── config.py       # モデル設定
│   ├── data/               # データ処理
│   │   ├── tokenizer.py    # SentencePiece BPEトークナイザー
│   │   ├── dataset.py      # PyTorch Dataset
│   │   └── generate.py     # 語彙拡充 & データ生成
│   ├── training/           # 訓練
│   │   ├── trainer.py      # 訓練ループ
│   │   └── config.py       # 訓練設定
│   └── inference/          # 推論
│       └── translate.py    # 翻訳 & Gradio UI
├── scripts/
│   ├── generate_data.py    # データ生成スクリプト
│   ├── train_tokenizer.py  # トークナイザー訓練
│   └── train.py            # モデル訓練
├── notebooks/
│   └── train.ipynb         # Google Colab用ノートブック
├── data/
│   ├── raw/                # 生データ
│   ├── processed/          # 訓練データ（train.jsonl, val.jsonl）
│   └── vocab/              # トークナイザー & 拡張語彙
└── requirements.txt        # 依存関係
```

## 訓練設定

### Colab T4 GPU向け最適化

| 設定 | 値 |
|------|-----|
| Batch size | 24 |
| Gradient accumulation | 4 |
| Effective batch size | 96 |
| Learning rate | 1e-4 |
| Warmup steps | 1000 |
| Scheduler | Cosine |
| Mixed precision | FP16 (AMP) |
| Label smoothing | 0.1 |

### チェックポイント

- Google Driveに自動保存
- セッション切断後も訓練再開可能
- ベストモデル自動保存

## 使い方

### 1. データ生成

```bash
python scripts/generate_data.py \
    --vocab-size 500 \
    --data-size 100000
```

### 2. トークナイザー訓練

```bash
python scripts/train_tokenizer.py \
    --vocab-size 8000
```

### 3. モデル訓練

```bash
python scripts/train.py \
    --model-config small \
    --epochs 20 \
    --batch-size 32
```

### 4. 推論

```python
from src.inference import Translator

translator = Translator(
    model_path="checkpoints/best.pt",
    tokenizer_path="data/vocab/lumira.model"
)

result = translator.translate("こんにちは")
print(result)  # → "Sola!"
```

## 技術スタック

- **言語**: Python 3.10+
- **フレームワーク**: PyTorch 2.0+
- **トークナイザー**: SentencePiece
- **UI**: Gradio
- **訓練環境**: Google Colab (T4 GPU)

## 今後の拡張予定

1. **会話型への拡張**: Decoder-onlyモデルへの移行
2. **データ品質向上**: LLM APIによる多様化
3. **評価指標**: BLEU, BERTScoreの導入
4. **マルチタスク**: 翻訳 + 会話の統合モデル

## 参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [SentencePiece](https://github.com/google/sentencepiece)
