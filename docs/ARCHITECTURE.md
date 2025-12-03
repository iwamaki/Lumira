# Lumira プロジェクト技術解説

このドキュメントは、Lumiraプロジェクトの技術的な仕組みを解説します。

## 目次
1. [プロジェクト概要](#1-プロジェクト概要)
2. [プロジェクト構成](#2-プロジェクト構成)
3. [モデルアーキテクチャ](#3-モデルアーキテクチャ)
4. [訓練パイプライン](#4-訓練パイプライン)
5. [データ処理](#5-データ処理)
6. [モデル設定オプション](#6-モデル設定オプション)
7. [使い方](#7-使い方)

---

## 1. プロジェクト概要

### 何をするプロジェクトか

Lumiraは、**日本語とルミラ語（人工言語）を相互翻訳する小規模LLM**を訓練するプロジェクトです。

```
日本語: こんにちは → Lumira: Salu
日本語: 私はあなたを愛しています → Lumira: Mi ama tu
```

### ルミラ語のコンセプト

ルミラ語は「いい意味しかない言語」という哲学を持つ人工言語です：

| ネガティブ概念 | ルミラでの捉え方 |
|---------------|-----------------|
| 失敗 | 学びの機会 |
| 悲しみ | 深い感受性の証 |
| 怒り | 変化のエネルギー |
| 別れ | 新章の始まり |

### 技術スタック

- **PyTorch**: ニューラルネットワークフレームワーク
- **Transformer**: Encoder-Decoderアーキテクチャ
- **SentencePiece**: サブワードトークナイザー
- **Google Colab T4 GPU**: 訓練環境

---

## 2. プロジェクト構成

```
Lumira/
├── src/                           # メインソースコード
│   ├── model/                    # Transformerモデル
│   │   ├── config.py             # モデル設定（3サイズ）
│   │   ├── transformer.py        # メインモデル
│   │   ├── encoder.py            # エンコーダー
│   │   ├── decoder.py            # デコーダー
│   │   └── attention.py          # アテンション機構
│   │
│   ├── data/                     # データ処理
│   │   ├── tokenizer.py          # トークナイザー
│   │   ├── dataset.py            # データセットクラス
│   │   └── generate.py           # データ生成
│   │
│   ├── training/                 # 訓練
│   │   ├── trainer.py            # 訓練ループ
│   │   └── config.py             # 訓練設定
│   │
│   └── inference/                # 推論
│       └── translate.py          # 翻訳 + Gradio UI
│
├── scripts/                       # 実行スクリプト
│   ├── generate_data.py          # データ生成
│   ├── train_tokenizer.py        # トークナイザー訓練
│   └── train.py                  # モデル訓練
│
├── notebooks/
│   └── train.ipynb               # Colab用ノートブック
│
└── data/
    ├── processed/                # 生成された訓練データ
    │   ├── train.jsonl           # 訓練データ
    │   └── val.jsonl             # 検証データ
    └── vocab/                    # 語彙ファイル
        └── vocabulary_expanded.json
```

---

## 3. モデルアーキテクチャ

### Encoder-Decoder Transformer

このプロジェクトは、**翻訳タスク用のEncoder-Decoder Transformer**を使用しています。

```
[入力文] → [Encoder] → [潜在表現] → [Decoder] → [出力文]

例：
"こんにちは" → Encoder → 中間表現 → Decoder → "Salu"
```

### アーキテクチャ詳細

```
┌─────────────────────────────────────────────────────────────┐
│                    LumiraTransformer                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │   Source Embed   │      │   Target Embed   │            │
│  │   (語彙→ベクトル)  │      │   (語彙→ベクトル)  │            │
│  └────────┬─────────┘      └────────┬─────────┘            │
│           │                         │                       │
│           ▼                         ▼                       │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │ Positional Enc.  │      │ Positional Enc.  │            │
│  │ (位置情報を付与)   │      │ (位置情報を付与)   │            │
│  └────────┬─────────┘      └────────┬─────────┘            │
│           │                         │                       │
│           ▼                         │                       │
│  ┌──────────────────┐               │                       │
│  │    Encoder       │               │                       │
│  │  (6層 x Self-Attn│───────────────┼──────────┐           │
│  │   + FFN)         │               │          │           │
│  └──────────────────┘               ▼          │           │
│                            ┌──────────────────┐│           │
│                            │    Decoder       ││           │
│                            │ (6層 x Self-Attn ││           │
│                            │  + Cross-Attn   ◄┘           │
│                            │  + FFN)         │            │
│                            └────────┬─────────┘            │
│                                     │                       │
│                                     ▼                       │
│                            ┌──────────────────┐            │
│                            │  Linear + Softmax│            │
│                            │  (ベクトル→語彙)   │            │
│                            └──────────────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 主要コンポーネント

| コンポーネント | 役割 | ファイル |
|---------------|------|----------|
| **Multi-Head Attention** | 入力の各部分間の関係を学習 | `src/model/attention.py` |
| **Encoder** | 入力文を意味表現にエンコード | `src/model/encoder.py` |
| **Decoder** | 意味表現から出力文を生成 | `src/model/decoder.py` |
| **Positional Encoding** | 単語の位置情報を付与 | `src/model/attention.py` |

### 技術的特徴

1. **Pre-norm アーキテクチャ**: LayerNormを各サブレイヤーの前に適用（訓練安定性向上）
2. **GELU活性化関数**: ReLUより滑らかな勾配
3. **Teacher Forcing**: 訓練時は正解トークンを入力として使用
4. **Label Smoothing (0.1)**: 過学習防止

---

## 4. 訓練パイプライン

### 全体フロー

訓練は以下の4ステップで行われます：

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: データ生成                                          │
│ ├─ docs/vocabulary.jsonから基本語彙を読み込み                 │
│ ├─ 音韻規則に従って語彙を500語に拡張                          │
│ ├─ テンプレートを使って10万の翻訳ペアを生成                    │
│ └─ train.jsonl (90%) / val.jsonl (10%) に分割                │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: トークナイザー訓練                                   │
│ ├─ 訓練データから全テキストを抽出                             │
│ ├─ SentencePiece BPEトークナイザーを訓練                     │
│ ├─ 語彙サイズ: 8,000トークン                                 │
│ └─ 特殊トークン: <pad>, <bos>, <eos>, <unk>                 │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: モデル訓練                                          │
│ ├─ SMALLモデル（~60Mパラメータ）を初期化                     │
│ ├─ AdamW + Cosine Annealing スケジューラー                  │
│ ├─ 20エポック、バッチサイズ24-32                             │
│ ├─ Mixed Precision (AMP) で高速化                           │
│ └─ ベストモデルをcheckpoints/に保存                          │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 推論                                                │
│ ├─ チェックポイントを読み込み                                 │
│ ├─ 自己回帰生成（1トークンずつ生成）                          │
│ ├─ Top-k (50) + Nucleus (0.9) サンプリング                  │
│ └─ Gradio UIでデモ可能                                      │
└─────────────────────────────────────────────────────────────┘
```

### 訓練設定（Colab T4 GPU用に最適化）

```python
# 学習パラメータ
epochs: 20                    # エポック数
batch_size: 24               # バッチサイズ（T4向け）
gradient_accumulation_steps: 4  # 勾配蓄積
learning_rate: 1e-4          # 学習率
max_seq_len: 128             # 最大シーケンス長

# 正則化
dropout: 0.1                 # ドロップアウト率
label_smoothing: 0.1         # ラベルスムージング
weight_decay: 0.01           # 重み減衰

# 最適化
warmup_steps: 1000           # ウォームアップステップ
scheduler: "cosine"          # コサインアニーリング
max_grad_norm: 1.0           # 勾配クリッピング
use_amp: True                # 混合精度訓練

# チェックポイント
save_every: 500              # 500ステップごとに保存
eval_every: 250              # 250ステップごとに評価
```

### なぜこれらの設定？

| 設定 | 理由 |
|------|------|
| **Mixed Precision (AMP)** | VRAM使用量を50%削減、訓練高速化 |
| **Gradient Accumulation** | 小バッチでも大きな実効バッチサイズを実現 |
| **Cosine Annealing** | 滑らかに学習率を下げ、収束を改善 |
| **Label Smoothing** | モデルの過信を防ぎ、汎化性能向上 |
| **Warmup** | 訓練初期の不安定さを回避 |

---

## 5. データ処理

### 語彙拡張システム

ルミラ語の語彙は、音韻規則に従って自動生成されます：

```python
# 音韻規則 (src/data/generate.py)
class LumiraPhonology:
    vowels = ['a', 'i', 'u', 'e', 'o']           # 母音
    soft_consonants = ['l', 'r', 'm', 'n', 's', 'v']  # 好まれる子音
    normal_consonants = ['b', 'd', 'f', 'h', 'k', 'p', 't', 'y', 'z']
    forbidden_patterns = ['kk', 'tt', 'pp', 'kr', 'tr', 'pr']  # 禁止パターン

    # 音節構造: 60% CV, 20% V, 20% CVC
    # 語尾は母音で終わることを優先
```

### 文テンプレート

107種類のテンプレートから翻訳ペアを生成：

```python
# テンプレート例
('{greeting}', '{greeting_l}')              # "こんにちは" → "Salu"
('私は{emotion}です', 'Mi senti {emotion_l}')  # "私は幸せです" → "Mi senti felira"
('{noun}は{adj}です', '{noun_l} {adj_l}')      # "空は青いです" → "Siel azura"
```

### トークナイゼーション

**SentencePiece BPE**を使用してサブワード分割：

```
入力: "こんにちは"
出力: [<bos>, token_123, token_456, <eos>]

入力: "Mi ama tu"
出力: [<bos>, token_789, token_101, token_202, <eos>]
```

| 特殊トークン | ID | 役割 |
|-------------|----|----|
| `<pad>` | 0 | パディング（長さ揃え） |
| `<bos>` | 1 | 文の開始 |
| `<eos>` | 2 | 文の終了 |
| `<unk>` | 3 | 未知語 |

---

## 6. モデル設定オプション

3つのプリセットサイズが用意されています：

### TINY_CONFIG（~3.4Mパラメータ）
```python
d_model=256, n_heads=4, n_encoder_layers=4, n_decoder_layers=4, d_ff=1024
```
- 用途: エッジデバイス、モバイル
- 訓練: 非常に高速

### SMALL_CONFIG（~60Mパラメータ）**← デフォルト**
```python
d_model=512, n_heads=8, n_encoder_layers=6, n_decoder_layers=6, d_ff=2048
```
- 用途: **Google Colab T4 GPU（12GB VRAM）**
- 訓練: 約2-3時間（20エポック）

### BASE_CONFIG（~148Mパラメータ）
```python
d_model=768, n_heads=12, n_encoder_layers=6, n_decoder_layers=6, d_ff=3072
```
- 用途: 大きなGPU（A100等）
- 訓練: より多くのVRAMが必要

---

## 7. 使い方

### Google Colabでの訓練

**Step 1: セットアップ**
```bash
!git clone https://github.com/iwamaki/Lumira.git
%cd /content/Lumira
!pip install -q torch sentencepiece tokenizers tqdm pandas gradio
```

**Step 2: データ生成**
```bash
python scripts/generate_data.py --vocab-size 500 --data-size 100000
```

**Step 3: トークナイザー訓練**
```bash
python scripts/train_tokenizer.py --vocab-size 8000
```

**Step 4: モデル訓練**
```bash
python scripts/train.py --model-config small --epochs 20 --batch-size 24
```

**Step 5: 推論**
```python
from src.inference import Translator

translator = Translator(
    model_path="checkpoints/best.pt",
    tokenizer_path="data/vocab/lumira.model",
)
result = translator.translate("こんにちは")
print(result)  # → "Salu"
```

### 主要ファイルクイックリファレンス

| やりたいこと | ファイル |
|------------|----------|
| モデル構造を変更 | `src/model/config.py` |
| 訓練設定を変更 | `src/training/config.py` |
| 語彙を追加 | `docs/vocabulary.json` → `scripts/generate_data.py` |
| 新しい文テンプレート追加 | `src/data/generate.py` |
| 推論パラメータ調整 | `src/inference/translate.py` |

---

## まとめ

このプロジェクトは以下を行っています：

1. **人工言語ルミラ語の語彙・文を自動生成**
2. **日本語↔ルミラ語の翻訳ペアを作成**
3. **Encoder-Decoder Transformerで翻訳モデルを訓練**
4. **訓練済みモデルで翻訳推論を実行**

コードは約2,380行で、Google Colab T4 GPUで効率的に訓練できるよう最適化されています。
