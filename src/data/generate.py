"""Data generation for Lumira training.

This module provides:
1. Vocabulary expansion based on Lumira phonology rules
2. Rule-based translation pair generation
3. Template-based sentence generation
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class LumiraPhonology:
    """Lumira phonological rules."""

    # Vowels
    vowels: List[str] = field(default_factory=lambda: ['a', 'i', 'u', 'e', 'o'])

    # Consonants by preference
    soft_consonants: List[str] = field(default_factory=lambda: ['l', 'r', 'm', 'n', 's', 'v'])
    normal_consonants: List[str] = field(default_factory=lambda: ['b', 'd', 'f', 'h', 'k', 'p', 't', 'y', 'z'])

    # Forbidden combinations
    forbidden: List[str] = field(default_factory=lambda: ['kk', 'tt', 'pp', 'kr', 'tr', 'pr'])

    def generate_syllable(self, prefer_soft: bool = True) -> str:
        """Generate a valid Lumira syllable."""
        consonants = self.soft_consonants if prefer_soft else self.soft_consonants + self.normal_consonants

        patterns = [
            ('CV', 0.6),   # Consonant + Vowel (most common)
            ('V', 0.2),    # Vowel only
            ('CVC', 0.2),  # Consonant + Vowel + Consonant
        ]

        pattern = random.choices([p[0] for p in patterns], weights=[p[1] for p in patterns])[0]

        syllable = ''
        for char in pattern:
            if char == 'C':
                syllable += random.choice(consonants)
            else:
                syllable += random.choice(self.vowels)

        return syllable

    def generate_word(self, syllables: int = 2, prefer_soft: bool = True) -> str:
        """Generate a valid Lumira word."""
        word = ''.join(self.generate_syllable(prefer_soft) for _ in range(syllables))

        # Check forbidden combinations
        for forbidden in self.forbidden:
            if forbidden in word:
                return self.generate_word(syllables, prefer_soft)

        # Prefer ending with vowel
        if word[-1] not in self.vowels and random.random() < 0.7:
            word += random.choice(self.vowels)

        return word

    def is_valid(self, word: str) -> bool:
        """Check if a word follows Lumira phonology."""
        for forbidden in self.forbidden:
            if forbidden in word:
                return False
        return True


@dataclass
class VocabularyEntry:
    """Single vocabulary entry."""
    lumira: str
    pronunciation: str
    pos: str  # Part of speech
    meaning: str
    etymology: str = ""
    positive_note: str = ""
    category: str = ""


class VocabularyExpander:
    """Expand Lumira vocabulary systematically."""

    # Word categories with Japanese meanings
    CATEGORIES = {
        'greetings': [
            ('おはよう', '朝の挨拶'),
            ('おやすみ', '夜の挨拶'),
            ('さようなら', '別れの挨拶'),
            ('ようこそ', '歓迎'),
            ('お元気ですか', '状態の確認'),
        ],
        'emotions': [
            ('嬉しい', '喜びの感情'),
            ('楽しい', '楽しみの感情'),
            ('穏やか', '平静の感情'),
            ('希望', '未来への期待'),
            ('感謝', '謝意'),
            ('勇気', '困難に立ち向かう力'),
            ('優しさ', '思いやり'),
            ('幸せ', '満足感'),
        ],
        'nature': [
            ('空', '大気'),
            ('海', '大きな水'),
            ('山', '高い大地'),
            ('森', '木々の集まり'),
            ('川', '流れる水'),
            ('星', '夜空の光'),
            ('風', '空気の流れ'),
            ('雨', '天からの水'),
            ('花', '植物の美'),
            ('木', '大きな植物'),
            ('雲', '空の綿'),
            ('虹', '色の架け橋'),
        ],
        'time': [
            ('今日', '現在の日'),
            ('明日', '次の日'),
            ('昨日', '前の日'),
            ('朝', '一日の始まり'),
            ('夜', '一日の終わり'),
            ('今', '現在'),
            ('永遠', '終わりなき時'),
            ('春', '再生の季節'),
            ('夏', '活力の季節'),
            ('秋', '収穫の季節'),
            ('冬', '休息の季節'),
        ],
        'verbs': [
            ('行く', '移動する'),
            ('来る', '近づく'),
            ('食べる', '栄養を取る'),
            ('飲む', '液体を取る'),
            ('寝る', '休息する'),
            ('起きる', '目覚める'),
            ('話す', '言葉を発する'),
            ('聞く', '音を受け取る'),
            ('見る', '視覚で認識する'),
            ('読む', '文字を理解する'),
            ('書く', '文字を残す'),
            ('作る', '創造する'),
            ('助ける', '支援する'),
            ('学ぶ', '知識を得る'),
            ('教える', '知識を伝える'),
            ('思う', '考える'),
            ('感じる', '感覚する'),
            ('笑う', '喜びを表す'),
            ('歩く', '足で移動する'),
            ('走る', '速く移動する'),
            ('待つ', '時を過ごす'),
            ('始める', '開始する'),
            ('終わる', '完了する'),
            ('変わる', '変化する'),
            ('続ける', '継続する'),
        ],
        'adjectives': [
            ('大きい', '規模が大'),
            ('小さい', '規模が小'),
            ('新しい', '時間的に近い'),
            ('古い', '時間的に遠い'),
            ('良い', '肯定的'),
            ('強い', '力がある'),
            ('優しい', '穏やか'),
            ('明るい', '光がある'),
            ('暖かい', '温度が心地よい'),
            ('涼しい', '温度が爽やか'),
            ('静か', '音が少ない'),
            ('深い', '奥行きがある'),
            ('高い', '位置が上'),
            ('広い', '面積が大'),
            ('長い', '距離・時間が大'),
            ('速い', 'スピードがある'),
            ('柔らかい', '触感が優しい'),
            ('清い', '純粋'),
        ],
        'nouns': [
            ('人', '人間'),
            ('友達', '親しい人'),
            ('家族', '血縁・絆の人々'),
            ('子供', '若い人'),
            ('親', '育てる人'),
            ('先生', '教える人'),
            ('夢', '眠りの中の世界'),
            ('道', '進む場所'),
            ('家', '住む場所'),
            ('国', '人々の集まり'),
            ('世界', 'すべての場所'),
            ('言葉', '意思伝達の手段'),
            ('音楽', '音の芸術'),
            ('歌', '声の音楽'),
            ('物語', '語られる話'),
            ('本', '知識の集まり'),
            ('光', '明るさ'),
            ('力', 'エネルギー'),
            ('愛', '深い感情'),
            ('心', '精神'),
            ('体', '肉体'),
            ('手', '掴む部分'),
            ('目', '見る器官'),
            ('耳', '聞く器官'),
        ],
        'pronouns': [
            ('彼', '男性三人称'),
            ('彼女', '女性三人称'),
            ('これ', '近い物'),
            ('それ', '中間の物'),
            ('あれ', '遠い物'),
            ('誰', '不明の人'),
            ('何', '不明の物'),
            ('どこ', '不明の場所'),
            ('いつ', '不明の時'),
            ('なぜ', '不明の理由'),
        ],
        'numbers': [
            ('一', '1'),
            ('二', '2'),
            ('三', '3'),
            ('四', '4'),
            ('五', '5'),
            ('六', '6'),
            ('七', '7'),
            ('八', '8'),
            ('九', '9'),
            ('十', '10'),
            ('百', '100'),
            ('千', '1000'),
            ('多い', '数量大'),
            ('少ない', '数量小'),
        ],
        'positive_transforms': [
            ('失敗', '学びの機会'),
            ('困難', '成長のチャンス'),
            ('別れ', '新しい出会いへの扉'),
            ('終わり', '新しい始まり'),
            ('涙', '心の浄化'),
            ('痛み', '癒しへの道'),
            ('迷い', '探求の旅'),
            ('弱さ', '成長の余地'),
            ('闘い', '乗り越える経験'),
            ('寂しさ', '自分と向き合う時間'),
        ],
    }

    def __init__(self, existing_vocab_path: str | Path | None = None):
        self.phonology = LumiraPhonology()
        self.existing_vocab: Dict[str, VocabularyEntry] = {}

        if existing_vocab_path:
            self._load_existing(existing_vocab_path)

    def _load_existing(self, path: str | Path):
        """Load existing vocabulary."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entry in data.get('vocabulary', []):
            self.existing_vocab[entry['meaning']] = VocabularyEntry(
                lumira=entry['lumira'],
                pronunciation=entry['pronunciation'],
                pos=entry['pos'],
                meaning=entry['meaning'],
                etymology=entry.get('etymology', ''),
                positive_note=entry.get('positive_note', ''),
            )

    def _japanese_to_pronunciation(self, lumira_word: str) -> str:
        """Convert Lumira word to katakana pronunciation."""
        # Simple romanji to katakana mapping
        mapping = {
            'a': 'ア', 'i': 'イ', 'u': 'ウ', 'e': 'エ', 'o': 'オ',
            'ka': 'カ', 'ki': 'キ', 'ku': 'ク', 'ke': 'ケ', 'ko': 'コ',
            'sa': 'サ', 'si': 'シ', 'su': 'ス', 'se': 'セ', 'so': 'ソ',
            'ta': 'タ', 'ti': 'チ', 'tu': 'ツ', 'te': 'テ', 'to': 'ト',
            'na': 'ナ', 'ni': 'ニ', 'nu': 'ヌ', 'ne': 'ネ', 'no': 'ノ',
            'ha': 'ハ', 'hi': 'ヒ', 'hu': 'フ', 'he': 'ヘ', 'ho': 'ホ',
            'ma': 'マ', 'mi': 'ミ', 'mu': 'ム', 'me': 'メ', 'mo': 'モ',
            'ya': 'ヤ', 'yu': 'ユ', 'yo': 'ヨ',
            'ra': 'ラ', 'ri': 'リ', 'ru': 'ル', 're': 'レ', 'ro': 'ロ',
            'la': 'ラ', 'li': 'リ', 'lu': 'ル', 'le': 'レ', 'lo': 'ロ',
            'wa': 'ワ', 'wo': 'ヲ', 'n': 'ン',
            'va': 'ヴァ', 'vi': 'ヴィ', 'vu': 'ヴ', 've': 'ヴェ', 'vo': 'ヴォ',
            'fa': 'ファ', 'fi': 'フィ', 'fu': 'フ', 'fe': 'フェ', 'fo': 'フォ',
            'ba': 'バ', 'bi': 'ビ', 'bu': 'ブ', 'be': 'ベ', 'bo': 'ボ',
            'da': 'ダ', 'di': 'ディ', 'du': 'ドゥ', 'de': 'デ', 'do': 'ド',
            'pa': 'パ', 'pi': 'ピ', 'pu': 'プ', 'pe': 'ペ', 'po': 'ポ',
            'za': 'ザ', 'zi': 'ジ', 'zu': 'ズ', 'ze': 'ゼ', 'zo': 'ゾ',
        }

        result = []
        word = lumira_word.lower()
        i = 0
        while i < len(word):
            # Try two-character match first
            if i + 1 < len(word) and word[i:i+2] in mapping:
                result.append(mapping[word[i:i+2]])
                i += 2
            elif word[i] in mapping:
                result.append(mapping[word[i]])
                i += 1
            else:
                i += 1

        return ''.join(result)

    def expand_vocabulary(self, target_count: int = 500) -> List[VocabularyEntry]:
        """Expand vocabulary to target count."""
        new_vocab = list(self.existing_vocab.values())
        used_words = {v.lumira for v in new_vocab}

        for category, items in self.CATEGORIES.items():
            for japanese, note in items:
                if japanese in self.existing_vocab:
                    continue

                # Generate a unique Lumira word
                attempts = 0
                while attempts < 100:
                    syllable_count = random.randint(2, 3)
                    word = self.phonology.generate_word(syllable_count)

                    if word not in used_words and self.phonology.is_valid(word):
                        used_words.add(word)
                        break
                    attempts += 1

                pos_map = {
                    'greetings': 'interjection',
                    'emotions': 'noun',
                    'nature': 'noun',
                    'time': 'noun',
                    'verbs': 'verb',
                    'adjectives': 'adjective',
                    'nouns': 'noun',
                    'pronouns': 'pronoun',
                    'numbers': 'numeral',
                    'positive_transforms': 'noun',
                }

                entry = VocabularyEntry(
                    lumira=word,
                    pronunciation=self._japanese_to_pronunciation(word),
                    pos=pos_map.get(category, 'noun'),
                    meaning=japanese,
                    etymology=f'Lumira造語',
                    positive_note=note,
                    category=category,
                )
                new_vocab.append(entry)

                if len(new_vocab) >= target_count:
                    break

            if len(new_vocab) >= target_count:
                break

        return new_vocab

    def save_vocabulary(self, vocab: List[VocabularyEntry], output_path: str | Path):
        """Save expanded vocabulary to JSON."""
        data = {
            'meta': {
                'language': 'Lumira',
                'version': '0.2.0',
                'description': 'Expanded Lumira vocabulary',
                'word_count': len(vocab),
            },
            'vocabulary': [
                {
                    'lumira': v.lumira,
                    'pronunciation': v.pronunciation,
                    'pos': v.pos,
                    'meaning': v.meaning,
                    'etymology': v.etymology,
                    'positive_note': v.positive_note,
                    'category': v.category,
                }
                for v in vocab
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class SentenceGenerator:
    """Generate translation pairs using templates."""

    # Sentence templates: (Japanese pattern, Lumira pattern)
    TEMPLATES = [
        # Greetings
        ('{greeting}', '{greeting_l}'),
        ('{greeting}、{name}さん', '{greeting_l}, {name}'),

        # Simple sentences
        ('私は{emotion}です', 'Mi senti {emotion_l}'),
        ('あなたは{adj}です', 'Tu {adj_l}'),
        ('{noun}は{adj}です', '{noun_l} {adj_l}'),

        # Actions
        ('私は{verb}', 'Mi {verb_l}'),
        ('あなたは{verb}', 'Tu {verb_l}'),
        ('私たちは{verb}', 'Noi {verb_l}'),
        ('{noun}を{verb}', 'Mi {verb_l} {noun_l}'),

        # Time expressions
        ('{time}、{action}', '{time_l}, {action_l}'),
        ('{time}は{adj}です', '{time_l} {adj_l}'),

        # Nature
        ('{nature}が{adj}です', '{nature_l} {adj_l}'),
        ('{nature}を見る', 'Mi mira {nature_l}'),
        ('{nature}が好きです', 'Mi ama {nature_l}'),

        # Positive transformations
        ('{negative}は{positive}です', '{negative_l} {positive_l}'),
        ('{negative}があっても大丈夫', '{negative_l} flori'),

        # Complex sentences
        ('私は{noun}を{verb}ます', 'Mi {verb_l} {noun_l}'),
        ('{adj}{noun}が{verb}', '{adj_l} {noun_l} {verb_l}'),
        ('あなたと一緒に{verb}たい', 'Mi {verb_l} sama tu'),
    ]

    def __init__(self, vocabulary: List[VocabularyEntry]):
        self.vocab = vocabulary
        self._build_indices()

    def _build_indices(self):
        """Build category indices for quick lookup."""
        self.by_category: Dict[str, List[VocabularyEntry]] = {}
        self.by_meaning: Dict[str, VocabularyEntry] = {}

        for v in self.vocab:
            cat = v.category or v.pos
            if cat not in self.by_category:
                self.by_category[cat] = []
            self.by_category[cat].append(v)
            self.by_meaning[v.meaning] = v

    def _get_word(self, category: str) -> Tuple[str, str]:
        """Get random word from category."""
        words = self.by_category.get(category, [])
        if not words:
            # Fallback
            words = self.vocab
        word = random.choice(words)
        return word.meaning, word.lumira

    def generate_pair(self) -> Tuple[str, str]:
        """Generate a single translation pair."""
        template_ja, template_lu = random.choice(self.TEMPLATES)

        # Fill in placeholders
        replacements_ja = {}
        replacements_lu = {}

        placeholders = [
            ('greeting', 'greetings'),
            ('emotion', 'emotions'),
            ('adj', 'adjectives'),
            ('noun', 'nouns'),
            ('verb', 'verbs'),
            ('time', 'time'),
            ('nature', 'nature'),
            ('negative', 'positive_transforms'),
            ('positive', 'emotions'),
            ('action', 'verbs'),
        ]

        for placeholder, category in placeholders:
            if '{' + placeholder + '}' in template_ja:
                ja, lu = self._get_word(category)
                replacements_ja[placeholder] = ja
                replacements_lu[placeholder + '_l'] = lu

        # Handle special cases
        if '{name}' in template_ja:
            names = ['太郎', '花子', '健', '美咲', 'みんな']
            name = random.choice(names)
            replacements_ja['name'] = name
            replacements_lu['name'] = name

        ja_sentence = template_ja.format(**replacements_ja)
        lu_sentence = template_lu.format(**replacements_lu)

        return ja_sentence, lu_sentence

    def generate_dataset(self, count: int = 100000) -> List[Dict[str, str]]:
        """Generate dataset of translation pairs."""
        pairs = []
        seen = set()

        while len(pairs) < count:
            ja, lu = self.generate_pair()
            key = (ja, lu)

            if key not in seen:
                seen.add(key)
                pairs.append({
                    'ja': ja,
                    'lumira': lu,
                })

            # Progress
            if len(pairs) % 10000 == 0:
                print(f"Generated {len(pairs):,} pairs...")

        return pairs

    def save_dataset(
        self,
        pairs: List[Dict[str, str]],
        output_path: str | Path,
        train_ratio: float = 0.9,
    ):
        """Save dataset to JSONL files (train/val split)."""
        random.shuffle(pairs)
        split_idx = int(len(pairs) * train_ratio)

        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save train
        with open(output_path / 'train.jsonl', 'w', encoding='utf-8') as f:
            for pair in train_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')

        # Save val
        with open(output_path / 'val.jsonl', 'w', encoding='utf-8') as f:
            for pair in val_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')

        print(f"Saved {len(train_pairs):,} train pairs")
        print(f"Saved {len(val_pairs):,} val pairs")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-input", default="docs/vocabulary.json")
    parser.add_argument("--vocab-output", default="data/vocab/vocabulary_expanded.json")
    parser.add_argument("--data-output", default="data/processed")
    parser.add_argument("--vocab-size", type=int, default=500)
    parser.add_argument("--data-size", type=int, default=100000)
    args = parser.parse_args()

    # Expand vocabulary
    print("Expanding vocabulary...")
    expander = VocabularyExpander(args.vocab_input)
    vocab = expander.expand_vocabulary(args.vocab_size)
    Path(args.vocab_output).parent.mkdir(parents=True, exist_ok=True)
    expander.save_vocabulary(vocab, args.vocab_output)
    print(f"Expanded vocabulary to {len(vocab)} words")

    # Generate dataset
    print("\nGenerating translation pairs...")
    generator = SentenceGenerator(vocab)
    pairs = generator.generate_dataset(args.data_size)
    generator.save_dataset(pairs, args.data_output)
    print("Done!")
