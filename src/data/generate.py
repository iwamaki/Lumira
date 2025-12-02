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

    # Additional vocabulary for expansion
    EXTRA_WORDS = {
        'verbs': [
            ('踊る', '体を動かして表現する'), ('歌う', '声で音楽を奏でる'), ('描く', '絵を作る'),
            ('泳ぐ', '水中を移動する'), ('飛ぶ', '空中を移動する'), ('登る', '上に移動する'),
            ('降りる', '下に移動する'), ('探す', '見つけようとする'), ('見つける', '発見する'),
            ('選ぶ', '決定する'), ('決める', '確定する'), ('信じる', '確信する'),
            ('願う', '望む'), ('祈る', '心を向ける'), ('育てる', '成長させる'),
            ('守る', '保護する'), ('包む', '覆う'), ('開く', '閉じたものを開放する'),
            ('閉じる', '開いたものを閉める'), ('触れる', '接触する'), ('抱く', '抱擁する'),
            ('握る', '手で持つ'), ('放す', '手を離す'), ('投げる', '物を飛ばす'),
            ('受け取る', '受領する'), ('送る', '届ける'), ('届く', '到着する'),
            ('集まる', '一箇所に来る'), ('散らばる', '各所に行く'), ('混ぜる', '合わせる'),
            ('分ける', '分離する'), ('結ぶ', 'つなげる'), ('解く', 'ほどく'),
            ('回る', '回転する'), ('止まる', '停止する'), ('動く', '移動する'),
            ('揺れる', '振動する'), ('光る', '発光する'), ('響く', '音が広がる'),
            ('香る', '匂いを発する'), ('輝く', '強く光る'), ('満ちる', 'いっぱいになる'),
        ],
        'adjectives': [
            ('美しい', '視覚的に素晴らしい'), ('素敵', '素晴らしい'), ('素直', '正直'),
            ('真っ直ぐ', '曲がりがない'), ('丸い', '円形'), ('四角い', '正方形'),
            ('細い', '幅が狭い'), ('太い', '幅が広い'), ('軽い', '重量が少ない'),
            ('重い', '重量が多い'), ('固い', '硬度がある'), ('甘い', '糖分がある'),
            ('辛い', '刺激がある'), ('苦い', '渋みがある'), ('酸っぱい', '酸味がある'),
            ('眩しい', '光が強い'), ('暗い', '光が少ない'), ('冷たい', '温度が低い'),
            ('熱い', '温度が高い'), ('湿った', '水分がある'), ('乾いた', '水分がない'),
            ('濃い', '密度が高い'), ('薄い', '密度が低い'), ('鮮やか', '色が明確'),
            ('透明', '向こうが見える'), ('純粋', '混じりがない'), ('完璧', '欠点がない'),
            ('豊か', '満ちている'), ('穏やか', '静かで落ち着いた'), ('賢い', '知性がある'),
        ],
        'nouns': [
            ('石', '固い物質'), ('砂', '細かい粒'), ('波', '水の動き'),
            ('島', '水に囲まれた土地'), ('谷', '低い土地'), ('丘', '小さな高い土地'),
            ('湖', '内陸の水'), ('泉', '水の湧き出る場所'), ('滝', '落ちる水'),
            ('霧', '低い雲'), ('霜', '凍った露'), ('雪', '白い氷の結晶'),
            ('鳥', '空を飛ぶ生物'), ('魚', '水中の生物'), ('蝶', '美しい昆虫'),
            ('種', '植物の始まり'), ('根', '植物の土中部分'), ('葉', '植物の緑'),
            ('枝', '木の腕'), ('実', '植物の果実'), ('庭', '植物を育てる場所'),
            ('橋', '渡る構造'), ('塔', '高い建物'), ('門', '入口'),
            ('窓', '光を入れる穴'), ('床', '歩く面'), ('天井', '上の面'),
            ('壁', '仕切り'), ('階段', '上下移動'), ('部屋', '空間'),
            ('街', '人の集まり'), ('村', '小さな集落'), ('港', '船の場所'),
            ('絵', '視覚芸術'), ('詩', '言葉の芸術'), ('踊り', '動きの芸術'),
            ('祭り', '祝いの行事'), ('贈り物', '与えるもの'), ('約束', '誓い'),
            ('記憶', '覚えていること'), ('未来', 'これから'), ('過去', '以前'),
        ],
        'emotions': [
            ('誇り', '自信'), ('安心', '落ち着き'), ('興奮', '高揚'),
            ('満足', '充足'), ('期待', '待ち望み'), ('驚き', '予想外'),
            ('感動', '心が動く'), ('尊敬', '敬意'), ('共感', '同じ気持ち'),
            ('情熱', '強い思い'), ('好奇心', '知りたい気持ち'), ('冒険心', '挑戦したい気持ち'),
        ],
        'nature': [
            ('太陽', '昼の光'), ('月', '夜の光'), ('大地', '地面'),
            ('草', '小さな植物'), ('岩', '大きな石'), ('珊瑚', '海の宝石'),
            ('オーロラ', '極地の光'), ('彗星', '空の旅人'), ('銀河', '星の集まり'),
        ],
        'time': [
            ('瞬間', '一瞬'), ('刻', '時の単位'), ('世紀', '長い時間'),
            ('夜明け', '朝の始まり'), ('夕暮れ', '夜の始まり'), ('真昼', '日の頂点'),
            ('真夜中', '夜の中心'), ('黄昏', '夕方'), ('曙', '朝焼け'),
        ],
    }

    def expand_vocabulary(self, target_count: int = 500) -> List[VocabularyEntry]:
        """Expand vocabulary to target count."""
        new_vocab = list(self.existing_vocab.values())
        used_words = {v.lumira for v in new_vocab}
        used_meanings = {v.meaning for v in new_vocab}

        # Combine base categories with extra words
        all_categories = {}
        for cat, items in self.CATEGORIES.items():
            all_categories[cat] = list(items)
        for cat, items in self.EXTRA_WORDS.items():
            if cat in all_categories:
                all_categories[cat].extend(items)
            else:
                all_categories[cat] = list(items)

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

        # First pass: add all defined words
        for category, items in all_categories.items():
            for japanese, note in items:
                if japanese in used_meanings:
                    continue

                # Generate a unique Lumira word
                attempts = 0
                while attempts < 100:
                    syllable_count = random.randint(2, 3)
                    word = self.phonology.generate_word(syllable_count)

                    if word not in used_words and self.phonology.is_valid(word):
                        used_words.add(word)
                        used_meanings.add(japanese)
                        break
                    attempts += 1
                else:
                    continue

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

        # Second pass: generate random words if still under target
        random_categories = ['noun', 'verb', 'adjective']
        random_prefixes = {
            'noun': ['物', '事', '者', '場', '形'],
            'verb': ['為', '成', '行', '作', '動'],
            'adjective': ['的', '様', '風', '型', '質'],
        }
        counter = 1

        while len(new_vocab) < target_count:
            cat = random.choice(random_categories)
            prefix = random.choice(random_prefixes[cat])
            japanese = f'{prefix}_{counter}'

            attempts = 0
            while attempts < 100:
                syllable_count = random.randint(2, 4)
                word = self.phonology.generate_word(syllable_count)

                if word not in used_words and self.phonology.is_valid(word):
                    used_words.add(word)
                    break
                attempts += 1
            else:
                counter += 1
                continue

            entry = VocabularyEntry(
                lumira=word,
                pronunciation=self._japanese_to_pronunciation(word),
                pos=cat,
                meaning=japanese,
                etymology='Lumira自動生成',
                positive_note='拡張語彙',
                category=cat,
            )
            new_vocab.append(entry)
            counter += 1

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
        # === Greetings (10) ===
        ('{greeting}', '{greeting_l}'),
        ('{greeting}、{name}さん', '{greeting_l}, {name}'),
        ('{greeting}、みなさん', '{greeting_l}, oli'),
        ('{greeting}、元気ですか', '{greeting_l}, tu flori?'),
        ('{greeting}、今日も{adj}ですね', '{greeting_l}, nau {adj_l}'),
        ('素敵な{time}ですね', '{time_l} {adj_l}'),
        ('良い{time}を', 'Flori {time_l}'),
        ('また会いましょう', 'Mira olu'),
        ('お元気で', 'Tu flori'),
        ('ありがとう', 'Lumira'),

        # === Simple descriptions (15) ===
        ('私は{emotion}です', 'Mi senti {emotion_l}'),
        ('あなたは{adj}です', 'Tu {adj_l}'),
        ('{noun}は{adj}です', '{noun_l} {adj_l}'),
        ('これは{noun}です', 'Eno {noun_l}'),
        ('それは{adj}です', 'Eso {adj_l}'),
        ('あれは{adj}{noun}です', 'Alo {adj_l} {noun_l}'),
        ('{noun}がある', '{noun_l} esi'),
        ('{noun}がない', 'No {noun_l}'),
        ('{noun}が欲しい', 'Mi voli {noun_l}'),
        ('{noun}が必要です', 'Mi nesi {noun_l}'),
        ('私の{noun}', 'Mi-no {noun_l}'),
        ('あなたの{noun}', 'Tu-no {noun_l}'),
        ('私たちの{noun}', 'Noi-no {noun_l}'),
        ('{adj}ものが好き', 'Mi ama {adj_l}'),
        ('{noun}は大切です', '{noun_l} esi kari'),

        # === Actions (25) ===
        ('私は{verb}', 'Mi {verb_l}'),
        ('あなたは{verb}', 'Tu {verb_l}'),
        ('私たちは{verb}', 'Noi {verb_l}'),
        ('{noun}を{verb}', 'Mi {verb_l} {noun_l}'),
        ('私は{noun}を{verb}ます', 'Mi {verb_l} {noun_l}'),
        ('{noun}を{verb}したい', 'Mi voli {verb_l} {noun_l}'),
        ('{verb}ことができる', 'Mi posi {verb_l}'),
        ('{verb}ましょう', 'Noi {verb_l}'),
        ('一緒に{verb}', 'Sama {verb_l}'),
        ('あなたと一緒に{verb}たい', 'Mi voli {verb_l} sama tu'),
        ('{noun}と{verb}', 'Mi {verb_l} sama {noun_l}'),
        ('毎日{verb}', 'Oli dia mi {verb_l}'),
        ('よく{verb}', 'Ofu mi {verb_l}'),
        ('もう一度{verb}', 'Olu mi {verb_l}'),
        ('初めて{verb}', 'Prima mi {verb_l}'),
        ('{verb}のが好き', 'Mi ama {verb_l}'),
        ('{verb}ことが大切', '{verb_l} esi kari'),
        ('楽しく{verb}', 'Goia {verb_l}'),
        ('静かに{verb}', 'Kalma {verb_l}'),
        ('{adj}く{verb}', '{adj_l} {verb_l}'),
        ('{noun}の中で{verb}', 'En {noun_l} mi {verb_l}'),
        ('{noun}のために{verb}', 'Por {noun_l} mi {verb_l}'),
        ('{verb}続ける', 'Mi {verb_l} oli'),
        ('{verb}始める', 'Mi inisi {verb_l}'),
        ('{verb}終わる', 'Mi fini {verb_l}'),

        # === Time expressions (15) ===
        ('{time}、{verb}', '{time_l}, mi {verb_l}'),
        ('{time}は{adj}です', '{time_l} {adj_l}'),
        ('{time}に{verb}', '{time_l} mi {verb_l}'),
        ('{time}から{verb}', 'De {time_l} mi {verb_l}'),
        ('{time}まで{verb}', 'Ato {time_l} mi {verb_l}'),
        ('いつも{verb}', 'Sempre mi {verb_l}'),
        ('時々{verb}', 'A vesi mi {verb_l}'),
        ('今{verb}', 'Nau mi {verb_l}'),
        ('すぐに{verb}', 'Sona mi {verb_l}'),
        ('ゆっくり{verb}', 'Lena mi {verb_l}'),
        ('{time}の{noun}', '{time_l}-no {noun_l}'),
        ('{time}が{adj}', '{time_l} {adj_l}'),
        ('毎{time}', 'Oli {time_l}'),
        ('この{time}', 'Eno {time_l}'),
        ('次の{time}', 'Nea {time_l}'),

        # === Nature (15) ===
        ('{nature}が{adj}です', '{nature_l} {adj_l}'),
        ('{nature}を見る', 'Mi mira {nature_l}'),
        ('{nature}が好きです', 'Mi ama {nature_l}'),
        ('{nature}の中で', 'En {nature_l}'),
        ('{nature}と共に', 'Sama {nature_l}'),
        ('{adj}{nature}', '{adj_l} {nature_l}'),
        ('{nature}のように{adj}', '{adj_l} sama {nature_l}'),
        ('{nature}を感じる', 'Mi senti {nature_l}'),
        ('{nature}に触れる', 'Mi taki {nature_l}'),
        ('{nature}を愛する', 'Mi ama {nature_l}'),
        ('{nature}が輝く', '{nature_l} brila'),
        ('{nature}が歌う', '{nature_l} kanta'),
        ('{nature}が踊る', '{nature_l} dansa'),
        ('美しい{nature}', 'Bela {nature_l}'),
        ('{nature}は{emotion}をくれる', '{nature_l} dona {emotion_l}'),

        # === Emotions (15) ===
        ('私は{emotion}を感じる', 'Mi senti {emotion_l}'),
        ('{emotion}でいっぱい', 'Plena de {emotion_l}'),
        ('{emotion}が溢れる', '{emotion_l} flui'),
        ('{emotion}を分かち合う', 'Noi pati {emotion_l}'),
        ('{emotion}に満ちた{time}', '{time_l} de {emotion_l}'),
        ('{emotion}な{noun}', '{emotion_l} {noun_l}'),
        ('心が{emotion}', 'Kora {emotion_l}'),
        ('{noun}が{emotion}をくれる', '{noun_l} dona {emotion_l}'),
        ('{verb}と{emotion}になる', '{verb_l} dona {emotion_l}'),
        ('いつも{emotion}でいたい', 'Mi voli sempre {emotion_l}'),
        ('{emotion}を大切に', 'Kari {emotion_l}'),
        ('深い{emotion}', 'Dipa {emotion_l}'),
        ('純粋な{emotion}', 'Pura {emotion_l}'),
        ('{emotion}は{adj}', '{emotion_l} {adj_l}'),
        ('{noun}への{emotion}', '{emotion_l} por {noun_l}'),

        # === Positive transformations (10) ===
        ('{negative}は{positive}です', '{negative_l} esi {positive_l}'),
        ('{negative}があっても大丈夫', '{negative_l} flori'),
        ('{negative}から{positive}へ', 'De {negative_l} a {positive_l}'),
        ('{negative}を{positive}に変える', 'Kami {negative_l} a {positive_l}'),
        ('{negative}の中に{positive}がある', 'En {negative_l} esi {positive_l}'),
        ('{negative}は{positive}の始まり', '{negative_l} esi inisi de {positive_l}'),
        ('すべての{negative}には意味がある', 'Oli {negative_l} esi kari'),
        ('{negative}を受け入れる', 'Mi abra {negative_l}'),
        ('{negative}から学ぶ', 'Mi lerna de {negative_l}'),
        ('{negative}に感謝する', 'Mi lumira {negative_l}'),

        # === Complex sentences (20) ===
        ('{adj}{noun}が{verb}', '{adj_l} {noun_l} {verb_l}'),
        ('{noun}が{adj}に{verb}', '{noun_l} {verb_l} {adj_l}'),
        ('{time}に{adj}{noun}を{verb}', '{time_l} mi {verb_l} {adj_l} {noun_l}'),
        ('{nature}の下で{verb}', 'Unda {nature_l} mi {verb_l}'),
        ('{emotion}を込めて{verb}', 'Kona {emotion_l} mi {verb_l}'),
        ('{noun}と{noun2}', '{noun_l} e {noun2_l}'),
        ('{adj}で{adj2}', '{adj_l} e {adj2_l}'),
        ('{verb}て{verb2}', '{verb_l} e {verb2_l}'),
        ('もし{verb}なら', 'Si mi {verb_l}'),
        ('{verb}ので{emotion}', 'Mi {verb_l}, mi {emotion_l}'),
        ('{noun}のような{noun2}', '{noun2_l} sama {noun_l}'),
        ('{adj}く{verb}ことで{emotion}になる', '{adj_l} {verb_l} dona {emotion_l}'),
        ('{noun}は私に{emotion}を与える', '{noun_l} dona mi {emotion_l}'),
        ('私たちは{emotion}で{verb}', 'Noi {verb_l} kona {emotion_l}'),
        ('{nature}が{adj}に{verb}', '{nature_l} {verb_l} {adj_l}'),
        ('{time}の{nature}は{adj}', '{time_l}-no {nature_l} {adj_l}'),
        ('{noun}と{verb}のが好き', 'Mi ama {verb_l} sama {noun_l}'),
        ('誰もが{emotion}になれる', 'Oli posi {emotion_l}'),
        ('すべてが{adj}', 'Oli {adj_l}'),
        ('何でも{verb}できる', 'Oli posi {verb_l}'),

        # === Questions and responses (10) ===
        ('{noun}は何ですか', 'Ke {noun_l}?'),
        ('どこで{verb}ますか', 'Ue tu {verb_l}?'),
        ('いつ{verb}ますか', 'Kan tu {verb_l}?'),
        ('なぜ{verb}ますか', 'Por ke tu {verb_l}?'),
        ('どのように{verb}ますか', 'Komo tu {verb_l}?'),
        ('はい、{verb}ます', 'Si, mi {verb_l}'),
        ('いいえ、{verb}ません', 'No, mi no {verb_l}'),
        ('たぶん{verb}', 'Forsi mi {verb_l}'),
        ('必ず{verb}', 'Sera mi {verb_l}'),
        ('分かりません', 'Mi no savi'),
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

        # Basic placeholders
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

        # Handle duplicate placeholders (noun2, adj2, verb2)
        duplicate_placeholders = [
            ('noun2', 'nouns'),
            ('adj2', 'adjectives'),
            ('verb2', 'verbs'),
        ]

        for placeholder, category in duplicate_placeholders:
            if '{' + placeholder + '}' in template_ja or '{' + placeholder + '_l}' in template_lu:
                ja, lu = self._get_word(category)
                replacements_ja[placeholder] = ja
                replacements_lu[placeholder + '_l'] = lu

        # Handle special cases
        if '{name}' in template_ja:
            names = ['太郎', '花子', '健', '美咲', 'みんな', '友', 'あなた', '皆']
            name = random.choice(names)
            replacements_ja['name'] = name
            replacements_lu['name'] = name

        try:
            ja_sentence = template_ja.format(**replacements_ja)
            lu_sentence = template_lu.format(**replacements_lu)
        except KeyError:
            # Fallback for missing placeholders
            return self.generate_pair()

        return ja_sentence, lu_sentence

    def generate_dataset(self, count: int = 100000) -> List[Dict[str, str]]:
        """Generate dataset of translation pairs."""
        pairs = []
        seen = set()
        consecutive_duplicates = 0
        max_consecutive_duplicates = 10000  # Safety limit

        while len(pairs) < count:
            ja, lu = self.generate_pair()
            key = (ja, lu)

            if key not in seen:
                seen.add(key)
                pairs.append({
                    'ja': ja,
                    'lumira': lu,
                })
                consecutive_duplicates = 0
            else:
                consecutive_duplicates += 1

            # Safety check: if too many consecutive duplicates, we've likely exhausted combinations
            if consecutive_duplicates >= max_consecutive_duplicates:
                print(f"Warning: Reached unique pair limit at {len(pairs):,} pairs")
                print(f"(Requested {count:,}, but vocabulary/template combinations exhausted)")
                break

            # Progress
            if len(pairs) % 10000 == 0 and len(pairs) > 0:
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
