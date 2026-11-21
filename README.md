# Python Voice Synthesis — Three Experiments

日本語 | English

> The three demos and their source files are available on the default branch.

## 日本語

Pythonによる音声合成実験3本。音声、音声合成コード、シーケンス、スペクトログラムをプロジェクト別に整理しています。

| 制作モデル | 実験 | 音声 |
|---|---|---|
| Claude Fable 5.2 (Medium) | Klatt型・規則ベースのフォルマント合成（NumPy/SciPy） | speech.wav / 約9.96秒 |
| Claude Fable 5.2 (Medium) | KORG DS-10相当の2シンセ・16分音符シーケンサー制約 | ds10_talk.wav / 11.52秒 |
| GPT-6.2 Astra (Ultra) | DS-10相当の2シンセ構成、譜面から再生成 | DS10_voice.wav / 10.00秒 |

### ディレクトリ構成

- `claude-fable-5.2-medium/klatt-formant/`：自作フォルマント合成器、音源、途中の比較資料
- `claude-fable-5.2-medium/ds10-style/`：制約付き2シンセ、音源、シーケンス
- `gpt-6.2-astra-ultra/ds10-style/`：音源、レンダラー、譜面、チェックコード

各音声の生成経路に、既存のTTSエンジン、学習済み音声モデル、録音素材は使用していません。音声認識器は、生成後の評価に使われた場合があります。

**重要**：DS-10系は実機の完全再現ではなく、音源数・操作系・シーケンサーの制約を参考にした独自の実装です。フィルター特性やノブ換算は近似であり、実機に同じ設定を入れて同一音色になるとは限りません。KORGによる公式プロジェクトではありません。

### 実行環境

Python 3.10以降推奨。共通の依存ライブラリは `requirements.txt`。各プロジェクト内の元READMEに個別の使用方法を記載しています。

**ライセンス**：公開と再利用許諾は別です。ライセンスファイルは、明示的な条件が確認できるまで追加しません。

## English

Three experiments in programmatic **speech synthesis with Python**, credited to the models reported by the user:

1. **Claude Fable 5.2 (Medium), Klatt-style formant TTS:** rule-based English speech with NumPy/SciPy (~9.96 s).
2. **Claude Fable 5.2 (Medium), DS-10-style constraint:** two monophonic synths with 16th-note sequencer control (11.52 s).
3. **GPT-6.2 Astra (Ultra), DS-10-style constraint:** two synthesizers driven by a stored note/gate/volume/KAOSS score (10.00 s).

The complete bundle includes WAV files, Python sources, sequencer data and diagnostic figures. Recorded samples and pre-trained TTS models are not used in the audio generation path; ASR tools may have been used for evaluation.

**Not a hardware emulator:** The DS-10-style projects use constrained synthesizer architectures inspired by the 2008 KORG DS-10, not a circuit-accurate or parameter-accurate simulation. They are independent experiments and are not affiliated with KORG.

Python 3.10+ recommended; shared dependencies are in `requirements.txt`. See the source-specific READMEs for commands and limitations. No software license is granted solely by making the repository public.
