# DS-10の2音で英語を喋る

先に試聴するなら `DS10_voice.wav` を開いてください。10.000秒、44.1 kHz、16-bit PCM、両トラック中央です。英文は末尾にあります。

## 再生・編集

Python 3.10以降とNumPyを使います。作業フォルダーで以下を実行すると、同梱の譜面からWAVを再生成できます。

```sh
python echo.py echo_score.json -o my_render.wav
python check.py my_render.wav
```

- `echo_score.json` に完成版の音色、BPM、パターン再生順、全ステップを収録しました。各行は単語、発音メモ、SYN 1、SYN 2の順。各SYNの配列は NOTE、GATE、VOLUME、PAN、X、Y の順です。NOTEはMIDI番号、休符はnull、GATEは百分率またはLEG、PANの0が中央です。単語や発音メモを消しても同じ音を再現できます。
- `echo.py` はこの譜面を直接再生します。`score_source.py` にある音素と単語の対応や目標周波数は参照しません。
- `score_source.py` は入力譜の作成用です。目標周波数を12段階のノブ位置に丸め、NOTEによるイントネーションと各音の長さを記譜しています。このファイルから組み直すときは `python score_source.py -o edited.json` を実行してください。
- `check.py` はPCMの再現性、歌詞メモをすべて空にした場合の一致、比較条件の周波数測定を検証します。周波数解析はこの検査スクリプトだけで行います。再生には `echo.py` とJSONの2ファイルだけが必要です。

## 音源と演奏

1ステップは16分音符。BPM 222で約67.57 ms、148ステップを11パターンに分けました。各パターンを1回ずつ順に再生します。両方のシンセでVCO1はノコギリ波、VCO2はホワイトノイズ。母音では同じNOTEで駆動し、各シンセに元からあるVCFの共鳴を、母音の低い山と高い山にそれぞれ合わせました。

| 固定設定 | SYN 1 | SYN 2 |
|---|---:|---:|
| VCF | BPF | BPF |
| CUTOFFの基準 | 415 Hz | 2,160 Hz |
| PEAKの実装値（Q） | 4.7 | 6.5 |
| EG（A / D / S / R） | 2 ms / 1.5 ms / 0.265 / 8 ms | 2 ms / 1.5 ms / 0.340 / 8 ms |
| VCAモード、レベル | EG、0.95 | EG、0.70 |
| ポルタメント（時定数） | 17 ms | 17 ms |
| KAOSS X | VCF EG INT、SMOOTH | VCF EG INT、HOLD |
| KAOSS Y | VCO BALANCE、SMOOTH | VCO BALANCE、HOLD |

Xを0～11の整数として、再生中のカットオフを `基準Hz × 2 ** ((2 × X / 11 − 1) × 4 × EG)` で近似しています。LEGATOでEGを維持すれば、SUSTAINの設定でX全体の周波数幅を絞れます。このため、ノブ全域を直接CUTOFFに当てる場合よりも母音付近に段を多く使えます。SYN 1のSUSTAIN中の範囲は約199～865 Hz、SYN 2は約841～5,544 Hzです。発音開始時とリリース中は同じADSRに従って変化します。両軸とも同一トラック内で同じHOLD/SMOOTH設定を使います。

Yが0ならノコギリ波、11ならノイズ、中間はその音量比です。/z/ では下の声を有声のまま残し、上にノイズを混ぜました。/k/ と /b/ では短いGATEと閉鎖を組み合わせています。25% GATEのキーオンは約16.9 ms、50%は約33.8 msです。/aɪ/ は3段の母音遷移で、/m/ は下寄りの弱い音と、次の母音へ開く動きで表しました。

VCO2 PITCHは0、SYNCはOFF、DRIVEは0。MG・パッチケーブル・FXは未使用です。マスターレベルは全曲固定0.83。WAVのサンプルピークは−1.45 dBFSです。

## 実測した比較

同じNOTE・GATE・VOLUME・PANと同じ乱数シードを使いました。両方SMOOTHの比較ではSYN 2の補間モードのみ変更。直接CUTOFFの比較では同じ目標帯域を30～16,000 Hzの対数カーブの12段に再量子化し、VCFのEG深さを中立にしました。この直接割当の周波数カーブも本実装の仮定です。

| 条件 | 持続 /i/ の上側ピーク¹ | /z/ の高域重心² |
|---|---:|---:|
| X＝CUTOFF、下SMOOTH・上HOLD | 2,853 Hz | 5,294 Hz |
| X＝VCF EG INT、上下ともSMOOTH | 2,361 Hz | 3,885 Hz |
| 完成版、下SMOOTH・上HOLD | 2,361 Hz | 5,778 Hz |

¹ /i/ が続く10ステップで、各ステップの15～70%の区間を解析。1.6～3.5 kHz内の最大パワーの周波数の中央値です。手で設定した目標は2,350 Hzです。  
² “is” と “music” の /z/ の同じ区間で、1.8～11 kHzのパワー重み付き重心を求めた中央値です。上側をHOLDにすると、1ステップの摩擦が次の母音へ早く移りすぎるのを避けられました。完成版の /s/ と /ʃ/ の高域重心は、それぞれ約5.74 kHzと3.63 kHzです。

直接CUTOFFの版も書き出せます。

```sh
python score_source.py --direct -o direct.json
python echo.py direct.json -o direct.wav
```

## 実機との差と自己監査

- 2台とも個別のVCO1・VCO2 → BALANCE → 単一VCF → VCAという経路です。各トラックのゲートとピッチは常に1音分です。サンプルレートで更新されるのは発振・フィルター状態、ADSR、ポルタメント、許可したSMOOTH補間です。ピッチ、ノブ、パッチの隠しレーンは設けていません。
- ノブ位置とHz・時間・Qの対応、とくにEG INTの最大±4オクターブ、2極の状態変数VCF、線形ADSR、指数ポルタメント、SMOOTHの直線補間はPython側の近似です。スピーカーやDS内部の演算精度はモデル化していません。使わないMGやFX等の汎用エミュレーションは含みません。実機へのセーブデータは生成しません。
- 再生成したPCMが配布WAVと全サンプル一致し、英文と音素名を空にした版も同じ波形になりました。ドラム、TTS、学習済み音声モデル、録音、外部API、追加VCF、ボコーダー、重ね録り、出力後のEQやコンプレッサーは使っていません。
- 難所は “hear” の /ɹ/、/m/ と /n/ の区別、“little” の弾き音。/ɹ/ の第3フォルマントを独立して下げる手段がなく、/m/ と /n/ に鼻音特有の反共鳴も作れないので、2本の共鳴と音量で近似しました。弾き音は1ステップの弱い有声音で代用しています。人によるブラインド書き取りは未実施です。

設定確認に使った資料（2026-11-21参照）

- [発売元のDS-10講座・KAOSSモード3](https://www.marv.jp/products/ds10/jp/lesson/chap03.html)
- [DS-10取扱説明書の音源ページ](https://www.manualslib.com/manual/4267609/Korg-Ds-10.html?page=30)
- [トラックレーン・12段階・LEGATO・HOLD/SMOOTH](https://www.tinyloops.com/doc/korg_ds-10/sequencer-kaoss.html)
- [実機のKAOSS割り当て画面](https://www.tinyloops.com/doc/korg_ds-10/images/ds-kaoss3.jpg)

## 英文

“Hello. My name is Echo. I'm a little blue machine. I live in a box. I make music. Can you hear me?”
