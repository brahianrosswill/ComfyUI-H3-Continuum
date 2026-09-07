# V3.8 初心者向けマニュアル設計・撮影台本

状態: Computer Use実画面検証PASS／4-step・0.30 MP・Spectrum OFFでContinue完走／README反映中（2026-09-07）

最新結果: 提供された00058を、Turbo4、実Steps 4、Euler/simple、Draft 0.30 MP、Spectrum OFF、2×5秒でUIから再検証した。Chunk 1の三択レビュー画面が出現し、`Use it and continue`選択後に右上の青い`Run`を押すと、受理済みChunk 1を再SamplingせずChunk 2だけをSampling（Sampler表示43.738秒）し、10秒の`Saved sequence is complete`へ到達した。修正後の`Try this chunk again`も別の実画面試験でPASS済み。現在フロントエンドの実行ボタンは右上の青い`Run`で、ノード内の`Run`設定やカード内の旧`Queue`表記と区別する。

## 現在の制作範囲（以下の旧撮影案より優先）

- 完成原稿は英語の `../README.md`。日本語READMEの翻訳は最終段階まで保留。単純な初回生成説明は増やさず、任意チャンク数での継続・再試行・残り生成・設定へ戻る・履歴・指定位置以降の再生成・完成後の延長を中心にする。
- 継続操作だけでなく、V3.8が公開する7ノードの全ユーザー向けsocket／control／outputをREADMEで説明する。SamplerのMain／Advanced／Review状態、Continuum Settings、Load Image／Audio／Video、Reference Audios、Finalize、Second Passを省略しない。非表示の内部IDは「UI」扱いせず、frontend管理であることだけを示す。
- UIの文字列、変更する値、維持する値、Queue操作、その回に新規生成される範囲、保存動画の長さを順番に説明する。2チャンクは例であり固定条件ではない。fixed seed、Chunksは追加数でなく合計数、Terminal Merge例外を省略しない。
- 提供された実画面6枚を加工せず `images/v38-manual/` へ保存済み。`seed-fixed.png`、`review-settings.png`、`review-actions.png`、`continue-selected.png`、`return-to-review.png`、`full-video-progress.png`。Spectrum設定の画像はこの章に使用しない。元画像とのSHA一致・PNG形式・READMEリンクを確認済み。
- 今回追加した公開用実画面はreview三択、continue選択、sequence完了、Back to Settings後の完了設定、Render History内部、Regenerate FromとVariation Nonce。既存画像を新規撮影と称さない。全画面ではなく操作部分の実スクリーンショットを使用する。finish選択は三択画面で場所と意味を説明し、必要になった場合だけ追加撮影する。
- 追加提供された7枚を、Video／Audio loaderとbypass、Reference Image 1–3、First／Last Image、First Image + Draft 0.30 MP、LightX2V Turbo LoRA一覧、First Image tooltip、Manual Width／Heightとして無加工コピーした。LoRA loader、Spectrum、画像中のAudio Switch／Fast Groups Bypasserは外部workflow要素として分離し、Continuum公開UIと誤認させない。
- README冒頭に「Web対応AIへこのREADMEを読ませる」案内を置く。ChatGPT／Gemini／Grokの現行公式情報に限定し、URLだけでrepository全文を読めるとは保証しない。GeminiのImport codeと、失敗時のREADME貼付／uploadを案内する。
- Size Source／Resolution／Manual Width・HeightはV3.8 Sampler本体に搭載されたため、通常workflowに外部MP／size nodeは不要と説明する。32×32は音声だけを聴く診断用であり、映像品質の推奨値にはしない。
- Issue #13はOpen、StandardにはProduction mitigationなし、R6-Aは一条件で改善したが最終4×8ではno-opだった、という証拠範囲を固定する。「解決済み」「全く改善方法がない」のどちらにも誇張しない。
- ユーザーが `http://127.0.0.6:8188/` で起動した `00058` ワークフローを使用した。ユーザー所有backendは停止しない。
- READMEの測定例は今回の4-step実測へ更新し、旧6-step値と混在させない。commit/pushは今後の明示依頼時のみ。

作成日: 2026-09-06

対象: `H3 Continuum Sampler V3.8`の現在のMain UI

再開時の正本: この文書と`PROJECT_STATE.md`

## この文書を読めば撮影準備から再開できる

この文書は、V3.8の初心者向け画像マニュアルを作るための設計書兼撮影台本です。完成マニュアルそのものではありません。ドリフト軽減策、公開UI、標準Workflowが固まったあと、この台本に沿ってComputer Useで実画面を確認し、スクリーンショットを撮影します。

撮影開始の指示を受けた担当者は、最初に次を読みます。

1. `AGENTS.md`
2. `PROJECT_STATE.md`
3. この文書
4. `README_JA.md`の「10秒を5秒ずつ確認しながら作る」
5. `examples/workflows/MiniMax_H3_Continuum_V38.json`

撮影開始前に、表示ラベル、標準Workflow、公開ノード数、Review／Take動作がこの台本と一致するか確認します。違いがある場合は、古い画面を撮影せず、先に台本を現在のUIへ合わせます。

## 完成マニュアルは初回手順と目的別の逆引きに分ける

読者はComfyUIの基本操作を知っているものの、Continuumのチャンク、Review、Take、保存再開は初めてという利用者を想定します。モデルの入手、ComfyUI本体の導入、一般的なSampler解説は別資料へ任せます。

完成版では、最初に読む範囲と必要時だけ読む範囲を分けます。

- 初回に読む: 1～4章
- 必要なときに読む: 5章以降

予定する章は次のとおりです。

1. V3.8で最初の動画を作る
2. 長さと出力サイズを決める
3. 10秒を5秒ずつ確認しながら作る
4. 完成動画をFinalizeして保存する
5. 現在のチャンクだけを作り直す
6. 保存されたTakeを選ぶ
7. 選んだ地点から続きを作る
8. List／Timeline Promptを使う
9. Audio Continuityと参照素材を使う
10. Second Passを完成後のsequenceへ使う
11. 症状から原因を探す
12. 用語集と報告テンプレート

マニュアル冒頭には、V3.8が「生成、チャンクごとの確認、部分再生成、保存再開」を行うProduction Samplerであることを短く示します。V3.7以前の画面やNode IDをV3.8の手順へ混ぜません。

## 1操作につき1枚の画面と1つの期待結果を示す

各手順は次の順番で書きます。

1. 目的を1文で示す
2. 変更する項目を画面上の正確な表示名で示す
3. 操作対象を番号付きスクリーンショットで示す
4. なぜ必要かを1文で添える
5. Queue後に見える正常な結果を示す
6. 次に選べる操作へつなぐ
7. 失敗時は対応する逆引き項目へ案内する

1枚へ多数の番号を詰め込みません。特にReviewでは「ボタンを選ぶ」と「ComfyUIのQueueを押す」を別ステップとして扱います。選択しただけで生成が始まらないことを、本文と画面の両方で明示します。

## 撮影はUIと長尺品質が固定されたあとに始める

本撮影は、次の条件がそろってから開始します。

- 公開対象のV3.8 UIラベルと配置が確定している
- 標準Workflowが確定している
- 現在進行中のドリフト調査について、ローンチ時の扱いが決まっている
- READMEと実画面のReview手順が一致している
- `Review Each Chunk`、再生成、Take選択、再開の正常系が通る
- 画面更新、警告表示、失敗時の案内が確認済みである
- 撮影に使うruntimeとsourceの一致を確認できる

構成と本文の下書きは先に進められます。UIが変わる可能性のある段階では、最終スクリーンショットと座標依存の注釈を確定しません。

## 撮影環境を固定して画面差し替えを減らす

撮影前に次の条件を記録します。

| 項目 | 撮影時に記録する内容 |
|---|---|
| Continuum | version、source commitまたはtree識別情報 |
| ComfyUI | versionまたはcommit |
| Workflow | 正本ファイル名とSHA-256 |
| Browser | 種類、表示倍率 |
| Viewport | 幅、高さ |
| Theme | light／dark |
| Sampler view | 現在のMain UIとAdvancedの開閉状態 |
| Input | 使用する画像、音声、動画の識別名 |
| Generation | Seed、Model、LoRA、Sampler、Steps、SIGMAS |
| Output | Width、Height、fps、Chunks、Seconds per Chunk |

撮影用Workflowは、個人名、ローカルの機密パス、過去の不要な履歴を含まない専用コピーを使います。ComfyUI Coreノードは標準表示名のまま撮影し、Continuum専用に見える名前へ変更しません。

## 撮影台本はQuick Startから始める

最初の撮影セットは、`2 × 5s = 10 seconds`のReview手順です。以下のShot IDを画像名にも使います。

| Shot ID | 撮影する状態 | 画面で確認する要素 | 本文で伝えること |
|---|---|---|---|
| V38-M01 | 標準Workflow全体 | Sampler、Core Decode、Finalize、Core Save Video | 最小の接続全体 |
| V38-M02 | Sampler初期Main | Prompt Format、Continuity、Base Seed、Control After Generate、Audio Continuity | 最初に確認する場所 |
| V38-M03 | 長さ設定後 | Chunks `2`、Seconds per Chunk `5`、Total Length `10 seconds` | 合計10秒の指定 |
| V38-M04 | 出力サイズ設定 | Size Sourceと適用中のWidth／Height | First ImageとManualの違い |
| V38-M05 | Review準備完了 | Run `Review Each Chunk`、Progress On、緑色のReadyカード | 最初のQueue準備完了 |
| V38-M06 | Chunk 1実行中 | ComfyUI Queueと進行表示 | 最初は5秒だけ作る |
| V38-M07 | Review Ready | `Chunk 1 is ready for review`のアンバー表示 | 5秒出力は正常 |
| V38-M08 | Reviewの3操作 | Use it and continue、Try this chunk again、Use it and finish the rest | 3つの分岐 |
| V38-M09 | Continue選択後 | 選択ボタンの`✓` | 選択だけでは生成しない |
| V38-M10 | 2回目のQueue | ComfyUI Queue | 採用済みChunk 1を再利用する |
| V38-M11 | 10秒完成後 | 完成状態、Finalize、保存結果 | 2チャンクの完成 |
| V38-M12 | Settingsへ戻る | Back to Settings | Reviewを失わず設定を見る |
| V38-M13 | Reviewへ戻る | Return to Review | 保留中のReviewへ戻る |

Quick Startの撮影が終わったら、同じ撮影用Runを再利用して再生成と履歴を撮ります。

| Shot ID | 撮影する状態 | 画面で確認する要素 | 本文で伝えること |
|---|---|---|---|
| V38-R01 | 再生成選択前 | 現在のChunkとTake | 作り直す対象 |
| V38-R02 | 再生成選択後 | Try this chunk againの`✓` | 同じChunkの別Takeを作る |
| V38-R03 | 再生成完了後 | 新しいTakeとReview状態 | 前のTakeは消えない |
| V38-H01 | Render Historyを開く | Take一覧と現在の選択 | 閲覧だけではcanonicalを変えない |
| V38-H02 | 過去Takeを選ぶ | Previous／Nextまたは現在の選択表示 | 候補を比較する |
| V38-H03 | Take採用前 | Use This Take | Queue後にcanonicalを変更する操作 |
| V38-H04 | 途中から継続する前 | Continue From Hereと再利用範囲 | 選択地点より後ろだけを作り直す |

## サイズ・入力・Advancedは目的別に追加撮影する

Quick Startへすべてを詰め込まず、次の画面は独立した章で撮ります。

| Shot ID | 撮影する状態 | 必須の比較 |
|---|---|---|
| V38-S01 | Size Source `First Image` | 画像の縦横比と参照表示のWidth／Height |
| V38-S02 | Size Source `Manual` | 編集可能な32-pixel単位のWidth／Height |
| V38-I01 | First Image接続 | I2VA／FL2VAでのサイズ決定 |
| V38-I02 | Reference Image接続 | First Imageとは異なる役割 |
| V38-I03 | Video Guide接続 | Video Guide SizeとFrames |
| V38-A01 | Reference Audios | Audio 1／2／3の順序 |
| V38-A02 | Audio Continuity | On／Offの意味と最終音声との違い |
| V38-X01 | Advancedを開く | Mainにない高度な項目の位置 |
| V38-X02 | Second Pass接続 | Review完了後のsequenceに使うこと |

## 異常系は実際の表示を再現して撮る

正常系の撮影後、データを壊さない条件で次の状態を意図的に再現します。各ケースは、表示された文言と解決手順が現在の実装に一致することを確認してから掲載します。

| Shot ID | 症状または表示 | マニュアルで案内する確認 |
|---|---|---|
| V38-T01 | 最初のQueueで5秒だけ出た | Review Each Chunkの正常動作 |
| V38-T02 | 次の操作が表示されない | Queue完了、Progress On、Chunks 2以上、画面更新 |
| V38-T03 | 操作を選んでも生成されない | 選択後にQueueが必要 |
| V38-T04 | Chunksが1 | 次のChunkが存在しない |
| V38-T05 | fixedを求める案内 | Control After GenerateとSeed lineage |
| V38-T06 | First Imageが届かない | 表示中のManual canvasへのfallback |
| V38-T07 | 保存済みprefixを再利用しない | Run Name、Seed、Prompt Plan、生成契約の差 |
| V38-T08 | Review途中でSecond Passを使おうとした | sequence確定後に実行 |
| V38-T09 | 音声境界や冒頭語の欠落を報告する | Workflowと再現条件を保存して報告 |
| V38-T10 | 長尺で色・質感・identityが変化する | 現行の既知制限と報告に必要な情報 |

T09とT10は原因を決めつけません。実際のバグ、設定差、説明不足を分けられるよう、報告テンプレートへ誘導します。

## スクリーンショットは同じ規則で保存する

画像は`docs/images/v38-manual/`へ置く想定です。撮影を始めるまでは空フォルダを作りません。

ファイル名は次の形式にします。

```text
v38-m01-standard-workflow.png
v38-m05-review-ready-to-queue.png
v38-m08-review-actions.png
v38-h04-continue-from-here.png
v38-t05-control-after-generate-fixed.png
```

元画像を保管する場合は`*-raw.png`、掲載用に注釈を加えた画像は通常名とします。注釈は番号、短いラベル、必要最小限の枠線に限定します。文字やsocketを覆わず、色だけに意味を持たせません。

## Computer Use撮影は観察と確認を1ステップずつ行う

撮影時はComputer Useで現在のComfyUIウィンドウを選び、操作前に画面状態を確認します。クリックや入力後は新しい画面状態を取得し、古い座標や要素番号を再利用しません。

各Shotは次の手順で確定します。

1. 台本の前提状態を作る
2. 表示ラベルと値を画面上で読む
3. スクリーンショットを取得する
4. Shot ID、Workflow、設定、実行前後を記録する
5. 画像と本文が同じ操作を示すか確認する
6. 次のShotへ進む

ログイン、権限変更、公開投稿、ファイル送信など、画面撮影に不要な外部操作は行いません。個人情報や認証情報が見えた場合は撮影を止め、撮影用環境を整え直します。

## 完成版には再現可能なバグ報告テンプレートを付ける

問い合わせ時には、少なくとも次を添えてもらいます。

- Continuum version
- ComfyUI version
- Workflow JSON
- Prompt全文とPrompt Format
- Model、LoRA、Sampler、Steps、SIGMAS
- SeedとControl After Generate
- ChunksとSeconds per Chunk
- ContinuityとAudio Continuity
- RunとProgress
- Width／HeightとSize Source
- 使用したFirst／Last／Reference Image、Video Guide、Driving／Reference Audioの有無
- どのChunk境界で起きたか
- Status／Report全文
- 問題が分かる短い出力またはスクリーンショット

音声の冒頭語が欠ける場合は、期待した台詞、実際に聞こえた台詞、欠落したChunk番号も記録します。ドリフトの場合は、開始Chunkと最も目立つChunkを示します。

## 撮影後は画面だけで手順を判断しない

完成判定では、画像の見た目に加えて実際の動作を確認します。

- すべての画像が現在のV3.8 UIである
- 表示名が実装と完全に一致する
- Quick Startを初見の利用者が順番どおり実行できる
- Review操作を選ぶ工程とQueueする工程が分かれている
- 再生成で前のTakeが保持される
- Continue From Hereの再利用範囲が正しい
- Coreノードの標準表示名を変更していない
- 画像内に個人情報や不要なローカルパスがない
- 英語版と日本語版で同じ画像を安全に共有できる
- README、完成マニュアル、実画面の説明が矛盾しない

撮影・本文作成後は、手順を最初から一度再実行します。画像と文章だけを見て再現できない箇所は、UI改善候補、マニュアル修正、実装バグのいずれかへ分類します。

## 次回の開始指示

ユーザーから「V3.8マニュアルの撮影を開始」「初心者向けマニュアルを作成」などの指示があった場合は、この文書を正本として再開します。まず撮影開始条件を確認し、未確定のUIやドリフト表示方針があれば、その部分だけを保留します。条件がそろっていればV38-M01から順に進めます。

公開、commit、push、Civitai掲載は、この台本の作成や撮影開始には含まれません。別の明示指示が必要です。
