# OpenScience Artifacts

このディレクトリは論文の再現性評価に必要なコード・設定・代表データをまとめたものです。
重いデータや一部の重みは同梱せず、取得方法や代替手段をREADMEに記載します。

## 目次
- 1. 構成
- 2. コード概要
- 3. 攻撃生成 (AttackGeneration)
- 4. 目標モデル (TargetModel)
- 5. Carla Driving (System-level)
- 6. データ（デジタル実験／物理実験）
- 7. 重み・外部依存
- 8. 実行例

## 1. 構成
```
OpenScience/
  AttackGeneration/
    1_ParameterSetting
    2_Optimization
        Altering/  (分類モデル攻撃)
        Hiding/    (物体検知攻撃)
  TargetModel/
    Detector/  (DetectorAPI)
    Classifier/
  Carla_TSR_Driver/  (System-level　評価)
```

## 2. コード概要
- AttackGeneration/Altering: 分類モデル向けの攻撃生成（STOP/SL65、Non-targeted/Targeted）
- AttackGeneration/Hiding: 物体検知向けの攻撃生成（STOP/SL65、Blender最適化）
- AttackGeneration/1_ParameterSetting: 反射再現のための Specular Tint 推定と色最適化
- TargetModel/Detector: DetectorAPI（YOLO/ARTSなどの推論）
- TargetModel/Classifier: 分類モデル推論（SimpleCNNなど）
- Carla_TSR_Driver: System-level評価（CARLA走行実験、設定/スクリプト一式）
- traffic_sign_classifiers: 推論用のノートブック

## 3. 準備
- Blender 4.1.1 を公式サイトから取得して展開しておくこと
- Blender の .blend は `Archive/2_Optimization copy/` 配下に格納（例: `Archive/2_Optimization copy/1_Altering/SL65/20240615_us_sl65_reflector_multiple_color.blend`）
- 攻撃生成スクリプトは `blender_file/` 配下の .blend を参照する想定。`BLEND_FILE` を設定するか、`Archive/2_Optimization copy/...` の .blend を `blender_file/` にコピー/リンクして使用
- Carla 0.9.15 公式のDocker から落とすこと．また追加アセットをインストールしておく．
- スクリプト内の `<PATH_TO_...>` は各自の環境に合わせて置換すること
- 背景画像は BDD100K を想定しているが、任意の走行画像データセットで代用可（`BACKGROUND_IMAGE_DIR` で指定）
### パス設定（Blender 実験用）
以下のスクリプトはパスを自動正規化し、環境変数で上書きできます。
対象: `AttackGeneration/2_Optimization/1_Altering/STOP copy/open_blender2.py`, `AttackGeneration/2_Optimization/1_Altering/STOP copy/alternate_attacker.py`

- `PROJECT_ROOT`: リポジトリルート。未指定時は `__file__` から自動推定。
- `HOME_GO`: `HOME` の代替。Desktop や `Code/...` 既定パスの基準。
- `BLENDER_PATH`: Blender 実行ファイルのパス。
- `BLEND_FILE`: `.blend` ファイルのパス。
- `ATTACKER_SCRIPT`: Blender に渡す Python スクリプトのパス。
- `BACKGROUND_IMAGE_DIR`: 背景画像ディレクトリ。
- `OUTPUT_CSV_ROOT`: 出力 CSV ルート。

`--background_image_path` と `--output_csv_dir` は相対パス指定が可能で、`PROJECT_ROOT` 基準で解決されます。
## 3. 攻撃生成 (AttackGeneration)
### 3.0 ParameterSetting（反射再現）
- `AttackGeneration/1_ParameterSetting/calc_specular_Tint.ipynb` で Specular Tint を推定（ヘッドライトの色などから算出）
- `AttackGeneration/1_ParameterSetting/20240613_no_opt_color.blend` 内の Python コードで色を最適化
- 補助スクリプト: `AttackGeneration/1_ParameterSetting/get_average_RGB.py`

### 3.1 Altering（分類モデル攻撃）
- STOP/SL65 に分けて配置
- Non-targeted / Targeted を含む
- デジタル評価は `open_blender.py` / `open_blender2.py` を起動すると Blender と Python スクリプトが連動して攻撃生成を実行
- パッチ材料パラメータは `1_ParameterSetting` で Specular Tint を決定し、`20240613_no_opt_color.blend` 内の Python で色を最適化した結果を利用

### 3.2 Hiding（物体検知攻撃）
- STOP/SL65 に分けて配置
- open_blender*.py と hiding_attacker*.py と .blend が連動

### 3.3 デジタル評価の出力（攻撃生成）
- `open_blender*.py` の実行で、各試行の結果が `--output_csv_dir` に JSON/CSV として出力される
- レンダリング画像は攻撃スクリプト内で指定した保存先に出力（既定は `blender_file/render/optuna/...`）
- 例: STOP の `open_blender2.py` は `attack_effectiveness_evaluation2/<patch>/log.json` に集計ログを出力

## 4. 攻撃対象モデル (TargetModel)
- DetectorAPI: YOLO/ARTS などの検出器推論
- Classifier: SimpleCNN 等の分類推論
- デジタル評価・物理評価の推論は TargetModel を使用
- Detector の場合は DetectorAPI を起動し、`test.ipynb` から最適化/評価を実行
- Classifier の場合は `infer.ipynb` を使用（手動で標識領域をクロップして入力）

## 5. Carla Driving (System-level)
- Carlaの走行実験に関するコード一式
- stopline_candidates.json 等の設定ファイルを含む
- `Carla_TSR_Driver/scripts/carla-server.sh` を使う前に以下を準備すること：
- Carla 0.9.15 を用意（公式 Docker を利用）
- 公式サイト配布の追加アセットをロードした状態の Carla を「サーバ用イメージ」として保存する
- `carla-server.sh` は上記イメージ（例: `carla:0.9.15-addassets`）を前提に起動する
- AdditionalMaps は公式ドキュメントを参照
- 実行時に CARLA PythonAPI を `pip` でインストールしておく必要がある
- サーバ起動後に以下を実行：
```
python Carla_TSR_Driver/src/place_stop_sign.py
```
- その後 `Carla_TSR_Driver/src/run_autopilot.py`（または `Carla_TSR_Driver/scripts/run_autopilot_batch.py`）で System-level 評価を実行
```
python3 src/place_stop_sign.py --host localhost --port 5555 --map Town04 --candidates stopline_candidates.json --mesh-path "/Game/Carla/Static/TrafficSign/Stop/Stop_v01/SM_stopSign.SM_stopSign" --lateral-offset 1.0 --forward-offset 0.0 --yaw-offset 90 --draw --z-offset 1.5
python3 src/run_autopilot.py --config configs/autopilot.json
```
- 別の評価環境を探索する場合は `find_straight_stop.py` を実行
- 既定の評価シチュエーションは `Carla_TSR_Driver/stopline_candidates.json` に記載
- `run_autopilot.py` のログは `logs/autopilot.csv`（設定で変更可）に出力され、`dist_to_stop_m` が停止線からの距離

## 6. データ（デジタル実験／物理実験）
### デジタル実験
- STOP/SL65 × パッチ種 × サイズごとに代表画像を1枚ずつ
- 全試行の画像は同梱しない（ログ/CSVで代表）
- 背景画像は BDD100K Evaluation Dataset を想定（任意データセットでも可）
- Mapillary/ARTS は Training Dataset で学習し、Evaluation Dataset で評価

### 物理実験
- STOP/SL65 の benign / attack 代表画像（物体検知・分類モデルは 15m 正面のみ）
- 最適化過程の動画は最小限のみ
- 走行デモ動画は全て同梱
- 防御前後の比較図は全て同梱
- Carla の結果は代表例のみ
- Perceptual Similarity（SSIM/LPIPS）と User Study のデータは別途まとめて提供
- Perceptual Similarity は標識領域をトリミングして比較すると安定
- User Study の集計は `perceptual_similarity.ipynb` を使用
- 物理評価は撮影動画を用い、TargetModel で推論して評価する
- 分類モデルは標識領域を手動でクロップして `infer.ipynb` に入力する

## 7. 重み・外部依存
- 主要重みのみ同梱（例: YOLO/ARTS/SimpleCNN の主モデル）
- それ以外は取得方法を別途記載
- サイズの都合で同梱していない重みは自前学習で利用可能
- Faster R-CNN は外部の学習パイプラインを用いて学習する（Mapillary/ARTS を使用）
- YOLOv11 は Ultralytics 系の学習パイプラインで学習する（Mapillary/ARTS を使用）
- ARTS/YOLOv5 は同梱、YOLOv11 と YOLOv5 (COCO) はオンラインから入手可能
- Ultralytics 8.4.12 を使用し、Python >= 3.8 / PyTorch >= 1.8 に合わせる
- PyTorch は Mapillary の学習環境に合うバージョンを選択
- 主要依存: PyTorch, Ultralytics, MMDetection, Optuna, Albumentations, FastAPI, CARLA PythonAPI
- CarlaのDocker/バージョン要件は追記予定
- 再学習手順の詳細はリンク先（Faster R-CNN / Ultralytics）に記載された手順に従う

### 依存関係（README にまとめる最小リスト）
- Python >= 3.8, Jupyter
- PyTorch >= 1.8 / torchvision
- Ultralytics 8.4.12
- Blender 4.1.1
- CARLA 0.9.15 + CARLA PythonAPI (pip)
- MMDetection (mmdet, mmcv, mmengine)
- OpenCV, Albumentations, Optuna
- FastAPI, Uvicorn
- NumPy, Pandas, Pillow, Matplotlib, scikit-learn
- LPIPS, scikit-image（Perceptual Similarity）


## 8. 実行例
### DetectorAPI
```
python OpenScience/TargetModel/Detector/DetectorAPI/app.py
```

### DetectorAPI（最適化例）
```
jupyter notebook OpenScience/TargetModel/Detector/test.ipynb
```

### Classifier 推論
```
jupyter notebook OpenScience/traffic_sign_classifiers/infer.ipynb
```

### Perceptual Similarity / User Study（例）
```
jupyter notebook OpenScience/perceptual_similarity.ipynb
```

### ParameterSetting（Specular Tint 推定）
```
jupyter notebook OpenScience/AttackGeneration/1_ParameterSetting/calc_specular_Tint.ipynb
```

### Blender 攻撃生成（例）
```
python OpenScience/AttackGeneration/Hiding/STOP/open_blender2.py
```

### Blender 攻撃生成（Altering/STOP copy の例）
```
PROJECT_ROOT=/path/to/OpenScience \
BLENDER_PATH=/path/to/blender \
BACKGROUND_IMAGE_DIR=/path/to/backgrounds \
OUTPUT_CSV_ROOT=/path/to/output_csv \
python 'AttackGeneration/2_Optimization/1_Altering/STOP copy/open_blender2.py'
```

---

今後の更新予定:
- Carla 実行手順とDocker要件の追記
- フォルダ名匿名化と対応表

## 外部データ・参考リンク
```
Blender 4.1: https://www.blender.org/download/releases/4-1/
BDD100K: https://bair.berkeley.edu/blog/2018/05/30/bdd/
Mapillary Traffic Sign Dataset: https://www.mapillary.com/dataset/trafficsign
ARTS Dataset: https://swshah.w3.uvm.edu/vail/datasets.php
Faster R-CNN training pipeline: https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline
CARLA 0.9.15 (AdditionalMaps): https://github.com/carla-simulator/carla/releases/tag/0.9.15
CARLA Standalone Asset Package: https://carla.readthedocs.io/en/latest/tuto_A_create_standalone/
```
