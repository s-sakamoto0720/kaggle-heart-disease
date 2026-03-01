# Predicting Heart Disease

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/playground-series-s6e2)

## Overview

| 項目 | 内容 |
|------|------|
| プラットフォーム | Kaggle (Playground Series S6 E2) |
| タスク | 二値分類 |
| 評価指標 | AUC (Area Under the ROC Curve) |
| 参加期間 | 2026/02/01 - 2026/02/28 |
| 最終順位 | 393 / 4371 (Top 9%) |

---

## Problem

患者の年齢、血圧、コレステロール値、心電図結果など13の健康指標から、心臓病の発症確率を予測する二値分類タスク。

データはKaggle Playground Seriesの合成データ（深層学習モデルにより生成）で、元データはCleveland Heart Disease Datasetに基づく。訓練データ63万件、テストデータ27万件。目的変数の分布は Absence 55% / Presence 45% で、ほぼ均等。

特徴量は数値5つ（Age, BP, Cholesterol, Max HR, ST depression）とカテゴリ8つ（Sex, Chest pain type, FBS over 120, EKG results, Exercise angina, Slope of ST, Number of vessels fluro, Thallium）。

---

## Approach

### EDA

まずは基本的なところから。

- 数値特徴量の分布をヒストグラム+KDEで確認。train/val/testで分布のずれがないかチェックした
- カテゴリ特徴量はvalue_countsでクラスの偏りを確認。EKG results=1 がかなり少ない（1,322件）ことに気づいた
- 相関行列のヒートマップを作って、特徴量間の関係を把握
- targetごとの分布を見て、Number of vessels fluro と Thallium が心臓病との関連が強いことを確認

### 特徴量エンジニアリング

特徴量エンジニアリングのコードは以下のノートブックを参考にした。
- [Heart: XGB, LightGBM, CatB baseline, K-fold](https://www.kaggle.com/code/kospintr/heart-xgb-lightgbm-catb-baseline-k-fold)

最終的に86特徴量を作った。ポイントは、trainval全体（63万件）の統計量を**分割前に**辞書として計算しておくこと。これでtestにもリークなく適用できる。

| カテゴリ | 特徴量数 | 内容 |
|---------|---------|------|
| cluster | 5 | KBinsDiscretizer（10ビン, uniform）で離散化 |
| frqc | 8 | カテゴリ特徴量の頻度エンコーディング |
| mean | 13 | 列ごとのtarget平均値 |
| median | 13 | 列ごとのtarget中央値 |
| std | 13 | 列ごとのtarget標準偏差 |
| skew | 13 | 列ごとのtarget歪度 |
| count | 13 | 列ごとのtargetカウント |
| raw categorical | 8 | カテゴリ変数をそのまま（CatBoostのネイティブ処理用） |

ターゲット統計量（mean, median, std, skew, count）は数値5列+カテゴリ8列=13列分を一括で作成。sklearnの`ColumnTransformer`でパイプラインにまとめた。

### モデル

CatBoost単体でのベストスコアを出した。ハイパーパラメータは以下の通り：

```
iterations: 1000
learning_rate: 0.03
depth: 6
eval_metric: AUC
early_stopping_rounds: 50
```

学習設定：
- StratifiedShuffleSplit で trainval を train（567K, 90%）/ val（63K, 10%）に分割
- StratifiedKFold（5-fold, shuffle=True）でCV
- OOF予測でCV-AUCを算出、独立したvalセットでも検証
- CatBoostには`cat_features`パラメータでカテゴリ列のインデックスを指定（ネイティブのターゲットエンコーディングが走る）

### Results

LightGBM、XGBoost、CatBoostの3モデルを同じ特徴量・同じfoldで比較した結果：

| Model | OOF AUC | Val AUC |
|-------|---------|---------|
| LightGBM | 0.955053 | 0.956029 |
| XGBoost | 0.955423 | 0.956498 |
| **CatBoost** | **0.955655** | **0.956643** |
| Stacking (LogReg) | 0.955550 | 0.956587 |
| SimpleAvg | 0.955541 | 0.956530 |

CatBoost単体が最強で、stackingでも超えられなかった。Public LB: **0.95387**。

---

## What I Learned

### 勾配ブースティング決定木はデータセットで優劣が変わる

「LightGBMが速くて強い」とよく聞くけど、今回はCatBoostが一番スコアが高かった。カテゴリ変数をネイティブに扱えるCatBoostの特性がこのデータセットに合っていたのかもしれない。データセットによって最適なモデルは変わるので、とりあえず3つ全部試すのが大事だなと思った。

### Stackingは万能ではない

LightGBM + XGBoost + CatBoost のOOF予測を使ってLogistic Regressionでスタッキングしたが、CatBoost単体のVal AUC 0.956647を超えられなかった（Stacking: 0.956587）。メタモデルの係数を見ると、CatBoostの重みが7.84でほぼCatBoostに依存していた。

振り返ると、メタモデルにLogistic Regressionを使ったのが良くなかったかもしれない。1位の解法ではリッジ回帰を使っていたので、線形モデルも複数試すべきだった。

### 特徴量を増やしても精度は上がらない場合がある

LLMに聞いてドメイン知識ベースの特徴量を大量に作り、合計190個くらいまで増やして回したけどスコアは改善しなかった。SHAP値やgainを見て重要度の低い特徴量を削っても変わらず、Optunaでハイパーパラメータ探索をしても大きな改善はなかった。

### 1位解法から学んだこと

1位の解法を見て衝撃を受けた。1つのモデルに対してパラメータ探索や特徴量を増やすのではなく：

- 特徴量をグループ分けして、**複数種類のモデル**に対してそれぞれ異なる特徴量グループを割り当てる
- モデルごとにハイパーパラメータも分ける
- **Optunaを活用したスタッキング**で最適な重みを探索する

つまり、1つのモデルを極限まで強くするよりも、複数のモデルを用意して特徴量・パラメータを分散させてアンサンブルする方が強い。特に今回のように1つのモデルだけでAUC 0.95を超えるようなタスクでは、単一モデルの改善余地が小さいので、多様性を持たせたアンサンブルの方が効果的なのだと思った。

---

## Notebooks

| No. | ファイル | 内容 |
|-----|---------|------|
| 01 | [01_eda_and_features.ipynb](notebook/01_eda_and_features.ipynb) | EDA・特徴量エンジニアリング |
| 02 | [02_modeling.ipynb](notebook/02_modeling.ipynb) | CatBoost 5-fold CV 学習・評価 |
