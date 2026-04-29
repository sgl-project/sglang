# Devlog: RelayKV SGLang MVP-0 args / skeleton 着手

## 日付

2026-04-29

## 対象

SGLang main から切り直した新ブランチで、RelayKV MVP-0 の最初の実装単位に入る。

## 方針

MVP-0 は shadow-only とする。

つまり、SGLang の起動引数と `python/sglang/srt/relaykv/` の最小 skeleton だけを追加し、実KV tensor、attention kernel、scheduler挙動はまだ変更しない。

## 追加するもの

### server args

- `--enable-relaykv`
- `--relaykv-mode {off,shadow}`
- `--relaykv-resident-budget-tokens`
- `--relaykv-recent-window`
- `--relaykv-anchor-pages`
- `--relaykv-log-interval`

### skeleton

```text
python/sglang/srt/relaykv/
  __init__.py
  config.py
  planner.py
  metrics.py
  profile.py
```

## まだやらないこと

- KV tensor の移動
- CPU cold backup
- GPU resident mapping
- attention差し替え
- scheduler制御
- full KV fallback

## 確認観点

- `server_args.py` の dataclass field と `add_cli_args` の順序が崩れていないこと
- `python -m compileall python/sglang/srt/relaykv` が通ること
- `--enable-relaykv --relaykv-mode shadow` 付きで起動引数として受理されること
- 通常生成が壊れていないこと

## 次の作業

1. `apply_relaykv_mvp0_args_and_skeleton.sh` を SGLang repo root で実行する。
2. `git diff` で差分を確認する。
3. `compileall` と server help を確認する。
4. 問題なければ `Add RelayKV shadow planner skeleton` としてコミットする。
5. 次コミットで runtime hook から shadow plan log を出す。
