# デバッグモード in VScode, SSH, Docker

## Docker build
1. Docker イメージをビルド
   ```bash
   docker build -t q-llp -f Dockerfile/Dockerfile .
   ```
2. 実行
   これができるまで確認！！
   ```bash
   sudo docker run --rm --shm-size=2g --gpus all -v $(pwd):/app -w /app q-llp python src/run.py
   ```

## テスト
`pytest` を実行してユニットテストを確認できます。
```bash
pytest
```
