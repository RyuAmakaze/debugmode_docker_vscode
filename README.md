# デバッグモード in VScode, SSH, Docker
F5を押したら[runファイル](https://github.com/RyuAmakaze/debugmode_docker_vscode/blob/main/.vscode/tasks.json#L7)をデバッグモードで実行

## Docker build
1. Docker イメージをビルド
   ```bash
   docker build -t q-llp -f Dockerfile/Dockerfile .
   ```
2. 実行 <br>
   これができるまで確認！！<br>
   以下の実行コマンドは[この行](https://github.com/RyuAmakaze/debugmode_docker_vscode/blob/main/.vscode/tasks.json#L7)に対応してます
   ```bash
   sudo docker run --rm --shm-size=2g --gpus all -v $(pwd):/app -w /app q-llp python src/run.py
   ```
