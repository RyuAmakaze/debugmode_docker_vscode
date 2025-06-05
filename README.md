# デバッグモード in VScode, SSH, Docker
F5を押したら[runファイル](https://github.com/RyuAmakaze/debugmode_docker_vscode/blob/main/.vscode/tasks.json#L7)をデバッグモードで実行

## Vscode setting
以下の拡張をインストール<br>
1. Dev Container install (Dockerデバッグモード用)
2. Python, Python Debugger install (ブレイク置く用)

## Docker build
1. Docker イメージをビルド
   ```bash
   docker build -t q-llp -f Dockerfile/Dockerfile .
   ```
2. 実行 <br>
   これができるまで確認！！(アカウントはDockerグループに入れ、sudoなし実行)<br>
   以下の実行コマンドは[この行](https://github.com/RyuAmakaze/debugmode_docker_vscode/blob/main/.vscode/tasks.json#L7)に対応してます
   ```bash
   docker run --rm --shm-size=2g --gpus all -v $(pwd):/app -w /app q-llp python src/run.py
   ```
## うまくいかんとき
ポート番号(ex:5678)既に使ってないかチェックしてください
   ```bash
   docker ps -a | grep 5678
   docker rm -f <該当ID>
   ```

