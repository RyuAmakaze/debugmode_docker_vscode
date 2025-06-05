# Q-LLP
Quantum Learning from Label Proportion

## 実行方法
1. 任意で仮想環境を作成します。
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. 依存パッケージをインストールします。
   ```bash
   pip install torch torchvision qiskit pytest
   ```
3. 学習を実行します。
   ```bash
   python src/run.py
   ```
   学習が完了すると `trained_quantum_llp.pt` が作成されます。
   CUDA が利用可能な環境では自動的に GPU を使用して計算します。

## Docker での実行
1. Docker イメージをビルドします。
   ```bash
   docker build -t q-llp -f Dockerfile/Dockerfile .
   ```
2. コンテナを起動して学習を実行します。
   ```bash
   docker run --rm q-llp
   ```

## テスト
`pytest` を実行してユニットテストを確認できます。
```bash
pytest
```
