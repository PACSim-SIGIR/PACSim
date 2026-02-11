## PACSim（论文发布版代码整理）

本仓库是 PACSim 的实验代码（用于论文复现）。为了便于匿名评审/复现，本 README 约定：

- **不要把 API Key 写进 YAML/代码**（用环境变量或命令行参数注入）
- **不要依赖个人绝对路径**（示例配置均使用相对路径或环境变量）
- **不随代码包上传数据/日志/模型权重**（提供生成与放置说明）

### 1) 安装

建议使用 conda（与你的 `HiSim` 环境一致）或直接 pip：

```bash
pip install -r requirements.txt
```

### 2) 环境变量（LLM / 数据）

当需要在线调用 LLM API（例如 `hisim_social.yaml`）时：

```bash
export TOGETHER_API_KEY="YOUR_KEY"
export HISIM_DATA_ROOT="/path/to/HiSim/data"
```

说明：
- `TOGETHER_API_KEY`：OpenAI-compatible key（脚本会从环境变量展开 `${TOGETHER_API_KEY}`）
- `HISIM_DATA_ROOT`：HiSim 原始数据目录（包含 `hisim_with_tweet/` 与 `user_data/`）

### 3) 在线社交仿真（LLM 生成 action JSON + post_text）

```bash
python src/train.py --config src/config/hisim_social.yaml
```

该配置会调用 executor LLM 生成 JSON：
`{"action_type","stance_id","post_text"}`，并在环境内解析/兜底后参与仿真。
