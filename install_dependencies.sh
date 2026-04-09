#!/bin/bash
set -e

echo "🔥 开始精确依赖安装..."

# 1. 首先升级pip到最新版（解决依赖解析问题）
pip install --upgrade pip==26.0.1

# 2. 按顺序安装核心依赖
echo "📦 安装核心依赖..."
pip install --no-cache-dir --prefer-binary --only-binary=:all: numpy==1.23.5
pip install --no-cache-dir --prefer-binary --only-binary=:all: spacy==3.5.4 thinc==8.1.10

# 3. 安装模型
echo "🧠 安装spacy模型..."
pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl

# 4. 强制安装Streamlit依赖（关键！）
echo "🚀 安装Streamlit及兼容依赖..."
pip install --no-cache-dir --prefer-binary --only-binary=:all: streamlit==1.28.0
pip install --no-cache-dir --prefer-binary --only-binary=:all: rich==13.7.1 markdown-it-py==2.2.0 mdurl==0.1.2 pygments==2.15.1

# 5. 安装其他依赖
echo "📦 安装其他依赖..."
pip install --no-cache-dir --prefer-binary --only-binary=:all: pandas==1.5.3 requests==2.31.0 plotly==5.18.0 altair==4.2.2

# 6. 验证安装
echo "✅ 验证依赖版本..."
pip list | grep -E "(streamlit|rich|numpy|spacy|thinc)"
echo "✅ 依赖安装完成！"
