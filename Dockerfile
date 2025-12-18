# Use the existing vLLM GH200 OpenAI compatible server image as base
FROM drikster80/vllm-gh200-openai:latest

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    datasets \
    evaluate \
    tree-sitter-language-pack==0.13.0 \
    codebleu==0.7.0 --no-deps \
    matplotlib \
    multiprocess \
    scikit-learn