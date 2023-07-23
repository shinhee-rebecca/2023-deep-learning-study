## IT ISSUE

### Intellij AI Assisstant
https://blog.jetbrains.com/ko/idea/2023/06/ai-assistant-in-jetbrains-ides/

### Llama 2 AI on Kubernetes
[License](https://ai.meta.com/llama/license/)

```
cat > values.yaml <<EOF
replicas: 1
deployment:
  image: quay.io/chenhunghan/ialacol:latest
  env:
    DEFAULT_MODEL_HG_REPO_ID: TheBloke/Llama-2-13B-chat-GGML
    DEFAULT_MODEL_FILE: llama-2-13b-chat.ggmlv3.q4_0.bin
    DEFAULT_MODEL_META: ""
    THREADS: 8
    BATCH_SIZE: 8
    CONTEXT_LENGTH: 1024
service:
  type: ClusterIP
  port: 8000
  annotations: {}
EOF
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install llama-2-13b-chat ialacol/ialacol -f values.yaml
```

```
kubectl port-forward svc/llama-2-13b-chat 8000:8000
```

```
curl -X POST -H 'Content-Type: application/json' \
  -d '{ "messages": [{"role": "user", "content": "Hello, are you better then llama version one?"}], "temperature":"1", "model": "llama-2-13b-chat.ggmlv3.q4_0.bin"}' \
  http://localhost:8000/v1/chat/completions
```

Llama 13B 모델을 사용하기 위해서는 10GB pvc와 16vCPU를 사용할것을 권장.

https://github.com/chenhunghan/ialacol/tree/main