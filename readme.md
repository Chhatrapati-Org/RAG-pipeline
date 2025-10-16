### check results
Query num = 54, 7044, 7048
req answer = doc_32.txt, doc_4030.txt, 4031.txt
Critical Query nums = _21, +29, 32, 57, 88, _7000, +7001, 7002, 7036, 7039, _7041, 7044, 7048, 7055, +7059, 7080, 7084, 7094

# steps
1. start venv
2. start docker image of qdrant on 6333 port
3. start ollama for evaluation
4. download models from huggingface if needed offline
5. run cli.py with appropriate argument and options

## Installing ollama
[text](https://github.com/ollama/ollama?tab=readme-ov-file#ollama)