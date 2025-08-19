1. Clone repo

git clone <your-repo-url> ai-poc-gpt
cd ai-poc-gpt


2. Create Python env & install (CPU-only Free Tier)

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # (create with packages below)
# or install minimal packages:
pip install transformers datasets sentencepiece accelerate peft fastapi uvicorn


(If you have a GPU and want bitsandbytes support, install bitsandbytes and torch with CUDA wheels.)

3. Quick local fine-tune on CPU (toy run)

python training/finetune_lora_minimal.py --config configs/train_small.yaml --data data/sample_train.jsonl


This will produce an adapter-saved model under ./outputs/distilgpt2_lora.

4. Run FastAPI server (management t2.micro or your laptop)

uvicorn inference.server_fastapi:app --host 0.0.0.0 --port 8080
# then POST to http://<host>:8080/generate with JSON {"prompt":"Hello","max_new_tokens":32}


5. Optional: Terraform deploy minimal infra (management EC2 + S3)

cd infra/terraform
# edit variables.tf or override via -var flags (replace s3 bucket with unique name)
terraform init
terraform apply
# follow prompts; after apply you'll get management_public_ip and s3_bucket outputs

SSH into management instance with your key, pull repo, and run the same steps there.

6. Optional cheap GPU spot (on-demand spot for experimentation)

Launch a g4dn.xlarge or p3.2xlarge spot via AWS console or use the EC2 launch wizard; attach your key.
SSH in, pull repo, build Docker image for GPU (use appropriate CUDA base image), and run training/finetune_lora_minimal.py with device_map="auto".

7. Tuning for <$50/mo

Only boot spot GPU when needed and shut it down afterwards.
Leave t2.micro always-on for orchestration (free tier).
Store small artifacts in S3; delete large checkpoint files after experiments.


8. requirements.txt (minimal)

requirements.txt

transformers==4.40.0
datasets
sentencepiece
peft==0.6.0
accelerate
fastapi
uvicorn
PyYAML


(Install bitsandbytes / torch with specific CUDA wheels only on GPU-capable instances.)

9. Final notes & safety

This scaled setup is explicitly for education, PoC, and small-scale fine-tuning. It will not produce production-grade LLMs — but it mirrors the workflows: tokenization → fine-tuning (LoRA) → merge/quantize → serving → eval → safety checks.

Replace s3_bucket default with a globally unique name before running Terraform.

On Free Tier AWS accounts, verify service limits (EC2 quotas, keypairs, regions). Spot instances may require quota increases for GPU instance types — use them sparingly.