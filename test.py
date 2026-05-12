"""诊断脚本：定位 step6 NaN 首次出现的层"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

# 强制使用 torch_dtype（修复 dtype bug）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # 注意：不是 dtype
    device_map="auto",
    attn_implementation="eager",
    trust_remote_code=True,
    cache_dir="./models",
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir="./models")

text = "What is 2+2? Answer:"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Hook 逐层检测
def make_hook(layer_idx):
    def hook(module, input, output):
        # output[0] 是 hidden_states
        h = output[0] if isinstance(output, tuple) else output
        if torch.isnan(h).any():
            print(f"  Layer {layer_idx} OUTPUT: NaN detected! {torch.isnan(h).sum().item()} NaN values")
        return output
    return hook

for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(make_hook(i))

print(f"Model dtype: {model.dtype}")
print(f"Running forward pass...")
with torch.no_grad():
    out = model(**inputs, output_attentions=True)

logits = out.logits[0, -1, :]
nan_count = torch.isnan(logits).sum().item()
print(f"\nFinal logits: {nan_count}/{logits.numel()} NaN")
if nan_count > 0:
    print("NaN persists even with bfloat16 + eager attention!")
else:
    print("No NaN - bfloat16 fixed the issue!")