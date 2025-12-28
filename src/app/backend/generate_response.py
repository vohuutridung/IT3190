import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
import torch
import ast
from peft import PeftModel

# ================================
# 1. Chọn thiết bị chạy (GPU / CPU)
# ================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =====================================================
# 2. Load tokenizer + base model + LoRA adapter
# =====================================================
MODEL_ID = 'Kiffaz11/qwen2.5-3b-absa-lora' # Cho tokenizer và LoRa


# Tokenizer dùng để:
# - chuyển text → token id
# - áp dụng chat template
eval_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


# Load model nền (base model)
base_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B-Instruct',
    torch_dtype="auto",
    device_map='balanced',
)


# Gắn LoRA weights vào base model
pre_eval_model = PeftModel.from_pretrained(base_model, MODEL_ID)


# Gộp LoRA vào model thật → inference nhanh hơn
eval_model = pre_eval_model.merge_and_unload()
eval_model.eval()



# =====================================================
# 3. Hàm sinh output từ LLM (ABSA)
# =====================================================
@torch.no_grad()
def generate_response_causallm(review, tokenizer, model,):
    SYSTEM_PROMPT = (
        "Bạn là hệ thống trích xuất khía cạnh và cảm xúc (Aspect-Based Sentiment Analysis) cho review sản phẩm.\n"
        "Nhiệm vụ: từ một đoạn review tiếng Việt, trích xuất các mục theo định dạng JSON:\n"
        '[[ "aspect_term", "aspect_category", "sentiment", "opinion_phrase" ], ...]\n'
        "Quy tắc:\n"
        "- Chỉ trả về JSON thuần (không giải thích, không markdown).\n"
        "- Giữ nguyên nhãn như dữ liệu (ví dụ: TỔNG_QUAN, PIN, ...; TÍCH_CỰC/TIÊU_CỰC/TRUNG_LẬP).\n"
        '- Nếu thiếu aspect_term hoặc opinion_phrase, dùng chuỗi "NULL" đúng như dữ liệu.\n'
    )

    USER_TEMPLATE = (
        "Hãy trích xuất các khía cạnh và cảm xúc từ review sau.\n"
        "Review:\n"
        "{text}\n\n"
        "Chỉ trả về JSON theo đúng format yêu cầu."
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=(review or "").strip())},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    input_len = inputs["attention_mask"].sum().item()

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_part = gen_ids[0][input_len:]
    response = tokenizer.decode(gen_part, skip_special_tokens=True)

    return response

# Ham de parse JSON tu text model tra ve
def parse_model_json(raw_text):
    if raw_text is None:
        return []
    s = str(raw_text).strip()

    # Find JSON array boundaries
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []

    candidate = s[start : end + 1].strip()

    # Try strict JSON
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, list) else []
    except Exception:
        pass

    # Try python literal fallback
    try:
        obj = ast.literal_eval(candidate)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []
    


text = "Máy bị lỗi mạng ko sử dụng dc 4g lúc mới mua thì gần 5tr giờ rớc giá quá nhiều ."
response = generate_response_causallm(text, eval_tokenizer, eval_model)
print("Raw model response:", response)
parsed = parse_model_json(response)
print("Parsed response:", parsed)
