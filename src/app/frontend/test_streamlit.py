import ast
import json

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from peft import PeftModel


@st.cache_resource()
def load_model():
    MODEL_ID = 'Kiffaz11/qwen2.5-3b-absa-lora'  # Cho tokenizer và LoRa
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-3B-Instruct',
        torch_dtype="auto",
        device_map='balanced',
    )

    model = PeftModel.from_pretrained(base_model, MODEL_ID)
    model = model.merge_and_unload()
    model.eval()

    return tokenizer, model


@torch.no_grad()
def generate_response_causallm(review, tokenizer, model, ):
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

    candidate = s[start: end + 1].strip()

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



def sentiment_color(s):
    if "TÍCH" in s:
        return "green"
    if "TIÊU" in s:
        return "red"
    return "gray"


# Streamlit Demo

st.title("Aspect-Based Sentiment Analysis")

review_input = st.text_area(
    "Nhập review sản phẩm:",
    height=150,
    placeholder="Ví dụ: Pin tốt, máy mạnh..."
)

if st.button("Phân tích"):
    if not review_input.strip():
        st.warning("Vui lòng nhập Review!")
    else:
        tokenize, model = load_model()

        with st.spinner("Đang phân tích..."):
            raw_output = generate_response_causallm(review_input, tokenize, model)
            parsed_output = parse_model_json(raw_output)

        print(parsed_output)
        st.subheader("Kết quả phân tích")

        st.write("")
        st.write("")

        # Cách hiển thị 1
        header_cols = st.columns(4)
        header_titles = ["Aspect", "Category", "Sentiment", "Opinion"]

        for col, title in zip(header_cols, header_titles):
            col.markdown(f"**{title}**")
        st.divider()

        for i, (term, category, sentiment, opinion) in enumerate(parsed_output):
            color = sentiment_color(sentiment)

            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.write(term)
                with c2:
                    st.write(category)
                with c3:
                    st.markdown(
                        f"<span style='color:{color}; font-weight:bold'>{sentiment}</span>",
                        unsafe_allow_html=True
                    )
                with c4:
                    st.write(opinion)
