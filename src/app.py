import ast
import json
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Cache load model
@st.cache_resource()
def load_model():
    MODEL_ID = 'vohuutridung/vit5-large-absa'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        # device_map='auto',
    ).to(device)
    model.eval()

    return tokenizer, model

@st.cache_resource()
def load_split_model():
    VIATOSS_ID = 'vohuutridung/vit5-base-split-sft-cpo'
    viatoss_tokenizer = AutoTokenizer.from_pretrained(VIATOSS_ID, use_fast=False)
    viatoss_model = AutoModelForSeq2SeqLM.from_pretrained(
        VIATOSS_ID,
        torch_dtype="auto",
        device_map='auto',
    )

    viatoss_model.eval()

    return viatoss_tokenizer, viatoss_model


### Core logic for app
@torch.no_grad()
def generate_response_seq2seqlm(review, tokenizer, model):
    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=256,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

CATEGORIES = {
    "TỔNG_QUAN","PIN","HIỆU_NĂNG","MÁY_ẢNH","MÀN_HÌNH",
    "GIÁ_CẢ","TÍNH_NĂNG","THIẾT_KẾ","DỊCH_VỤ&PHỤ_KIỆN","LƯU TRỮ"
}
SENTIMENT = {"TÍCH_CỰC","TIÊU_CỰC","TRUNG_LẬP"}
QUAD_RE = re.compile(r"\[\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*'([^']*)'\s*\]")

def parse_model_json(raw_text):
    """
    Robust parser for ViT5 / seq2seq ABSA output.
    - Ignores broken / truncated quadruples
    - Extracts only fully-formed quadruples
    - NEVER hallucinates fields
    """
    if not raw_text:
        return []
    s = str(raw_text)

    results = []
    for a, c, se, o in QUAD_RE.findall(s):
        a, c, se, o = a.strip(), c.strip(), se.strip(), o.strip()

        if c not in CATEGORIES:
            continue
        if se not in SENTIMENT:
            continue

        results.append([a, c, se, o])

    return results

def sentiment_color(s):
    if "TÍCH" in s:
        return "green"
    if "TIÊU" in s:
        return "red"
    return "gray"

@torch.no_grad()
def split(reviews, tokenizer, model):
    inputs = tokenizer(
        reviews,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_length=256,
    )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

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
        viatoss_tokenizer, viatoss_model = load_split_model()

        with st.spinner("Đang phân tích..."):
            split_output = split(review_input, viatoss_tokenizer, viatoss_model)
            raw_output = generate_response_seq2seqlm(review_input, tokenize, model)
            parsed_output = parse_model_json(raw_output)

        print(parsed_output)

        st.subheader("Kết quả tách câu từ Vi-ATOSS")

        st.write("")
        st.write("")

        for i, sent in enumerate(split_output, 1):
            st.markdown(
                # f"<span style='font-size:20px'>- <b>Câu {i}:</b>&nbsp;&nbsp;&nbsp;&nbsp;{sent}</span>",
                f"<span style='font-size:20px'>{sent}</span>",
                unsafe_allow_html=True
            )
        print("Split output: ", split_output)
        st.write("")
        st.write("")

        st.subheader("Kết quả phân tích từ mô hình ABSA")

        st.write("")
        st.write("")

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
