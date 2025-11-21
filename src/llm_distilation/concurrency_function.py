import os
import json
import time
import re

from datasets import load_dataset
from together import Together
from dotenv import load_dotenv

unicode_pattern = re.compile(r'\\u[0-9a-zA-Z]{4}')

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

INPUT_DATASET = "vohuutridung/3190-data"

load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

def create_generate_prompt(sentence, aspects):
    prompt = f"""
You are a sentence splitting expert. You will be provided with a review sentence and a few [aspect, category, sentiment, opinion] quadruplets from that review sentence. Here is the definition of each element in the quadruplet:
- The ‘aspect’ refers to a specific feature, attribute, or aspect of a product or service that a user may express an opinion about. The aspect term might be ‘null’ for an implicit aspect.
- The ‘opinion’ refers to the sentiment or attitude expressed by a user towards a particular aspect or feature of a product or service. The opinion term might be ‘null’ for an implicit opinion.
- The ‘category’ refers to the category that the aspect belongs to (e.g. food quality, restaurant general, etc.).
- The ‘sentiment’ refers to the sentiment class of the aspect (e.g. positive, negative, neutral).

You need to split the sentence into shorter sentences such that each short sentence contains one aspect term. When splitting, sentences connected by conjunctions must be divided into individual sentences along with their conjunctions. This process must specify the subject in every sentence. This process must retain the existing spellings exactly as in the original sentence. This process must also retain the existing spacings exactly as in the original sentence. If the sentence is too short to split or does not need to be split, use the original sentence as is. No numbering, line breaks, or explanations are needed.

ORIGINAL SENTENCE:
{sentence}

ASPECT TERMS:
{aspects}
"""

    return prompt


def create_filter_prompt(sentence, aspects, candidates, K):

    prompt = f"""
You are a strict evaluator of Aspect-Term-Oriented Sentence Splitting (ATOSS).

Your task:
Given:
- the ORIGINAL sentence,
- ASPECT terms,
- 10 CANDIDATE split versions S′ (each S′ is a SINGLE STRING containing several shorter sentences),

Select EXACTLY {K} BEST versions that follow ALL splitting rules.

A valid split version S′ MUST satisfy:

RULES:
1. S′ must be ONE SINGLE STRING that includes several shorter sentences.
2. Each shorter sentence MUST contain EXACTLY ONE aspect term.
3. All spellings must match the original EXACTLY (no substitutions).
4. All spacing must match the original EXACTLY (no extra/missing spaces).
5. No rewriting, no paraphrasing, no synonym replacements.
6. No missing content and no added content.
7. No reordering of any part of the original sentence.
8. Every shorter sentence MUST contain an explicit subject.
9. Conjunctions ("and", "or", "but", commas) may appear ONLY if they appear in the original.

INVALID candidates should be discarded:
- If any sentence has zero aspects or more than one → invalid.
- If spelling/spacing changes → invalid.
- If subject is missing → invalid.
- If content is removed, merged, or reordered → invalid.

------------------------------------------
### EXAMPLES OF CORRECT S′ FORMAT (from ATOSS paper)

Correct S′ example:
very immature bartender, didnt know how to make specific drinks. service was so slowwwww. the food was not fresh or warm. waitresses were busy flirting with men at the bar and werent very attentive to all the customers .

Another valid S′:
i swore never to return for a warm beer. i swore never to return for a mediocre meal.

------------------------------------------

OUTPUT REQUIREMENT:
- Return EXACTLY {K} valid S′ versions.
- Each version on its own line.
- NO JSON, NO numbering, NO markdown, NO explanation.

ORIGINAL SENTENCE:
{sentence}

ASPECT TERMS:
{aspects}

CANDIDATES SPLITS:
{json.dumps(candidates, indent=2)}
"""

    return prompt

def generate_splits(sentence, aspects):

    prompt = create_generate_prompt(sentence, aspects)

    versions = []

    for i in range(10):

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1.0
        )

        text = response.choices[0].message.content
        versions.append(text.strip().replace("\n", ". "))

    print("first_raw_output", versions)
    if len(versions) != 10:
        print(f"WARNING: Gemini did not return 10 versions, only {len(versions)} versions")
        print("Versions: ", versions)
        return versions

    return versions

def filter_split(sentence, aspects, candidates, K=2):

    prompt = create_filter_prompt(sentence, aspects, candidates, K)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0
    )

    text = response.choices[0].message.content
    selections = [line.strip() for line in text.split("\n") if line.strip() != ""]

    if len(selections) != K:
        print(f"WARNING: Gemini did not return {K} selections:", len(selections))
        print("Selections: ", selections)
        return selections

    print("Selections: ", selections)
    return selections


unicode_pattern = re.compile(r'\\u[0-9a-zA-Z]{4}')
def safe_decode(s: str):
    if not unicode_pattern.search(s):
        return s

    try:
        return s.encode('utf-8').decode('unicode_escape')
    except:
        return s

def build_dataset(start, end, OUTPUT_FILE, OUTPUT_RAW_FILE):

    ds = load_dataset(INPUT_DATASET, split="train")
    len_ds = len(ds)
    count_process = 0

    fout_raw = open(OUTPUT_RAW_FILE, "a", encoding="utf-8")
    fout = open(OUTPUT_FILE, "a", encoding="utf-8")

    # batch = ds[start:end]
    batch = ds.select(range(start, end))

    for row in batch:

        if count_process == 200:
            break

        start_time = time.time()
        count_process += 1
        print("count:", count_process)

        sentence = row["text"]
        aspects = row["labels"]

        # Step 1: generate 10 s'
        candidates = generate_splits(sentence, aspects)
        if not candidates:
            print(f"This {sentence} cant be processed in generate_splits function")
            continue
        for s_raw_out in candidates:
            s_raw_out = safe_decode(s_raw_out)
            fout_raw.write(sentence + "####" + s_raw_out + "\n")
            fout_raw.flush()


        # Step 2: Select K s'
        best = filter_split(sentence, aspects, candidates, 2)
        if not best:
            print(f"This {sentence} cant be processed in filter_split function")
            continue

        # Step 3: Export
        for s_output in best:
            # print("Output 1: ", s_output)
            s_output = safe_decode(s_output)
            # print("Output 2: ", s_output)
            fout.write(sentence + "####" + s_output + "\n")
            fout.flush()

        end_time = time.time()
        process_time = end_time - start_time
        print(f"process_time: {process_time}")

        if count_process % 100 == 0:
            print(f"Processed {count_process}/{len_ds} sentences")


    fout.close()
    fout_raw.close()
    print("Process successfully")
