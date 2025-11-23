FINAL_TRAIN_OUTPUT_FILE="final_output/train.txt"
with open(FINAL_TRAIN_OUTPUT_FILE, "w", encoding="utf-8") as f:

    for i in range(1, 11):
        file_path = f"""output/atoss_sft_dataset_{i}.txt"""
        with open(file_path, "r", encoding="utf-8") as fr:
            data = fr.read()
            f.write(data)

