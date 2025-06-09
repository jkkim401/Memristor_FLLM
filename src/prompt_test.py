import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

PROMPT_TEMPLATE = (
    "You are a medical artificial intelligence assistant. "
    "You directly diagnose patients based on the provided information to assist a doctor in his clinical duties. "
    "Your goal is to correctly diagnose the patient. Based on the provided information you will provide a final diagnosis of the most severe pathology. "
    "Don't write any further information. Give only a single diagnosis. "
    "{fewshot_examples} "
    "Provide the most likely final diagnosis of the following patient. {input} {diagnostic_criteria} "
    "Final Diagnosis:"
)

def prompt_test(model, tokenizer, test_dataset, fewshot_examples="", diagnostic_criteria="", device="cuda"):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    references = []

    for sample in tqdm(test_dataset):
        # sample이 dict 또는 pandas.Series일 수 있음
        if isinstance(sample, dict):
            text = sample["text"]
            gold_diagnosis = sample["diagnosis"]
        else:
            text = sample.text
            gold_diagnosis = sample.diagnosis
        # 프롬프트에서 진단명 부분 제거
        prompt_input = re.split(r'The patient was diagnosed with', text)[0].strip()

        prompt = PROMPT_TEMPLATE.format(
            fewshot_examples=fewshot_examples,
            input=prompt_input,
            diagnostic_criteria=diagnostic_criteria
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "Final Diagnosis:" in output_text:
            pred = output_text.split("Final Diagnosis:")[-1].strip().split("\n")[0]
        else:
            pred = output_text.strip().split("\n")[0]

        predictions.append(pred)
        references.append(gold_diagnosis)
        if pred.lower() == gold_diagnosis.lower():
            correct += 1
        total += 1

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    accuracy = accuracy_score([l.lower() for l in references], [p.lower() for p in predictions])
    f1 = f1_score([l.lower() for l in references], [p.lower() for p in predictions], average='macro')
    precision = precision_score([l.lower() for l in references], [p.lower() for p in predictions], average='macro', zero_division=0)
    recall = recall_score([l.lower() for l in references], [p.lower() for p in predictions], average='macro', zero_division=0)

    print(f"Prompt-based Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Confusion matrix 출력
    labels = sorted(list(set([l.lower() for l in references] + [p.lower() for p in predictions])))
    cm = confusion_matrix([l.lower() for l in references], [p.lower() for p in predictions], labels=labels)
    print("Confusion Matrix:")
    print(cm)

    return predictions, references, accuracy, precision, recall, f1, cm, labels

def save_results(predictions, references, accuracy, precision, recall, f1, cm=None, labels=None, save_path="results/prompt_test_results.csv"):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame({
        "prediction": predictions,
        "reference": references
    })
    df.to_csv(save_path, index=False)
    # 지표는 별도 파일에 저장
    with open(save_path.replace('.csv', '_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
    print(f"결과가 {save_path} 및 metrics.txt로 저장되었습니다.")
    # confusion matrix 저장
    if cm is not None and labels is not None:
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_path = save_path.replace('.csv', '_confusion_matrix.csv')
        cm_df.to_csv(cm_path, encoding='utf-8')
        print(f"Confusion matrix가 {cm_path}로 저장되었습니다.")

# --- main 실행 예시 ---
if __name__ == "__main__":
    # test set 불러오기
    test_df = pd.read_csv('data/test_set_for_prompt.csv')

    # 1. 파인튜닝 전(pretrained) 모델 테스트
    pretrained_model_dir = "bitnet-b1.58-2B-4T-bf16"  # 예: "facebook/opt-1.3b" 또는 "bitnet-b1.58-2B-4T-bf16"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    preds, refs, acc, prec, rec, f1, cm, labels = prompt_test(model, tokenizer, test_df.itertuples(index=False), device=device)
    save_results(preds, refs, acc, prec, rec, f1, cm, labels, save_path="results/pretrained_prompt_test_results.csv")

    # 2. 파인튜닝 후(finetuned) 모델 테스트
    finetuned_model_dir = "results/finetuned_model"
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_dir)
    model = AutoModelForCausalLM.from_pretrained(finetuned_model_dir)
    model.to(device)
    preds, refs, acc, prec, rec, f1, cm, labels = prompt_test(model, tokenizer, test_df.itertuples(index=False), device=device)
    save_results(preds, refs, acc, prec, rec, f1, cm, labels, save_path="results/finetuned_prompt_test_results.csv")