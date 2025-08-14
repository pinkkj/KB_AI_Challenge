# -*- coding: utf-8 -*-
import os, json, re, random, math
import numpy as np
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    LlamaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

# =========================
# 0) 기본 설정 / 재현성
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# A100/H100에서 속도↑(수치 안정성 크게 해치지 않음)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =========================
# 1) 데이터 로드
# =========================
LABELS = ["분노", "기쁨", "슬픔", "불안", "상처", "당황", "중립"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

train_df_bal = pd.read_excel("./datasets/train_balanced_v3.xlsx")
val_df = pd.read_excel("./datasets/val.xlsx")
test_df = pd.read_excel("./datasets/test.xlsx")

def to_hf(ds: pd.DataFrame):
    # Sentence -> text, Emotion -> label_str
    dset = Dataset.from_pandas(
        ds[["Sentence", "Emotion"]].rename(columns={"Sentence": "text", "Emotion": "label_str"}),
        preserve_index=False
    )
    return dset.map(lambda x: {"label": label2id[x["label_str"]]})

raw_dset = DatasetDict({
    "train": to_hf(train_df_bal),
    "validation": to_hf(val_df),
    "test": to_hf(test_df),
})

# =========================
# 2) 모델/토크나이저
# =========================
MODEL_NAME = "openlm-research/open_llama_3b"   # 필요시 교체
MAX_LEN = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tok_fn(batch):
    out = tokenizer(batch["text"], padding=False, truncation=True, max_length=MAX_LEN)
    out["labels"] = batch["label"]
    return out

tokenized = raw_dset.map(tok_fn, batched=True, remove_columns=["text", "label_str", "label"])
collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

# 하드웨어 감지 (bf16/fp16/optimizer)
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8  # A100/H100 계열
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
use_fused = ("A100" in device_name) or ("H100" in device_name)
fp16 = (not use_bf16) and torch.cuda.is_available()
bf16 = use_bf16
optim_name = "adamw_torch_fused" if use_fused else "adamw_hf"

print("모델 로드 전")
# 모델 로드
model = LlamaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32),
)
print("모델 로드 후")
# 분류 문제 타입 명시
model.config.problem_type = "single_label_classification"

# 메모리 최적화 옵션
model.config.use_cache = False      # gradient checkpointing과 충돌 방지
model.gradient_checkpointing_enable()

# =========================
# 3) 메트릭
# =========================
import evaluate
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        "macro_f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],  # <-- 키 이름 통일
    }
print("step 3 완료")
# =========================
# 4) 스텝 자동 계산
# =========================
def auto_steps(n_train_examples, per_device_bsz, grad_acc, n_gpus=1, evals_per_epoch=3):
    global_bsz = per_device_bsz * grad_acc * max(1, n_gpus)
    steps_per_epoch = max(1, math.floor(n_train_examples / global_bsz))
    step = max(50, steps_per_epoch // max(1, evals_per_epoch))  # 너무 촘촘하지 않게 하한 50
    return step, steps_per_epoch

N_TRAIN = len(tokenized["train"])
PER_DEVICE_BSZ = 8      # 좋은 GPU면 16까지 시도 가능 (OOM 나면 8)
GRAD_ACC = 2            # 전역 배치 = bsz * acc * n_gpus
EVAL_STEPS, STEPS_PER_EPOCH = auto_steps(N_TRAIN, PER_DEVICE_BSZ, GRAD_ACC, n_gpus=torch.cuda.device_count())
SAVE_STEPS = EVAL_STEPS
print(f"[INFO] steps/epoch ≈ {STEPS_PER_EPOCH}, eval_steps={EVAL_STEPS}, save_steps={SAVE_STEPS}")
print(f"[INFO] device: {device_name}, bf16={bf16}, fp16={fp16}, optim={optim_name}")
print("step 4 완료")
# =========================
# 5) 학습 인자
# =========================
OUTPUT_DIR = "./llama-emotion-clf"
LR = 2e-5
EPOCHS = 3

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,

    # 주요 HP
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BSZ,
    per_device_eval_batch_size=PER_DEVICE_BSZ,
    gradient_accumulation_steps=GRAD_ACC,
    weight_decay=0.05,
    max_grad_norm=1.0,

    # 스케줄러/워밍업
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    # 평가/저장 주기
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    logging_steps=100,

    # 베스트 모델 로드
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",   # compute_metrics의 키와 일치
    greater_is_better=True,

    # 정밀도/최적화기
    fp16=fp16,
    bf16=bf16,
    optim=optim_name,

    # 기타
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    report_to="none",
    seed=SEED,
)
print("step 5 완료")
# =========================
# 6) Trainer
# =========================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
print("step 6 완료")
# =========================
# 7) 학습/평가/저장
# =========================
train_result = trainer.train()
print("Best checkpoint:", trainer.state.best_model_checkpoint)

print("\n[Evaluate: TRAIN]")
print(trainer.evaluate(tokenized["train"]))

print("\n[Evaluate: VAL]")
print(trainer.evaluate(tokenized["validation"]))

print("\n[Evaluate: TEST]")
print(trainer.evaluate(tokenized["test"]))

final_dir = os.path.join(OUTPUT_DIR, "final")
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)
print("step 7/학습 완료")
# =========================
# (옵션) 8) Optuna 스윕으로 튜닝 (개선판)
# =========================
print("step 8 튜닝 시작 완료")
DO_TUNE = True

if DO_TUNE:
    import copy, optuna

    # 각 trial 로그 적재용
    trial_logs = []

    orig_sd = copy.deepcopy(model.state_dict())

    def build_args(trial):
        lr = trial.suggest_float("learning_rate", 8e-6, 4e-5, log=True)
        wd = trial.suggest_float("weight_decay", 0.01, 0.08)
        warmup = trial.suggest_float("warmup_ratio", 0.05, 0.2)
        sched = trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts"])
        grad_norm = trial.suggest_float("max_grad_norm", 0.6, 1.5)
        epochs = trial.suggest_categorical("num_train_epochs", [2, 3, 4])
        patience = trial.suggest_categorical("early_stopping_patience", [2, 3])

        a = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/optuna-trial-{trial.number}",
            overwrite_output_dir=True,
            learning_rate=lr,
            weight_decay=wd,
            warmup_ratio=warmup,
            lr_scheduler_type=sched,

            per_device_train_batch_size=PER_DEVICE_BSZ,
            per_device_eval_batch_size=PER_DEVICE_BSZ,
            gradient_accumulation_steps=GRAD_ACC,
            num_train_epochs=epochs,
            max_grad_norm=grad_norm,

            evaluation_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=1,

            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,

            fp16=fp16,
            bf16=bf16,
            optim=optim_name,
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            report_to="none",
            seed=SEED,
        )
        return a, patience

    def objective(trial):
        # 모델 초기화
        model.load_state_dict(orig_sd)

        a, patience = build_args(trial)
        t = Trainer(
            model=model,
            args=a,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )

        # 학습
        t.train()

        # 평가(Trainer.evaluate는 eval_ 접두사를 붙여 반환)
        train_metrics = t.evaluate(tokenized["train"])
        val_metrics = t.evaluate(tokenized["validation"])

        # 콘솔 로그
        print(
            f"[Trial {trial.number}] "
            f"Train Acc: {train_metrics['eval_accuracy']:.4f}, "
            f"Train F1: {train_metrics['eval_macro_f1']:.4f} | "
            f"Val Acc: {val_metrics['eval_accuracy']:.4f}, "
            f"Val F1: {val_metrics['eval_macro_f1']:.4f}"
        )

        # Optuna 내부에도 저장(나중에 UI에서 보기 편함)
        trial.set_user_attr("train_acc", train_metrics["eval_accuracy"])
        trial.set_user_attr("train_f1", train_metrics["eval_macro_f1"])
        trial.set_user_attr("val_acc", val_metrics["eval_accuracy"])
        trial.set_user_attr("val_f1", val_metrics["eval_macro_f1"])

        # CSV 저장용 누적
        trial_logs.append({
            "trial": trial.number,
            **trial.params,  # 샘플링된 하이퍼파라미터들
            "train_acc": train_metrics["eval_accuracy"],
            "train_f1": train_metrics["eval_macro_f1"],
            "val_acc": val_metrics["eval_accuracy"],
            "val_f1": val_metrics["eval_macro_f1"],
        })

        # 최적화 대상(Validation Macro-F1)
        return val_metrics["eval_macro_f1"]

    # 스터디 생성 및 최적화
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    # 결과 요약 출력
    print("Best trial:", study.best_trial.number, study.best_trial.value)
    print("Best params:", study.best_trial.params)

    # CSV 저장
    import pandas as pd
    df_trials = pd.DataFrame(trial_logs)
    df_trials.to_csv(f"{OUTPUT_DIR}/optuna_results.csv", index=False)

    # 상위 N개 요약(예: 상위 5개)
    topk = df_trials.sort_values("val_f1", ascending=False).head(5)
    print("\n[Top-5 by Val F1]")
    print(topk[[
        "trial","val_f1","val_acc","train_f1","train_acc",
        "learning_rate","weight_decay","warmup_ratio","lr_scheduler_type",
        "max_grad_norm","num_train_epochs","early_stopping_patience"
    ]].to_string(index=False))