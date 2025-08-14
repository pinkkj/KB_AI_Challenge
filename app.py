# -*- coding: utf-8 -*-
import os, json, re, random, math
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # 포크 경고/데드락 회피

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

# Ampere(3090)에서 TF32 허용
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =========================
# (스위치) 정밀도 설정
# =========================
USE_FP16 = True   # 문제가 계속되면 False로 내려서 먼저 정상 동작 확인

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
MODEL_NAME = "openlm-research/open_llama_3b"
MAX_LEN = 96

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)  # (원하면 legacy=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tok_fn(batch):
    out = tokenizer(batch["text"], padding=False, truncation=True, max_length=MAX_LEN)
    out["labels"] = batch["label"]
    return out

tokenized = raw_dset.map(tok_fn, batched=True, remove_columns=["text", "label_str", "label"])
collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

# 하드웨어 감지
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
bf16 = False  # 3090은 bf16 미지원
fp16 = torch.cuda.is_available() and USE_FP16

print("모델 로드 전")
# ✅ FP16로 로드(핵심)
model = LlamaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.float16 if fp16 else torch.float32,
    low_cpu_mem_usage=True,
)
print("모델 로드 후")

# 분류 문제 타입 명시 + 체크포인팅 충돌 방지
model.config.problem_type = "single_label_classification"
model.config.use_cache = False
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
        "macro_f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
    }
print("step 3 완료")

# =========================
# 4) 스텝 자동 계산(정보 출력용)
# =========================
def auto_steps(n_train_examples, per_device_bsz, grad_acc, n_gpus=1, evals_per_epoch=1):
    global_bsz = per_device_bsz * grad_acc * max(1, n_gpus)
    steps_per_epoch = max(1, math.floor(n_train_examples / global_bsz))
    step = max(50, steps_per_epoch // max(1, evals_per_epoch))
    return step, steps_per_epoch

N_TRAIN = len(tokenized["train"])
PER_DEVICE_BSZ = 2
GRAD_ACC = 8
EVAL_STEPS, STEPS_PER_EPOCH = auto_steps(
    N_TRAIN, PER_DEVICE_BSZ, GRAD_ACC, n_gpus=torch.cuda.device_count(), evals_per_epoch=1
)
print(f"[INFO] steps/epoch ≈ {STEPS_PER_EPOCH}, (epoch 기반 평가/저장)")
print(f"[INFO] device: {device_name}, bf16={bf16}, fp16={fp16}")
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

    learning_rate=LR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BSZ,
    per_device_eval_batch_size=PER_DEVICE_BSZ,
    gradient_accumulation_steps=GRAD_ACC,
    weight_decay=0.05,
    max_grad_norm=1.0,

    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    evaluation_strategy="epoch",   # 간단/안정
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=100,

    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,

    fp16=fp16,          # ✅ FP16 사용
    bf16=bf16,          # 3090이라 False
    optim="adamw_torch",

    gradient_checkpointing=True,
    # ✅ AMP와의 충돌 줄이기
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # ✅ 슬럼/포크 환경 안정화
    dataloader_num_workers=0,
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
# (옵션) 8) Optuna 스윕
# =========================
print("step 8 튜닝 시작 완료")
DO_TUNE = True

if DO_TUNE:
    import optuna

    trial_logs = []

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

            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,

            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,

            fp16=fp16,
            bf16=bf16,
            optim="adamw_torch",

            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},

            dataloader_num_workers=0,
            report_to="none",
            seed=SEED,
        )
        return a, patience

    def objective(trial):
        m = LlamaForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABELS),
            id2label=id2label,
            label2id=label2id,
            torch_dtype=torch.float16 if fp16 else torch.float32,
            low_cpu_mem_usage=True,
        )
        m.config.problem_type = "single_label_classification"
        m.config.use_cache = False
        m.gradient_checkpointing_enable()

        a, patience = build_args(trial)
        t = Trainer(
            model=m,
            args=a,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )

        t.train()

        train_metrics = t.evaluate(tokenized["train"])
        val_metrics = t.evaluate(tokenized["validation"])

        print(
            f"[Trial {trial.number}] "
            f"Train Acc: {train_metrics['eval_accuracy']:.4f}, "
            f"Train F1: {train_metrics['eval_macro_f1']:.4f} | "
            f"Val Acc: {val_metrics['eval_accuracy']:.4f}, "
            f"Val F1: {val_metrics['eval_macro_f1']:.4f}"
        )

        trial.set_user_attr("train_acc", train_metrics["eval_accuracy"])
        trial.set_user_attr("train_f1", train_metrics["eval_macro_f1"])
        trial.set_user_attr("val_acc", val_metrics["eval_accuracy"])
        trial.set_user_attr("val_f1", val_metrics["eval_macro_f1"])

        trial_logs.append({
            "trial": trial.number,
            **trial.params,
            "train_acc": train_metrics["eval_accuracy"],
            "train_f1": train_metrics["eval_macro_f1"],
            "val_acc": val_metrics["eval_accuracy"],
            "val_f1": val_metrics["eval_macro_f1"],
        })

        # 메모리 정리
        del t, m
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return val_metrics["eval_macro_f1"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print("Best trial:", study.best_trial.number, study.best_trial.value)
    print("Best params:", study.best_trial.params)

    df_trials = pd.DataFrame(trial_logs)
    df_trials.to_csv(f"{OUTPUT_DIR}/optuna_results.csv", index=False)

    topk = df_trials.sort_values("val_f1", ascending=False).head(5)
    print("\n[Top-5 by Val F1]")
    print(topk[[
        "trial","val_f1","val_acc","train_f1","train_acc",
        "learning_rate","weight_decay","warmup_ratio","lr_scheduler_type",
        "max_grad_norm","num_train_epochs","early_stopping_patience"
    ]].to_string(index=False))
