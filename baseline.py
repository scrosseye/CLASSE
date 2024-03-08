import csv

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import AutoTokenizer, TFAutoModel
from wandb.keras import WandbMetricsLogger

import wandb

def read_prompts():
    prompts = {}
    with open("prompts.csv", "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts[row["prompt_id"]] = row["prompt_text"]
    return prompts

def read_dataset(split):
    dataset = []
    prompts = read_prompts()
    with open("summaries.csv", "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue
            entry = {
                "student": row["student_id"],
                "text": prompts[row["prompt_id"]],
                "summary": row["text"],
                "content": float(row["content"]),
                "wording": float(row["wording"]),
            }
            dataset.append(entry)
    return dataset

def process_data(tokenizer, dataset, config):
    result = {"input_ids": [], "attention_mask": [], "summary_mask": []}
    outputs = []
    for entry in dataset:
        if config["include_text"]:
            tokenized = tokenizer([(entry["summary"], entry["text"])], return_tensors="np")
        else:
            tokenized = tokenizer([entry["summary"]], return_tensors="np")
        result["input_ids"].append(tokenized.input_ids.squeeze())
        result["attention_mask"].append(tokenized.attention_mask.squeeze())
        if config["include_text"]:
            summary_mask = tokenizer(entry["summary"], return_tensors="np").attention_mask.squeeze()
            summary_mask = np.concatenate([summary_mask, np.zeros((tokenized.attention_mask.shape[1]-len(summary_mask),), dtype=np.int32)])
        else:
            summary_mask = tokenized.attention_mask.squeeze()
        result["summary_mask"].append(summary_mask)
        outputs.append((entry["content"], entry["wording"]))
        
    result["input_ids"] = tf.ragged.constant(result["input_ids"])
    result["attention_mask"] = tf.ragged.constant(result["attention_mask"])
    result["summary_mask"] = tf.ragged.constant(result["summary_mask"])
    
    def to_tensor(inputs, outputs):
        return {
            "input_ids": inputs["input_ids"], 
            "attention_mask": inputs["attention_mask"], 
            "summary_mask": inputs["summary_mask"], 
        }, {"content": outputs[0], "wording": outputs[1]}
        
    dataset = tf.data.Dataset.from_tensor_slices((result, outputs)).map(to_tensor)
    return dataset

def create_model(encoder, config):
    input_ids = tf.keras.layers.Input((None,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input((None,), dtype=tf.int32)
    summary_mask = tf.keras.layers.Input((None,), dtype=tf.int32)
    bert_output = encoder(input_ids=input_ids, attention_mask=attention_mask)
    if config["avg_summary"]:
        emb = tf.keras.layers.GlobalAveragePooling1D()(bert_output.last_hidden_state, mask=summary_mask)
    else:
        # emb = tf.keras.layers.GlobalAveragePooling1D()(embs, mask=attention_mask)
        emb = bert_output.pooler_output
    emb = tf.keras.layers.Dropout(config["dropout"])(emb)
    if config["hidden"] > 0:
        emb = tf.keras.layers.Dense(config["hidden"], activation=config["activation"])(emb)
    content = tf.keras.layers.Dense(1, name="content", activation=None)(emb)
    wording = tf.keras.layers.Dense(1, name="wording", activation=None)(emb)
    return tf.keras.Model(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask, "summary_mask": summary_mask}, 
        outputs={"content": content, "wording": wording})

def train():
    config = {
        "avg_summary": True,
        "include_text": True,
        "hidden": 0, 
        "activation": "relu",
        "dropout": 0.2,
        "lr": 1e-5,
    }
    encoder = TFAutoModel.from_pretrained("allenai/longformer-base-4096")
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    train_dataset = process_data(tokenizer, read_dataset("train"), config).shuffle(8096).padded_batch(8)
    dev_dataset = process_data(tokenizer, read_dataset("public_test"), config).padded_batch(8)
    model = create_model(encoder, config)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config["lr"])
    model.compile(optimizer=optimizer, loss="mse", metrics="mae")
    wandb.init(entity="readerbench", project="summary scoring", config=config)
    wandb_logger = WandbMetricsLogger()
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f"models/{wandb.run.name}", save_best_only=True, save_weights_only=True)
    model.fit(train_dataset, epochs=10, validation_data=dev_dataset, callbacks=[wandb_logger, checkpoint])

def test(run_name):
    config = {
        "avg_summary": True,
        "include_text": True,
        "hidden": 0, 
        "activation": "tanh",
        "dropout": 0.2,
        "lr": 1e-5,
    }
    encoder = TFAutoModel.from_pretrained("allenai/longformer-base-4096")
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    test_dataset = process_data(tokenizer, read_dataset("private_test"), config).padded_batch(8)
    model = create_model(encoder, config)
    model.load_weights(f"models/{run_name}").expect_partial()
    target_content = []
    target_wording = []
    predicted_content = []
    predicted_wording = []
    for x, y in tqdm(test_dataset):
        prediction = model(x)
        target_content += y["content"].numpy().tolist()
        target_wording += y["wording"].numpy().tolist()
        predicted_content += prediction["content"][:, 0].numpy().tolist()
        predicted_wording += prediction["wording"][:, 0].numpy().tolist()
    content_rmse = np.sqrt(np.mean((np.array(target_content) - np.array(predicted_content))**2))
    wording_rmse = np.sqrt(np.mean((np.array(target_wording) - np.array(predicted_wording))**2))
    print("Content RMSE: ", content_rmse)
    print("Wording RMSE: ", wording_rmse)
    print("MCRMSE: ", (content_rmse + wording_rmse) / 2)
    with open(f"{run_name}.csv", "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["student_id", "Predicted Content", "Target Content", "Predicted Wording", "Target Wording"])
        for entry, pc, tc, pw, tw in zip(read_dataset("private_test"), predicted_content, target_content, predicted_wording, target_wording):
            writer.writerow([entry["student"], pc, tc, pw, tw])
    
if __name__ == "__main__":
    # train()
    test()
    
    