from tqdm.notebook import tqdm
import transformers
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import random

import torch, gc

gc.collect()
torch.cuda.empty_cache()

def f1_score_calculation(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def evaluate(dataloader_test, model, device):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
    for batch in dataloader_test:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    loss_val_avg = loss_val_total / len(dataloader_test)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_val_avg, predictions, true_vals


def _evaluation(dataloader_test, model, device, encoder, epoch, model_name):
    # Validation Loss and Validation F-1 Score
    val_loss, predictions, true_vals = evaluate(dataloader_test, model, device)
    val_f1 = f1_score_calculation(predictions, true_vals)
    print('Val Loss = ', val_loss)
    print('Val F1 = ', val_f1)
    # Validation Accuracy
    encoded_classes = encoder.classes_
    predicted_category = [encoded_classes[np.argmax(x)] for x in predictions]
    true_category = [encoded_classes[x] for x in true_vals]
    # accuracy score
    print('Accuracy Score = ', accuracy_score(true_category, predicted_category))


def model_tuning_bert(model_name, training_path, testing_path, text_col, sentiment_col, batch_size, epochs, max_length):
    # data
    training = pd.read_csv(training_path)
    testing = pd.read_csv(testing_path)

    # X
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    X_train = training[text_col]
    X_test = testing[text_col]
    encoded_data_train = tokenizer.batch_encode_plus(
        X_train,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=max_length,
        return_tensors='pt'
    )
    encoded_data_test = tokenizer.batch_encode_plus(
        X_test,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=max_length,
        return_tensors='pt'
    )

    # y
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(training[sentiment_col])
    y_test = encoder.fit_transform(testing[sentiment_col])

    # Instantiate TensorDataset
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(y_train)
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(y_test)
    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

    dataloader_train = DataLoader(dataset_train,
                                  sampler=RandomSampler(dataset_train),
                                  batch_size=int(batch_size))
    dataloader_test = DataLoader(dataset_test,
                                 sampler=SequentialSampler(dataset_test),
                                 batch_size=int(batch_size))

    # model
    model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                       num_labels=5,
                                                                       output_attentions=False,
                                                                       output_hidden_states=False)

    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)

    # train
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    device = torch.device('cuda')
    model.to(device)

    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        loss_train_total = 0
        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            inputs = {'input_ids': batch[0].to(device),
                      'attention_mask': batch[1].to(device),
                      'labels': batch[2].to(device),
                      }
            outputs = model(**inputs)
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
        # progress bar
        tqdm.write(f'\nEpoch {epoch}')
        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')
        # evaluate the model
        _evaluation(dataloader_test, model, device, encoder, epoch, model_name)

    # save the model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_name + '.tar')
