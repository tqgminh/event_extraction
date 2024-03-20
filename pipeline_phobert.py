from models2 import *

import os
import json

from pyvi import ViTokenizer
import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

event_question = ["sự_kiện"]
argument_nth_question = 5
tokenizer = AutoTokenizer.from_pretrained('phobert-base')

max_length = 300

with open('json/event2id.json', 'r', encoding='utf-8') as r:
    event2id = json.load(r)

with open('json/id2event.json', 'r', encoding='utf-8') as r:
    id2event = json.load(r)

with open('json/argument2id.json', 'r', encoding='utf-8') as r:
    argument2id = json.load(r)

with open('json/id2argument.json', 'r', encoding='utf-8') as r:
    id2argument = json.load(r)

def preprocess_for_event(sentence):
    words = ViTokenizer.tokenize(sentence).split()

    tokens = []
    token_type_ids = []
    in_sentence = []

    tokens.append('<s>')
    token_type_ids.append(0)
    in_sentence.append(0)

    for token in event_question:
        sub_tokens = tokenizer.tokenize(token)
        tokens.append(sub_tokens[0])
        token_type_ids.append(0)
        in_sentence.append(0)
    
    tokens.extend(['</s>', '</s>'])
    token_type_ids.extend([0, 0])
    in_sentence.extend([0, 0])

    for j, token in enumerate(words):
        if token.strip() != '':
            sub_tokens = tokenizer.tokenize(token)
            tokens.append(sub_tokens[0])
            token_type_ids.append(0)
            in_sentence.append(1)
        else:
            tokens.append(tokenizer.unk_token_id)
            token_type_ids.append(0)
            in_sentence.append(1)

    # add </s>
    tokens.append('</s>')
    token_type_ids.append(0)
    in_sentence.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)

    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        token_type_ids.append(0)
        in_sentence.append(0)
    
    tmp = dict()
    tmp['input_ids'] = torch.tensor([input_ids])
    tmp['token_type_ids'] = torch.tensor([token_type_ids])
    tmp['in_sentence'] = torch.tensor([in_sentence])
    tmp['attention_mask'] = torch.tensor([attention_mask])
    tmp['sentence_id'] = torch.tensor([0])

    return tmp

def make_prediction_event(inps, in_sentence, sentence_ids, id2event):
    inps = inps.tolist()
    in_sentence.tolist()
    sentence_ids = sentence_ids.tolist()

    result = []
    
    for i, in_sent in enumerate(in_sentence):
        inp = inps[i]
        events_of_sent = []

        start_sent = 0
        while in_sent[start_sent] == 0 and start_sent+1 < len(in_sent):
            start_sent += 1
        end_sent = start_sent
        while in_sent[end_sent] == 1 and end_sent+1 < len(in_sent):
            end_sent += 1
        
        inp = inp[start_sent:end_sent]
        j = 0
        while j < len(inp):
            event_id = inp[j]
            event_type = id2event[event_id]
            if event_type != 'O':
                tmp = [event_type, j]
                while j+1 < len(inp) and inp[j+1] == inp[j]:
                    j += 1
                tmp.append(j)
                events_of_sent.append(tuple(tmp))
            j += 1

        result.append(events_of_sent)
    
    return result

event_model = EventClassifyBert(AutoModel.from_pretrained('vinai/phobert-base'), len(event2id))

event_model.load_state_dict(torch.load('weight/phobert-base-event-0-th-question.pth', map_location=device))

def predict_event(sentence):
    inputs = preprocess_for_event(sentence)

    input_ids = inputs['input_ids'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)
    in_sentence = inputs['in_sentence'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    sentence_ids = inputs['sentence_id'].to(device)

    event_model.to(device)

    event_model.eval()
    with torch.no_grad():
        logits = event_model(input_ids, attention_mask, token_type_ids)

    preds = torch.argmax(logits, dim=-1)
    preds = make_prediction_event(preds, in_sentence, sentence_ids, id2event)[0]

    return preds

def read_question_templates(folder):
    files = os.listdir(folder)
    files = sorted(files)
    question_templates = dict()
    for file in files:
        if file.startswith('template'):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as r:
                lines = r.read().split('\n')[:-1]
                for line in lines:
                    items = line.split(',')
                    event_argument = items[0].split('_')
                    event = event_argument[0]
                    argument = event_argument[1]
                    question = items[1]
                    if event not in question_templates:
                        question_templates[event] = dict()
                    if argument not in question_templates[event]:
                        question_templates[event][argument] = []
                    question_templates[event][argument].append(question)
                    sc_question = question
                    if sc_question.endswith('?'):
                        sc_question = sc_question[:-1] + "trong [trigger] ?"
                    else:
                        sc_question = sc_question + " trong [trigger]"
                    question_templates[event][argument].append(sc_question)

    return question_templates

question_templates = read_question_templates('question_templates')

# print(question_templates['Attack'])

def preprocess_for_argument(sentence, events):
    words = ViTokenizer.tokenize(sentence).split()

    tmp = dict()
    tmp['input_ids'] = []
    tmp['token_type_ids'] = []
    tmp['in_sentence'] = []
    tmp['attention_mask'] = []
    tmp['event_id'] = []
    tmp['start_trigger'] = []
    tmp['end_trigger'] = []
    tmp['argument_id'] = []
    tmp['sentence_id'] = []

    for event in events:
        event_type = event[0]
        start_trigger = event[1]
        end_trigger = event[2]
        trigger = []
        for j in range(start_trigger, end_trigger+1):
            trigger.append(words[j])

        trigger = ' '.join(trigger)

        for argument_type in question_templates[event_type]:
            question = question_templates[event_type][argument_type][argument_nth_question]
            question = question.replace("[trigger]", trigger)
            # print(question)

            tokens = []
            token_type_ids = []
            in_sentence = []

            tokens.append('<s>')
            token_type_ids.append(0)
            in_sentence.append(0)

            # add question
            question_tokens = tokenizer.tokenize(question)
            for token in question_tokens:
                tokens.append(token)
                token_type_ids.append(0)
                in_sentence.append(0)
                    
            # add </s> and </s>
            tokens.extend(['</s>', '</s>'])
            token_type_ids.extend([0, 0])
            in_sentence.extend([0, 0])

            # add sentence
            for j, token in enumerate(words):
                if token.strip() != '':
                    sub_tokens = tokenizer.tokenize(token)
                    tokens.append(sub_tokens[0])
                    token_type_ids.append(0)
                    in_sentence.append(1)
                else:
                    tokens.append(tokenizer.unk_token_id)
                    token_type_ids.append(0)
                    in_sentence.append(1)
                    
            # add </s>
            tokens.append('</s>')
            token_type_ids.append(0)
            in_sentence.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            while len(input_ids) < max_length:
                input_ids.append(tokenizer.pad_token_id)
                attention_mask.append(0)
                token_type_ids.append(0)
                in_sentence.append(0)

            sentence_offset = len(question_tokens) + 3
            n_start_trigger = start_trigger + sentence_offset
            n_end_trigger = end_trigger + sentence_offset

            tmp['input_ids'].append(input_ids)
            tmp['token_type_ids'].append(token_type_ids)
            tmp['in_sentence'].append(in_sentence)
            tmp['attention_mask'].append(attention_mask)
            tmp['event_id'].append(event2id[event_type])
            tmp['start_trigger'].append(n_start_trigger)
            tmp['end_trigger'].append(n_end_trigger)
            tmp['argument_id'].append(argument2id[argument_type])
            tmp['sentence_id'].append(0)
    
    tmp['input_ids'] = torch.tensor(tmp['input_ids'])
    tmp['token_type_ids'] = torch.tensor(tmp['token_type_ids'])
    tmp['in_sentence'] = torch.tensor(tmp['in_sentence'])
    tmp['attention_mask'] = torch.tensor(tmp['attention_mask'])
    tmp['event_id'] = torch.tensor(tmp['event_id'])
    tmp['start_trigger'] = torch.tensor(tmp['start_trigger'])
    tmp['end_trigger'] = torch.tensor(tmp['end_trigger'])
    tmp['argument_id'] = torch.tensor(tmp['argument_id'])
    tmp['sentence_id'] = torch.tensor(tmp['sentence_id'])

    return tmp, words

def make_prediction_argument(start_inps, end_inps, in_sentence, event_ids, start_triggers, end_triggers, argument_ids, sentence_ids, id2event, id2argument):
    
    start_inps = start_inps.tolist()
    end_inps = end_inps.tolist()
    in_sentence = in_sentence.tolist()
    event_ids = event_ids.tolist()
    start_triggers = start_triggers.tolist()
    end_triggers = end_triggers.tolist()
    argument_ids = argument_ids.tolist()
    sentence_ids = sentence_ids.tolist()

    result = []
    for i, sentence_id in enumerate(sentence_ids):
        sentence_id = sentence_id
        start_inp = start_inps[i]
        end_inp = end_inps[i]
        in_sent = in_sentence[i]
        event_id = event_ids[i]
        start_trigger = start_triggers[i]
        end_trigger = end_triggers[i]
        argument_id = argument_ids[i]

        if in_sent[start_inp] == 0 or in_sent[end_inp] == 0:
            start_inp = 0
            end_inp = 0

        sent_offset = 0
        while in_sent[sent_offset] == 0:
            sent_offset += 1

        if start_inp != 0 and end_inp != 0:
            event_type = id2event[event_id]
            argument_type = id2argument[argument_id]
            tmp = (sentence_id, event_type, start_trigger-sent_offset, end_trigger-sent_offset, \
                   argument_type, start_inp-sent_offset, end_inp-sent_offset)
            result.append(tmp)

    return result

argument_model = ArgumentMRCBert(AutoModel.from_pretrained('phobert-base'))                   

argument_model.load_state_dict(torch.load('weight/phobert-base-qa-argument-5-th-question.pth', map_location=device))

def predict_argument(sentence):

    events = predict_event(sentence)

    words = ViTokenizer.tokenize(sentence).split(' ')

    if len(events) == 0:
        return [], words

    inputs, words = preprocess_for_argument(sentence, events)

    input_ids = inputs['input_ids'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)
    in_sentence = inputs['in_sentence'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    event_ids = inputs['event_id'].to(device)
    start_triggers = inputs['start_trigger'].to(device)
    end_triggers = inputs['end_trigger'].to(device)
    argument_ids = inputs['argument_id'].to(device)
    sentence_ids = inputs['sentence_id'].to(device)

    argument_model.to(device)

    argument_model.eval()
    with torch.no_grad():
        start_logits, end_logits = argument_model(input_ids, attention_mask, token_type_ids)
    
    start_preds = torch.argmax(start_logits, dim=-1)
    end_preds = torch.argmax(end_logits, dim=-1)
    preds = make_prediction_argument(start_preds, end_preds, in_sentence, event_ids, \
        start_triggers, end_triggers, argument_ids, sentence_ids, id2event, id2argument)
    
    # print(words)

    # for pred in preds:
    #     event_type = pred[1]
    #     start_trigger = pred[2]
    #     end_trigger = pred[3]
    #     argument_type = pred[4]
    #     start_arg = pred[5]
    #     end_arg = pred[6]

    #     print(f'{event_type}: {" ".join(words[start_trigger:end_trigger+1])}\t{start_trigger}\t{end_trigger}\t-\t{argument_type}: {" ".join(words[start_arg:end_arg+1])}\t{start_arg}\t{end_arg}')

    return preds, words

# predict_argument("Minh sẽ rời Hà Nội để tới thành phố Hồ Chí Minh vào ngày mai.")