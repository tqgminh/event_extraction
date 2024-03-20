from models1 import *

import json

from pyvi import ViTokenizer, ViPosTagger
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

word_embedding_dim = 300
position_embedding_dim = 25
event_embedding_dim = 25
postag_embedding_dim = 25
max_length = 124

hidden_size = 512
dropout_rate = 0.5

with open('json/word2id.json', 'r', encoding='utf-8') as r:
    word2id = json.load(r)

with open('json/id2word.json', 'r', encoding='utf-8') as r:
    id2word = json.load(r)

with open('json/postag2id.json', 'r', encoding='utf-8') as r:
    postag2id = json.load(r)

with open('json/id2postag.json', 'r', encoding='utf-8') as r:
    id2postag = json.load(r)

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
    postags_of_words = ViPosTagger.postagging_tokens(words)[1]

    input_ids = []
    postag_ids = []
    in_sentence = []

    for j in range(max_length):
        if j < len(words):
            if words[j] in word2id:
                input_ids.append(word2id[words[j]])
            else:
                input_ids.append(word2id['<unk>'])
            postag_ids.append(postag2id[postags_of_words[j]])
            in_sentence.append(1)
        else:
            input_ids.append(word2id['<pad>'])
            postag_ids.append(postag2id['O'])
            in_sentence.append(0)
    
    tmp = dict()
    tmp['input_ids'] = torch.tensor([input_ids])
    tmp['postag_ids'] = torch.tensor([postag_ids])
    tmp['in_sentence'] = torch.tensor([in_sentence])
    tmp['sentence_id'] = torch.tensor([0])

    return tmp

def make_prediction_event(inps, in_sentence, sentence_ids, id2event):
    
    inps = inps.tolist()
    in_sentence = in_sentence.tolist()
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

event_model = EventSentenceBiLSTMWithPostag(
    num_labels=len(event2id),
    num_words=len(word2id),
    word_embedding_dim=word_embedding_dim,
    num_postags=len(postag2id),
    postag_embedding_dim=postag_embedding_dim,
    hidden_size=hidden_size,
    dropout_rate=dropout_rate,
    use_pretrained=False,
    embedding_weight=None
)

event_model.load_state_dict(torch.load('weight/bilstm-event-sentence-True-word2vec-True-postag.pth', map_location=device))

def predict_event(sentence):
    inputs = preprocess_for_event(sentence)
    input_ids = inputs['input_ids'].to(device)
    postag_ids = inputs['postag_ids'].to(device)
    in_sentence = inputs['in_sentence'].to(device)
    sentence_ids = inputs['sentence_id'].to(device)

    event_model.to(device)

    event_model.eval()
    with torch.no_grad():
        logits = event_model(input_ids, postag_ids, in_sentence)

    preds = torch.argmax(logits, dim=-1)
    preds = make_prediction_event(preds, in_sentence, sentence_ids, id2event)[0]
    
    return preds

def preprocess_for_argument(sentence, events):
    words = ViTokenizer.tokenize(sentence).split()
    postags_of_words = ViPosTagger.postagging_tokens(words)[1]

    tmp = dict()
    tmp['input_ids'] = []
    tmp['trigger_position_ids'] = []
    tmp['postag_ids'] = []
    tmp['in_sentence'] = []
    tmp['event_ids'] = []
    tmp['start_trigger'] = []
    tmp['end_trigger'] = []
    tmp['sentence_id'] = []

    for event in events:
        event_type = event[0]
        start_trigger = event[1]
        end_trigger = event[2]
        event_ids = [event2id[event_type]] * max_length

        input_ids = []
        trigger_position_ids = []
        postag_ids = []
        in_sentence = []

        for j in range(max_length):
            if j < len(words):
                if words[j] in word2id:
                    input_ids.append(word2id[words[j]])
                else:
                    input_ids.append(word2id['<unk>'])
                postag_ids.append(postag2id[postags_of_words[j]])
                in_sentence.append(1)
            else:
                input_ids.append(word2id['<pad>'])
                postag_ids.append(postag2id['O'])
                in_sentence.append(0)
            trigger_position_ids.append(abs(j-start_trigger))
        
        tmp['input_ids'].append(input_ids)
        tmp['trigger_position_ids'].append(trigger_position_ids)
        tmp['postag_ids'].append(postag_ids)
        tmp['in_sentence'].append(in_sentence)
        tmp['event_ids'].append(event_ids)
        tmp['start_trigger'].append(start_trigger)
        tmp['end_trigger'].append(end_trigger)
        tmp['sentence_id'].append(0)
        
    tmp['input_ids'] = torch.tensor(tmp['input_ids'])
    tmp['trigger_position_ids'] = torch.tensor(tmp['trigger_position_ids'])
    tmp['postag_ids'] = torch.tensor(tmp['postag_ids'])
    tmp['in_sentence'] = torch.tensor(tmp['in_sentence'])
    tmp['event_ids'] = torch.tensor(tmp['event_ids'])
    tmp['start_trigger'] = torch.tensor(tmp['start_trigger'])
    tmp['end_trigger'] = torch.tensor(tmp['end_trigger'])
    tmp['sentence_id'] = torch.tensor(tmp['sentence_id'])

    return tmp, words

def make_prediction_argument(inps, in_sentence, sentence_ids, event_ids, start_triggers, end_triggers, id2event, id2argument):
    inps = inps.tolist()
    in_sentence = in_sentence.tolist()
    sentence_ids = sentence_ids.tolist()
    event_ids = [event_id[0] for event_id in event_ids.tolist()]
    start_triggers = start_triggers.tolist()
    end_triggers = end_triggers.tolist()
    
    result = []
    for i, sentence_id in enumerate(sentence_ids):
        inp = inps[i]
        in_sent = in_sentence[i]
        event_id = event_ids[i]
        start_trigger = start_triggers[i]
        end_trigger = end_triggers[i]
        event_type = id2event[event_id]
        
        start_sent = 0
        while in_sent[start_sent] == 0 and start_sent+1 < len(in_sent):
            start_sent += 1
        end_sent = start_sent
        while in_sent[end_sent] == 1 and end_sent+1 < len(in_sent):
            end_sent += 1
        
        inp = inp[start_sent:end_sent]
        j = 0
        while j < len(inp):
            argument_id = inp[j]
            argument_type = id2argument[argument_id]
            if argument_type != 'O':
                tmp = [sentence_id, event_type, start_trigger, end_trigger, argument_type, j]
                while j+1 < len(inp) and inp[j+1] == inp[j]:
                    j += 1
                tmp.append(j)
                result.append(tuple(tmp))
            j += 1
    
    return result

argument_model = ArgumentSentenceBiLSTMWithPostag(
    num_labels=len(argument2id),
    num_words=len(word2id),
    word_embedding_dim=word_embedding_dim,
    max_length=max_length,
    position_embedding_dim=position_embedding_dim,
    num_events=len(event2id),
    event_embedding_dim=event_embedding_dim,
    num_postags=len(postag2id),
    postag_embedding_dim=postag_embedding_dim,
    hidden_size=hidden_size,
    dropout_rate=dropout_rate,
    use_pretrained=False,
    embedding_weight=None
)

argument_model.load_state_dict(torch.load('weight/bilstm-argument-sentence-True-word2vec-True-postag.pth', map_location=device))

def predict_argument(sentence):

    events = predict_event(sentence)

    words = ViTokenizer.tokenize(sentence).split(' ')

    if len(events) == 0:
        return [], words

    inputs, words = preprocess_for_argument(sentence, events)

    input_ids = inputs['input_ids'].to(device)
    trigger_position_ids = inputs['trigger_position_ids'].to(device)
    postag_ids = inputs['postag_ids'].to(device)
    in_sentence = inputs['in_sentence'].to(device)
    event_ids = inputs['event_ids'].to(device)
    start_triggers = inputs['start_trigger'].to(device)
    end_triggers = inputs['end_trigger'].to(device)
    sentence_ids = inputs['sentence_id'].to(device)

    argument_model.to(device)

    argument_model.eval()
    with torch.no_grad():
        logits = argument_model(input_ids, trigger_position_ids, event_ids, postag_ids, in_sentence)

    preds = torch.argmax(logits, dim=-1)
    preds = make_prediction_argument(preds, in_sentence, sentence_ids, event_ids, \
        start_triggers, end_triggers, id2event, id2argument)
    
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
