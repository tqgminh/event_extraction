from models1 import *

import json

from pyvi import ViTokenizer, ViPosTagger
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 15
word_embedding_dim = 300
position_embedding_dim = 25
postag_embedding_dim = 25
event_embedding_dim = 25
max_length = 124

kernel_sizes = [2, 3, 4, 5]
num_filters = 150
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

    tmp = dict()
    tmp['input_ids'] = []
    tmp['position_ids'] = []
    tmp['postag_ids'] = []
    tmp['sentence_id'] = []
    tmp['current_position_id'] = []

    for pos, word in enumerate(words):
        input_ids = []
        position_ids = []
        postag_ids = []

        for j in range(pos-window_size, pos+window_size+1):
            if j < 0 or j >= len(words):
                input_ids.append(word2id['<pad>'])
                postag_ids.append(postag2id['O'])
            else:
                if words[j] in word2id:
                    input_ids.append(word2id[words[j]])
                else:
                    input_ids.append(word2id['<unk>'])
                postag_ids.append(postag2id[postags_of_words[j]])
            position_ids.append(abs(j-pos))
        
        tmp['input_ids'].append(input_ids)
        tmp['position_ids'].append(position_ids)
        tmp['postag_ids'].append(postag_ids)
        tmp['sentence_id'].append(0)
        tmp['current_position_id'].append(pos)
    
    tmp['input_ids'] = torch.tensor(tmp['input_ids'])
    tmp['position_ids'] = torch.tensor(tmp['position_ids'])
    tmp['postag_ids'] = torch.tensor(tmp['postag_ids'])
    tmp['sentence_id'] = torch.tensor(tmp['sentence_id'])
    tmp['current_position_id'] = torch.tensor(tmp['current_position_id'])

    return tmp

def make_prediction_event(inps, sentence_ids, id2event):
    event_ids_of_sents = []
    event_ids_of_sent = []
    
    for i in range(len(sentence_ids)):
        if i == 0:
            event_ids_of_sent = [inps[i]]
        else:
            if sentence_ids[i] == sentence_ids[i-1]:
                event_ids_of_sent.append(inps[i])
            else:
                event_ids_of_sents.append(event_ids_of_sent)
                event_ids_of_sent = [inps[i]]
    event_ids_of_sents.append(event_ids_of_sent)
    
    result = []
    for event_ids_of_sent in event_ids_of_sents:
        result_sent = []
        j = 0
        while j < len(event_ids_of_sent):
            event_id = event_ids_of_sent[j]
            event_type = id2event[event_id]
            if event_type != 'O':
                tmp = [event_type, j]
                while j+1 < len(event_ids_of_sent) and event_ids_of_sent[j+1] == event_ids_of_sent[j]:
                    j += 1
                tmp.append(j)
                result_sent.append(tuple(tmp))
            j += 1
        result.append(result_sent)
    
    return result

event_model = EventWordCNNWithPostag(
    num_labels=len(event2id),
    num_words=len(word2id),
    word_embedding_dim=word_embedding_dim,
    window_size=window_size,
    position_embedding_dim=position_embedding_dim,
    num_postags=len(postag2id),
    postag_embedding_dim=postag_embedding_dim,
    kernel_sizes=kernel_sizes,
    num_filters=num_filters,
    dropout_rate=dropout_rate,
    use_pretrained=False,
    embedding_weight=None
)

event_model.load_state_dict(torch.load('weight/cnn-event-word-True-word2vec-True-postag.pth', map_location=device))

def predict_event(sentence):
    inputs = preprocess_for_event(sentence)
    input_ids = inputs['input_ids'].to(device)
    position_ids = inputs['position_ids'].to(device)
    postag_ids = inputs['postag_ids'].to(device)
    sentence_ids = inputs['sentence_id'].to(device)
    # current_position_id = inputs['current_position_id']

    event_model.to(device)

    event_model.eval()
    with torch.no_grad():
        logits = event_model(input_ids, position_ids, postag_ids)

    preds = torch.argmax(logits, dim=-1)
    preds = make_prediction_event(preds, sentence_ids, id2event)[0]
    
    return preds

def preprocess_for_argument(sentence, events):
    words = ViTokenizer.tokenize(sentence).split()
    postags_of_words = ViPosTagger.postagging_tokens(words)[1]

    tmp = dict()
    tmp['input_ids'] = []
    tmp['candidate_position_ids'] = []
    tmp['trigger_position_ids'] = []
    tmp['postag_ids'] = []
    tmp['event_ids'] = []
    tmp['start_trigger'] = []
    tmp['end_trigger'] = []
    tmp['sentence_id'] = []
    tmp['current_position_id'] = []

    for event in events:
        event_type = event[0]
        start_trigger = event[1]
        end_trigger = event[2]
        event_ids = [event2id[event_type]] * max_length

        for pos, word in enumerate(words):
            input_ids = []
            candidate_position_ids = []
            trigger_position_ids = []
            postag_ids = []

            for j in range(max_length):
                if j < len(words):
                    if words[j] in word2id:
                        input_ids.append(word2id[words[j]])
                    else:
                        input_ids.append(word2id['<unk>'])
                    postag_ids.append(postag2id[postags_of_words[j]])
                else:
                    input_ids.append(word2id['<pad>'])
                    postag_ids.append(postag2id['O'])
                            
                candidate_position_ids.append(abs(j-pos))
                trigger_position_ids.append(abs(j-start_trigger))
        
            tmp['input_ids'].append(input_ids)
            tmp['candidate_position_ids'].append(candidate_position_ids)
            tmp['trigger_position_ids'].append(trigger_position_ids)
            tmp['postag_ids'].append(postag_ids)
            tmp['event_ids'].append(event_ids)
            tmp['start_trigger'].append(start_trigger)
            tmp['end_trigger'].append(end_trigger)
            tmp['sentence_id'].append(0)
            tmp['current_position_id'].append(pos)
    
    tmp['input_ids'] = torch.tensor(tmp['input_ids'])
    tmp['candidate_position_ids'] = torch.tensor(tmp['candidate_position_ids'])
    tmp['trigger_position_ids'] = torch.tensor(tmp['trigger_position_ids'])
    tmp['postag_ids'] = torch.tensor(tmp['postag_ids'])
    tmp['event_ids'] = torch.tensor(tmp['event_ids'])
    tmp['start_trigger'] = torch.tensor(tmp['start_trigger'])
    tmp['end_trigger'] = torch.tensor(tmp['end_trigger'])
    tmp['sentence_id'] = torch.tensor(tmp['sentence_id'])
    tmp['current_position_id'] = torch.tensor(tmp['current_position_id'])

    return tmp, words

def make_prediction_argument(inps, event_ids, start_triggers, end_triggers, sentence_ids, current_position_ids, id2event, id2argument):
    inps = inps.tolist()
    event_ids = [event_id[0] for event_id in event_ids.tolist()]
    start_triggers = start_triggers.tolist()
    end_triggers = end_triggers.tolist()
    sentence_ids = sentence_ids.tolist()
    current_position_ids = current_position_ids.tolist()

    argument_ids_of_sents = []
    argument_ids_of_sent = []
    event_id_of_sents = []
    start_triggers_of_sents =[]
    start_triggers_of_sent = []
    end_triggers_of_sents = []
    end_triggers_of_sent = []
    sentence_id_of_sents = []
    
    for i, current_position_id in enumerate(current_position_ids):
        if i == 0:
            argument_ids_of_sent = [inps[i]]
            start_triggers_of_sent = [start_triggers[i]]
            end_triggers_of_sent = [end_triggers[i]]
        else:
            if current_position_id == 0:
                argument_ids_of_sents.append(argument_ids_of_sent)
                event_id_of_sents.append(event_ids[i-1])
                start_triggers_of_sents.append(start_triggers_of_sent)
                end_triggers_of_sents.append(end_triggers_of_sent)
                sentence_id_of_sents.append(sentence_ids[i-1])

                argument_ids_of_sent = [inps[i]]
                start_triggers_of_sent = [start_triggers[i]]
                end_triggers_of_sent = [end_triggers[i]]
            else:
                argument_ids_of_sent.append(inps[i])
                start_triggers_of_sent.append(start_triggers[i])
                end_triggers_of_sent.append(end_triggers[i])

    argument_ids_of_sents.append(argument_ids_of_sent)
    event_id_of_sents.append(event_ids[len(event_ids)-1])
    start_triggers_of_sents.append(start_triggers_of_sent)
    start_trigger_of_sents = [start_triggers_of_sent[0] for start_triggers_of_sent in start_triggers_of_sents]
    end_triggers_of_sents.append(end_triggers_of_sent)
    end_trigger_of_sents = [end_triggers_of_sent[0] for end_triggers_of_sent in end_triggers_of_sents]
    sentence_id_of_sents.append(sentence_ids[len(sentence_ids)-1])
    
    result = []
    for i, argument_ids_of_sent in enumerate(argument_ids_of_sents):
        result_sent = []
        j = 0
        event_id = event_id_of_sents[i]
        event_type = id2event[event_id]
        start_trigger = start_trigger_of_sents[i]
        end_trigger = end_trigger_of_sents[i]
        sentence_id = sentence_id_of_sents[i]

        while j < len(argument_ids_of_sent):
            argument_id = argument_ids_of_sent[j]
            argument_type = id2argument[argument_id]
            if argument_type != 'O':
                tmp = [sentence_id, event_type, start_trigger, end_trigger, argument_type, j]
                while j+1 < len(argument_ids_of_sent) and argument_ids_of_sent[j+1] == argument_ids_of_sent[j]:
                    j += 1
                tmp.append(j)
                result_sent.append(tuple(tmp))
            j += 1
        result.extend(result_sent)
    
    return result

argument_model = ArgumentWordCNNWithPostag(
    num_labels=len(argument2id),
    num_words=len(word2id),
    word_embedding_dim=word_embedding_dim,
    max_length=max_length,
    position_embedding_dim=position_embedding_dim,
    num_events=len(event2id),
    event_embedding_dim=event_embedding_dim,
    num_postags=len(postag2id),
    postag_embedding_dim=postag_embedding_dim,
    kernel_sizes=kernel_sizes,
    num_filters=num_filters,
    dropout_rate=dropout_rate,
    use_pretrained=False,
    embedding_weight=None
)

argument_model.load_state_dict(torch.load('weight/cnn-argument-word-True-word2vec-True-postag.pth', map_location=device))

def predict_argument(sentence):

    events = predict_event(sentence)

    words = ViTokenizer.tokenize(sentence).split(' ')

    if len(events) == 0:
        return [], words

    inputs, words = preprocess_for_argument(sentence, events)
    
    input_ids = inputs['input_ids'].to(device)
    candidate_position_ids = inputs['candidate_position_ids'].to(device)
    trigger_position_ids = inputs['trigger_position_ids'].to(device)
    postag_ids = inputs['postag_ids'].to(device)
    event_ids = inputs['event_ids'].to(device)
    start_triggers = inputs['start_trigger'].to(device)
    end_triggers = inputs['end_trigger'].to(device)
                
    sentence_ids = inputs['sentence_id'].to(device)
    current_position_ids = inputs['current_position_id'].to(device)

    argument_model.to(device)

    argument_model.eval()
    with torch.no_grad():
        logits = argument_model(input_ids, candidate_position_ids, trigger_position_ids, event_ids, postag_ids)

    preds = torch.argmax(logits, dim=-1)
    preds = make_prediction_argument(preds, event_ids, start_triggers, end_triggers, \
        sentence_ids, current_position_ids, id2event, id2argument)
    
    # res = ' '.join(words) + '\n'
    
    # print(words)

    # for pred in preds:
    #     event_type = pred[1]
    #     start_trigger = pred[2]
    #     end_trigger = pred[3]
    #     argument_type = pred[4]
    #     start_arg = pred[5]
    #     end_arg = pred[6]

    #     res += f'{event_type}: {" ".join(words[start_trigger:end_trigger+1])}\t{start_trigger}\t{end_trigger}\t-\t{argument_type}: {" ".join(words[start_arg:end_arg+1])}\t{start_arg}\t{end_arg}' + '\n'

    return preds, words


# predict_argument("Minh sẽ rời Hà Nội để tới thành phố Hồ Chí Minh vào ngày mai.")


