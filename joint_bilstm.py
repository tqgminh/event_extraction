from models1 import *

import json
from copy import deepcopy

from pyvi import ViTokenizer, ViPosTagger
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

word_embedding_dim = 300
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

def preprocess(sentence):
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
    
    return tmp, words

def make_prediction(event_inps, arguments_inps, in_sentence, sentence_ids, id2event, id2argument):
    event_inps = event_inps.tolist()
    arguments_inps = arguments_inps.tolist()
    in_sentence = in_sentence.tolist()
    sentence_ids = sentence_ids.tolist()

    event_result = []
    argument_result = []
    for i in range(len(sentence_ids)):
        sentence_id = sentence_ids[i]
        event_inp = event_inps[i]
        argument_inps = arguments_inps[i]
        in_sent = in_sentence[i]

        start_sent = 0
        while in_sent[start_sent] == 0 and start_sent+1 < len(in_sent):
            start_sent += 1
        end_sent = start_sent
        while in_sent[end_sent] == 1 and end_sent+1 < len(in_sent):
            end_sent += 1
        
        event_inp = event_inp[start_sent:end_sent]
        argument_inps = argument_inps[start_sent:end_sent]
        
        j = 0
        while j < len(event_inp):
            event_id = event_inp[j]
            event_type = id2event[event_id]
            if event_type != 'O':
                tmp = [sentence_id, event_type, j]
                event_tmp = deepcopy(tmp)
                argument_inp = argument_inps[j][start_sent:end_sent]
                while j+1 < len(event_inp) and event_inp[j+1] == event_inp[j]:
                    j += 1
                    # arguments_inp_tmp.append(arguments_inp[j])
                event_tmp.append(j)
                event_result.append(tuple(event_tmp))
                
                k = 0
                while k < len(argument_inp):
                    argument_id = argument_inp[k]
                    argument_type = id2argument[argument_id]
                    if argument_type != 'O':
                        argument_tmp = deepcopy(event_tmp)
                        argument_tmp.extend([argument_type, k])
                        while k+1 < len(argument_inp) and argument_inp[k+1] == argument_inp[k]:
                            k += 1
                        argument_tmp.append(k)
                        argument_result.append(tuple(argument_tmp))
                    k += 1
            j += 1
    
    return event_result, argument_result

model = JointSentenceBiLSTM2_WithPostag(
    num_events=len(event2id),
    num_arguments=len(argument2id),
    num_words=len(word2id),
    word_embedding_dim=word_embedding_dim,
    max_length=max_length,
    num_postags=len(postag2id),
    postag_embedding_dim=postag_embedding_dim,
    hidden_size=hidden_size,
    dropout_rate=dropout_rate,
    use_pretrained=False,
    embedding_weight=None
)

model.load_state_dict(torch.load('weight/bilstm-joint-sentence2_-True-word2vec-True-postag.pth', map_location=device))

def predict(sentence):
    
    inputs, words = preprocess(sentence)

    input_ids = inputs['input_ids'].to(device)
    postag_ids = inputs['postag_ids'].to(device)
    in_sentence = inputs['in_sentence'].to(device)
    sentence_ids = inputs['sentence_id'].to(device)

    model.to(device)

    model.eval()
    with torch.no_grad():
        event_logits, arguments_logits = model(input_ids, postag_ids, in_sentence)
    
    event_preds = torch.argmax(event_logits, dim=-1)
    arguments_preds = torch.argmax(arguments_logits, dim=-1)
    event_preds, arguments_preds = make_prediction(event_preds, arguments_preds, in_sentence, sentence_ids, id2event, id2argument)

    # print(words)

    # for pred in arguments_preds:
    #     event_type = pred[1]
    #     start_trigger = pred[2]
    #     end_trigger = pred[3]
    #     argument_type = pred[4]
    #     start_arg = pred[5]
    #     end_arg = pred[6]

    #     print(f'{event_type}: {" ".join(words[start_trigger:end_trigger+1])}\t{start_trigger}\t{end_trigger}\t-\t{argument_type}: {" ".join(words[start_arg:end_arg+1])}\t{start_arg}\t{end_arg}')

    return arguments_preds, words

# predict("Minh sẽ rời Hà Nội để tới thành phố Hồ Chí Minh vào ngày mai.")
