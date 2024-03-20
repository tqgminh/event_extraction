import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F


class EventWordCNN(nn.Module):
    def __init__(self, num_labels, num_words, word_embedding_dim, window_size, position_embedding_dim,
                kernel_sizes, num_filters, dropout_rate, use_pretrained=False, embedding_weight=None):
        
        super(EventWordCNN, self).__init__()

        self.num_labels = num_labels
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.window_size = window_size
        self.position_embedding_dim = position_embedding_dim  
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        
        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.position_embedding = nn.Embedding(num_embeddings=window_size+1, embedding_dim=position_embedding_dim)
        self.embedding_dim = word_embedding_dim + position_embedding_dim

        self.dropout = nn.Dropout(dropout_rate)

        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, [kernel_size, self.embedding_dim])
                                    for kernel_size in kernel_sizes])

        self.linear = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, input_ids, position_ids, labels=None):
        
        word_emb = self.word_embedding(input_ids)
        position_emb = self.position_embedding(position_ids)
        emb = torch.cat((word_emb, position_emb), 2)
        
        emb = torch.unsqueeze(emb, 1)

        cnns = []
        for conv in self.convs:
            cnn = F.relu(conv(emb))
            cnn = torch.squeeze(cnn, -1)
            cnn = F.max_pool1d(cnn, cnn.size(2))
            cnns.append(cnn)
        
        cnn = torch.cat(cnns, 2)
        flat = cnn.view(cnn.size(0), -1)
        flat = self.dropout(flat)
        logits = self.linear(flat)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class EventWordCNNWithPostag(nn.Module):
    def __init__(self, num_labels, num_words, word_embedding_dim, window_size, 
                position_embedding_dim, num_postags, postag_embedding_dim, kernel_sizes, 
                num_filters, dropout_rate, use_pretrained=False, embedding_weight=None):
        
        super(EventWordCNNWithPostag, self).__init__()

        self.num_labels = num_labels
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.window_size = window_size
        self.position_embedding_dim = position_embedding_dim
        self.num_postags = num_postags
        self.postag_embedding_dim = postag_embedding_dim  
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        
        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.position_embedding = nn.Embedding(num_embeddings=window_size+1, embedding_dim=position_embedding_dim)
        self.postag_embedding = nn.Embedding(num_embeddings=num_postags, embedding_dim=postag_embedding_dim)    
        self.embedding_dim = word_embedding_dim + position_embedding_dim + postag_embedding_dim

        self.dropout = nn.Dropout(dropout_rate)

        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, [kernel_size, self.embedding_dim])
                                    for kernel_size in kernel_sizes])

        self.linear = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, input_ids, position_ids, postag_ids, labels=None):
        
        word_emb = self.word_embedding(input_ids)
        position_emb = self.position_embedding(position_ids)
        postag_emb = self.postag_embedding(postag_ids)
        emb = torch.cat((word_emb, position_emb, postag_emb), 2)

        emb = torch.unsqueeze(emb, 1)

        cnns = []
        for conv in self.convs:
            cnn = F.relu(conv(emb))
            cnn = torch.squeeze(cnn, -1)
            cnn = F.max_pool1d(cnn, cnn.size(2))
            cnns.append(cnn)
        
        cnn = torch.cat(cnns, 2)
        flat = cnn.view(cnn.size(0), -1)
        flat = self.dropout(flat)
        logits = self.linear(flat)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class ArgumentWordCNN(nn.Module):
    def __init__(self, num_labels, num_words, word_embedding_dim, max_length, 
                position_embedding_dim, num_events, event_embedding_dim, kernel_sizes,
                num_filters, dropout_rate, use_pretrained=False, embedding_weight=None):
        
        super(ArgumentWordCNN, self).__init__()

        self.num_labels = num_labels
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.max_length = max_length
        self.position_embedding_dim = position_embedding_dim
        self.num_events = num_events
        self.event_embedding_dim = event_embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.candidate_position_embedding = nn.Embedding(num_embeddings=max_length, embedding_dim=position_embedding_dim)
        self.trigger_position_embedding = nn.Embedding(num_embeddings=max_length, embedding_dim=position_embedding_dim)
        self.event_embedding = nn.Embedding(num_embeddings=num_events, embedding_dim=event_embedding_dim)    
        self.embedding_dim = word_embedding_dim + 2*position_embedding_dim + event_embedding_dim

        self.dropout = nn.Dropout(dropout_rate)

        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, [kernel_size, self.embedding_dim])
                                    for kernel_size in kernel_sizes])

        self.linear = nn.Linear(num_filters * len(kernel_sizes), num_labels)
    
    def forward(self, input_ids, candidate_position_ids, trigger_position_ids, event_ids, labels=None):

        word_emb = self.word_embedding(input_ids)
        candidate_position_emb = self.candidate_position_embedding(candidate_position_ids)
        trigger_position_emb = self.trigger_position_embedding(trigger_position_ids)
        event_emb = self.event_embedding(event_ids)
        
        emb = torch.cat((word_emb, candidate_position_emb, trigger_position_emb, event_emb), 2)

        emb = torch.unsqueeze(emb, 1)

        cnns = []
        for conv in self.convs:
            cnn = F.relu(conv(emb))
            cnn = torch.squeeze(cnn, -1)
            cnn = F.max_pool1d(cnn, cnn.size(2))
            cnns.append(cnn)
        
        cnn = torch.cat(cnns, 2)
        flat = cnn.view(cnn.size(0), -1)
        flat = self.dropout(flat)
        logits = self.linear(flat)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class ArgumentWordCNNWithPostag(nn.Module):
    def __init__(self, num_labels, num_words, word_embedding_dim, max_length, position_embedding_dim,
                num_events, event_embedding_dim, num_postags, postag_embedding_dim, 
                kernel_sizes, num_filters, dropout_rate, use_pretrained=False, embedding_weight=None):
        
        super(ArgumentWordCNNWithPostag, self).__init__()

        self.num_labels = num_labels
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.max_length = max_length
        self.position_embedding_dim = position_embedding_dim
        self.num_events = num_events
        self.event_embedding_dim = event_embedding_dim
        self.num_postags = num_postags
        self.postag_embedding_dim = postag_embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.candidate_position_embedding = nn.Embedding(num_embeddings=max_length, embedding_dim=position_embedding_dim)
        self.trigger_position_embedding = nn.Embedding(num_embeddings=max_length, embedding_dim=position_embedding_dim)
        self.event_embedding = nn.Embedding(num_embeddings=num_events, embedding_dim=event_embedding_dim)
        self.postag_embedding = nn.Embedding(num_embeddings=num_postags, embedding_dim=postag_embedding_dim)
        self.embedding_dim = word_embedding_dim + 2*position_embedding_dim + event_embedding_dim + postag_embedding_dim

        self.dropout = nn.Dropout(dropout_rate)

        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, [kernel_size, self.embedding_dim])
                                    for kernel_size in kernel_sizes])

        self.linear = nn.Linear(num_filters * len(kernel_sizes), num_labels)
    
    def forward(self, input_ids, candidate_position_ids, trigger_position_ids, event_ids, postag_ids, labels=None):

        word_emb = self.word_embedding(input_ids)
        candidate_position_emb = self.candidate_position_embedding(candidate_position_ids)
        trigger_position_emb = self.trigger_position_embedding(trigger_position_ids)
        event_emb = self.event_embedding(event_ids)
        postag_emb = self.postag_embedding(postag_ids)
        emb = torch.cat((word_emb, candidate_position_emb, trigger_position_emb, event_emb, postag_emb), 2)

        emb = torch.unsqueeze(emb, 1)

        cnns = []
        for conv in self.convs:
            cnn = F.relu(conv(emb))
            cnn = torch.squeeze(cnn, -1)
            cnn = F.max_pool1d(cnn, cnn.size(2))
            cnns.append(cnn)
        
        cnn = torch.cat(cnns, 2)
        flat = cnn.view(cnn.size(0), -1)
        flat = self.dropout(flat)
        logits = self.linear(flat)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class EventSentenceBiLSTM(nn.Module):
    def __init__(self, num_labels, num_words, word_embedding_dim, hidden_size,
                dropout_rate, use_pretrained=False, embedding_weight=None):

        super(EventSentenceBiLSTM, self).__init__()

        self.num_labels = num_labels
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.embedding_dim = word_embedding_dim

        self.dropout = nn.Dropout(dropout_rate)

        self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        
        self.linear = nn.Linear(hidden_size * 2, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):

        word_emb = self.word_embedding(input_ids)
        emb = word_emb
        # emb = torch.cat((word_emb, entity_emb, postag_emb), 2)

        hidden_states, _ = self.bilstm(emb)
        hidden_states = self.dropout(hidden_states)
        logits = self.linear(hidden_states)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits


class EventSentenceBiLSTMWithPostag(nn.Module):
    def __init__(self, num_labels, num_words, word_embedding_dim, num_postags, postag_embedding_dim, \
                hidden_size, dropout_rate, use_pretrained=False, embedding_weight=None):

        super(EventSentenceBiLSTMWithPostag, self).__init__()

        self.num_labels = num_labels
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.num_postags = num_postags
        self.postag_embedding_dim = postag_embedding_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.postag_embedding = nn.Embedding(num_embeddings=num_postags, embedding_dim=postag_embedding_dim)
        self.embedding_dim = word_embedding_dim + postag_embedding_dim

        self.dropout = nn.Dropout(dropout_rate)

        self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        
        self.linear = nn.Linear(hidden_size * 2, num_labels)
    
    def forward(self, input_ids, postag_ids, attention_mask=None, labels=None):

        word_emb = self.word_embedding(input_ids)
        postag_emb = self.postag_embedding(postag_ids)
        emb = torch.cat((word_emb, postag_emb), 2)

        hidden_states, _ = self.bilstm(emb)
        hidden_states = self.dropout(hidden_states)
        logits = self.linear(hidden_states)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits


class ArgumentSentenceBiLSTM(nn.Module):
    def __init__(self, num_labels, num_words, word_embedding_dim, max_length, 
                position_embedding_dim, num_events, event_embedding_dim, \
                hidden_size, dropout_rate, use_pretrained=False, embedding_weight=None):
            
        super(ArgumentSentenceBiLSTM, self).__init__()

        self.num_labels = num_labels
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.max_length = max_length
        self.position_embedding_dim = position_embedding_dim
        self.num_events = num_events
        self.event_embedding_dim = event_embedding_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.trigger_position_embedding = nn.Embedding(num_embeddings=max_length, embedding_dim=position_embedding_dim)
        self.event_embedding = nn.Embedding(num_embeddings=num_events, embedding_dim=event_embedding_dim)
        self.embedding_dim = word_embedding_dim + position_embedding_dim + event_embedding_dim

        self.dropout = nn.Dropout(dropout_rate)

        self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        
        self.linear = nn.Linear(hidden_size * 2, num_labels)
    
    def forward(self, input_ids, trigger_position_ids, event_ids, attention_mask=None, labels=None):

        word_emb = self.word_embedding(input_ids)
        trigger_position_emb = self.trigger_position_embedding(trigger_position_ids)
        event_emb = self.event_embedding(event_ids)
        emb = torch.cat((word_emb, trigger_position_emb, event_emb), 2)

        hidden_states, _ = self.bilstm(emb)
        hidden_states = self.dropout(hidden_states)
        logits = self.linear(hidden_states)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits


class ArgumentSentenceBiLSTMWithPostag(nn.Module):
    def __init__(self, num_labels, num_words, word_embedding_dim, max_length, position_embedding_dim, 
                num_events, event_embedding_dim, num_postags, postag_embedding_dim,\
                hidden_size, dropout_rate, use_pretrained=False, embedding_weight=None):
            
        super(ArgumentSentenceBiLSTMWithPostag, self).__init__()

        self.num_labels = num_labels
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.max_length = max_length
        self.position_embedding_dim = position_embedding_dim
        self.num_events = num_events
        self.event_embedding_dim = event_embedding_dim
        self.num_postags = num_postags
        self.postag_embedding_dim = postag_embedding_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.trigger_position_embedding = nn.Embedding(num_embeddings=max_length, embedding_dim=position_embedding_dim)
        self.event_embedding = nn.Embedding(num_embeddings=num_events, embedding_dim=event_embedding_dim)
        self.postag_embedding = nn.Embedding(num_embeddings=num_postags, embedding_dim=postag_embedding_dim)
        self.embedding_dim = word_embedding_dim + position_embedding_dim + event_embedding_dim + postag_embedding_dim

        self.dropout = nn.Dropout(dropout_rate)

        self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        
        self.linear = nn.Linear(hidden_size * 2, num_labels)
    
    def forward(self, input_ids, trigger_position_ids, event_ids, postag_ids, attention_mask=None, labels=None):

        word_emb = self.word_embedding(input_ids)
        trigger_position_emb = self.trigger_position_embedding(trigger_position_ids)
        event_emb = self.event_embedding(event_ids)
        postag_emb = self.postag_embedding(postag_ids)
        emb = torch.cat((word_emb, trigger_position_emb, event_emb, postag_emb), 2)

        hidden_states, _ = self.bilstm(emb)
        hidden_states = self.dropout(hidden_states)
        logits = self.linear(hidden_states)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits


class JointSentenceBiLSTM(nn.Module):
    def __init__(self, num_events, num_arguments, num_words, word_embedding_dim, 
                max_length, hidden_size, dropout_rate,
                use_pretrained=False, embedding_weight=None):
        
        super(JointSentenceBiLSTM, self).__init__()

        self.num_events = num_events
        self.num_arguments = num_arguments
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.embedding_dim = word_embedding_dim
        
        self.event_dropout = nn.Dropout(dropout_rate)
        self.argument_dropout = nn.Dropout(dropout_rate)

        self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        
        self.event_linear = nn.Linear(hidden_size * 2, num_events)

        self.argument_linear = nn.Linear(hidden_size * 4, num_arguments)

    def forward(self, input_ids, attention_mask=None, event_labels=None, arguments_labels=None):

        word_emb = self.word_embedding(input_ids)
        emb = word_emb

        hidden_states, _ = self.bilstm(emb)
        
        list_event_logits = []
        list_arguments_logits = []
        
        for i in range(self.max_length):
            trigger_hidden_state = hidden_states[:, i]
            event_hidden_state = self.event_dropout(trigger_hidden_state)
            event_logit = self.event_linear(event_hidden_state)
            list_event_logits.append(event_logit)
            
            trigger_hidden_states = torch.stack([trigger_hidden_state] * self.max_length, dim=1)
            argument_hidden_states = torch.cat((hidden_states, trigger_hidden_states), dim=2)
            argument_hidden_states = self.argument_dropout(argument_hidden_states)
            argument_logits = self.argument_linear(argument_hidden_states)
            list_arguments_logits.append(argument_logits)
        
        event_logits = torch.stack(list_event_logits, dim=1)
        arguments_logits = torch.stack(list_arguments_logits, dim=1)

        if event_labels is not None and arguments_labels is not None:
            event_loss_fct = CrossEntropyLoss()
            argument_loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_event_logits = event_logits.view(-1, self.num_events)[active_loss]
                active_event_labels = event_labels.view(-1)[active_loss]
                event_loss = event_loss_fct(active_event_logits, active_event_labels)

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)[active_loss]
                active_arguments_labels = arguments_labels.view(-1, self.max_length)[active_loss]
                
                active_argument_loss = active_event_labels != 0
                active_arguments_logits = active_arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = active_arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            else:
                event_loss = event_loss_fct(event_logits.view(-1, self.num_events), event_labels.view(-1))

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)
                active_arguments_labels = arguments_labels.view(-1, self.max_length)

                active_argument_loss = event_labels != 0
                active_arguments_logits = arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            
            return loss, event_logits, arguments_logits
        else:
            return event_logits, arguments_logits

# g_trg_arg
class JointSentenceBiLSTM1(nn.Module):
    def __init__(self, num_events, num_arguments, num_words, word_embedding_dim, 
                max_length, hidden_size, dropout_rate,
                use_pretrained=False, embedding_weight=None):
        
        super(JointSentenceBiLSTM1, self).__init__()

        self.num_events = num_events
        self.num_arguments = num_arguments
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.embedding_dim = word_embedding_dim
        
        self.event_dropout = nn.Dropout(dropout_rate)
        self.argument_dropout = nn.Dropout(dropout_rate)

        self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        
        self.event_linear = nn.Linear(hidden_size * 2, num_events)

        self.argument_linear = nn.Linear(hidden_size*4 + num_events-1, num_arguments)

        self.initial_g_trg_arg_row = torch.zeros((max_length, num_events-1), dtype=torch.int64)
    
    def update_g_trg_arg(self, g_trg_arg, event_pred, argument_preds):

        """
            g_tr_arg: batch_size * max_length * num_events
            event_pred: batch_size
            argument_preds: batch_size * max_length 
        """
        
        new_g_trg_arg = g_trg_arg # batch_size * max_length * num_events
        event_pred_bool = event_pred > 0 
        updated_batch_indices = event_pred_bool.nonzero().squeeze(-1) # n
        
        if updated_batch_indices.shape == torch.Size([0]):
            return new_g_trg_arg
        
        updated_g_trg_arg = new_g_trg_arg.index_select(0, updated_batch_indices) # n * max_length * num_events
        kept_event_pred = event_pred[event_pred_bool] # n
        kept_argument_preds = argument_preds.index_select(0, updated_batch_indices) # n * max_length

        kept_event_pred_ = torch.sub(kept_event_pred, 1)
        for i in range(kept_event_pred_.size(0)):
            updated_g_trg_arg_i = updated_g_trg_arg[i] # max_length * num_events
            kept_event_pred_i = kept_event_pred_[i] # 1
            kept_argument_preds_i = kept_argument_preds[i] # max_length 
            
            updated_g_trg_arg_i_column = updated_g_trg_arg_i.index_select(1, kept_event_pred_i) # max_length * 1
            kept_argument_preds_i = (kept_argument_preds_i > 0).long().unsqueeze(-1) # max_length * 1
            updated_g_trg_arg_i_column = torch.where(kept_argument_preds_i > 0, kept_argument_preds_i, updated_g_trg_arg_i_column)
            updated_g_trg_arg[i] = updated_g_trg_arg[i].index_copy(dim=1, index=kept_event_pred_i, source=updated_g_trg_arg_i_column)

        new_g_trg_arg = new_g_trg_arg.index_copy_(dim=0, index=updated_batch_indices, source=updated_g_trg_arg)
        
        return new_g_trg_arg

    def forward(self, input_ids, attention_mask=None, event_labels=None, arguments_labels=None):
        
        device = input_ids.device
        batch_size = input_ids.size(0)
        g_trg_arg = torch.stack([self.initial_g_trg_arg_row] * batch_size, dim=0).to(device)

        word_emb = self.word_embedding(input_ids)
        emb = word_emb

        hidden_states, _ = self.bilstm(emb)
        
        list_event_logits = []
        list_arguments_logits = []
        
        for i in range(self.max_length):
            trigger_hidden_state = hidden_states[:, i]
            event_hidden_state = self.event_dropout(trigger_hidden_state)
            event_logit = self.event_linear(event_hidden_state)
            list_event_logits.append(event_logit)
            event_pred = torch.argmax(event_logit, dim=-1)
            
            trigger_hidden_states = torch.stack([trigger_hidden_state] * self.max_length, dim=1)
            argument_hidden_states = torch.cat((hidden_states, trigger_hidden_states, g_trg_arg), dim=2)
            argument_hidden_states = self.argument_dropout(argument_hidden_states)
            argument_logits = self.argument_linear(argument_hidden_states)
            list_arguments_logits.append(argument_logits)
            argument_preds = torch.argmax(argument_logits, dim=-1)
            
            g_trg_arg = self.update_g_trg_arg(g_trg_arg, event_pred, argument_preds)
        
        event_logits = torch.stack(list_event_logits, dim=1)
        arguments_logits = torch.stack(list_arguments_logits, dim=1)

        if event_labels is not None and arguments_labels is not None:
            event_loss_fct = CrossEntropyLoss()
            argument_loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_event_logits = event_logits.view(-1, self.num_events)[active_loss]
                active_event_labels = event_labels.view(-1)[active_loss]
                event_loss = event_loss_fct(active_event_logits, active_event_labels)

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)[active_loss]
                active_arguments_labels = arguments_labels.view(-1, self.max_length)[active_loss]
                
                active_argument_loss = active_event_labels != 0
                active_arguments_logits = active_arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = active_arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            else:
                event_loss = event_loss_fct(event_logits.view(-1, self.num_events), event_labels.view(-1))

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)
                active_arguments_labels = arguments_labels.view(-1, self.max_length)

                active_argument_loss = event_labels != 0
                active_arguments_logits = arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            
            return loss, event_logits, arguments_logits
        else:
            return event_logits, arguments_logits

# g_arg
class JointSentenceBiLSTM2(nn.Module):
    def __init__(self, num_events, num_arguments, num_words, word_embedding_dim, 
                max_length, hidden_size, dropout_rate,
                use_pretrained=False, embedding_weight=None):
        
        super(JointSentenceBiLSTM2, self).__init__()

        self.num_events = num_events
        self.num_arguments = num_arguments
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.embedding_dim = word_embedding_dim
        
        self.event_dropout = nn.Dropout(dropout_rate)
        self.argument_dropout = nn.Dropout(dropout_rate)

        self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        
        self.event_linear = nn.Linear(hidden_size * 2, num_events)

        self.argument_linear = nn.Linear(hidden_size*4 + num_arguments-1, num_arguments)

        self.initial_g_arg_row = torch.zeros((max_length, num_arguments-1), dtype=torch.int64)
    
    def update_g_arg(self, g_arg, event_pred, argument_preds):

        """
            g_arg: batch_size * max_length * num_arguments
            event_pred: batch_size
            argument_preds: batch_size * max_length 
        """

        device = g_arg.device
        
        new_g_arg = g_arg # batch_size * max_length * num_arguments
        event_pred_bool = event_pred > 0 
        updated_batch_indices = event_pred_bool.nonzero().squeeze(-1) # n
        
        if updated_batch_indices.shape == torch.Size([0]):
            return new_g_arg
        
        updated_g_arg = new_g_arg.index_select(0, updated_batch_indices) # n * max_length * num_arguments
        kept_argument_preds = argument_preds.index_select(0, updated_batch_indices) # n * max_length

        for i in range(kept_argument_preds.size(0)):
            updated_g_arg_i = updated_g_arg[i] # max_length * num_arguments
            kept_argument_preds_i = kept_argument_preds[i] # max_length 
            
            for j in range(kept_argument_preds_i.size(0)):
                if kept_argument_preds_i[j].sub(1) == -1:
                    continue
                updated_g_arg_i[j] = updated_g_arg_i[j].index_copy_(dim=0, index=kept_argument_preds_i[j].sub(1), source=torch.tensor(1).to(device))
            updated_g_arg[i] = updated_g_arg_i

        new_g_arg = new_g_arg.index_copy_(dim=0, index=updated_batch_indices, source=updated_g_arg)
        
        return new_g_arg

    def forward(self, input_ids, attention_mask=None, event_labels=None, arguments_labels=None):
        
        device = input_ids.device
        batch_size = input_ids.size(0)
        g_arg = torch.stack([self.initial_g_arg_row] * batch_size, dim=0).to(device)

        word_emb = self.word_embedding(input_ids)
        emb = word_emb

        hidden_states, _ = self.bilstm(emb)
        
        list_event_logits = []
        list_arguments_logits = []
        
        for i in range(self.max_length):
            trigger_hidden_state = hidden_states[:, i]
            event_hidden_state = self.event_dropout(trigger_hidden_state)
            event_logit = self.event_linear(event_hidden_state)
            list_event_logits.append(event_logit)
            event_pred = torch.argmax(event_logit, dim=-1)
            
            trigger_hidden_states = torch.stack([trigger_hidden_state] * self.max_length, dim=1)
            argument_hidden_states = torch.cat((hidden_states, trigger_hidden_states, g_arg), dim=2)
            argument_hidden_states = self.argument_dropout(argument_hidden_states)
            argument_logits = self.argument_linear(argument_hidden_states)
            list_arguments_logits.append(argument_logits)
            argument_preds = torch.argmax(argument_logits, dim=-1)
            
            g_arg = self.update_g_arg(g_arg, event_pred, argument_preds)
        
        event_logits = torch.stack(list_event_logits, dim=1)
        arguments_logits = torch.stack(list_arguments_logits, dim=1)

        if event_labels is not None and arguments_labels is not None:
            event_loss_fct = CrossEntropyLoss()
            argument_loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_event_logits = event_logits.view(-1, self.num_events)[active_loss]
                active_event_labels = event_labels.view(-1)[active_loss]
                event_loss = event_loss_fct(active_event_logits, active_event_labels)

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)[active_loss]
                active_arguments_labels = arguments_labels.view(-1, self.max_length)[active_loss]
                
                active_argument_loss = active_event_labels != 0
                active_arguments_logits = active_arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = active_arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            else:
                event_loss = event_loss_fct(event_logits.view(-1, self.num_events), event_labels.view(-1))

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)
                active_arguments_labels = arguments_labels.view(-1, self.max_length)

                active_argument_loss = event_labels != 0
                active_arguments_logits = arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            
            return loss, event_logits, arguments_logits
        else:
            return event_logits, arguments_logits


class JointSentenceBiLSTM12(nn.Module):
    def __init__(self, num_events, num_arguments, num_words, word_embedding_dim, 
                max_length, hidden_size, dropout_rate,
                use_pretrained=False, embedding_weight=None):
        
        super(JointSentenceBiLSTM12, self).__init__()

        self.num_events = num_events
        self.num_arguments = num_arguments
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.embedding_dim = word_embedding_dim
        
        self.event_dropout = nn.Dropout(dropout_rate)
        self.argument_dropout = nn.Dropout(dropout_rate)

        self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        
        self.event_linear = nn.Linear(hidden_size * 2, num_events)

        self.argument_linear = nn.Linear(hidden_size*4 + num_arguments-1 + num_events-1 , num_arguments)

        self.initial_g_trg_arg_row = torch.zeros((max_length, num_events-1), dtype=torch.int64)

        self.initial_g_arg_row = torch.zeros((max_length, num_arguments-1), dtype=torch.int64)
    
    def update_g_trg_arg(self, g_trg_arg, event_pred, argument_preds):

        """
            g_tr_arg: batch_size * max_length * num_events
            event_pred: batch_size
            argument_preds: batch_size * max_length 
        """
        
        new_g_trg_arg = g_trg_arg # batch_size * max_length * num_events
        event_pred_bool = event_pred > 0 
        updated_batch_indices = event_pred_bool.nonzero().squeeze(-1) # n
        
        if updated_batch_indices.shape == torch.Size([0]):
            return new_g_trg_arg
        
        updated_g_trg_arg = new_g_trg_arg.index_select(0, updated_batch_indices) # n * max_length * num_events
        kept_event_pred = event_pred[event_pred_bool] # n
        kept_argument_preds = argument_preds.index_select(0, updated_batch_indices) # n * max_length

        kept_event_pred_ = torch.sub(kept_event_pred, 1)
        for i in range(kept_event_pred_.size(0)):
            updated_g_trg_arg_i = updated_g_trg_arg[i] # max_length * num_events
            kept_event_pred_i = kept_event_pred_[i] # 1
            kept_argument_preds_i = kept_argument_preds[i] # max_length 
            
            updated_g_trg_arg_i_column = updated_g_trg_arg_i.index_select(1, kept_event_pred_i) # max_length * 1
            kept_argument_preds_i = (kept_argument_preds_i > 0).long().unsqueeze(-1) # max_length * 1
            updated_g_trg_arg_i_column = torch.where(kept_argument_preds_i > 0, kept_argument_preds_i, updated_g_trg_arg_i_column)
            updated_g_trg_arg[i] = updated_g_trg_arg[i].index_copy(dim=1, index=kept_event_pred_i, source=updated_g_trg_arg_i_column)

        new_g_trg_arg = new_g_trg_arg.index_copy_(dim=0, index=updated_batch_indices, source=updated_g_trg_arg)
        
        return new_g_trg_arg
    
    def update_g_arg(self, g_arg, event_pred, argument_preds):

        """
            g_arg: batch_size * max_length * num_arguments
            event_pred: batch_size
            argument_preds: batch_size * max_length 
        """

        device = g_arg.device
        
        new_g_arg = g_arg # batch_size * max_length * num_arguments
        event_pred_bool = event_pred > 0 
        updated_batch_indices = event_pred_bool.nonzero().squeeze(-1) # n
        
        if updated_batch_indices.shape == torch.Size([0]):
            return new_g_arg
        
        updated_g_arg = new_g_arg.index_select(0, updated_batch_indices) # n * max_length * num_arguments
        kept_argument_preds = argument_preds.index_select(0, updated_batch_indices) # n * max_length

        for i in range(kept_argument_preds.size(0)):
            updated_g_arg_i = updated_g_arg[i] # max_length * num_arguments
            kept_argument_preds_i = kept_argument_preds[i] # max_length 
            
            for j in range(kept_argument_preds_i.size(0)):
                if kept_argument_preds_i[j].sub(1) == -1:
                    continue
                updated_g_arg_i[j] = updated_g_arg_i[j].index_copy_(dim=0, index=kept_argument_preds_i[j].sub(1), source=torch.tensor(1).to(device))
            updated_g_arg[i] = updated_g_arg_i

        new_g_arg = new_g_arg.index_copy_(dim=0, index=updated_batch_indices, source=updated_g_arg)
        
        return new_g_arg

    def forward(self, input_ids, attention_mask=None, event_labels=None, arguments_labels=None):

        device = input_ids.device
        batch_size = input_ids.size(0)
        g_trg_arg = torch.stack([self.initial_g_trg_arg_row] * batch_size, dim=0).to(device)
        g_arg = torch.stack([self.initial_g_arg_row] * batch_size, dim=0).to(device)

        word_emb = self.word_embedding(input_ids)
        emb = word_emb

        hidden_states, _ = self.bilstm(emb)
        
        list_event_logits = []
        list_arguments_logits = []
        
        for i in range(self.max_length):
            trigger_hidden_state = hidden_states[:, i]
            event_hidden_state = self.event_dropout(trigger_hidden_state)
            event_logit = self.event_linear(event_hidden_state)
            list_event_logits.append(event_logit)
            event_pred = torch.argmax(event_logit, dim=-1)
            
            trigger_hidden_states = torch.stack([trigger_hidden_state] * self.max_length, dim=1)
            argument_hidden_states = torch.cat((hidden_states, trigger_hidden_states, g_arg, g_trg_arg), dim=2)
            argument_hidden_states = self.argument_dropout(argument_hidden_states)
            argument_logits = self.argument_linear(argument_hidden_states)
            list_arguments_logits.append(argument_logits)
            argument_preds = torch.argmax(argument_logits, dim=-1)
            
            g_trg_arg = self.update_g_trg_arg(g_trg_arg, event_pred, argument_preds)
            g_arg = self.update_g_arg(g_arg, event_pred, argument_preds)
        
        event_logits = torch.stack(list_event_logits, dim=1)
        arguments_logits = torch.stack(list_arguments_logits, dim=1)

        if event_labels is not None and arguments_labels is not None:
            event_loss_fct = CrossEntropyLoss()
            argument_loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_event_logits = event_logits.view(-1, self.num_events)[active_loss]
                active_event_labels = event_labels.view(-1)[active_loss]
                event_loss = event_loss_fct(active_event_logits, active_event_labels)

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)[active_loss]
                active_arguments_labels = arguments_labels.view(-1, self.max_length)[active_loss]
                
                active_argument_loss = active_event_labels != 0
                active_arguments_logits = active_arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = active_arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            else:
                event_loss = event_loss_fct(event_logits.view(-1, self.num_events), event_labels.view(-1))

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)
                active_arguments_labels = arguments_labels.view(-1, self.max_length)

                active_argument_loss = event_labels != 0
                active_arguments_logits = arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            
            return loss, event_logits, arguments_logits
        else:
            return event_logits, arguments_logits

class JointSentenceBiLSTM_(nn.Module):
    def __init__(self, num_events, num_arguments, num_words, word_embedding_dim, 
                max_length, hidden_size, dropout_rate,
                use_pretrained=False, embedding_weight=None):
        
        super(JointSentenceBiLSTM_, self).__init__()

        self.num_events = num_events
        self.num_arguments = num_arguments
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.embedding_dim = word_embedding_dim
        
        self.event_dropout = nn.Dropout(dropout_rate)
        self.argument_dropout = nn.Dropout(dropout_rate)

        self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        
        self.event_linear = nn.Linear(hidden_size*2 + num_events-1, num_events)

        self.argument_linear = nn.Linear(hidden_size*4, num_arguments)

        self.initial_g_trg_row = torch.zeros((num_events-1), dtype=torch.int64)
    
    def update_g_trg(self, g_trg, event_pred):

        """
            g_trg: batch_size * num_events
            event_pred: batch_size
        """
        
        device = g_trg.device

        new_g_trg = g_trg
        event_pred_bool = event_pred > 0
        updated_batch_indices = event_pred_bool.nonzero().squeeze(-1) # n

        if updated_batch_indices.shape == torch.Size([0]):
            return new_g_trg
        
        updated_g_trg = new_g_trg.index_select(0, updated_batch_indices) # n * num_events
        kept_event_pred = event_pred.index_select(0, updated_batch_indices) # n

        for i in range(kept_event_pred.size(0)):
            # updated_g_trg_i = updated_g_trg[i] # num_events
            kept_event_pred_i = kept_event_pred[i] # 1

            if kept_event_pred_i.sub(1) == -1:
                continue
            
            updated_g_trg[i] = updated_g_trg[i].index_copy_(dim=0, index=kept_event_pred_i.sub(1), source=torch.tensor(1).to(device))

        new_g_trg = new_g_trg.index_copy_(dim=0, index=updated_batch_indices, source=updated_g_trg)

        return new_g_trg

    def forward(self, input_ids, attention_mask=None, event_labels=None, arguments_labels=None):

        device = input_ids.device
        batch_size = input_ids.size(0)
        g_trg = torch.stack([self.initial_g_trg_row] * batch_size, dim=0).to(device)

        word_emb = self.word_embedding(input_ids)
        emb = word_emb

        hidden_states, _ = self.bilstm(emb)
        
        list_event_logits = []
        list_arguments_logits = []
        
        for i in range(self.max_length):
            trigger_hidden_state = hidden_states[:, i]
            event_hidden_state = torch.cat((trigger_hidden_state, g_trg), dim=1)
            event_hidden_state = self.event_dropout(event_hidden_state)
            event_logit = self.event_linear(event_hidden_state)
            list_event_logits.append(event_logit)
            event_pred = torch.argmax(event_logit, dim=-1)
            
            trigger_hidden_states = torch.stack([trigger_hidden_state] * self.max_length, dim=1)
            argument_hidden_states = torch.cat((hidden_states, trigger_hidden_states), dim=2)
            argument_hidden_states = self.argument_dropout(argument_hidden_states)
            argument_logits = self.argument_linear(argument_hidden_states)
            list_arguments_logits.append(argument_logits)

            g_trg = self.update_g_trg(g_trg, event_pred)
        
        event_logits = torch.stack(list_event_logits, dim=1)
        arguments_logits = torch.stack(list_arguments_logits, dim=1)

        if event_labels is not None and arguments_labels is not None:
            event_loss_fct = CrossEntropyLoss()
            argument_loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_event_logits = event_logits.view(-1, self.num_events)[active_loss]
                active_event_labels = event_labels.view(-1)[active_loss]
                event_loss = event_loss_fct(active_event_logits, active_event_labels)

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)[active_loss]
                active_arguments_labels = arguments_labels.view(-1, self.max_length)[active_loss]
                
                active_argument_loss = active_event_labels != 0
                active_arguments_logits = active_arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = active_arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            else:
                event_loss = event_loss_fct(event_logits.view(-1, self.num_events), event_labels.view(-1))

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)
                active_arguments_labels = arguments_labels.view(-1, self.max_length)

                active_argument_loss = event_labels != 0
                active_arguments_logits = arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            
            return loss, event_logits, arguments_logits
        else:
            return event_logits, arguments_logits

# g_trg_arg
class JointSentenceBiLSTM1_(nn.Module):
    def __init__(self, num_events, num_arguments, num_words, word_embedding_dim, 
                max_length, hidden_size, dropout_rate,
                use_pretrained=False, embedding_weight=None):
        
        super(JointSentenceBiLSTM1_, self).__init__()

        self.num_events = num_events
        self.num_arguments = num_arguments
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.embedding_dim = word_embedding_dim
        
        self.event_dropout = nn.Dropout(dropout_rate)
        self.argument_dropout = nn.Dropout(dropout_rate)

        self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        
        self.event_linear = nn.Linear(hidden_size*2 + num_events-1, num_events)

        self.argument_linear = nn.Linear(hidden_size*4 + num_events-1, num_arguments)

        self.initial_g_trg_row = torch.zeros((num_events-1), dtype=torch.int64)

        self.initial_g_trg_arg_row = torch.zeros((max_length, num_events-1), dtype=torch.int64)
    
    def update_g_trg(self, g_trg, event_pred):

        """
            g_trg: batch_size * num_events
            event_pred: batch_size
        """
        
        device = g_trg.device

        new_g_trg = g_trg
        event_pred_bool = event_pred > 0
        updated_batch_indices = event_pred_bool.nonzero().squeeze(-1) # n

        if updated_batch_indices.shape == torch.Size([0]):
            return new_g_trg
        
        updated_g_trg = new_g_trg.index_select(0, updated_batch_indices) # n * num_events
        kept_event_pred = event_pred.index_select(0, updated_batch_indices) # n

        for i in range(kept_event_pred.size(0)):
            # updated_g_trg_i = updated_g_trg[i] # num_events
            kept_event_pred_i = kept_event_pred[i] # 1

            if kept_event_pred_i.sub(1) == -1:
                continue
            
            updated_g_trg[i] = updated_g_trg[i].index_copy_(dim=0, index=kept_event_pred_i.sub(1), source=torch.tensor(1).to(device))

        new_g_trg = new_g_trg.index_copy_(dim=0, index=updated_batch_indices, source=updated_g_trg)

        return new_g_trg
    
    def update_g_trg_arg(self, g_trg_arg, event_pred, argument_preds):

        """
            g_tr_arg: batch_size * max_length * num_events
            event_pred: batch_size
            argument_preds: batch_size * max_length 
        """
        
        new_g_trg_arg = g_trg_arg # batch_size * max_length * num_events
        event_pred_bool = event_pred > 0 
        updated_batch_indices = event_pred_bool.nonzero().squeeze(-1) # n
        
        if updated_batch_indices.shape == torch.Size([0]):
            return new_g_trg_arg
        
        updated_g_trg_arg = new_g_trg_arg.index_select(0, updated_batch_indices) # n * max_length * num_events
        kept_event_pred = event_pred[event_pred_bool] # n
        kept_argument_preds = argument_preds.index_select(0, updated_batch_indices) # n * max_length

        kept_event_pred_ = torch.sub(kept_event_pred, 1)
        for i in range(kept_event_pred_.size(0)):
            updated_g_trg_arg_i = updated_g_trg_arg[i] # max_length * num_events
            kept_event_pred_i = kept_event_pred_[i] # 1
            kept_argument_preds_i = kept_argument_preds[i] # max_length 
            
            updated_g_trg_arg_i_column = updated_g_trg_arg_i.index_select(1, kept_event_pred_i) # max_length * 1
            kept_argument_preds_i = (kept_argument_preds_i > 0).long().unsqueeze(-1) # max_length * 1
            updated_g_trg_arg_i_column = torch.where(kept_argument_preds_i > 0, kept_argument_preds_i, updated_g_trg_arg_i_column)
            updated_g_trg_arg[i] = updated_g_trg_arg[i].index_copy(dim=1, index=kept_event_pred_i, source=updated_g_trg_arg_i_column)

        new_g_trg_arg = new_g_trg_arg.index_copy_(dim=0, index=updated_batch_indices, source=updated_g_trg_arg)
        
        return new_g_trg_arg

    def forward(self, input_ids, attention_mask=None, event_labels=None, arguments_labels=None):

        device = input_ids.device
        batch_size = input_ids.size(0)
        g_trg = torch.stack([self.initial_g_trg_row] * batch_size, dim=0).to(device)
        g_trg_arg = torch.stack([self.initial_g_trg_arg_row] * batch_size, dim=0).to(device)

        word_emb = self.word_embedding(input_ids)
        emb = word_emb

        hidden_states, _ = self.bilstm(emb)
        
        list_event_logits = []
        list_arguments_logits = []
        
        for i in range(self.max_length):
            trigger_hidden_state = hidden_states[:, i]
            event_hidden_state = torch.cat((trigger_hidden_state, g_trg), dim=1)
            event_hidden_state = self.event_dropout(event_hidden_state)
            event_logit = self.event_linear(event_hidden_state)
            list_event_logits.append(event_logit)
            event_pred = torch.argmax(event_logit, dim=-1)
            
            trigger_hidden_states = torch.stack([trigger_hidden_state] * self.max_length, dim=1)
            argument_hidden_states = torch.cat((hidden_states, trigger_hidden_states, g_trg_arg), dim=2)
            argument_hidden_states = self.argument_dropout(argument_hidden_states)
            argument_logits = self.argument_linear(argument_hidden_states)
            list_arguments_logits.append(argument_logits)
            argument_preds = torch.argmax(argument_logits, dim=-1)

            g_trg = self.update_g_trg(g_trg, event_pred)

            g_trg_arg = self.update_g_trg_arg(g_trg_arg, event_pred, argument_preds)
        
        event_logits = torch.stack(list_event_logits, dim=1)
        arguments_logits = torch.stack(list_arguments_logits, dim=1)

        if event_labels is not None and arguments_labels is not None:
            event_loss_fct = CrossEntropyLoss()
            argument_loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_event_logits = event_logits.view(-1, self.num_events)[active_loss]
                active_event_labels = event_labels.view(-1)[active_loss]
                event_loss = event_loss_fct(active_event_logits, active_event_labels)

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)[active_loss]
                active_arguments_labels = arguments_labels.view(-1, self.max_length)[active_loss]
                
                active_argument_loss = active_event_labels != 0
                active_arguments_logits = active_arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = active_arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            else:
                event_loss = event_loss_fct(event_logits.view(-1, self.num_events), event_labels.view(-1))

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)
                active_arguments_labels = arguments_labels.view(-1, self.max_length)

                active_argument_loss = event_labels != 0
                active_arguments_logits = arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            
            return loss, event_logits, arguments_logits
        else:
            return event_logits, arguments_logits


class JointSentenceBiLSTM2_(nn.Module):
    def __init__(self, num_events, num_arguments, num_words, word_embedding_dim, 
                max_length, hidden_size, dropout_rate,
                use_pretrained=False, embedding_weight=None):
        
        super(JointSentenceBiLSTM2_, self).__init__()

        self.num_events = num_events
        self.num_arguments = num_arguments
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.embedding_dim = word_embedding_dim
        
        self.event_dropout = nn.Dropout(dropout_rate)
        self.argument_dropout = nn.Dropout(dropout_rate)

        self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        
        self.event_linear = nn.Linear(hidden_size*2 + num_events-1, num_events)

        self.argument_linear = nn.Linear(hidden_size*4 + num_arguments-1, num_arguments)

        self.initial_g_trg_row = torch.zeros((num_events-1), dtype=torch.int64)

        self.initial_g_arg_row = torch.zeros((max_length, num_arguments-1), dtype=torch.int64)
    
    def update_g_trg(self, g_trg, event_pred):

        """
            g_trg: batch_size * num_events
            event_pred: batch_size
        """
        
        device = g_trg.device

        new_g_trg = g_trg
        event_pred_bool = event_pred > 0
        updated_batch_indices = event_pred_bool.nonzero().squeeze(-1) # n

        if updated_batch_indices.shape == torch.Size([0]):
            return new_g_trg
        
        updated_g_trg = new_g_trg.index_select(0, updated_batch_indices) # n * num_events
        kept_event_pred = event_pred.index_select(0, updated_batch_indices) # n

        for i in range(kept_event_pred.size(0)):
            # updated_g_trg_i = updated_g_trg[i] # num_events
            kept_event_pred_i = kept_event_pred[i] # 1

            if kept_event_pred_i.sub(1) == -1:
                continue
            
            updated_g_trg[i] = updated_g_trg[i].index_copy_(dim=0, index=kept_event_pred_i.sub(1), source=torch.tensor(1).to(device))

        new_g_trg = new_g_trg.index_copy_(dim=0, index=updated_batch_indices, source=updated_g_trg)

        return new_g_trg
    
    def update_g_arg(self, g_arg, event_pred, argument_preds):

        """
            g_arg: batch_size * max_length * num_arguments
            event_pred: batch_size
            argument_preds: batch_size * max_length 
        """

        device = g_arg.device
        
        new_g_arg = g_arg # batch_size * max_length * num_arguments
        event_pred_bool = event_pred > 0 
        updated_batch_indices = event_pred_bool.nonzero().squeeze(-1) # n
        
        if updated_batch_indices.shape == torch.Size([0]):
            return new_g_arg
        
        updated_g_arg = new_g_arg.index_select(0, updated_batch_indices) # n * max_length * num_arguments
        kept_argument_preds = argument_preds.index_select(0, updated_batch_indices) # n * max_length

        for i in range(kept_argument_preds.size(0)):
            updated_g_arg_i = updated_g_arg[i] # max_length * num_arguments
            kept_argument_preds_i = kept_argument_preds[i] # max_length 
            
            for j in range(kept_argument_preds_i.size(0)):
                if kept_argument_preds_i[j].sub(1) == -1:
                    continue
                updated_g_arg_i[j] = updated_g_arg_i[j].index_copy_(dim=0, index=kept_argument_preds_i[j].sub(1), source=torch.tensor(1).to(device))
            updated_g_arg[i] = updated_g_arg_i

        new_g_arg = new_g_arg.index_copy_(dim=0, index=updated_batch_indices, source=updated_g_arg)
        
        return new_g_arg

    def forward(self, input_ids, attention_mask=None, event_labels=None, arguments_labels=None):

        device = input_ids.device
        batch_size = input_ids.size(0)
        g_trg = torch.stack([self.initial_g_trg_row] * batch_size, dim=0).to(device)
        g_arg = torch.stack([self.initial_g_arg_row] * batch_size, dim=0).to(device)

        word_emb = self.word_embedding(input_ids)
        emb = word_emb

        hidden_states, _ = self.bilstm(emb)
        
        list_event_logits = []
        list_arguments_logits = []
        
        for i in range(self.max_length):
            trigger_hidden_state = hidden_states[:, i]
            event_hidden_state = torch.cat((trigger_hidden_state, g_trg), dim=1)
            event_hidden_state = self.event_dropout(event_hidden_state)
            event_logit = self.event_linear(event_hidden_state)
            list_event_logits.append(event_logit)
            event_pred = torch.argmax(event_logit, dim=-1)
            
            trigger_hidden_states = torch.stack([trigger_hidden_state] * self.max_length, dim=1)
            argument_hidden_states = torch.cat((hidden_states, trigger_hidden_states, g_arg), dim=2)
            argument_hidden_states = self.argument_dropout(argument_hidden_states)
            argument_logits = self.argument_linear(argument_hidden_states)
            list_arguments_logits.append(argument_logits)
            argument_preds = torch.argmax(argument_logits, dim=-1)

            g_trg = self.update_g_trg(g_trg, event_pred)

            g_arg = self.update_g_arg(g_arg, event_pred, argument_preds)
        
        event_logits = torch.stack(list_event_logits, dim=1)
        arguments_logits = torch.stack(list_arguments_logits, dim=1)

        if event_labels is not None and arguments_labels is not None:
            event_loss_fct = CrossEntropyLoss()
            argument_loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_event_logits = event_logits.view(-1, self.num_events)[active_loss]
                active_event_labels = event_labels.view(-1)[active_loss]
                event_loss = event_loss_fct(active_event_logits, active_event_labels)

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)[active_loss]
                active_arguments_labels = arguments_labels.view(-1, self.max_length)[active_loss]
                
                active_argument_loss = active_event_labels != 0
                active_arguments_logits = active_arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = active_arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            else:
                event_loss = event_loss_fct(event_logits.view(-1, self.num_events), event_labels.view(-1))

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)
                active_arguments_labels = arguments_labels.view(-1, self.max_length)

                active_argument_loss = event_labels != 0
                active_arguments_logits = arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            
            return loss, event_logits, arguments_logits
        else:
            return event_logits, arguments_logits


class JointSentenceBiLSTM12_(nn.Module):
    def __init__(self, num_events, num_arguments, num_words, word_embedding_dim, 
                max_length, hidden_size, dropout_rate,
                use_pretrained=False, embedding_weight=None):
        
        super(JointSentenceBiLSTM12_, self).__init__()

        self.num_events = num_events
        self.num_arguments = num_arguments
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.embedding_dim = word_embedding_dim
        
        self.event_dropout = nn.Dropout(dropout_rate)
        self.argument_dropout = nn.Dropout(dropout_rate)

        self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        
        self.event_linear = nn.Linear(hidden_size*2 + num_events-1, num_events)

        self.argument_linear = nn.Linear(hidden_size*4 + num_arguments-1 + num_events-1 , num_arguments)

        self.initial_g_trg_row = torch.zeros((num_events-1), dtype=torch.int64)

        self.initial_g_trg_arg_row = torch.zeros((max_length, num_events-1), dtype=torch.int64)

        self.initial_g_arg_row = torch.zeros((max_length, num_arguments-1), dtype=torch.int64)
    
    def update_g_trg(self, g_trg, event_pred):

        """
            g_trg: batch_size * num_events
            event_pred: batch_size
        """
        
        device = g_trg.device

        new_g_trg = g_trg
        event_pred_bool = event_pred > 0
        updated_batch_indices = event_pred_bool.nonzero().squeeze(-1) # n

        if updated_batch_indices.shape == torch.Size([0]):
            return new_g_trg
        
        updated_g_trg = new_g_trg.index_select(0, updated_batch_indices) # n * num_events
        kept_event_pred = event_pred.index_select(0, updated_batch_indices) # n

        for i in range(kept_event_pred.size(0)):
            # updated_g_trg_i = updated_g_trg[i] # num_events
            kept_event_pred_i = kept_event_pred[i] # 1

            if kept_event_pred_i.sub(1) == -1:
                continue
            
            updated_g_trg[i] = updated_g_trg[i].index_copy_(dim=0, index=kept_event_pred_i.sub(1), source=torch.tensor(1).to(device))

        new_g_trg = new_g_trg.index_copy_(dim=0, index=updated_batch_indices, source=updated_g_trg)

        return new_g_trg
    
    def update_g_trg_arg(self, g_trg_arg, event_pred, argument_preds):

        """
            g_tr_arg: batch_size * max_length * num_events
            event_pred: batch_size
            argument_preds: batch_size * max_length 
        """
        
        new_g_trg_arg = g_trg_arg # batch_size * max_length * num_events
        event_pred_bool = event_pred > 0 
        updated_batch_indices = event_pred_bool.nonzero().squeeze(-1) # n
        
        if updated_batch_indices.shape == torch.Size([0]):
            return new_g_trg_arg
        
        updated_g_trg_arg = new_g_trg_arg.index_select(0, updated_batch_indices) # n * max_length * num_events
        kept_event_pred = event_pred[event_pred_bool] # n
        kept_argument_preds = argument_preds.index_select(0, updated_batch_indices) # n * max_length

        kept_event_pred_ = torch.sub(kept_event_pred, 1)
        for i in range(kept_event_pred_.size(0)):
            updated_g_trg_arg_i = updated_g_trg_arg[i] # max_length * num_events
            kept_event_pred_i = kept_event_pred_[i] # 1
            kept_argument_preds_i = kept_argument_preds[i] # max_length 
            
            updated_g_trg_arg_i_column = updated_g_trg_arg_i.index_select(1, kept_event_pred_i) # max_length * 1
            kept_argument_preds_i = (kept_argument_preds_i > 0).long().unsqueeze(-1) # max_length * 1
            updated_g_trg_arg_i_column = torch.where(kept_argument_preds_i > 0, kept_argument_preds_i, updated_g_trg_arg_i_column)
            updated_g_trg_arg[i] = updated_g_trg_arg[i].index_copy(dim=1, index=kept_event_pred_i, source=updated_g_trg_arg_i_column)

        new_g_trg_arg = new_g_trg_arg.index_copy_(dim=0, index=updated_batch_indices, source=updated_g_trg_arg)
        
        return new_g_trg_arg
    
    def update_g_arg(self, g_arg, event_pred, argument_preds):

        """
            g_arg: batch_size * max_length * num_arguments
            event_pred: batch_size
            argument_preds: batch_size * max_length 
        """

        device = g_arg.device
        
        new_g_arg = g_arg # batch_size * max_length * num_arguments
        event_pred_bool = event_pred > 0 
        updated_batch_indices = event_pred_bool.nonzero().squeeze(-1) # n
        
        if updated_batch_indices.shape == torch.Size([0]):
            return new_g_arg
        
        updated_g_arg = new_g_arg.index_select(0, updated_batch_indices) # n * max_length * num_arguments
        kept_argument_preds = argument_preds.index_select(0, updated_batch_indices) # n * max_length

        for i in range(kept_argument_preds.size(0)):
            updated_g_arg_i = updated_g_arg[i] # max_length * num_arguments
            kept_argument_preds_i = kept_argument_preds[i] # max_length 
            
            for j in range(kept_argument_preds_i.size(0)):
                if kept_argument_preds_i[j].sub(1) == -1:
                    continue
                updated_g_arg_i[j] = updated_g_arg_i[j].index_copy_(dim=0, index=kept_argument_preds_i[j].sub(1), source=torch.tensor(1).to(device))
            updated_g_arg[i] = updated_g_arg_i

        new_g_arg = new_g_arg.index_copy_(dim=0, index=updated_batch_indices, source=updated_g_arg)
        
        return new_g_arg

    def forward(self, input_ids, attention_mask=None, event_labels=None, arguments_labels=None):

        device = input_ids.device
        batch_size = input_ids.size(0)
        g_trg = torch.stack([self.initial_g_trg_row] * batch_size, dim=0).to(device)
        g_trg_arg = torch.stack([self.initial_g_trg_arg_row] * batch_size, dim=0).to(device)
        g_arg = torch.stack([self.initial_g_arg_row] * batch_size, dim=0).to(device)

        word_emb = self.word_embedding(input_ids)
        emb = word_emb

        hidden_states, _ = self.bilstm(emb)
        
        list_event_logits = []
        list_arguments_logits = []
        
        for i in range(self.max_length):
            trigger_hidden_state = hidden_states[:, i]
            event_hidden_state = torch.cat((trigger_hidden_state, g_trg), dim=1)
            event_hidden_state = self.event_dropout(event_hidden_state)
            event_logit = self.event_linear(event_hidden_state)
            list_event_logits.append(event_logit)
            event_pred = torch.argmax(event_logit, dim=-1)
            
            trigger_hidden_states = torch.stack([trigger_hidden_state] * self.max_length, dim=1)
            argument_hidden_states = torch.cat((hidden_states, trigger_hidden_states, g_arg, g_trg_arg), dim=2)
            argument_hidden_states = self.argument_dropout(argument_hidden_states)
            argument_logits = self.argument_linear(argument_hidden_states)
            list_arguments_logits.append(argument_logits)
            argument_preds = torch.argmax(argument_logits, dim=-1)

            g_trg = self.update_g_trg(g_trg, event_pred)
            g_trg_arg = self.update_g_trg_arg(g_trg_arg, event_pred, argument_preds)
            g_arg = self.update_g_arg(g_arg, event_pred, argument_preds)
        
        event_logits = torch.stack(list_event_logits, dim=1)
        arguments_logits = torch.stack(list_arguments_logits, dim=1)

        if event_labels is not None and arguments_labels is not None:
            event_loss_fct = CrossEntropyLoss()
            argument_loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_event_logits = event_logits.view(-1, self.num_events)[active_loss]
                active_event_labels = event_labels.view(-1)[active_loss]
                event_loss = event_loss_fct(active_event_logits, active_event_labels)

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)[active_loss]
                active_arguments_labels = arguments_labels.view(-1, self.max_length)[active_loss]
                
                active_argument_loss = active_event_labels != 0
                active_arguments_logits = active_arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = active_arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            else:
                event_loss = event_loss_fct(event_logits.view(-1, self.num_events), event_labels.view(-1))

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)
                active_arguments_labels = arguments_labels.view(-1, self.max_length)

                active_argument_loss = event_labels != 0
                active_arguments_logits = arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            
            return loss, event_logits, arguments_logits
        else:
            return event_logits, arguments_logits

class JointSentenceBiLSTM2_WithPostag(nn.Module):
    def __init__(self, num_events, num_arguments, num_words, word_embedding_dim, 
                max_length, hidden_size, num_postags, postag_embedding_dim, 
                dropout_rate, use_pretrained=False, embedding_weight=None):
        
        super(JointSentenceBiLSTM2_WithPostag, self).__init__()

        self.num_events = num_events
        self.num_arguments = num_arguments
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.num_postags = num_postags
        self.postag_embedding_dim = postag_embedding_dim
        self.dropout_rate = dropout_rate

        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.postag_embedding = nn.Embedding(num_embeddings=num_postags, embedding_dim=postag_embedding_dim)
        self.embedding_dim = word_embedding_dim + postag_embedding_dim
        
        self.event_dropout = nn.Dropout(dropout_rate)
        self.argument_dropout = nn.Dropout(dropout_rate)

        self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        
        self.event_linear = nn.Linear(hidden_size*2 + num_events-1, num_events)

        self.argument_linear = nn.Linear(hidden_size*4 + num_arguments-1, num_arguments)

        self.initial_g_trg_row = torch.zeros((num_events-1), dtype=torch.int64)

        self.initial_g_arg_row = torch.zeros((max_length, num_arguments-1), dtype=torch.int64)
    
    def update_g_trg(self, g_trg, event_pred):

        """
            g_trg: batch_size * num_events
            event_pred: batch_size
        """
        
        device = g_trg.device

        new_g_trg = g_trg
        event_pred_bool = event_pred > 0
        updated_batch_indices = event_pred_bool.nonzero().squeeze(-1) # n

        if updated_batch_indices.shape == torch.Size([0]):
            return new_g_trg
        
        updated_g_trg = new_g_trg.index_select(0, updated_batch_indices) # n * num_events
        kept_event_pred = event_pred.index_select(0, updated_batch_indices) # n

        for i in range(kept_event_pred.size(0)):
            # updated_g_trg_i = updated_g_trg[i] # num_events
            kept_event_pred_i = kept_event_pred[i] # 1

            if kept_event_pred_i.sub(1) == -1:
                continue
            
            updated_g_trg[i] = updated_g_trg[i].index_copy_(dim=0, index=kept_event_pred_i.sub(1), source=torch.tensor(1).to(device))

        new_g_trg = new_g_trg.index_copy_(dim=0, index=updated_batch_indices, source=updated_g_trg)

        return new_g_trg
    
    def update_g_arg(self, g_arg, event_pred, argument_preds):

        """
            g_arg: batch_size * max_length * num_arguments
            event_pred: batch_size
            argument_preds: batch_size * max_length 
        """

        device = g_arg.device
        
        new_g_arg = g_arg # batch_size * max_length * num_arguments
        event_pred_bool = event_pred > 0 
        updated_batch_indices = event_pred_bool.nonzero().squeeze(-1) # n
        
        if updated_batch_indices.shape == torch.Size([0]):
            return new_g_arg
        
        updated_g_arg = new_g_arg.index_select(0, updated_batch_indices) # n * max_length * num_arguments
        kept_argument_preds = argument_preds.index_select(0, updated_batch_indices) # n * max_length

        for i in range(kept_argument_preds.size(0)):
            updated_g_arg_i = updated_g_arg[i] # max_length * num_arguments
            kept_argument_preds_i = kept_argument_preds[i] # max_length 
            
            for j in range(kept_argument_preds_i.size(0)):
                if kept_argument_preds_i[j].sub(1) == -1:
                    continue
                updated_g_arg_i[j] = updated_g_arg_i[j].index_copy_(dim=0, index=kept_argument_preds_i[j].sub(1), source=torch.tensor(1).to(device))
            updated_g_arg[i] = updated_g_arg_i

        new_g_arg = new_g_arg.index_copy_(dim=0, index=updated_batch_indices, source=updated_g_arg)
        
        return new_g_arg

    def forward(self, input_ids, postag_ids, attention_mask=None, event_labels=None, arguments_labels=None):

        device = input_ids.device
        batch_size = input_ids.size(0)
        g_trg = torch.stack([self.initial_g_trg_row] * batch_size, dim=0).to(device)
        g_arg = torch.stack([self.initial_g_arg_row] * batch_size, dim=0).to(device)

        word_emb = self.word_embedding(input_ids)
        postag_emb = self.postag_embedding(postag_ids)
        emb = torch.cat((word_emb, postag_emb), 2)

        hidden_states, _ = self.bilstm(emb)
        
        list_event_logits = []
        list_arguments_logits = []
        
        for i in range(self.max_length):
            trigger_hidden_state = hidden_states[:, i]
            event_hidden_state = torch.cat((trigger_hidden_state, g_trg), dim=1)
            event_hidden_state = self.event_dropout(event_hidden_state)
            event_logit = self.event_linear(event_hidden_state)
            list_event_logits.append(event_logit)
            event_pred = torch.argmax(event_logit, dim=-1)
            
            trigger_hidden_states = torch.stack([trigger_hidden_state] * self.max_length, dim=1)
            argument_hidden_states = torch.cat((hidden_states, trigger_hidden_states, g_arg), dim=2)
            argument_hidden_states = self.argument_dropout(argument_hidden_states)
            argument_logits = self.argument_linear(argument_hidden_states)
            list_arguments_logits.append(argument_logits)
            argument_preds = torch.argmax(argument_logits, dim=-1)

            g_trg = self.update_g_trg(g_trg, event_pred)

            g_arg = self.update_g_arg(g_arg, event_pred, argument_preds)
        
        event_logits = torch.stack(list_event_logits, dim=1)
        arguments_logits = torch.stack(list_arguments_logits, dim=1)

        if event_labels is not None and arguments_labels is not None:
            event_loss_fct = CrossEntropyLoss()
            argument_loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_event_logits = event_logits.view(-1, self.num_events)[active_loss]
                active_event_labels = event_labels.view(-1)[active_loss]
                event_loss = event_loss_fct(active_event_logits, active_event_labels)

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)[active_loss]
                active_arguments_labels = arguments_labels.view(-1, self.max_length)[active_loss]
                
                active_argument_loss = active_event_labels != 0
                active_arguments_logits = active_arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = active_arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            else:
                event_loss = event_loss_fct(event_logits.view(-1, self.num_events), event_labels.view(-1))

                # Only keep parts of trigger of the loss
                active_arguments_logits = arguments_logits.view(-1, self.max_length, self.num_arguments)
                active_arguments_labels = arguments_labels.view(-1, self.max_length)

                active_argument_loss = event_labels != 0
                active_arguments_logits = arguments_logits[active_argument_loss].view(-1, self.num_arguments)
                active_arguments_labels = arguments_labels[active_argument_loss].view(-1)
                arguments_loss = argument_loss_fct(active_arguments_logits, active_arguments_labels)

                if torch.isnan(arguments_loss):
                    loss = event_loss
                else:
                    loss = event_loss + arguments_loss
            
            return loss, event_logits, arguments_logits
        else:
            return event_logits, arguments_logits

# model = JointSentenceBiLSTM2_(num_events=3, num_arguments=4, max_length=5, \
#     num_words=10, word_embedding_dim=100, hidden_size=64, dropout_rate=0.5)

# # num_events, num_arguments, num_words, word_embedding_dim, 
# #                max_length, hidden_size, dropout_rate

# input_ids = torch.tensor([[4, 1, 2, 3, 4],
#                           [3, 2, 1, 0, 0]])

# postag_ids = torch.tensor([[0, 1, 1, 0, 0],
#                            [1, 0, 0, 2, 0]])

# attention_mask = torch.tensor([[1, 1, 1, 1, 1],
#                                [1, 1, 1, 0, 0]])

# event_labels = torch.tensor([[0, 2, 0, 1, 0],
#                              [1, 0, 0, 0, 0]])
    
# arguments_labels = torch.tensor([[[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, 0]],
#                                  [[0, 0, 0, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])

# g_trg_arg = torch.tensor([[[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 
#                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]])

# event_pred = torch.tensor([1, 0])

# argument_preds = torch.tensor([[0, 0, 1, 2, 0, 0],
#                                [0, 0, 0, 0, 0, 0]])

# model(input_ids, attention_mask)

# print(event_logits.shape)
# print(argument_logits.shape)