import streamlit as st
import pipeline_cnn, pipeline_bilstm, pipeline_phobert, joint_bilstm

st.title('Vietnamese Event Extraction')

text = st.text_area('Type a sentence:')

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    a = st.button('Pipeline CNN')
with col2:
    b = st.button('Pipeline BiLSTM')
with col3:
    c = st.button('Pipeline PhoBERT')
with col4:
    d = st.button('Joint BiLSTM')

preds = []
words = []

if a:
    preds, words = pipeline_cnn.predict_argument(text)

    st.write(' '.join(words))
    
    for pred in preds:
        event_type = pred[1]
        start_trigger = pred[2]
        end_trigger = pred[3]
        argument_type = pred[4]
        start_arg = pred[5]
        end_arg = pred[6]
        st.write(f'{event_type}: {" ".join(words[start_trigger:end_trigger+1])} ({start_trigger} - {end_trigger}) - {argument_type}: {" ".join(words[start_arg:end_arg+1])} ({start_arg} - {end_arg})')

elif b:
    preds, words = pipeline_bilstm.predict_argument(text)

    st.write(' '.join(words))
    
    for pred in preds:
        event_type = pred[1]
        start_trigger = pred[2]
        end_trigger = pred[3]
        argument_type = pred[4]
        start_arg = pred[5]
        end_arg = pred[6]
        st.write(f'{event_type}: {" ".join(words[start_trigger:end_trigger+1])} ({start_trigger} - {end_trigger}) - {argument_type}: {" ".join(words[start_arg:end_arg+1])} ({start_arg} - {end_arg})')

elif c:
    preds, words = pipeline_phobert.predict_argument(text)

    st.write(' '.join(words))
    
    for pred in preds:
        event_type = pred[1]
        start_trigger = pred[2]
        end_trigger = pred[3]
        argument_type = pred[4]
        start_arg = pred[5]
        end_arg = pred[6]
        st.write(f'{event_type}: {" ".join(words[start_trigger:end_trigger+1])} ({start_trigger} - {end_trigger}) - {argument_type}: {" ".join(words[start_arg:end_arg+1])} ({start_arg} - {end_arg})')

elif d:
    preds, words = joint_bilstm.predict(text)

    st.write(' '.join(words))
    
    for pred in preds:
        event_type = pred[1]
        start_trigger = pred[2]
        end_trigger = pred[3]
        argument_type = pred[4]
        start_arg = pred[5]
        end_arg = pred[6]
        st.write(f'{event_type}: {" ".join(words[start_trigger:end_trigger+1])} ({start_trigger} - {end_trigger}) - {argument_type}: {" ".join(words[start_arg:end_arg+1])} ({start_arg} - {end_arg})')

# st.text(str(preds))