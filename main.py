from json import encoder
import numpy as np
import streamlit as st
import requests
import json
import re
import torch
import sys
from pydantic import BaseModel
import os
from typing import List, Dict
from torch import cuda
from annotated_text import annotated_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from pyserini.dsearch import SimpleDenseSearcher, TctColBertQueryEncoder, AutoQueryEncoder
from pyserini.hsearch import HybridSearcher
from pyserini.search import SimpleSearcher
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed', do_lower_case=False)
model = AutoModel.from_pretrained('monologg/biobert_v1.1_pubmed')

anserini_root = '.'

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
sys.path += [os.path.join(anserini_root, 'src/main/python')]

from pyserini.search import SimpleSearcher



COVID_INDEX = 'lucene-index-cord19-full-text-2020-05-01/'
searcher = SimpleSearcher(COVID_INDEX)

class input_query(BaseModel):
    text: str
    topk: int


class base1(BaseModel):
    Document: str
    Doc_Scores: int
    Doc_Titles: str
    Document_Body: str

class base2(BaseModel):
    passages: List[base1]



def show_document(idx,doc):
    have_body_text = 'body_text' in json.loads(doc.raw)
    body_text = ' Full text available.' if have_body_text else ''
    return {
        'Document': doc.docid,
        'Doc_Score': doc.score,
        'Doc_Title': doc.lucene_document.get('title'),
        'Document_Body': body_text
    }


import json
dense_passages = []
with open('pyserini_corpus.jsonl','r') as f:
  # lines = f.readlines()
  for line in f:
    dense_passages.append(json.loads(line))


def predict(data):
    query = data['text']
    top_k = data['topk']
    sys.stderr.write(query)
    hits = searcher.search((query))
    output_dict = dict()
    doc_ids = []
    doc_scores = []
    doc_titles = []
    docs = []
    for i, hit in enumerate(hits[:top_k]):
        doc_ids.append(hit.docid)
        doc_scores.append(hit.score)
        doc_titles.append(hit.lucene_document.get('title'))
        if hit.lucene_document.get('doi'):
            docs.append(f"Link: https://doi.org/{hit.lucene_document.get('doi')}")
        else:
            docs.append(f"Link not available")
    output_dict['Document'] = doc_ids
    output_dict['Doc_Scores'] = doc_scores
    output_dict['Doc_Titles'] = doc_titles
    output_dict['Document_Body'] = docs
    return output_dict



def get_highlights_entail_labels(data):
    query = data['text']
    top_k = data['topk']
    hits = searcher.search((query))
    top_hit = hits[0]
    json_doc = json.loads(top_hit.raw)
    encoded_query = tokenizer(query,truncation=True,padding=False,return_tensors='pt')
    # encoded_query = encoded_query.to(device)
    with torch.no_grad():
        query_state = torch.squeeze(model(**encoded_query)[0][:,1:-1,:],dim=0)
    par_states = []
    for par in json_doc['body_text']:
        encoded_par = tokenizer(par['text'],stride=100,truncation=True,return_tensors='pt')
        # encoded_par = encoded_par.to(device)
        with torch.no_grad():
            last_hid_out = model(**encoded_par)
        last_hid_out = last_hid_out[0][:,1:-1,:]
        par_states.append(torch.squeeze(last_hid_out,dim=0))
    sim_matrix = []
    for state in par_states:
        cos_sim = torch.nn.CosineSimilarity(dim=-1)
        sim_matrix.append(cos_sim(query_state.unsqueeze(1),state).mean())
    order = np.argsort(sim_matrix)
    

        # final_state = torch.cat(par_states,1)
    
    return json_doc['body_text'][0]['text'], order



def denseSearch(data):
    query = data['text']
    top_k = data['topk']
    encoder = AutoQueryEncoder(encoder_dir='gsarti/covidbert-nli',tokenizer_name='gsarti/covidbert-nli')
    searcher = SimpleDenseSearcher.from_prebuilt_index(COVID_INDEX,encoder)
    hits = searcher.search((query))
    output_dict = dict()
    doc_ids = []
    doc_scores = []
    doc_titles = []
    docs = []
    for i, hit in enumerate(hits[:top_k]):
        doc_ids.append(hit.docid)
        doc_scores.append(hit.score)
        doc_titles.append(hit.lucene_document.get('title'))
        if hit.lucene_document.get('doi'):
            docs.append(f"Link: https://doi.org/{hit.lucene_document.get('doi')}")
        else:
            docs.append(f"Link not available")
    output_dict['Document'] = doc_ids
    output_dict['Doc_Scores'] = doc_scores
    output_dict['Doc_Titles'] = doc_titles
    output_dict['Document_Body'] = docs
    return output_dict


def bm25search(data):
    query = data['text']
    top_k = data['topk']
    searcher = SimpleSearcher('indexes/covid_index')
    hits = searcher.search((query))
    output_dict = dict()
    doc_ids = []
    doc_scores = []
    doc_titles = []
    docs = []
    for i in range(top_k):
        json_doc = json.loads(hits[i].raw)
        doc_ids.append(json_doc['id'].split(' ')[0])
        doc_scores.append(hits[i].score)
        doc_titles.append(json_doc['id'].split(' ')[-1])
        docs.append(json_doc['contents'])
    output_dict['Document'] = doc_ids
    output_dict['Doc_Scores'] = doc_scores
    output_dict['Doc_Titles'] = doc_titles
    output_dict['Document_Body'] = docs
    return output_dict


def dense_search_ourdata(data):
    query = data['text']
    top_k = data['topk']
    encoder = TctColBertQueryEncoder('gsarti/covidbert-nli')
    searcher = SimpleDenseSearcher('dense_index_covidBERT/',encoder)
    hits = searcher.search((query))
    output_dict = dict()
    doc_ids = []
    doc_scores = []
    doc_titles = []
    docs = []
    for i in range(top_k):
        # json_doc = json.loads(hits[i].raw)
        doc_ids.append(hits[i].docid.split(' ')[0])
        doc_scores.append(hits[i].score)
        doc_titles.append(hits[i].docid.split(' ')[-1])
        docs.append(dense_passages[int(hits[i].docid.split(' ')[0][3:])]['contents'])
    output_dict['Document'] = doc_ids
    output_dict['Doc_Scores'] = doc_scores
    output_dict['Doc_Titles'] = doc_titles
    output_dict['Document_Body'] = docs
    return output_dict


def hybrid_search_ourdata(data):
    query = data['text']
    top_k = data['topk']
    ssearcher = SimpleSearcher('indexes/covid_index')
    encoder = TctColBertQueryEncoder('gsarti/covidbert-nli')
    dsearcher = SimpleDenseSearcher('dense_index_covidBERT/',encoder)
    hsearcher = HybridSearcher(dsearcher,ssearcher)
    hits = hsearcher.search((query))
    output_dict = dict()
    doc_ids = []
    doc_scores = []
    doc_titles = []
    docs = []
    for i in range(top_k):
        doc_ids.append(hits[i].docid.split(' ')[0])
        doc_scores.append(hits[i].score)
        doc_titles.append(hits[i].docid.split(' ')[-1])
        docs.append(dense_passages[int(hits[i].docid.split(' ')[0][3:])]['contents'])
    output_dict['Document'] = doc_ids
    output_dict['Doc_Scores'] = doc_scores
    output_dict['Doc_Titles'] = doc_titles
    output_dict['Document_Body'] = docs
    return output_dict

def get_entailment_labels(premise,hypothesis):
    id2class = {0:'contradiction',1:'neutral',2:'entailment'}
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
    model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
    x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                     truncation_strategy='only_second')
    logits = model(x)[0]
    probs = torch.softmax(logits,axis=-1)
    category = torch.argmax(probs).item()
    confidence = probs[:,category].item()
    return {'category':id2class[category],'confidence':confidence}




st.title('Claim Verification')
st.write('Developed by LCS2 and Accenture')
claim:str = st.text_area('Enter your claim')
top_k:int = st.number_input('Number of articles to retrieve',min_value=1,max_value=10)
input_data = {'text': claim,'topk':top_k}
# data_frame = pd.DataFrame(columns=['id','model_used','claim','article_link','article_text'])
data_frame = pd.read_csv('responses.csv')

if st.button('Search on CORD-19 Corpus Using BM25') and claim != None:
    with st.spinner('Getting Results...'):
        res = predict(input_data)
        result = res
        for i in range(top_k):
            st.write(f"Document Id: {result['Document'][i]}")
            st.write(f"Document Score: {result['Doc_Scores'][i]}")
            st.write(f"Title of the Document: {result['Doc_Titles'][i]}")
            st.write(f"{result['Document_Body'][i]}")
            with st.spinner("Verifying Veracity..."):
                out = get_entailment_labels(input_data['text'],result['Document_Body'][i])
            st.write(f"The article belongs to {out['category']} category with a confidence of {out['confidence']}")
            st.markdown('---')
            new_row = {'id': result['Document'][i],
                        'model_used': 'BM25_CORD19',
                        'claim': input_data['text'],
                        'article_link': result['Doc_Titles'][i],
                        'article_text': result['Document_Body'][i],
                        'entailment_label': out['category'],
                        'entailment_confidence': out['confidence']}
            data_frame = data_frame.append(new_row,ignore_index=True)

if st.button('Search on In house data Using BM25'):
    with st.spinner('Getting Results'):
        res = bm25search(input_data)
        result = res
        for i in range(top_k):
            st.write(f"Document Id: {result['Document'][i]}")
            st.write(f"Document Score: {result['Doc_Scores'][i]:.2f}")
            st.write(f"Link of the Document: {result['Doc_Titles'][i]}")
            st.write(f"{result['Document_Body'][i][:100]}...")
            with st.spinner("Verifying Veracity..."):
                out = get_entailment_labels(input_data['text'],result['Document_Body'][i])
            st.write(f"The article belongs to {out['category']} category with a confidence of {out['confidence']}")
            st.markdown('---')
            new_row = {'id': result['Document'][i],
            'model_used': 'BM25_InHouse',
            'claim': input_data['text'],
            'article_link': result['Doc_Titles'][i],
            'article_text': result['Document_Body'][i],
            'entailment_label': out['category'],
            'entailment_confidence': out['confidence']}
            data_frame = data_frame.append(new_row,ignore_index=True)

# if st.button('Dense Search Using COVID-BERT on CORD-19 data'):
#     with st.spinner('Getting Results'):
#         res = denseSearch(input_data)
#         result = res
#         for i in range(top_k):
#             st.write(f"Document Id: {result['Document'][i]}")
#             st.write(f"Document Score: {result['Doc_Scores'][i]}")
#             st.write(f"Title of the Document: {result['Doc_Titles'][i]}")
#             st.write(f"{result['Document_Body'][i]}")
#             if st.button('Get Entailment Labels'):
#                 out = get_entailment_labels(input_data['text'],result['Document_Body'][i])
#                 st.write(f"The article belongs to {out['category']} category with a confidence of {out['confidence']}")
#             st.markdown('---')


if st.button('Dense Search Using COVID-BERT on In House data'):
    with st.spinner('Getting Results'):
        res = dense_search_ourdata(input_data)
        result = res
        for i in range(top_k):
            st.write(f"Document Id: {result['Document'][i]}")
            st.write(f"Document Score: {result['Doc_Scores'][i]:.2f}")
            st.write(f"Link of the Document: {result['Doc_Titles'][i]}")
            st.write(f"{result['Document_Body'][i][:30]}...")
            with st.spinner("Verifying Veracity..."):
                out = get_entailment_labels(input_data['text'],result['Document_Body'][i])
            st.write(f"The article belongs to {out['category']} category with a confidence of {out['confidence']}")
            st.markdown('---')
            new_row = {'id': result['Document'][i],
                        'model_used': 'DenseSearch_COVID_BERT_InHouse',
                        'claim': input_data['text'],
                        'article_link': result['Doc_Titles'][i],
                        'article_text': result['Document_Body'][i],
                        'entailment_label': out['category'],
                        'entailment_confidence': out['confidence']}
            data_frame = data_frame.append(new_row,ignore_index=True)


if st.button('Hybrid Search on In House data'):
    with st.spinner('Getting Results'):
        res = hybrid_search_ourdata(input_data)
        result = res
        for i in range(top_k):
            st.write(f"Document Id: {result['Document'][i]}")
            st.write(f"Document Score: {result['Doc_Scores'][i]:.2f}")
            st.write(f"Link of the Document: {result['Doc_Titles'][i]}")
            st.write(f"{result['Document_Body'][i][:100]}...")
            with st.spinner("Verifying Veracity..."):
                out = get_entailment_labels(input_data['text'],result['Document_Body'][i])
            st.write(f"The article belongs to {out['category']} category with a confidence of {out['confidence']}")
            st.markdown('---')
            new_row = {'id': result['Document'][i],
                        'model_used': 'HybridSearch_InHouse',
                        'claim': input_data['text'],
                        'article_link': result['Doc_Titles'][i],
                        'article_text': result['Document_Body'][i],
                        'entailment_label': out['category'],
                        'entailment_confidence': out['confidence']}
            data_frame = data_frame.append(new_row,ignore_index=True)

if st.button('Get finegrained results'):
    with st.spinner('Getting fine grained results. Hold Tight...'):    
        out,order = get_highlights_entail_labels(input_data)
        st.write(annotated_text((out,'imp','#afa')))
        # st.write(order)s  
            # st.write(f"Output is {len(shape)}")



data_frame.to_csv('responses.csv')
