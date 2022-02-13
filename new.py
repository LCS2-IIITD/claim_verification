from json import encoder
import numpy as np
import streamlit as st
import requests
import json
import re
from streamlit.proto.Selectbox_pb2 import Selectbox
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
import tqdm
from tqdm.auto import tqdm
import pandas as pd

@st.cache(allow_output_mutation=True)
def load_finegrain_model():
    tokenizer = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed', do_lower_case=False)
    model = AutoModel.from_pretrained('monologg/biobert_v1.1_pubmed')
    return tokenizer, model


@st.cache(allow_output_mutation=True)
def load_searchers():
    COVID_INDEX = 'lucene-index-cord19-full-text-2020-05-01/'
    cord19_searcher = SimpleSearcher(COVID_INDEX)
    bm25our_searcher = SimpleSearcher('indexes/covid_index')
    encoder = TctColBertQueryEncoder('gsarti/covidbert-nli')
    dense_searcher = SimpleDenseSearcher('dense_index_covidBERT/',encoder)
    ssearcher = SimpleSearcher('indexes/covid_index')
    encoder = TctColBertQueryEncoder('gsarti/covidbert-nli')
    dsearcher = SimpleDenseSearcher('dense_index_covidBERT/',encoder)
    hsearcher = HybridSearcher(dsearcher,ssearcher)
    return cord19_searcher, bm25our_searcher, dense_searcher, hsearcher


@st.cache(allow_output_mutation=True)
def load_entailment_model():
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
    model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
    return tokenizer, model


##### Load all models and searchers ######
entailment_tokenizer, entailment_model = load_entailment_model()
cord19_searcher, bm25our_searcher, dense_searcher, hsearcher = load_searchers()
tokenizer, model = load_finegrain_model()



anserini_root = '.'

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
sys.path += [os.path.join(anserini_root, 'src/main/python')]







class input_query(BaseModel):
    text: str
    topk: int


class base1(BaseModel):
    Document: str
    Doc_Scores: int
    Doc_Titles: str
    Document_Body: str
    json_doc: str

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


@st.cache(suppress_st_warning=True)
def predict(data):
    
    query = data['text']
    top_k = data['topk']
    sys.stderr.write(query)
    hits = cord19_searcher.search((query))
    top_hit = hits[0]
    json_doc = json.loads(top_hit.raw)
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
            docs.append(f"Link of the document: https://doi.org/{hit.lucene_document.get('doi')}")
        else:
            docs.append(f"Link not available")
    output_dict['Document'] = doc_ids
    output_dict['Doc_Scores'] = doc_scores
    output_dict['Doc_Titles'] = doc_titles
    output_dict['Document_Body'] = docs
    output_dict['json_doc'] = json_doc
    return output_dict


@st.cache(suppress_st_warning=True)
def denseSearch(data):
    COVID_INDEX = 'lucene-index-cord19-full-text-2020-05-01/'

    query = data['text']
    top_k = data['topk']
    encoder = AutoQueryEncoder(encoder_dir='gsarti/covidbert-nli',tokenizer_name='gsarti/covidbert-nli')
    searcher = SimpleDenseSearcher.from_prebuilt_index(COVID_INDEX,encoder)
    hits = searcher.search((query))
    top_hit = hits[0]
    json_doc = json.loads(top_hit.raw)
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
    output_dict['json_doc'] = json_doc
    return output_dict


@st.cache(suppress_st_warning=True)
def bm25search(data):
    query = data['text']
    top_k = data['topk']
    hits = bm25our_searcher.search((query))
    top_hit = hits[0]
    json_doc = json.loads(top_hit.raw)
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
    output_dict['json_doc'] = json_doc
    return output_dict


@st.cache(suppress_st_warning=True)
def dense_search_ourdata(data):
    query = data['text']
    top_k = data['topk']

    hits = dense_searcher.search((query))
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


@st.cache(suppress_st_warning=True)
def hybrid_search_ourdata(data):
    query = data['text']
    top_k = data['topk']
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


@st.cache(suppress_st_warning=True)
def get_entailment_labels(premise,hypothesis):
    id2class = {0:'contradiction',1:'neutral',2:'entailment'}
    x = entailment_tokenizer.encode(premise, hypothesis, return_tensors='pt',
                     truncation_strategy='only_second')
    logits = entailment_model(x)[0]
    probs = torch.softmax(logits,axis=-1)
    category = torch.argmax(probs).item()
    confidence = probs[:,category].item()
    return {'category':id2class[category],'confidence':confidence}




st.title('Claim Verification')
st.write('Developed by LCS2 and Accenture')
st.subheader('Welcome!')
annotated_text(('Problem Statement',' ','#B0C4DE'))
st.write("The spread of misinformation through online media has increased manifold during the COVID-19 pandemic. Claims made on such online platforms often mislead people to believe in rumors. To combat this, we present a claim veracity verification pipeline consisting of two major steps: evidence retrieval and claim verification.")
st.markdown('---')
annotated_text(('Architecture',' ','#B0C4DE'))
st.write('For our evidence retrieval pipeline, we use employ methods like BM25, Dense Search using COVID-BERT and Hybrid Search using ColBERT based architecture. Our veracity verification step uses a fine-tuned version of BART Transformer model.')
st.markdown('---')
annotated_text(('Datasets',' ','#B0C4DE'))
st.write('We provide 2 datasets as our central knowledge base: CORD-19 and In-House data. CORD-19 has over 500,000 scientific articles related to the medical aspects of the COVID-19 pandemic, whereas our In House data contains articles which are collected from trustworthy news sources to debunk false claims and rumors')
st.markdown('---')
annotated_text(('Usage',' ','#B0C4DE'))
st.write('Enter the claim and wait for the model to generate responses. **Press the button to generate fine grained evidences only after all other computation is finished as this may cause the system to crash.**')
st.write('Please fill in the questions in the form that follows as you use and explore the app. Thanks for your participation!')
st.markdown('---')







# page_names = ['Tool','Form']
# pages = st.radio('Navigation',page_names)

# if pages == 'Overview':
#     st.subheader('Welcome!')
#     annotated_text(('Problem Statement',' ','#B0C4DE'))
#     st.write("The spread of misinformation through online media has increased manifold during the COVID-19 pandemic. Claims made on such online platforms often mislead people to believe in rumors. To combat this, we present a claim veracity verification pipeline consisting of two major steps: evidence retrieval and claim verification.")
#     st.markdown('---')
#     annotated_text(('Architecture',' ','#B0C4DE'))
#     st.write('For our evidence retrieval pipeline, we use employ methods like BM25, Dense Search using COVID-BERT and Hybrid Search using ColBERT based architecture. Our veracity verification step uses a fine-tuned version of BART Transformer model.')
#     st.markdown('---')
#     annotated_text(('Datasets',' ','#B0C4DE'))
#     st.write('We provide 2 datasets as our central knowledge base: CORD-19 and In-House data. CORD-19 has over 500,000 scientific articles related to the medical aspects of the COVID-19 pandemic, whereas our In House data contains articles which are collected from trustworthy news sources to debunk false claims and rumors')
#     st.markdown('---')
#     annotated_text(('Tool',' ','#B0C4DE'))
#     st.write('Owing to the large size of the total knowledge base and the use of large transformer models, our tool runs significantly slower than large search engines. Also, due to the large size of CORD-19 corpus, it currently only supports BM25 based sparse search whereas all methods are available to use in our In House data. To use the tool, just enter a claim and press submit. The app then provides you with the retrieved articles and their veracity scores. For BM25 based methods on either dataset, we also provide the option of geting fine-grained evidences which select and present a section of the article which is more closely aligned with the claim given as input.')
#     st.write('The veracity cerification pipeline outputs a category out of neutral, contradiction or entailment as the labels for that particular article along with the probability score.')
#     st.markdown('---')
#     annotated_text(('Usage',' ','#B0C4DE'))
#     st.write('Enter the claim and wait for the model to generate responses. **Press the button to generate fine grained evidences only after all other computation is finished as this may cause the system to crash.**')
#     st.write('Please fill in the questions in the form that follows as you use and explore the app. Thanks for your participation!')

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
claim:str = st.text_area('Enter your claim')
top_k:int = 1
input_data = {'text': claim,'topk':top_k}
# data_frame = pd.DataFrame(columns=['id','model_used','claim','article_link','article_text'])
data_frame = pd.read_csv('generated_responses.csv')
df = pd.read_csv('collected_responses.csv')
# df = pd.read_csv('form.csv')




submit = st.checkbox('Submit',key='submit')
st.markdown('---')

if submit:
    with st.spinner('Getting Results...'):
        res = predict(input_data)
        result = res
        for i in range(top_k):
            
            st.write(f"{result['Document_Body'][i]}")
            st.write(f"{result['Doc_Titles'][i]}")

            at = [('CORD-19','Dataset','#8A2BE2'),'  ',('BM25','IR Technique','#faa')]
            annotated_text(*at)
            st.write(' ')
            with st.spinner("Verifying Veracity..."):
                out = get_entailment_labels(input_data['text'],result['Document_Body'][i])
            st.markdown(f"**The article belongs to {out['category']} category with a confidence of {out['confidence']}**")
            
            new_row = {'id': result['Document'][i],
                        'data': 'CORD19',
                        'model_used': 'BM25',
                        'claim': input_data['text'],
                        'article_link': result['Doc_Titles'][i],
                        'article_text': result['Document_Body'][i],
                        'entailment_label': out['category'],
                        'entailment_confidence': out['confidence']}
            data_frame = data_frame.append(new_row,ignore_index=True)
            


        fine_grained = st.checkbox('Get Fine Grained Evidences',key='bm25cord19checkbox')
        finegrained_cord19 = 'NA'
        
        if fine_grained:
            with st.spinner('Getting fine grained results'):
                json_doc = result['json_doc']
                query = input_data['text']
                encoded_query = tokenizer(query,truncation=True,padding=False,return_tensors='pt')
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
                st.write(json_doc['body_text'][0]['text'])
    form = st.form('User Study')
    slider1 = form.slider('Rate the semantic relevancy of the article generated above',0,5,0,1,key='s1')
    selectbox1 = form.selectbox('Do you agree with the veracity label generated for this article?',('Yes','No','Not Enough Information'),key='select1')
    selectbox2 = form.selectbox('Do you think that the fine grained results provide better evidences?',('Yes','No','Not Enough Information'),key='select2')
    submit1 = form.form_submit_button('Submit')
    if submit1:
        nr = {'data':'CORD19','model_name':'BM25','model_perf':slider1,'ver_perf':selectbox1,'fine_perf':selectbox2}
        df = df.append(nr,ignore_index=True)

    st.markdown('---')


if submit:
    with st.spinner('Getting Results...'):
        res = bm25search(input_data)
        result = res
        for i in range(top_k):
            st.write(f"Link of the Document: {result['Doc_Titles'][i]}")
            st.write(f"{result['Document_Body'][i][:100]}...")
            at = [('In-House','Dataset','#8A2BE2'),'  ',('BM25','IR Technique','#faa')]
            annotated_text(*at)
            st.write(' ')
            with st.spinner("Verifying Veracity..."):
                out = get_entailment_labels(input_data['text'],result['Document_Body'][i])
            st.markdown(f"**The article belongs to {out['category']} category with a confidence of {out['confidence']}**")
            new_row = {'id': result['Document'][i],
            'data':'In_House',
            'model_used': 'BM25',
            'claim': input_data['text'],
            'article_link': result['Doc_Titles'][i],
            'article_text': result['Document_Body'][i],
            'entailment_label': out['category'],
            'entailment_confidence': out['confidence']}
            data_frame = data_frame.append(new_row,ignore_index=True)

        fine_grained = st.checkbox('Get Fine Grained Evidences',key='bm25Inhousecheckbox')
        finegrained_inHouse = 'NA'
        if fine_grained:
            with st.spinner('Getting fine grained results'):
                json_doc = result['json_doc']
                query = input_data['text']
                encoded_query = tokenizer(query,truncation=True,padding=False,return_tensors='pt')
                with torch.no_grad():
                    query_state = torch.squeeze(model(**encoded_query)[0][:,1:-1,:],dim=0)
                par_states = []
                sents = json_doc['contents'].split('.')
                for sent in sents:
                    encoded_par = tokenizer(sent,stride=100,truncation=True,return_tensors='pt')
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
                st.write(sents[order[0]])
    form2 = st.form(key='f2')
    slider2 = form2.slider('Rate the semantic relevancy of the article above',0,5,0,1,key='s2')
    selectbox3 = form2.selectbox('Do you agree with the veracity label generated for this article?',('Yes','No','Not Enough Information'),key='select3')
    selectbox4 = form2.selectbox('Do you think that the fine grained results provide better evidences in this case?',('Yes','No','Not Enough Information'),key='select4')
    submit2 = form2.form_submit_button('Submit')
    if submit2:
        nr = {'data':'In_House','model_name':'BM25','model_perf':slider2,'ver_perf':selectbox3,'fine_perf':selectbox4}
        df = df.append(nr,ignore_index=True)
    st.markdown('---')
    



if submit:
    with st.spinner('Getting Results...'):
        res = dense_search_ourdata(input_data)
        result = res
        for i in range(top_k):
            st.write(f"Link of the Document: {result['Doc_Titles'][i]}")
            st.write(f"{result['Document_Body'][i][:30]}...")
            at = [('In-House','Dataset','#8A2BE2'),'  ',('Dense Search','IR Technique','#faa')]
            annotated_text(*at)
            st.write(' ')
            with st.spinner("Verifying Veracity..."):
                out = get_entailment_labels(input_data['text'],result['Document_Body'][i])
            st.markdown(f"**The article belongs to {out['category']} category with a confidence of {out['confidence']}**")

            st.markdown('---')
            new_row = {'id': result['Document'][i],
                        'data':'In_House',
                        'model_used': 'DenseSearch',
                        'claim': input_data['text'],
                        'article_link': result['Doc_Titles'][i],
                        'article_text': result['Document_Body'][i],
                        'entailment_label': out['category'],
                        'entailment_confidence': out['confidence']}
            data_frame = data_frame.append(new_row,ignore_index=True)
    form3 = st.form('f3')
    slider3 = form3.slider('Rate the semantic relevancy of the article generated above',0,5,0,1,key='s3')
    selectbox5 = form3.selectbox('Do you agree with the veracity label generated for this article?',('Yes','No','Not Enough Information'),key='select5')
    submit3 = form3.form_submit_button('Submit')
    if submit3:
        nr = {'data':'In_House','model_name':'Dense','model_perf':slider3,'ver_perf':selectbox5,'fine_perf':'NA'}
        df = df.append(nr,ignore_index=True)
    st.markdown('---')
    





if submit:
    with st.spinner('Getting Results...'):
        res = hybrid_search_ourdata(input_data)
        result = res
        for i in range(top_k):
            st.write(f"Link of the Document: {result['Doc_Titles'][i]}")
            st.write(f"{result['Document_Body'][i][:100]}...")
            at = [('In-House','Dataset','#8A2BE2'),'  ',('Hybrid Search','IR Technique','#faa')]
            annotated_text(*at)
            st.write(' ')
            with st.spinner("Verifying Veracity..."):
                out = get_entailment_labels(input_data['text'],result['Document_Body'][i])
            st.markdown(f"**The article belongs to {out['category']} category with a confidence of {out['confidence']}**")
            st.markdown('---')
            new_row = {'id': result['Document'][i],
                        'data': 'In_House',
                        'model_used': 'HybridSearch',
                        'claim': input_data['text'],
                        'article_link': result['Doc_Titles'][i],
                        'article_text': result['Document_Body'][i],
                        'entailment_label': out['category'],
                        'entailment_confidence': out['confidence']}
            data_frame = data_frame.append(new_row,ignore_index=True)
    form4 = st.form('f4')
    slider4 = form4.slider('Rate the semantic relevancy of the article generated above',0,5,0,1,key='s3')
    selectbox6 = form4.selectbox('Do you agree with the veracity label generated for this article?',('Yes','No','Not Enough Information'),key='select5')
    submit4 = form4.form_submit_button('Submit')
    if submit4:
        nr = {'data':'In_House','model_name':'Hybrid','model_perf':slider4,'ver_perf':selectbox6,'fine_perf':'NA'}
        df = df.append(nr,ignore_index=True)
    st.markdown('---')


if submit:
    final_submit = st.checkbox('Record Responses and Finish',key='final_submit')

    if final_submit:
        df.to_csv('collected_responses.csv',index=False)
        data_frame.to_csv('generated_responses.csv',index=False)
        st.success('Successfully recorded the responses! Thanks for participation.')


# if pages=='Form':
#     df = pd.read_csv('form.csv')
#     form = st.form('User Study')
#     slider1 = form.slider('Rate the semantic relevancy of the article generated using the BM25 technique on CORD-19 data',0,5,0,1,key='s1')
#     selectbox1 = form.selectbox('Do you agree with the veracity label generated for this article?',('Yes','No','Not Enough Information'),key='select1')
#     selectbox2 = form.selectbox('Do you think that the fine grained results provide better evidences?',('Yes','No','Not Enough Information'),key='select2')
#     st.markdown('---')

#     slider2 = form.slider('Rate the semantic relevancy of the article generated using the BM25 technique on In House data',0,5,0,1,key='s2')
#     selectbox3 = form.selectbox('Do you agree with the veracity label generated for this article?',('Yes','No','Not Enough Information'),key='select3')
#     selectbox4 = form.selectbox('Do you think that the fine grained results provide better evidences in this case?',('Yes','No','Not Enough Information'),key='select4')
#     st.markdown('---')
    
#     slider3 = form.slider('Rate the semantic relevancy of the article generated using the Dense Search technique on In House data',0,5,0,1,key='s3')
#     selectbox5 = form.selectbox('Do you agree with the veracity label generated for this article?',('Yes','No','Not Enough Information'),key='select5')
#     # selectbox2 = form.selectbox('Do you think that the fine grained results provide better evidences?',('Yes','No','Not Enough Information'),key='select2')
#     st.markdown('---')
    
#     slider4 = form.slider('Rate the semantic relevancy of the article generated using the Hybrid Search technique on In House data',0,5,0,1,key='s4')
#     selectbox6 = form.selectbox('Do you agree with the veracity label generated for this article?',('Yes','No','Not Enough Information'),key='select6')
#     # selectbox2 = form.selectbox('Do you think that the fine grained results provide better evidences?',('Yes','No','Not Enough Information'),key='select2')
#     st.markdown('---')
    
#     submit = form.form_submit_button('Submit')
#     if submit:
#         nr = {'perf_cord19':slider1,'perf_inhouseBM25':slider2,'perf_dense':slider3,'perf_hybrid':slider4,'ver_cord19':selectbox1,'fg_cord19':selectbox2,'ver_inhouseBM25':selectbox3,'fg_inhouseBM25':selectbox4,'ver_dense':selectbox5,'ver_hybrid':selectbox6}
#         df = df.append(nr,ignore_index=True)
#         df.to_csv('form.csv',index=False)
#         st.success('Successfully recorded the responses! Thanks for participation.')
