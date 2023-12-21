from tiktoken import get_encoding, encoding_for_model
from weaviate_interface import WeaviateClient, WhereFilter
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from openai import BadRequestError
from app_features import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_content_cache, load_data,
                          prepare_expanded_content, expand_response_content)
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import sys
import json
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)
 
## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)
##############
# START CODE #
##############
data_path = './data/impact_theory_data.json'

## RETRIEVER

api_key = os.environ['WEAVIATE_API_KEY']
url = os.environ['WEAVIATE_ENDPOINT']
client = WeaviateClient(api_key, url)
available_classes=sorted(client.show_classes())
logger.info(available_classes)

## RERANKER
reranker = ReRanker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')

## LLM 

model_ids = ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613']
model_name = model_ids[1]
llm = GPT_Turbo(model=model_name, api_key=os.environ['OPENAI_API_KEY'])

## ENCODING

encoding = encoding_for_model(model_name)

## INDEX NAME

##############
#  END CODE  #
##############

## DATA + CACHE
data_path = 'data/impact_theory_data.json'
# cache_path = '../impact-theory-cache-window2.parquet'
cache_path = 'impact-theory-minilmL6-256.parquet'
data = load_data(data_path)
# cache = load_content_cache(cache_path)

expanded_content_map = prepare_expanded_content(cache_path, 1)

#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

def main():
        
    with st.sidebar:
        # guest = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')
        guest_input = st.selectbox(label='Select Guest', options=guest_list, index=None, placeholder='Select Guest')
        alpha_input = st.slider(label='Alpha for Hybrid', min_value=0.00, max_value=1.00, value=0.45, step=0.05)
        retrieval_limit = st.slider(label='Hybrid Search Retrieval Results', min_value=10, max_value=300, value=10, step=10)
        reranker_topk = st.slider(label='Reranker Top K', min_value=1, max_value=5, value=3, step=1)
        temperature_input = st.slider(label='Temperature of LLM', min_value=0.0, max_value=2.0, value=0.10, step=0.10)
        class_name = st.selectbox(label='Class Name:', options=available_classes, index=None, placeholder='Select Class Name')

    client.display_properties.append("summary")

    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

        if query:
            ##############
            # START CODE #
            ##############
            
            if class_name is None:
                st.write('You have not selected a class.')
                return

            # st.write('Hmmm...this app does not seem to be working yet.  Please check back later.')
            # if guest:
            #    st.write(f'However, it looks like you selected {guest} as a filter.')

            # make hybrid call to weaviate
            guest_filter = WhereFilter(path=['guest'], operator='Equal', valueText=guest_input).todict() if guest_input else None
            hybrid_response = client.hybrid_search(query, 
                                                   class_name, 
                                                   alpha=alpha_input, 
                                                   limit=retrieval_limit, 
                                                   display_properties=client.display_properties, 
                                                   where_filter=guest_filter)
            
            # rerank results
            ranked_response = reranker.rerank(hybrid_response, 
                                              query, 
                                              top_k=reranker_topk, 
                                              apply_sigmoid=True)
            

            expanded_response = expand_response_content(ranked_response, expanded_content_map)

            # validate token count is below threshold
            token_threshold = 8000 if model_name == model_ids[0] else 3500
            valid_response = validate_token_threshold(
                ranked_results=expanded_response,
                base_prompt=question_answering_prompt_series,
                query=query,
                tokenizer=encoding,
                token_threshold=token_threshold,
                verbose=True
            )
            
            ##############
            #  END CODE  #
            ##############

            # generate LLM prompt

            use_llm = True

            if use_llm:
                st.subheader("Response from Impact Theory (context)")

                # prep for streaming response

                with st.spinner('Generating Response...'):
                    st.markdown("----")
                    # creates container for LLM response
                    chat_container, response_box = [], st.empty()
            
                    prompt = generate_prompt_series(query=query, results=valid_response)

                    try:
                            for resp in llm.get_chat_completion(
                                prompt=prompt,
                                temperature=temperature_input,
                                max_tokens=350, # expand for more verbose answers
                                show_response=True,
                                stream=True):

                                # inserts chat stream from LLM
                                with response_box:
                                    content = resp.choices[0].delta.content
                                    if content:
                                        chat_container.append(content)
                                        result = "".join(chat_container).strip()
                                        st.write(f'{result}')
                    except BadRequestError:
                        logger.info('Making request with smaller context...')
                        valid_response = validate_token_threshold(
                            ranked_results=ranked_response,
                            base_prompt=question_answering_prompt_series,
                            query=query,
                            tokenizer=encoding,
                            token_threshold=token_threshold,
                            verbose=True
                        )

                        # generate LLM prompt
                        prompt = generate_prompt_series(query=query, results=valid_response)
                        for resp in llm.get_chat_completion(
                            prompt=prompt,
                            temperature=temperature_input,
                            max_tokens=350, # expand for more verbose answers
                            show_response=True,
                            stream=True):

                            try:
                                # inserts chat stream from LLM
                                with response_box:
                                    content = resp.choices[0].delta.content
                                    if content:
                                        chat_container.append(content)
                                        result = "".join(chat_container).strip()
                                        st.write(f'{result}')
                            except Exception as e:
                                print(e)

            else:
                st.subheader("Use of LLM is switched off.")


            
            st.subheader("Search Results")
            for i, hit in enumerate(valid_response):
                col1, col2 = st.columns([7, 3], gap='large')
                image = hit['thumbnail_url']
                episode_url = hit['episode_url']
                title = hit['title']
                show_length = hit['length']
                time_string = convert_seconds(show_length)
                summary = hit['summary']

                with col1:
                    st.write( search_result(i=i,  
                                            url=episode_url, 
                                            guest=hit['guest'], 
                                            title=title,
                                    content=hit['content'], 
                                    length=time_string),
                            unsafe_allow_html=True)
                    st.write('\n\n')
                    with st.expander("View Episode Summary:"):
                        ep_summary = summary
                        st.write(ep_summary)
                with col2:
                    # st.markdown(f"<a href='{episode_url}' <img src={image} width='200'></a>",  unsafe_allow_html=True)
                    # st.write(f"{episode_url}",  unsafe_allow_html=True)
                    st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)


if __name__ == '__main__':
    main()