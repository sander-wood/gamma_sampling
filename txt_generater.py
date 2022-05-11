import torch
import numpy as np
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from samplings import gamma_sampling
from transformers import GPT2Tokenizer, GPT2LMHeadModel

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id).to(device)

def get_topic(word_embeddings, key_words, top_n):

    words_vec = torch.zeros(1,768).to(device)

    for key_word in key_words:
        key_idx = tokenizer.encode(key_word,add_prefix_space=True)[0]
        words_vec += word_embeddings[key_idx]

    words_vec /= len(key_words)

    sim_list = dict(zip(range(50257),torch.cosine_similarity(words_vec, word_embeddings).cpu().detach().numpy().tolist()))

    word_list = []

    for word in sorted(sim_list.items(), key = lambda kv:(kv[1], kv[0]),reverse=True):
        
        idx = np.array(word[0])
        idx_id = torch.from_numpy(idx[np.newaxis, np.newaxis, ...])
        word = tokenizer.decode(idx_id[0], skip_special_tokens=True)

        if len(word)>1 and word[0]==' ' and word not in word_list:
            word_list.append(word)
            if len(word_list)==top_n:
                break
    
    for i in range(len(word_list)):
        word_list[i] = tokenizer.encode(word_list[i],add_prefix_space=False)[0]

    return word_list


def get_sentiment_tokens(path):

    tokens = []
    f = open(path,encoding = "utf-8")
    lines = f.readlines()

    for line in lines:
        token = tokenizer.encode(line[:-1],add_prefix_space=True)[0]
        tokens.append(token)
    
    return tokens


def generate_txt(text,
                 key_word=None,
                 sentence_gamma=0.5,
                 rep_gamma=0.9,
                 topic_gamma=0.5,
                 relate_gamma=0.3,
                 sentiment_gamma=0.5,
                 rep_num=100,
                 nn_num=10,
                 end_num=20,
                 top_n=100,
                 top_k=0,
                 top_p=0.8,
                 temperature=1,
                 max_len=100,
                 seed=None):

    recent_nn = []
    pos_list = pos_tag(word_tokenize(text))

    for elememt in pos_list:
        if 'NN' in elememt[1] and elememt[0].isalpha():
            recent_nn.append(elememt[0])

    if len(recent_nn)>nn_num:
        recent_nn = recent_nn[-nn_num:]

    if sentiment_gamma>=0.5:
        sentiment_gamma = 1-sentiment_gamma
        sentiment_tokens = get_sentiment_tokens('positive.txt')

    else:
        sentiment_gamma = sentiment_gamma
        sentiment_tokens = get_sentiment_tokens('negative.txt')

    input_ids = tokenizer.encode(text, return_tensors='pt')
    output_ids = input_ids
    word_embeddings = model.transformer.wte.weight

    if key_word!=None:
        topic_tokens = get_topic(word_embeddings, [key_word], top_n=top_n)
    else:
        topic_tokens = []

    related_tokens = get_topic(word_embeddings, recent_nn, top_n=top_n)
    end_tokens = [13, 30, 0, 526, 1701, 2474, 737, 19427, 31520]
    recent_tokens = list(set(input_ids[0][-rep_num:].numpy().tolist()).difference(set(end_tokens)))

    tokens = [end_tokens]+[recent_tokens]+[topic_tokens]+[related_tokens]+[sentiment_tokens]
    gamma = [sentence_gamma, rep_gamma, topic_gamma, relate_gamma, sentiment_gamma]
    new_top_p = top_p

    for i in range(max_len): 

        if i>=max_len-end_num:
            gamma[0] = (max_len-i-1)/end_num*sentence_gamma
            new_top_p = top_p+(1+i-max_len+end_num)/end_num*(1-top_p)
        
        output_ids = output_ids.to(device)
        output = model(output_ids)
        predictions = output[0][0][-1]
        probs = torch.nn.Softmax(dim=-1)(predictions).cpu().detach().numpy()

        index = gamma_sampling(probs, 
                               tokens, 
                               gamma, 
                               activation='tan',
                               top_k=top_k,
                               top_p=new_top_p, 
                               temperature=temperature,
                               seed=seed)

        if index not in tokens[1] and index not in end_tokens:
            tokens[1].append(index)
            
        if len(tokens[1])>rep_num:
            tokens[1] = tokens[1][1:]

        index_id = torch.from_numpy(index[np.newaxis, np.newaxis, ...]).to(device)
        output_ids = torch.cat((output_ids, index_id), -1)

        index_word = tokenizer.decode(index_id[0]).lstrip()

        if index_word.isalpha() and 'NN' in pos_tag([index_word])[0][1]:

            recent_nn.append(index_word)

            if len(recent_nn)>nn_num:
                recent_nn = recent_nn[1:]
            
            tokens[-2] = get_topic(word_embeddings, recent_nn, top_n=top_n)

        if i>max_len-end_num and index in end_tokens:
            break

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == '__main__':

    text = 'The issue focused'
    key_word = 'computers'
    topic_gamma = 0.1
    sentiment_gamma=0.5
    sentence_gamma=0.5
    max_len = 100

    generated_txt = generate_txt(text=text,
                                 key_word=key_word, 
                                 topic_gamma=topic_gamma, 
                                 sentiment_gamma=sentiment_gamma, 
                                 sentence_gamma=sentence_gamma,
                                 max_len=max_len,
                                 seed=0)
    print(generated_txt)