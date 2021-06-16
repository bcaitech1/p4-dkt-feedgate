import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import copy
import math
import sys
import types

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel    
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel



def str_to_class(field):
    try:
        identifier = getattr(sys.modules[__name__], field)
    except AttributeError:
        raise NameError("%s doesn't exist." % field)
    if isinstance(identifier, type):
        return identifier
    raise TypeError("%s is not a class." % field)

class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.drop_out = self.args.drop_out

        # Embedding 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//9)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//9)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//9)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//9)

        #Linear
        self.linear_time = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))
        self.linear_elapsed_time = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))
        self.linear_test_ans = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))
        self.linear_test_cnt = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))
        self.linear_user_ans = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))
        self.linear_user_cnt = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//9)*9, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True,
                            bidirectional =True,
                            dropout = self.drop_out)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim*2, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers*2,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers*2,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        test, time, question, tag, elapsed_time, test_ans, user_ans, user_cnt, _, mask, interaction, _ = input

        batch_size = interaction.size(0)
        elapsed_time = torch.unsqueeze(elapsed_time,2) ## [64,20] -> [64,20,1]
        time = torch.unsqueeze(time,2)
        test_ans = torch.unsqueeze(test_ans,2)
        user_ans = torch.unsqueeze(user_ans,2)
        user_cnt = torch.unsqueeze(user_cnt,2)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction) ## [64,20] -> [64,20,21]
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        #Linear

        linear_elapsed_time = self.linear_elapsed_time(elapsed_time.float()) ## [64,20,1] -> [64,20,21] -> [64,20,21] (layer normalize)
        linear_time = self.linear_time(time.float())
        linear_test_ans = self.linear_test_ans(test_ans.float())
        # linear_test_cnt = self.linear_test_cnt(test_cnt.float())
        linear_user_ans = self.linear_user_ans(user_ans.float())
        linear_user_cnt = self.linear_user_cnt(user_cnt.float())

        embed = torch.cat([embed_interaction,
                           embed_test,
                           linear_time,
                           embed_question,
                           embed_tag,
                           linear_elapsed_time,
                           linear_test_ans,
                           linear_user_ans,
                           linear_user_cnt,
                           ], 2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim*2)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds




class LSTMATTN(nn.Module):

    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//9)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//9)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//9)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//9)

        #Linear
        self.linear_time = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))
        self.linear_elapsed_time = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))
        self.linear_test_ans = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))
        self.linear_test_cnt = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))

        self.linear_user_ans = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))
        self.linear_user_cnt = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))
        self.linear_cate_ans = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))
        

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//9)*9, self.hidden_dim)
        
        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True,
                            bidirectional =True,
                            dropout = self.drop_out)
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim*2,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim*2,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)            
    
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim*2, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers*2,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers*2,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        test, time, question, tag, elapsed_time, test_ans, user_ans, user_cnt, _, mask, interaction, _ = input

        batch_size = interaction.size(0)
        elapsed_time = torch.unsqueeze(elapsed_time,2) ## [64,20] -> [64,20,1]
        time = torch.unsqueeze(time,2)
        test_ans = torch.unsqueeze(test_ans,2)
        # tag_ans = torch.unsqueeze(tag_ans,2)
        user_ans = torch.unsqueeze(user_ans,2)
        user_cnt = torch.unsqueeze(user_cnt,2)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction) ## [64,20] -> [64,20,21]
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        #Linear

        linear_elapsed_time = self.linear_elapsed_time(elapsed_time.float()) ## [64,20,1] -> [64,20,21] -> [64,20,21] (layer normalize)
        linear_time = self.linear_time(time.float())
        linear_test_ans = self.linear_test_ans(test_ans.float())
        linear_user_ans = self.linear_user_ans(user_ans.float())
        linear_user_cnt = self.linear_user_cnt(user_cnt.float())

        embed = torch.cat([embed_interaction,
                           embed_test,
                           linear_time,
                           embed_question,
                           embed_tag,
                           linear_elapsed_time,
                           linear_test_ans,
                           linear_user_ans,
                           linear_user_cnt,
                           ], 2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim*2)
                
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers
        
        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)        
        sequence_output = encoded_layers[-1]
        
        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class Bert(nn.Module):

    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_category = nn.Embedding(self.args.n_category + 1, self.hidden_dim//3)
        self.embedding_number = nn.Embedding(self.args.n_number + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)

        # Linear
        self.linear_feature = nn.Sequential(nn.Linear(1, self.hidden_dim//3), nn.LayerNorm(self.hidden_dim//3))

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//3)*10, self.hidden_dim)

        # Bert config
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len          
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)  

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)
       
        self.activation = nn.Sigmoid()


    def forward(self, input):
        test, category, number, tag, soltime, time, user_acc, ItemID_mean, test_mean, _, mask, interaction, _ = input

        batch_size = interaction.size(0)
        soltime = torch.unsqueeze(soltime,2) ## [batch_size,max_seq_len] [64,20] -> [64,20,1]
        time = torch.unsqueeze(time,2)
        user_acc = torch.unsqueeze(user_acc,2)
        ItemID_mean = torch.unsqueeze(ItemID_mean,2)
        test_mean = torch.unsqueeze(test_mean,2)


        # Embedding

        embed_interaction = self.embedding_interaction(interaction) ## [64,20] -> [64,20,21]
        embed_test = self.embedding_test(test)
        embed_category = self.embedding_category(category)
        embed_number = self.embedding_number(number)
        # embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        # Linear
        linear_soltime = self.linear_feature(soltime.float()) ## [64,20,1] -> [64,20,21] -> [64,20,21] (layer normalize)
        linear_time = self.linear_feature(time.float())
        linear_user_acc = self.linear_feature(user_acc.float())
        linear_ItemID_mean = self.linear_feature(ItemID_mean.float())
        linear_test_mean = self.linear_feature(test_mean.float()) 

        embed = torch.cat([embed_interaction,
                           embed_test,
                        #    embed_question,
                           embed_category,
                           embed_number,
                           embed_tag,
                           linear_soltime,
                           linear_time,
                           linear_user_acc,
                           linear_ItemID_mean,
                           linear_test_mean,], 2)

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


class Saint(nn.Module):
    
    def __init__(self, args):
        super(Saint, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.dropout = self.args.dropout
        #self.dropout = 0.
        
        # Embedding 
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//9)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//9)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//9)

        #Linear
        self.linear_time = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))
        self.linear_elapsed_time = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))
        self.linear_test_ans = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))
        self.linear_test_cnt = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))

        self.linear_user_ans = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))
        self.linear_user_cnt = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))
        self.linear_cate_ans = nn.Sequential(nn.Linear(1, self.hidden_dim//9), nn.LayerNorm(self.hidden_dim//9))

        #encoder combination projection
        self.enc_comb_proj = nn.Linear((self.hidden_dim//9)*8, self.hidden_dim)

        # DECODER embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//9)
        
        # decoder combination projection
        self.dec_comb_proj = nn.Linear((self.hidden_dim//9)*9, self.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        

        self.transformer = nn.Transformer(
            d_model=self.hidden_dim, 
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers, 
            num_decoder_layers=self.args.n_layers, 
            dim_feedforward=self.hidden_dim, 
            dropout=self.dropout, 
            activation='relu')

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None
    
    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, input):
        test, time, question, tag, elapsed_time, test_ans, user_ans, user_cnt, _, mask, interaction, _ = input

        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        elapsed_time = torch.unsqueeze(elapsed_time,2) ## [64,20] -> [64,20,1]
        time = torch.unsqueeze(time,2)
        test_ans = torch.unsqueeze(test_ans,2)
        user_ans = torch.unsqueeze(user_ans,2)
        user_cnt = torch.unsqueeze(user_cnt,2)

        # Embedding
        embed_test = self.embedding_test(test)  ## [64,20] -> [64,20,21]
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        #Linear

        linear_elapsed_time = self.linear_elapsed_time(elapsed_time.float()) ## [64,20,1] -> [64,20,21] -> [64,20,21] (layer normalize)
        linear_time = self.linear_time(time.float())
        linear_test_ans = self.linear_test_ans(test_ans.float())
        linear_user_ans = self.linear_user_ans(user_ans.float())
        linear_user_cnt = self.linear_user_cnt(user_cnt.float())

        embed_enc = torch.cat([embed_test,
                            linear_time,
                            embed_question,
                        #    embed_category,
                        #    embed_number,
                            embed_tag,
                        #    
                            linear_elapsed_time,
                            linear_test_ans,
                        #    linear_test_cnt,
                            linear_user_ans,
                            linear_user_cnt,
                           ], 2)

        embed_enc = self.enc_comb_proj(embed_enc)

        # Decoder    
        # Embedding
        embed_interaction = self.embedding_interaction(interaction) ## [64,20] -> [64,20,21]
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        #Linear

        linear_elapsed_time = self.linear_elapsed_time(elapsed_time.float()) ## [64,20,1] -> [64,20,21] -> [64,20,21] (layer normalize)
        linear_time = self.linear_time(time.float())
        linear_test_ans = self.linear_test_ans(test_ans.float())
        linear_user_ans = self.linear_user_ans(user_ans.float())
        linear_user_cnt = self.linear_user_cnt(user_cnt.float())


        embed_dec = torch.cat([embed_interaction,
                           embed_test,
                           linear_time,
                           embed_question,
                        #    embed_category,
                        #    embed_number,
                           embed_tag,
                        #    
                           linear_elapsed_time,
                           linear_test_ans,
                        #    linear_test_cnt,
                           linear_user_ans,
                           linear_user_cnt,
                           ], 2)

        embed_dec = self.dec_comb_proj(embed_dec)

        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)
            
        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)
            
        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)
            
  
        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)
        
        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)
        
        out = self.transformer(embed_enc, embed_dec,
                               src_mask=self.enc_mask,
                               tgt_mask=self.dec_mask,
                               memory_mask=self.enc_dec_mask)

        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds



class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self,ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))

class LastQuery(nn.Module):
    def __init__(self, args):
        super(LastQuery, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.drop_out = self.args.drop_out
        
       # Embedding 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        # self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_category = nn.Embedding(self.args.n_category + 1, self.hidden_dim//3)
        self.embedding_number = nn.Embedding(self.args.n_number + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)
        # self.embedding_soltime = nn.Embedding(self.args.n_cate_time + 1, self.hidden_dim//3)
        self.linear_soltime = nn.Sequential(nn.Linear(1, self.hidden_dim//3), nn.LayerNorm(self.hidden_dim//3))
        self.linear_time = nn.Sequential(nn.Linear(1, self.hidden_dim//3), nn.LayerNorm(self.hidden_dim//3))
        
        # self.linear_user_acc = nn.Sequential(nn.Linear(1, self.hidden_dim//3), nn.LayerNorm(self.hidden_dim//3))
        # self.linear_ItemID_mean = nn.Sequential(nn.Linear(1, self.hidden_dim//3), nn.LayerNorm(self.hidden_dim//3))
        # self.linear_test_mean = nn.Sequential(nn.Linear(1, self.hidden_dim//3), nn.LayerNorm(self.hidden_dim//3))
        # self.linear_tag_mean = nn.Sequential(nn.Linear(1, self.hidden_dim//3), nn.LayerNorm(self.hidden_dim//3))
        # self.linear_sol_num = nn.Sequential(nn.Linear(1, self.hidden_dim//3), nn.LayerNorm(self.hidden_dim//3))

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//3)*7, self.hidden_dim)

        # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # 하지만 사용 여부는 자유롭게 결정해주세요 :)
        # self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)
        
        # Encoder
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.args.n_heads)
        self.mask = None # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block(self.hidden_dim)      

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            self.args.n_layers,
            batch_first=True,
            dropout=self.drop_out)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
       
        self.activation = nn.Sigmoid()

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)
 
    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        c = c.to(self.device)

        return (h, c)


    def get_mask(self, seq_len, mask, batch_size):
        new_mask = torch.zeros_like(mask)
        new_mask[mask == 0] = 1
        new_mask[mask != 0] = 0
        mask = new_mask
    
        # batchsize * n_head 수만큼 각 mask를 반복하여 증가시킨다
        mask = mask.repeat(1, self.args.n_heads).view(batch_size*self.args.n_heads, -1, seq_len)
        return mask.masked_fill(mask==1, float('-inf'))


    def forward(self, input):
        
        test, category, number, tag, soltime, time, _, mask, interaction, _ = input
        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        batch_size = interaction.size(0)
        soltime = torch.unsqueeze(soltime,2) ## [64,20] -> [64,20,1]
        time = torch.unsqueeze(time,2)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction) ## [64,20] -> [64,20,21]
        embed_test = self.embedding_test(test)
        embed_category = self.embedding_category(category)
        embed_number = self.embedding_number(number)
        # embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        linear_soltime = self.linear_soltime(soltime.float()) ## [64,20,1] -> [64,20,21] -> [64,20,21] (layer normalize)
        # embed_soltime = self.embedding_soltime(soltime)
        linear_time = self.linear_time(time.float())
        embed = torch.cat([embed_interaction,
                           embed_test,
                           embed_category,
                           embed_number,
                           embed_tag,
                        #    
                        #    linear_time,
                        #    linear_user_acc,
                        #    linear_ItemID_mean,
                        #    linear_test_mean,
                        #    linear_tag_mean,
                           linear_soltime,
                           linear_time,], 2)

        embed = self.comb_proj(embed)

        # Positional Embedding
        # last query에서는 positional embedding을 하지 않음
        # position = self.get_pos(seq_len).to('cuda')
        # embed_pos = self.embedding_position(position)
        # embed = embed + embed_pos

        ####################### ENCODER #####################
        q = self.query(embed)[:, -1:, :].permute(1, 0, 2)
        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        self.mask = self.get_mask(seq_len, mask, batch_size).to(self.device)
        out, _ = self.attn(q, k, v, attn_mask=self.mask)
        
        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out
        out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


class Saint(nn.Module):
    
    def __init__(self, args):
        super(Saint, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.dropout = self.args.drop_out
        #self.dropout = 0.
        
        ### Embedding 
        # ENCODER embedding
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)
        
        # encoder combination projection
        self.enc_comb_proj = nn.Linear((self.hidden_dim//3)*3, self.hidden_dim)

        # DECODER embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        
        # decoder combination projection
        self.dec_comb_proj = nn.Linear((self.hidden_dim//3)*4, self.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        

        self.transformer = nn.Transformer(
            d_model=self.hidden_dim, 
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers, 
            num_decoder_layers=self.args.n_layers, 
            dim_feedforward=self.hidden_dim, 
            dropout=self.dropout, 
            activation='relu')

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None
    
    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, input):
        test, question, tag, _, mask, interaction, _ = input

        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        # 신나는 embedding
        # ENCODER
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed_enc = torch.cat([embed_test,
                               embed_question,
                               embed_tag,], 2)

        embed_enc = self.enc_comb_proj(embed_enc)
        
        # DECODER     
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed_interaction = self.embedding_interaction(interaction)

        embed_dec = torch.cat([embed_test,
                               embed_question,
                               embed_tag,
                               embed_interaction], 2)

        embed_dec = self.dec_comb_proj(embed_dec)

        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)
            
        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)
            
        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)
            
  
        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)
        
        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)
        
        out = self.transformer(embed_enc, embed_dec,
                               src_mask=self.enc_mask,
                               tgt_mask=self.dec_mask,
                               memory_mask=self.enc_dec_mask)

        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds



class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self,ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))

class LastQuery(nn.Module):
    def __init__(self, args):
        super(LastQuery, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        
        # Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)
        self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//3)*4, self.hidden_dim)

        # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # 하지만 사용 여부는 자유롭게 결정해주세요 :)
        # self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)
        
        # Encoder
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.args.n_heads)
        self.mask = None # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block(self.hidden_dim)      

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            self.args.n_layers,
            batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
       
        self.activation = nn.Sigmoid()

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)
 
    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        c = c.to(self.device)

        return (h, c)


    def forward(self, input):
        test, question, tag, _, mask, interaction, index = input
        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        # 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat([embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,], 2)

        embed = self.comb_proj(embed)

        # Positional Embedding
        # last query에서는 positional embedding을 하지 않음
        # position = self.get_pos(seq_len).to('cuda')
        # embed_pos = self.embedding_position(position)
        # embed = embed + embed_pos

        ####################### ENCODER #####################

        q = self.query(embed).permute(1, 0, 2)
        
        
        q = self.query(embed)[:, -1:, :].permute(1, 0, 2)
        
        
        
        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        out, _ = self.attn(q, k, v)
        
        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out
        out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        print(preds)

        return preds