# coding: utf-8

import discord
from gensim import corpora, matutils
from gensim.models import TfidfModel
from janome.tokenizer import Tokenizer
import json
import csv
import numpy as np
import random
import warnings
import time

"""
PuTTyKey:
8UH3T3WWza1t8CfGqFE0vZXX5qHb8aJp1RTox9WlEtSHAxSAhReoTBbbN7U2tY3Y3R4BbOLuH+641b0ycYTq8jJfbSx2bRjH4dIQy3+tkV0QI39SqRmWlHcePe3ZMKM0cE2uc2vrbcjbKQhDCL2XFGQirYz8t4r4teTxXVA8zEUlkqlv8grxfhat4O7y+76lVZjpJNSeTEfwQ== usagibotv2
"""

warnings.simplefilter('ignore')
client = discord.Client()

@client.event
async def on_ready():
    print("-"*20)
    print("ユーザー名：", client.user.name)
    print("ユーザーID：", client.user.id)
    print("-"*20)

meigen = []
with open('daigo_meigen11.csv', encoding="utf-8_sig") as f:
    obj = csv.reader(f)
    for row in obj:
        try:
            meigen.append(row[0])
        except:
            pass # null

with open('text_processed.csv', encoding="utf-8_sig") as f:
    reader = csv.reader(f)
    text_processed = [obj for obj in reader]

dictionary = corpora.Dictionary(text_processed)

@client.event
async def on_message(message):
    if not message.author.id == client.user.id:
        #await client.send_message(message.channel, message.content)
        await message.channel.send("ふむ、なるほどね。(20秒くらい待ってね..(˘ω˘)..)")


    target = message.content

    outcome = meigen + target # outcomeはtargetを含む全テキストデータのリスト

    stop_words = ['lik','retweets','replies','1','2','3','4','5','6','7','8','9']
    numbers = []
    for i in range(3000):
        numbers.append(str(i))

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    sonota = ["'"]

    def token_generator(text,stopwords=numbers+months+sonota):
        tokens = []
        t = Tokenizer(mmap=True)

        num = 0
        for token in t.tokenize(text):
            if token.surface not in stopwords:
                if (token.part_of_speech.split(',')[0] == '名詞')            or (token.part_of_speech.split(',')[0] == '形容詞')            or (token.part_of_speech.split(',')[0] == '動詞')            or (token.part_of_speech.split(',')[0] == '副詞'):
                    tokens.append(token.surface)
        return tokens

    # targetの形態素解析
    targettoken = [token_generator(target[0])]

    text_processed3 = text_processed + targettoken

    corpus = [dictionary.doc2bow(doc) for doc in text_processed3]
    model = TfidfModel(corpus)
    tfidf = [model[i] for i in corpus]
    doc_matrix = matutils.corpus2csc(tfidf).transpose()

    c=len(outcome)
    cos_sim = np.zeros([c, c]) # 初期化0の行列を作成
    var_SDGs = doc_matrix.dot(doc_matrix.transpose()).toarray()

    for j in range(c):
        cos_sim[-1,j] = var_SDGs[-1,j]/(np.sqrt(var_SDGs[-1, -1])*np.sqrt(var_SDGs[j, j]))

    simirality_for_the_target3 =cos_sim[c-1:c,:-1] # input自身の要素だけ抜いた最後の行
    A2_result = np.argsort(-simirality_for_the_target3)

    if not message.author.id == client.user.id:
        if A2_result[0,0] != 0:
            await message.channel.send("\n\n" + meigen[A2_result[0,0]] +"\n\n" + "参考になったかな？またおいで。")
        else:
            print("全然違うこと言うけど、")
            print(meigen[random.randint(0,len(meigen)-1)])
            await message.channel.send("\n\n" + meigen[A2_result[0,0]] +"\n\n" + "参考になったかな？またおいで。")

client.run("")
