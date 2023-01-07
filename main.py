#from bigbird.core import modeling
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizerFast,BertTokenizer
from sklearn.utils import shuffle
from bigbird.core.modeling import BigbirdForTokenClassification,CustomNonPaddingTokenLoss
import sys
from bigbird.core.tokenization import is_number,searchInsert




label2id={
    "TIME":1,
    "LOC":3,
    "TITLE":5,
    "NEGATIVE TIME":7,
    }


df=pd.read_csv("G:/bigbird2/bigbird-master/trainset_withadditionaldata.csv")

df["label3"]=""
df["xid3"]=""
df["yid3"]=""
df["string3"]=""
for i in df.index:
    #这个循环是用来对齐数字的
    label=eval(df.at[i,"label2"])
    string3=""
    numposition=[]
    numcumulate=[]
    numcount=0
    xid2=eval(df.at[i,"xid2"])
    yid2=eval(df.at[i,"yid2"])
    xid3=[]
    yid3=[]
    for j in range(len(df.at[i,"string2"])):
        if is_number(ord(df.at[i,"string2"][j])) is True:
                     string3=string3+" "+df.at[i,"string2"][j]+" "
                     numposition.append(j)
                     numcumulate.append(numcount*2)
                     numcount+=1
                     xid3.extend([xid2[j]]*3)
                     yid3.extend([yid2[j]]*3)

        else:
            string3+=df.at[i,"string2"][j]
            xid3.append(xid2[j])
            yid3.append(yid2[j])
    for k in range(len(label)):
        sepid=searchInsert(numposition,label[k]["start"])
        if sepid==len(numcumulate):
            label[k]["start"]+=(numcumulate[-1]+2)
        elif label[k]["start"]==numposition[sepid]:
            label[k]["start"]+=(numcumulate[sepid]+1)
        else:
            label[k]["start"]+=numcumulate[sepid]
        if df.at[i,"string2"][label[k]["end"]-1]==" ":
            label[k]["end"]-=1
        sepid=searchInsert(numposition,label[k]["end"]-1)
        if sepid == len(numcumulate):
            label[k]["end"] += (numcumulate[-1] +2)
        elif (label[k]["end"]-1) == numposition[sepid]:
            label[k]["end"] += (numcumulate[sepid] + 1)
        else:
            label[k]["end"] += numcumulate[sepid]

    df.at[i,"string3"]=string3
    df.at[i,"xid3"]=xid3
    df.at[i,"yid3"]=yid3
    df.at[i,"label3"]=label

df.to_csv("trainfilev3_withadditionaldata.csv")

df=pd.read_csv("G:/bigbird2/bigbird-master/trainfilev3_withadditionaldata.csv")
tokenizer=BertTokenizerFast.from_pretrained("bert-base-chinese")


df["labellist"]=""
for i in df.index:
    #这个循环是用来对齐subword和label之间的关系的
    rawsplittext=tokenizer(df.at[i,"string3"])
    labellist =[0]*len(rawsplittext["input_ids"])
    label3=eval(df.at[i,"label3"])
    for k in range(len(label3)):
        start=rawsplittext.char_to_token(batch_or_char_index=label3[k]["start"])
        m=1
        while start is None:
            start = rawsplittext.char_to_token(batch_or_char_index=(label3[k]["start"]+m))
            m+=1
        end=rawsplittext.char_to_token(batch_or_char_index=(label3[k]["end"]-1))
        m=1
        while end is None:
            end = rawsplittext.char_to_token(batch_or_char_index=(label3[k]["end"]-m))
            m-=1
        labellist[start] = label2id[label3[k]["labels"][0]]
        for l in range(start+1,end):
            labellist[l]=label2id[label3[k]["labels"][0]]+1
    df.at[i,"labellist"]=labellist
#df.to_csv("trainfilev4.csv")


df["xid4"]=""
df["yid4"]=""
df["tokenlist"]=""
df["tokenidlist"]=""

for i in df.index:
    #这个循环是用来对齐subword和xid和yid之间的关系的
    rawsplittext=tokenizer(df.at[i,"string3"])
    df.at[i,"tokenlist"]=rawsplittext.tokens
    df.at[i,"tokenidlist"] =rawsplittext["input_ids"]
    xid4=[]
    yid4=[]
    xid3=eval(df.at[i,"xid3"])
    yid3=eval(df.at[i,"yid3"])
    for k in range(len(rawsplittext["input_ids"])):
        try:
            pos=rawsplittext.token_to_chars(batch_or_token_index=k)
            if xid3[pos.start]<0:
                print("遇到异常值")
                print(i)
            xid4.append(xid3[pos.start])
            yid4.append(yid3[pos.start])
        except:
            xid4.append(xid3[0])
            yid4.append(yid3[0])
    df.at[i,"xid4"]=xid4
    df.at[i,"yid4"]=yid4


df["positionid"]=""
for i in df.index:
    initpos=0
    positionids=[0,]
    for j in range(1,len(df.at[i,"xid4"])):
        if df.at[i,"xid4"][j-1]==df.at[i,"xid4"][j]:
            initpos+=1
            positionids.append(initpos)
        else:
            initpos=0
            positionids.append(initpos)
    df.at[i,"positionid"]=positionids




maxx=0.0
maxy=0.0

for i in df.index:
    if max(df.at[i,"xid4"])>maxx:
        maxx=max(df.at[i,"xid4"])
    if max(df.at[i,"yid4"])>maxy:
        maxy=max(df.at[i,"yid4"])

len(df.at[1,"xid4"])==len(df.at[1,"labellist"])


xmaxlength=1550
ymaxlength=2048

df=shuffle(df,random_state=0)
xidbatch=list(df["xid4"])
yidbatch=list(df["yid4"])
tokenbatch=list(df["tokenidlist"])
xidbatch=[[int(j) if j <1550 else 1549 for j in i]for i in xidbatch]
yidbatch=[[int(j) if j <2048 else 2047 for j in i]for i in yidbatch]
positionbatch=list(df["positionid"])
positionbatch=[[int(j) if j <1200 else 1199  for j in i]for i in positionbatch]
labelbatch=list(df["labellist"])

#check max sequence length
maxlength=0
maxx=0
maxy=0
minx=0
miny=0
maxpos=0
minpos=0

for i in range(len(tokenbatch)):
    if len(tokenbatch[i])>maxlength:
        maxlength=len(tokenbatch[i])
    if max(yidbatch[i])>maxy:
        maxy=max(yidbatch[i])
    if min(yidbatch[i])<miny:
        miny=min(yidbatch[i])
    if max(xidbatch[i])>maxx:
        maxx=max(xidbatch[i])
    if min(xidbatch[i])<minx:
        minx=min(xidbatch[i])
    if max(positionbatch[i])>maxpos:
        maxpos=max(positionbatch[i])
    if min(positionbatch[i])<minpos:
        minpos=min(positionbatch[i])


maxtokenlength=2048
#pad and truncate sequence
tokenbatch=tf.keras.preprocessing.sequence.pad_sequences(
    tokenbatch,
    maxlen=maxtokenlength,
    dtype='int64',
    padding='post',
    truncating='post',
    value=0.0
)
xidbatch=tf.keras.preprocessing.sequence.pad_sequences(
    xidbatch,
    maxlen=maxtokenlength,
    dtype='int64',
    padding='post',
    truncating='post',
    value=1549
)
yidbatch=tf.keras.preprocessing.sequence.pad_sequences(
    yidbatch,
    maxlen=maxtokenlength,
    dtype='int64',
    padding='post',
    truncating='post',
    value=2047
)
labelbatch=tf.keras.preprocessing.sequence.pad_sequences(
    labelbatch,
    maxlen=maxtokenlength,
    dtype='int32',
    padding='post',
    truncating='post',
    value=0
)
positionbatch=tf.keras.preprocessing.sequence.pad_sequences(
    positionbatch,
    maxlen=maxtokenlength,
    dtype='int32',
    padding='post',
    truncating='post',
    value=0
)



#i=random(0,len(tokenbatch))
#np.not_equal(label_batch[i],0)
#np.argwhere(condition)




trainratio=0.8
batch_size=2
# Prepare the training dataset.
trainsize=int(trainratio*tokenbatch.shape[0])
train_dataset = tf.data.Dataset.from_tensor_slices((tokenbatch[:trainsize],xidbatch[:trainsize],yidbatch[:trainsize],positionbatch[:trainsize],labelbatch[:trainsize]))
train_dataset = train_dataset.shuffle(buffer_size=1024,seed=0).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((tokenbatch[trainsize:],xidbatch[trainsize:],yidbatch[trainsize:],positionbatch[trainsize:],labelbatch[trainsize:]))
val_dataset = val_dataset.batch(batch_size)








params = {
      # transformer basic configs
      "attention_probs_dropout_prob": 0.01,
      "embedding_dropout":0.01,
      "hidden_act": "relu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 768,
      "max_position_embeddings": 1200,
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "type_vocab_size": 1,
      "use_bias": True,
      "use_token_type":False,
      "scale_emb": False,
      "scope": "bert",
      "use_position_embeddings":True,
      "use_2dposition_embedding":True,
      # sparse mask configs
      "attention_type": "block_sparse",
      "norm_type": "postnorm",
      "block_size": 32,
      "num_rand_blocks": 3,
      # common bert configs
      "word_embedding_path":"G:/bigbird2/bigbird-master/bert_word_embedding.dat",
      "word_embedding_train":False,
      "max_encoder_length": maxtokenlength,
      "max_2dyposition":ymaxlength,
      "max_2dxposition":xmaxlength,
      "max_decoder_length": 64,
      "couple_encoder_decoder": False,
      "beam_size": 5,
      "alpha": 0.7,
      "label_smoothing": 0.1,
      "weight_decay_rate": 0.01,
      "optimizer_beta1": 0.9,
      "optimizer_beta2": 0.999,
      "optimizer_epsilon": 1e-6,
      "num_labels":9,
      "classifier_dropout":0.01,
      "vocab_size":21128,
      
      # TPU settings
      'use_gradient_checkpointing':False,
      "use_tpu": False,
      "tpu_name": None,
      "tpu_zone": None,
      "tpu_job_name": None,
      "gcp_project": None,
      "master": None,
      "num_tpu_cores": 8,
      "iterations_per_loop": "1000",
}


model = BigbirdForTokenClassification(params)

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# Instantiate a loss function.
#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)#这样每个token后面就不用加softmax了
loss_fn = CustomNonPaddingTokenLoss()




class CustomNonPaddingTokenLoss(tf.keras.layers.Layer):
    def __init__(self, name="custom_ner_loss",from_logits=True):
        super().__init__(name=name)
        self.from_logits=from_logits
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=self.from_logits, reduction=tf.keras.losses.Reduction.NONE
        )
    def call(self, loss,mask):

        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)






#x是未聚合的loss
x=tf.constant([[1,0,0,0,0],
               [1,0,0,0,0],
               [1,0,0,0,0],
],dtype=tf.float32)
#x=tf.tile(x,(10,1))
#y是mask
y=tf.constant([[1,1,0,0,0],
               [1,1,0,0,0],
               [1,1,0,0,0]],dtype=tf.float32)
#y=tf.tile(y,[10])
loss =x*y
tf.reduce_sum(loss) / tf.reduce_sum(y)
loss_fn(x,y)




loss_fn(y,x)
lengthlist=[2,3,4,3]
maxtokenlength=5
tf.sequence_mask(lengthlist,maxtokenlength,dtype=tf.float32)







epochs = 3

tvs = model.trainable_weights
#创建变量用于记录每个变量的累积梯度
gradient_accumulation = [tf.Variable(tf.zeros_like(tv),trainable=False) for tv in tvs]
shapelist=[]
for i in tvs:
    shapelist.append(i.shape)



n_gradients =3
n_acum_step =0
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    for step, (input_id,x_id,y_id,positionid,label) in enumerate(train_dataset):
        n_acum_step+=1
        with tf.GradientTape() as tape:
            logits = model(input_ids=input_id,x_2dposition=x_id,y_2dposition=y_id,positionid=positionid,training=True)  # Logits for this minibatch
            loss_value = loss_fn(label,logits)/n_gradients
            print(loss_value)
        gradients = tape.gradient(loss_value, model.trainable_weights)
        for i in range(len(gradient_accumulation)):
            if gradients[i] is None:
                continue
            gradient_accumulation[i].assign_add(gradients[i]) 

        #清零梯度
        if n_acum_step==n_gradients:
            optimizer.apply_gradients(zip(gradient_accumulation, model.trainable_weights))
            zero_gradient = [tv.assign(tf.zeros_like(tv)) for tv in gradient_accumulation]
            n_acum_step=0
            print("完成一次梯度")
            print("当前损失值"+str(loss_value))
        # Log every 200 batches.
        if step % 20 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value)))
            print("Seen so far: %s samples" % ((step + 1) * batch_size))

#只保存/加载模型的权重
model.save_weights('model_weights.h5')
model.load_weights('model_weights.h5')
#保存模型结构和权重
#model.save(filepath)






#模型预测
for step, (input_id,x_id,y_id,positionid,label) in enumerate(val_dataset):
    predict=model(input_id,x_id,y_id,positionid,training=False)
    where_index2 = tf.where(tf.equal(predict, 2))
    label_ids = tf.gather_nd(input_ids,where_index2)
    tokenizer.decode(token_ids=label_ids,skip_special_tokens=True)
    tf.where(predict=1)


#tensorflow提供了tf.gather()和tf.gather_nd()函数。
















