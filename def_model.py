import numpy as np
import tensorflow as tf
from tensorflow import keras


########################################## Building blocks ###############################################
class my_cnn2d():
    def __init__(self,k_size=3,in_chs=1,n_filters=32):
        # Determine initialization parameters
        fan_in=k_size*k_size*in_chs
        fan_out=k_size*k_size*n_filters
        #Xavior initialization
        self.w=tf.Variable(tf.random.normal([k_size, k_size, in_chs,n_filters],stddev=np.sqrt(2/(fan_in+fan_out)) ,dtype=tf.float32))
#         self.w=tf.Variable(tf.random.uniform([k_size, k_size, in_chs,n_filters],minval=-np.sqrt(6/(fan_in+fan_out)),maxval=np.sqrt(6/(fan_in+fan_out)),dtype=tf.float32))
#         self.b=tf.Variable(tf.random.normal([n_filters,],stddev=xavior_std,dtype=tf.float32))
        self.b=tf.Variable(tf.zeros([n_filters,],dtype=tf.float32))
        self.strides=[1,1,1,1]
    def __call__(self,x):
        xx=tf.nn.conv2d(x, self.w, strides=self.strides, padding='VALID')+self.b
        xx=tf.keras.activations.relu(xx)
        y=tf.nn.max_pool2d(xx,ksize=2,strides=2,padding="VALID")
        return y
class my_conv_1x1():
    def __init__(self,in_chs=32,n_filters=32):
        # Determine initialization parameters
        k_size=1
        fan_in=k_size*k_size*in_chs
        fan_out=k_size*k_size*n_filters
        #Xavior initialization
        self.w=tf.Variable(tf.random.normal([k_size, k_size, in_chs,n_filters],stddev=np.sqrt(2/(fan_in+fan_out)) ,dtype=tf.float32))
#         self.w=tf.Variable(tf.random.uniform([k_size, k_size, in_chs,n_filters],minval=-np.sqrt(6/(fan_in+fan_out)),maxval=np.sqrt(6/(fan_in+fan_out)),dtype=tf.float32))
#         self.b=tf.Variable(tf.random.normal([n_filters,],stddev=xavior_std,dtype=tf.float32))
        self.b=tf.Variable(tf.zeros([n_filters,],dtype=tf.float32))
        self.strides=[1,1,1,1]
    def __call__(self,x):
        y=tf.nn.conv2d(x, self.w, strides=self.strides, padding='VALID')+self.b
#         y=tf.keras.activations.relu(y)
        return y    
class my_dense():
    def __init__(self,in_chs,out_chs):
        # Determine initialization parameters
        fan_in=in_chs
        fan_out=out_chs
        #Xavior initialization
        self.w=tf.Variable(tf.random.normal([in_chs,out_chs],stddev=np.sqrt(2/(fan_in+fan_out)) ,dtype=tf.float32))
#         self.w=tf.Variable(tf.random.uniform([in_chs,out_chs],minval=-np.sqrt(6/(fan_in+fan_out)),maxval=np.sqrt(6/(fan_in+fan_out)),dtype=tf.float32))
#         self.b=tf.Variable(tf.random.normal([out_chs,],stddev=np.sqrt(2/(fan_in+fan_out)),dtype=tf.float32))
        self.b=tf.Variable(tf.zeros([out_chs,],dtype=tf.float32))
    def __call__(self,x):
        y=tf.matmul(x,self.w)+self.b
        return y
    
########################################## Model ###############################################
class cnn_model():
    def __init__(self,way,n_filter=32,drop_rate=0.3):
#         super().__init__()
        self.cnn1=my_cnn2d(3,1,n_filter)
        self.cnn1_1=my_conv_1x1(n_filter,n_filter//2)
        self.cnn2=my_cnn2d(3,n_filter//2,n_filter)
        self.cnn2_1=my_conv_1x1(n_filter,n_filter//2)
        self.cnn3=my_cnn2d(3,n_filter//2,n_filter)
        self.cnn3_1=my_conv_1x1(n_filter,n_filter//2)
#         self.dense=my_dense(1600,way)
        self.dense=my_dense(n_filter//2,way)
        self.drop_rate=drop_rate
    def __call__(self,x):
        xx=self.cnn1(x)
        xx=self.cnn1_1(xx)
        xx=self.cnn2(xx)
        xx=self.cnn2_1(xx)
        xx=self.cnn3(xx)
        xx=self.cnn3_1(xx)
        # global average pooling
        xx=tf.nn.avg_pool(xx, ksize=[1]+xx.shape[1:-1]+[1], strides=[1, 1, 1, 1], padding='VALID')
        xx=tf.reshape(xx,[xx.shape[0],xx.shape[-1]])
#         xx=tf.reshape(xx,[xx.shape[0],-1])
        xx=tf.nn.dropout(xx, self.drop_rate)
        y=self.dense(xx)
        return tf.keras.activations.softmax(y)
    def trainable_weights(self):
        wt=[]
        for ii in [self.cnn1,self.cnn1_1,self.cnn2,self.cnn2_1,self.cnn3,self.cnn3_1,self.dense]:
            wt.append(ii.w)
            wt.append(ii.b)
        return wt
    def assign_update(self,tensors):
        nn=0
        for ww in [self.cnn1,self.cnn1_1,self.cnn2,self.cnn2_1,self.cnn3,self.cnn3_1,self.dense]:
            flt_len=np.prod(ww.w.shape)
            ww.w.assign_sub(tf.reshape(tensors[nn:nn+flt_len],ww.w.shape))
            nn+=flt_len
            
            flt_len=np.prod(ww.b.shape)
            ww.b.assign_sub(tf.reshape(tensors[nn:nn+flt_len],ww.b.shape))
            nn+=flt_len
    def apply_gradients(self,grad):
        nn=0
        for ww in [self.cnn1,self.cnn1_1,self.cnn2,self.cnn2_1,self.cnn3,self.cnn3_1,self.dense]:
            ww.w=ww.w+grad[nn]
            ww.b=ww.b+grad[nn+1]
            nn+=2
            
    
    def print_shape(self):
        params=0
        for idx,ii in enumerate([self.cnn1,self.cnn1_1,self.cnn2,self.cnn2_1,self.cnn3,self.cnn3_1,self.dense]):
            params+=np.prod(ii.w.shape)+np.prod(ii.b.shape)
            print(f"layer {idx}: w:{ii.w.shape} + b: {ii.b.shape} = {np.prod(ii.w.shape)+np.prod(ii.b.shape)}")
        print(f"totla {params} parameters")
        
class LSTM_model(keras.Model):
    def __init__(self,n_units):
        super().__init__()
        self.initializer = keras.initializers.lecun_uniform()
        
        self.lstm1 = keras.layers.LSTMCell(n_units,kernel_initializer=self.initializer,use_bias=False)
        self.lstm2 = keras.layers.LSTMCell(n_units,kernel_initializer=self.initializer,use_bias=False)
        self.rnn= keras.layers.RNN([self.lstm1,self.lstm2],return_state=True,stateful=True)
        self.dense=keras.layers.Dense(1,kernel_initializer=self.initializer)
    def forward(self, x,state=None):
        x,*states = self.rnn(x,initial_state=state)
        y = self.dense(x)
        return y,states