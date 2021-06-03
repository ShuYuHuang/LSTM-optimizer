import numpy as np
from tensorflow.keras import utils

class DataGenerator(utils.Sequence):
    def __init__(self,data,label,way,batch_size=32,shuffle=False):
        super().__init__()
        self.dataset=np.empty((list(data.shape)+[1]),dtype=np.float32)
        self.dataset[...,0]=data
        self.labels=utils.to_categorical(label,way)
        self.batch_size=batch_size
        self.shuffle=shuffle
    def __len__(self):
        return len(self.dataset)//self.batch_size
    def __getitem__(self,index):
        x=self.dataset[index*self.batch_size:(index+1)*self.batch_size]/255.
        y=self.labels[index*self.batch_size:(index+1)*self.batch_size]
        return x,y
    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        if self.shuffle:
            order=np.random.permutation(len(self.dataset))
            self.dataset=self.dataset[order]
            self.labels=self.labels[order]
        for item in (self[i] for i in range(self.__len__())):
            yield item