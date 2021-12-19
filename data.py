# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import numpy as np
from paddlenlp.datasets import MapDataset

class NewsDataset(paddle.io.Dataset):
    def __init__(self,data_path,mode='train'):
        assert mode in ['train','dev','test']
        is_test=True if mode=='test' else False
        self.label_map={lbl:idx for idx,lbl in enumerate(self.label_list)}
        self.samples=self._read_file(data_path,is_test)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

    def _read_file(self,data_path,is_test):
        samples=[]
        with open(data_path,'r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip()
                if is_test:
                    samples.append((line,))
                else:
                    text,label=line.split('\t')
                    label=self.label_map[label]
                    samples.append((text,label))
        return samples

    @property # 注册为属性
    def label_list(self):
        return ['科技', '股票', '体育', '娱乐', '时政', '社会', '教育', '财经', '家居', '游戏', '房产', '时尚', '彩票', '星座']


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    assert mode in ['train', 'dev', 'test']
    if not isinstance(dataset,MapDataset):
        dataset=MapDataset(dataset)
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)



def convert_example(example, tokenizer, max_seq_length=512, is_test=False):

    title= example[0]

    encoded_inputs = tokenizer(
        text=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label=example[1]
        label = np.array(label, dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids


if __name__ == '__main__':
    data=NewsDataset('news/train.txt',mode='train')
    import paddlenlp as ppnlp
    from functools import partial
    from paddlenlp.data import Stack, Tuple, Pad

    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(
        'ernie-1.0')

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=128)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # token idx
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment idx
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

    loader=create_dataloader(data,
                              mode='train',
                              batch_size=2,
                              batchify_fn=batchify_fn,
                              trans_fn=trans_func
                             )
    # for data in loader:
    #     print(data)
    #     break
    print(data.label_list)