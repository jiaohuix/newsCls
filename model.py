# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import paddle.nn as nn
import paddle.nn.functional as F

import paddlenlp as ppnlp


class PoolerWithClassifier(nn.Layer):
    def __init__(self, hidden_size,num_classes):
        super().__init__()
        self.dense = nn.Linear(3*hidden_size, hidden_size)
        # self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        # self.activation = nn.ReLU()
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.avg_pool=nn.AdaptiveAvgPool1D(output_size=1)
        self.max_pool=nn.AdaptiveMaxPool1D(output_size=1)

    def forward(self, sequence_output):
        ''' sequence_output: [bsz,seq_len,dim] '''
        pooled_avg=self.avg_pool(sequence_output[:,1:].transpose((0,2,1))).squeeze()  #[bsz dim]
        pooled_max=self.avg_pool(sequence_output[:,1:].transpose((0,2,1))).squeeze()
        pooled_output = paddle.concat([sequence_output[:,0],pooled_avg,pooled_max],axis=1) #[bsz,3dim] concate有助于保存信息，能用拼接就用拼接
        pooled_output=self.activation(self.dense(pooled_output))
        pooled_output=self.classifier(self.dropout(pooled_output))

        return pooled_output


class TextCls(nn.Layer):
    def __init__(self, pretrained_model,num_classes, dropout=None, rdrop_coef=0.0):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # self.classifier = nn.Linear(self.ptm.config["hidden_size"], num_classes)
        self.classifier = PoolerWithClassifier(self.ptm.config["hidden_size"], num_classes)
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                do_evaluate=False):
        # [bsz,seq_len,dim] ** , [bsz,dim]
        sequence_output, cls_embedding1 = self.ptm(input_ids, token_type_ids, position_ids,
                                     attention_mask)
        logits1 = self.classifier(sequence_output)

        # For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
        # Original implementation please refer to this code: https://github.com/dropreg/R-Drop
        if self.rdrop_coef > 0 and not do_evaluate:
            sequence_output2, _ = self.ptm(input_ids, token_type_ids, position_ids,
                                         attention_mask)
            logits2 = self.classifier(self.dropout(sequence_output2))
            kl_loss = self.rdrop_loss(logits1, logits2)
        else:
            kl_loss = 0.0

        return logits1, kl_loss
