import os
def vote(file_ls=[],out_path=''):
    for file in file_ls:
        assert os.path.exists(file),f'file {file} not exists'
    files=[]
    for file in file_ls:
        with open(file,'r',encoding='utf-8') as f:
            lines=f.readlines()
            files.append(lines)
    for data in zip(*files):
        pass

def vote3(file_a,file_b,file_c,out_path=''):
    with open(file_a,'r',encoding='utf-8') as fa,open(file_b,'r',encoding='utf-8') as fb,open(file_c,'r',encoding='utf-8') as fc:
            linesa=fa.readlines()
            linesb=fb.readlines()
            linesc=fc.readlines()
            ensenble_lines=[]
            for linea,lineb,linec in zip(linesa,linesb,linesc):
                vote_tmp={}
                vote_tmp[linea.strip()]=vote_tmp.get(lineb.strip(),0)+0.4
                vote_tmp[lineb.strip()]=vote_tmp.get(lineb.strip(),0)+0.3
                vote_tmp[linec.strip()]=vote_tmp.get(linec.strip(),0)+0.2
                ens_cls=sorted(vote_tmp,reverse=True)[0]
                ensenble_lines.append(ens_cls)
    with open(out_path,'w',encoding='utf-8') as f:
        f.write('\n'.join(ensenble_lines))
# vote3('result_ernie_pool.txt','pool_gate.txt','result_ernie2.txt', out_path='result.txt')

    import torch
    from transformers import BertModel,BertTokenizer,RobertaModel,RobertaTokenizer
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    new_s=torch.load('ROB3/pytorch_model.bin')
    for k,v in new_s.items():
        print(k,v.shape)
    model = RobertaModel.from_pretrained('ROB3')
    print('ueueeueue')
    skip_keys=['cls.predictions.bias','cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.weight', 'cls.predictions.decoder.bias']
    map_key={
        'bert.':'',
        'cls.predictions.transform.':'pooler.',
    }
    need_state=model.state_dict()
    # print(len(need_state),'need')
    import torch
    law_state=torch.load('ROB2/pytorch_model.bin')
    # print(len(law_state),'law_state')

    for lawk,lawv in law_state.items():
        if lawk in skip_keys:
            continue
        # 改名
        for mk,mv in map_key.items():
            new_key=lawk.replace(mk,mv)
            need_state[new_key]=law_state[lawk]
    # torch.save(need_state,'pytorch_model.bin')
    # for k,v in model.state_dict().items():
    #     print(model.state_dict()[k]==law_state[k])
    # print(loadstate==Mstate)
    # model.save_pretrained('ROB')
    # tokenizer = BertTokenizer.from_pretrained("hfl/chinese-macbert-large")
    # model = BertModel.from_pretrained("hfl/chinese-macbert-large")

    print(model)
    # model.save_pretrained('bert')
    import torch
    # torch.save(model.state_dict(),'macbert.pth')
    # tokenizer.save_pretrained('macbert')
    # tokenizer.save_vocabulary('vocab.txt')