import torch
import numpy as np
import os

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from transformers.configuration_electra import ElectraConfig
from transformers.modeling_electra import ElectraPreTrainedModel, ElectraModel
from transformers.optimization import AdamW

from tokenization_kocharelectra import KoCharElectraTokenizer

class ElectraMRC(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 분류할 라벨의 개수
        self.num_labels = config.num_labels

        # ELECTRA 모델
        self.electra = ElectraModel(config)

        # Span 범위 예측을 위한 linear
        self.projection_layer = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # input_ids, attention_mask, token_type_ids 형태: [batch, seq_len]
        # electra_outputs 형태: [1, batch, seq_len, hidden]
        electra_outputs = self.electra(input_ids, attention_mask, token_type_ids)

        # hypothesis 형태: [batch, seq_len, 2 (start/end)]
        hypothesis = self.projection_layer(electra_outputs[0])

        # start, end 형태: [batch, seq_len, 1] -> [batch, seq_len]
        p_start, p_end = hypothesis.split(1, dim=-1)
        p_start = p_start.squeeze(-1)
        p_end = p_end.squeeze(-1)

        return p_start, p_end

def convert_data2feature(config, input_sequence, tokenizer):
    # input_sequence : [CLS] Question [SEP] Context [SEP]
    # => [CLS] 세 종 대 왕 은 _ 몇 _ 대 _ 왕 인 가 ? [SEP] 세 종 대 왕 은 _ 조 선 의 _ ~ [SEP]"

    # 고정 길이 벡터 생성
    input_ids = np.zeros(config["max_length"], dtype=np.int)
    attention_mask = np.zeros(config["max_length"], dtype=np.int)
    segment_ids = np.zeros(config["max_length"], dtype=np.int)

    is_context = False
    for idx, token in enumerate(input_sequence.split()):
        input_ids[idx] = tokenizer._convert_token_to_id(token)
        attention_mask[idx] = 1
        if is_context:
            segment_ids[idx] = 1
        if token == '[SEP]':
            is_context = True

    return input_ids, attention_mask, segment_ids

# 데이터 읽기 함수
def read_data(file_path, tokenizer):
    with open(file_path, "r", encoding="utf8") as inFile:
        lines = inFile.readlines()

    # 데이터를 저장하기 위한 리스트 생성
    all_input_ids, all_attention_mask, all_segment_ids, start_indexes, end_indexes = [], [], [], [], []
    for idx, line in enumerate(lines):
        input_sequence, start_idx, end_idx = line.strip().split("\t")
        input_ids, attention_mask, segment_ids = convert_data2feature(config, input_sequence, tokenizer)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_segment_ids.append(segment_ids)
        start_indexes.append(int(start_idx))
        end_indexes.append(int(end_idx))

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
    start_indexes = torch.tensor(start_indexes, dtype=torch.long)
    end_indexes = torch.tensor(end_indexes, dtype=torch.long)

    return all_input_ids, all_attention_mask, all_segment_ids, start_indexes, end_indexes


def tensor2list(input_tensor):
    return input_tensor.cpu().detach().numpy().tolist()

def do_test(model, tokenizer):
    # 평가 모드 셋팅
    model.eval()

    # 평가 데이터 Load
    all_input_ids, all_attention_mask, all_segment_ids, start_indexes, end_indexes = \
        read_data(tokenizer=tokenizer, file_path=config["test_data_path"])

    # TensorDataset/DataLoader를 통해 배치(batch) 단위로 데이터를 나누고 셔플(shuffle)
    test_features = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, start_indexes, end_indexes)
    test_dataloader = DataLoader(test_features, shuffle = True, batch_size=1)

    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, attention_mask, segment_ids, a_start, a_end = batch

        # 입력 데이터에 대한 출력과 loss 생성
        # p_start, p_end 형태:[1, seq_len]
        p_start, p_end = model(input_ids, attention_mask, segment_ids)

        p_start = p_start.argmax(dim=-1)
        p_start = tensor2list(p_start)[0]
        p_end = p_end.argmax(dim=-1)
        p_end_ = tensor2list(p_end)[0]

        a_start = tensor2list(a_start)[0]
        a_end = tensor2list(a_end)[0]

        # 입력 Text 생성
        input_token_ids = tensor2list(input_ids)[0]
        input_tokens = [tokenizer._convert_id_to_token(e) for e in input_token_ids]

        # 입력 Text에서 예측/정답 Span 추출
        predict_span = input_tokens[p_start:p_end+1]
        answer_span = input_tokens[a_start:a_end+1]

        # 입력 Seqquence의 질문, 단락 위치 저장
        segment_positions = [position for position, token in enumerate(input_tokens) if token == "[SEP]"]

        # 모델 예측 확인
        if step < 5:
            question = ''.join(input_tokens[1:segment_positions[0]]).replace("_", " ")
            context = ''.join(input_tokens[segment_positions[0] + 1:segment_positions[1]]).replace("_", " ")
            print("\n\n######################################")
            print("Context : ", context)
            print("Question : ", question)
            print("Answer Span : ", ''.join(predict_span).replace("_", " "))
            print("Predict Span : ", ''.join(answer_span).replace("_", " "))
        else:
            break

def test(config):
    # electra config 객체 생성
    electra_config = ElectraConfig.from_pretrained(
        os.path.join(config["output_dir"], "checkpoint-{0:d}".format(config["checkpoint"])),
        num_labels=config["num_labels"])

    # electra tokenizer 객체 생성
    electra_tokenizer = KoCharElectraTokenizer.from_pretrained(
        os.path.join(config["output_dir"], "checkpoint-{0:d}".format(config["checkpoint"])),
        do_lower_case=False)

    # electra model 객체 생성
    model = ElectraMRC.from_pretrained(
        os.path.join(config["output_dir"], "checkpoint-{0:d}".format(config["checkpoint"])),
        config=electra_config).cuda()

    do_test(model=model, tokenizer=electra_tokenizer)


def train(config):
    # electra config 객체 생성
    electra_config = ElectraConfig.from_pretrained("monologg/kocharelectra-base-discriminator",
                                                   num_labels=config["num_labels"])
    # electra tokenizer 객체 생성
    electra_tokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-discriminator",
                                                               do_lower_case=False)
    # electra model 객체 생성
    model = ElectraMRC.from_pretrained("monologg/kocharelectra-base-discriminator",
                                                                     config=electra_config).cuda()
    # 데이터 읽기
    all_input_ids, all_attention_mask, all_token_type_ids, start_indexes, end_indexes = \
        read_data(tokenizer=electra_tokenizer, file_path=config["train_data_path"])

    # TensorDataset/DataLoader를 통해 배치(batch) 단위로 데이터를 나누고 셔플(shuffle)
    train_features = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, start_indexes, end_indexes)
    train_dataloader = DataLoader(train_features, batch_size=config["batch_size"])

    # 크로스 엔트로피 손실 함수
    loss_func = nn.CrossEntropyLoss()

    # 옵티마이저 함수 지정
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["epoch"]):
        for step, batch in enumerate(train_dataloader):
            # 학습 모드 셋팅
            model.train()
            # batch = (input_ids[step], attention_mask[step], segment_ids[step],
            #                                    start_index[step], end_index[step])*batch_size
            # .cuda()를 통해 메모리에 업로드
            batch = tuple(t.cuda() for t in batch)

            # 변화도를 0으로 변경
            optimizer.zero_grad()

            # p_start, p_end 형식: [batch, seq_len]
            # a_start, a_end 형식: [batch]
            input_ids, attention_mask, segment_ids, a_start, a_end = batch
            p_start, p_end = model(input_ids, attention_mask, segment_ids)

            start_loss = loss_func(p_start, a_start)
            end_loss = loss_func(p_end, a_end)

            total_loss = start_loss + end_loss

            # 손실 역전파 수행
            total_loss.backward()
            optimizer.step()

            # 50 batch_step 마다 Loss 출력
            if (step + 1) % 50 == 0:
                print("Current Step : {0:d} / {1:d}\tCurrent Loss : {2:.4f}".format(step+1, int(len(all_input_ids) / config['batch_size']), total_loss.item()))

            # 500 batch_step 마다 결과 출력
            if (step + 1) % 500 == 0:
                do_test(model, electra_tokenizer)

        # 에폭마다 가중치 저장
        output_dir = os.path.join(config["output_dir"], "checkpoint-{}".format(epoch+1))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        electra_config.save_pretrained(output_dir)
        electra_tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        
if __name__ == "__main__":
    root_dir = "c:/Users/82108/Desktop/스터디 폴더/Transfer learn/mrc"
    output_dir = os.path.join(root_dir, "output")

    if (not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    config = {"mode": "test",
              "train_data_path": os.path.join(root_dir, "mrc_train.txt"),
              "test_data_path": os.path.join(root_dir, "mrc_dev.txt"),
              "output_dir": output_dir,
              "checkpoint": 3,
              "epoch": 3,
              "learning_rate": 5e-5,
              "batch_size": 16,
              "max_length": 512,
              "num_labels": 2,
              }

    if (config["mode"] == "train"):
        train(config)
    else:
        test(config)
