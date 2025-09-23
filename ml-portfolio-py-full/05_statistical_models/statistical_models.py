# 간단 HMM(비터비) 데모: 관측열 O에 대해 가장 그럴듯한 상태열 S 추정
import numpy as np
states = ['H','C']  # Healthy, Cold
obs = ['normal','cold','dizzy']

start_p = {'H':0.6,'C':0.4}
trans_p = {'H':{'H':0.7,'C':0.3}, 'C':{'H':0.4,'C':0.6}}
emit_p  = {'H':{'normal':0.5,'cold':0.4,'dizzy':0.1},
           'C':{'normal':0.1,'cold':0.3,'dizzy':0.6}}

def viterbi(obs_seq):
    V = [{}]; path = {}
    # 초기화
    for s in states:
        V[0][s] = np.log(start_p[s]) + np.log(emit_p[s][obs_seq[0]])
        path[s] = [s]
    # 동적 프로그래밍
    for t in range(1, len(obs_seq)):
        V.append({}); new_path = {}
        for s in states:
            (prob, prev) = max((V[t-1][s0] + np.log(trans_p[s0][s]) + np.log(emit_p[s][obs_seq[t]]), s0) for s0 in states)
            V[t][s] = prob
            new_path[s] = path[prev] + [s]
        path = new_path
    # 종료
    n = len(obs_seq)-1
    (prob, state) = max((V[n][s], s) for s in states)
    return prob, path[state]

sequence = ['normal','cold','dizzy']
prob, state_seq = viterbi(sequence)
print("Obs:", sequence)
print("Viterbi best path:", state_seq)
