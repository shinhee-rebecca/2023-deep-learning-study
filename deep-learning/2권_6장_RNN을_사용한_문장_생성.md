# RNN을 사용한 문장 생성

목차

# 요약

- 언어 모델을 사용한 문장 생성
- seq2seq 구조의 신경망

---

# 7.1. 언어 모델을 사용한 문장 생성

## 7.1.1. RNN을 사용한 문장 생성의 순서

- 앞 장에서 구현한 언어 모델의 확률분포 출력 과정
    - 단어 → Embedding → LSTM → Affine → Softmax → 확률분포
- 위와 같은 과정으로 다음 단어를 새롭게 생성하려면 다음과 같은 방법을 떠올릴 수 있음.
    1. 확률이 가장 높은 단어를 선택하는 방법 (결정적,  deterministic). 매 실행마다 결과가 변하지 않음.
    2. 각 후보 단어의 확률에 맞게 선택하는 방법 (확률적, probablistic). 매 실행마다 결과가 달라질 수 있음.
- <eos>와 같은 종결 토큰이 나타날 때까지 다음 단어 생성을 반복하여 새로운 문장을 생성함.

## 7.1.2. 문장 생성 구현

- 앞 장에서 구현한 Rnnlm 클래스 상속

```python
import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm

class RnnlmGen(Rnnlm):
	def generate(self, start_id, skip_ids=None, sample_size=100):
		word_ids = [start_id]
		
		x = start_id
		while len(word_ids) < sample_size: # 샘플링하는 단어
			x = np.arrapy(x).reshape(1, 1) # (batch_size, 1)
			score = self.predict(x) # 다음 단어 후보들의 점수들
			p = softmax(score.flatten()) # softmax를 통해 확률 분포를 구함
			
			sampled = np.random.choice(len(p), size=1, p=p) # p의 확률로 다음 단어 선택
			if (skip_ids is None) or (sampled not in skip_ids): # skip_ids: 샘플링하지 않을 단어 리스트
				x = sampled # x를 다음 단어로 업데이트
				word_ids.append(int(x))
			
		return word_ids # 생성된 단어들의 id
			
```

```python
# 생성 실행
import sys
sys.path.append('..')
from rnnlm_gen import RnnlmGen
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RnnlmGen()
# model.load_params('../ch06/Rnnlm.pkl')

# 시작 토큰, 스킵할 토큰 설정
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]

# 문장 생성
word_ids = model.generate(start_id, skip_ids)
txt = ' '.join(id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print(txt)
```

# 7.2. seq2seq

- seq2seq(sequence to sequence) or Encoder-Decoder model: 시계열 데이터를 또 다른 시계열 데이터로 변환하는 모델

## 7.2.1. seq2seq의 원리

- Encoder는 입력 데이터를 인코딩하고, Decoder는 인코딩된 데이터를 디코딩한다.
- Encoder
    - `나는 고양이이다.` → Encoder → Decoder → `I am a cat`
        
        ![image](https://github.com/shinhee-rebecca/2023-deep-learning-study/assets/42907231/be0f7c16-9379-4809-8f92-5fb94bb23403)
        
    - RNN의 인코딩: 임의의 길이의 문장을 고정된 길이의 벡터로 변환한다.

- Decoder
    
    ![image](https://github.com/shinhee-rebecca/2023-deep-learning-study/assets/42907231/52f5ed08-03eb-43e0-a5ad-6a0f670d804a)
    
    - 인코더-디코더 구조에서는 LSTM 계층이 벡터 h를 입력받는다는 특징이 있음.
- seq2seq
    
    ![fig 7-9.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8f3ba993-80b3-4678-9f84-a8e257120db7/fig_7-9.png)
    
    - 인코더의 LSTM 계층의 은닉 상태가 인코더와 디코더를 이어주는 역할 수행

## 7.2.2. 시계열 데이터 변환용 장난감 문제

- 더하기를 할 줄 아는 모델 학습하기
    
    ![image](https://github.com/shinhee-rebecca/2023-deep-learning-study/assets/42907231/13fb2f3d-add3-4b4d-ab73-1d368b2a9258)

    

## 7.2.3. 가변 길이 시계열 데이터

- 각 샘플마다 길이가 상이함. 따라서 미니배치 처리를 위해, 한 미니배치 내의 샘플들의 데이터 형상이 모두 동일하도록 해주는 ‘패딩(padding)’이 필요함.
    - 패딩: 원래의 데이터에 의미 없는 데이터를 채워 모든 데이터의 길이를 균일하게 맞추는 기법
        
        ![image](https://github.com/shinhee-rebecca/2023-deep-learning-study/assets/42907231/ec441acc-bd84-48c2-829c-d4e6ea33871c)

        
    - 패딩을 통해 가변 길이의 시계열 데이터를 처리할 수 있음.
        - (정확성을 위해 패딩에는 손실의 결과가 반영되지 않도록 ‘마스크’ 기능을 추가하기도 함.)

# 7.3. seq2seq 구현

## 7.3.1. Encoder 클래스

- 문자열을 입력 받아 은닉 상태 벡터 h를 출력
    
    ![image](https://github.com/shinhee-rebecca/2023-deep-learning-study/assets/42907231/db56a84d-473e-4de5-9597-ac8b1dce124b)

    
    ```python
    class Encoder:
    	def __init__(self, vocab_size, wordvec_size, hidden_size):
    		V, D, H = vocab_size, wordvec_size, hidden_size # 어휘 수, 문자벡터의 차원 수, LSTM 은닉 상태 벡터의 차원 수 
    		rn = np.random.randn
    
    		embed_W = (rn(V, D) / 100).astype('f')
    		lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
    		lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
    		lstm_b = np.zeros(4*H).astype('f')
    
    		self.embed = TimeEmbedding(embed_W)
    		self.lstm = TimeLSTM(lstm_W, lstm_Wh, sltm_b, statueful=False) # 은닉 상태를 저장하지 않음.
    		
    		self.params = self.embed.params + self.lstm.params # 가중치 보관
    		self.grads = self.embed.grads + self.lstm.grads # 기울기 보관
    		self.hs = None
    
    	def forward(self, xs):
    		xs = self.embed.forward(xs)
    		hs = self.lstm.forward(xs)
    		self.hs = hs
    		return hs[:, -1, :] # 마지막 은닉 상태만 추출
    
    	def backward(self, dh):
    		dhs = np.zeros_like(self.hs)
    		dhs[:, -1, :] = dh # dh: LSTM 계층의 마지막 은닉상태에 대한 기울기. 즉, 디코더에서 전해지는 기울기
    		
    		dout = self.lstm.backward(dhs) # LSTM의 역전파
    		dout = self.embed.backward(dout) # 임베딩 계층의 역전파
    		return dout
    ```
    
    ## 7.3.2. Decoder 클래스
    
    ![image](https://github.com/shinhee-rebecca/2023-deep-learning-study/assets/42907231/c0c62875-7e27-405e-82b1-36c2b9702f5a)

    
    ![image](https://github.com/shinhee-rebecca/2023-deep-learning-study/assets/42907231/df8649ae-edf5-4f85-bb99-2fcc2755afcc)

    
    ![image](https://github.com/shinhee-rebecca/2023-deep-learning-study/assets/42907231/47ce8b23-34e4-4301-8f63-336a31488026)

    
    ![image](https://github.com/shinhee-rebecca/2023-deep-learning-study/assets/42907231/5b7a2ef4-db8d-4caf-9d31-78bddc306285)

    
    ```python
    class Decoder:
    	def __init__(self, vocab_size, wordvec_size, hidden_size):
    		V, D, H = vocab_size, wordvec_size, hidden_size
    		rn = np.random.randn
    
    		embed_W = (rn(V, D) / 100).astype('f')
    		lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
    		lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
    		lstm_b = np.zeros(4*H).astype('f')
    		affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
    		affine_b = np.zeros(V).astype('f')
    
    		self.embed = TimeEmbedding(embed_W)
    		self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
    		self.affine = TimeAffine(affine-W, affine_b)
    		self.params, self.grads = [], []
    		for layer in (self.embed, self.lstm, self.affine):
    			self.params += layer.params
    			self.grads += layer.grads
    
    	# 학습 시의 순전파
    	def forward(self, xs, h):
    		self.lstm.set_state(h)
    		
    		out = self.embed.forward(xs)
    		out = self.lstm.forward(out)
    		score = self.affine.forward(out)
    		return score # 각 단어의 확률 분포
    	
    	def backward(self, dscore):
    		dout = self.affine.backward(dscore)
    		dout = self.lstm.backward(dout)
    		dout = self.embed.backward(dout)
    		dh = self.lstm.dh
    		return dh
    ```
    
    - 역전파 과정
        - Softmax with Loss 계층으로부터 dscore를 받고, Affine, LSTM, Embedding 계층 순서로 전파한다.
        - `lstm.dh`에는 시간 방향의 기울기가 저장되어 있음. 이를 Decoder 클래스의 역전파의 출력으로 반환함.
        
- 추론용 함수
    
    ```python
    def generate(self, h, start_id, sample_size):
    	sampled = []
    	sample_id = start_id
    	self.lstm.set_state(h)
    	
    	for _ in range(sample_size): # 문장 길이 만큼 반복
    		x = np.array(sample_id).reshape(1, 1)
    		out = self.embed.forward(x)
    		out = self.lstm.forward(out)
    		score = self.affine.forward(out)
    	
    		sample_id = np.argmax(score.flatten())
    		sampled.append(int(sample_id))
    
    	return sampled
    ```
    
    - 문자를 입력 받아, affine 계층이 출력하는 점수가 가장 큰 문자 id를 선택하는 작업을 반복함.

## 7.3.3. Seq2seq 클래스

```python
class Seq2seq(BaseModel):
	def __init__(self, vocab_size, wordvec_size, hisdden_size):
		V, D, H = vocab_size, wordvec_size, hidden_size
		self.encoder = Encoder(V, D, H)
		self.decoder = Decoder(V, D, H)
		self.softmax = TimeSoftmaxWithLoss()
		
		self.params = self.encoder.params + self.decoder.params
		self.grads = self.encoder.grads + self.decoder.grads

	def forward(self, xs, ts):
		decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:] # 마지막 토큰 직전, 다음 토큰부터 쭉
		h = self.encoder.forward(xs)
		score = self.decoder.forward(decoder_xs, h)
		loss = self.softmax.forward(score, decoder_ts)
		return loss

	def backward(self, dout=1):
		dout = self.softmax.backward(dout)
		dh = self.decoder.backward(dout)
		dout = self.encoder.backward(dh)
		return dout

	def generate(self, xs, start_id, sample_size):
		h = self.encoder.forward(xs)
		sampled = self.decoder.generate(h, start_id, sample_size)
		return sampled
```

# 7.4. seq2seq 개선

## 7.4.1. 입력 데이터 반전(Reverse)

- 입력 데이터의 순서를 뒤집는다.
    
    ![image](https://github.com/shinhee-rebecca/2023-deep-learning-study/assets/42907231/698e4b06-140d-491f-bcb3-4b5fd32d1428)

    
    ```python
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
    ```
    

## 7.4.2. 엿보기(Peeky)

- 디코더의 최초 LSTM 계층뿐만 아니라 디코더의 다른 계층들도 인코더의 출력인 벡터 h를 사용하게끔 해주어서 인코더의 정보를 최대한 활용
    
    ![image](https://github.com/shinhee-rebecca/2023-deep-learning-study/assets/42907231/c75bf8a3-2fa6-4114-aa08-8fa423270e39)

    
    - Affine 계층의 입력으로, h와 디코더 LSTM의 출력 벡터가 concatenate되어 들어감.

```python
class PeekyDecoder:
	def __init__(self, vocab_size, wordvec_size, hidden_size):
		V, D, H = vocab_size, wordvec_size, hidden_size
		rn = np.random.randn

		embed_W = (rn(V, D) / 100).astype('f')
		lstm_Wx = (rn(H+D, 4*H) / np.sqrt(D)).astype('f') # 형상 변경. D -> D+H 
		lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
		lstm_b = np.zeros(4*H).astype('f')
		affine_W = (rn(H+H, V) / np.sqrt(H)).astype('f') # 형상 변경. H -> H+H
		affine_b = np.zeros(V).astype('f')

		self.embed = TimeEmbedding(embed_W)
		self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
		self.affine = TimeAffine(affine-W, affine_b)
		self.params, self.grads = [], []
		for layer in (self.embed, self.lstm, self.affine):
			self.params += layer.params
			self.grads += layer.grads

	# 학습 시의 순전파
	def forward(self, xs, h):
		self.lstm.set_state(h)
		
		out = self.embed.forward(xs)
		hs = np.repeat(h, T, axis=0).reshape(N, T, H) # 추가. h를 시계열만큼 복제해 hs에 저장
		out = np.concatenate((hs, out), axis=2) #추가 # hs와 임베딩 계층의 출력을 연결

		out = self.lstm.forward(out)
		out = np.concatenate((hs, out), axis=2) # 추가. affine 계층에 hs와 LSTM 계층의 출력을 연결하여 입력해줌

		score = self.affine.forward(out)
		return score # 각 단어의 확률 분포
	
	def backward(self, dscore):
		dout = self.affine.backward(dscore)
		dout = self.lstm.backward(dout)
		dout = self.embed.backward(dout)
		dh = self.lstm.dh
		return dh
```

```python
from seq2seq import Seq2seq, Encoder
class PeekySeq2seq(Seq2seq):
	def __init__(self, vocab_size, wordvec_size, hidden_size):
		V, D, H = vocab_size, wordvec_size, hidden_size
		self.encoder = Encoder(V, D, H)
		self.decoder = PeekyDecoder(V, D, H) # PeekyDecoder 초기화
		self.softmax = TimeSoftmaxWithLoss()
		
		self.params = self.encoder.params + self.decoder.params
		self.grads = self.encoder.grads + self.decoder.grads
```

# 7.5. seq2seq를 이용하는 애플리케이션

- 챗봇
- 알고리즘 학습
- 이미지 캡셔닝
