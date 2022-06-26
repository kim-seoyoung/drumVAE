# 데이터 전처리
## Groove MIDI Dataset
MIDI파일, wav파일, 각 음원의 meta info를 저장한 csv로 구성되어 있다.

이 데이터셋의 MIDI파일은 한 마디를 16칸으로 나눈 것을 기본 단위로 가지며 이 구간에서 사용된 악기(hit, pitch), 해당 악기를 얼마나 세게 쳤는지(velocity), 기준 시점에서 얼마나 차이나는 시점에서 소리가 시작되었는지(offset)과 같은 정보를 가진다. 

Hit은 자연수이고 드럼 소리가 숫자에 매핑되어있다. 데이터에는 22개의 소리가 다르게 저장되어 있지만, 학습에 사용시에는 9개의 카테고리로 비슷한 소리끼리 모아서 사용한다. 이 데이터셋에서는 드럼 소리만 사용하지만 피아노, 기타와 같은 다른 소리들도 숫자로 매핑되어 다양한 악기가 사용된 음원도 같은 방식으로 자연수로 표현 가능하다.

Offset은 [-0.5, 0. 5)의 값을 가진다. 위에서 말한 단위 구간의 실제 길이는 bpm에 따라 달라지므로 offset은 단위 구간을 1로 보았을 때 가지는 상대적인 시간 위치이다. 음수를 가지면 기준 시점보다 빠르게, 양수를 가지면 기준 시점보다 느리게 소리가 났다는 의미이다.

Velocity는 0~127의 값을 가진다. 얼마나 세게 쳤는지를 나타내는 것으로 소리의 크기에 영향을 주지만 단순히 소리의 크기라고 할 수 는 없고, 부드러운 연주와 같은 보다 추상적인 영향을 포함하고 있다고 한다. 이 값은 사용될 때는 [0, 1]사이의 값으로 변환하여 사용한다.

<br>

## 행렬로 변환해 pkl파일로 저장
Pytorch를 사용하였으므로 tfrecord대신 pickle파일을 사용했다. MIDI파일의 정보를 행렬(numpy)로 표현하고, 4마디의 길이로 잘라 저장했다.

MIDI파일을 Python에서 사용할 수 있게 해주는 다양한 library가 있다.(note_seq, Pretty-midi, mido 등) 참고한 코드에서 pretty-midi를 사용해 이 library를 사용했다. 하나의 단위 구간마다 존재하는 hit값을 일종의 one-hot encoding으로 9개의 채널을 0과 1로 표현한다. 하지만 하나의 단위 구간에 여러개의 악기가 존재할 수 있어 이를 다시 one-hot encoding을 해준다. 예를 들어 [1, 0, 0, 0, 1, 0, 1, 1, 0]과 같은 드럼 연주가 나왔다고 하자. 그려면 이 값을 2진수로 생각하고 다시 10진수로 바꿔준다. 그러면 278이 되고, 길이 512의 encoding된 vector의 278번째 숫자가 1이 되는 것이다. 이렇게 encoding된 벡터가 쌓여 있는 것을 64개(=16/마디 * 4마디)씩 잘라서 저장한다. 이 때 행렬을 그대로 저장하면 0이 매우 많은 상태로 크기가 큰 파일이 생성되므로 sparse library를 이용해 COO(값이 있는 coordinate을 정하는 방식)형식으로 변환해 저장했다.

<br>
<br>

# 학습
## Model - MusicVAE
___
### **VAE (Variational Auto-Encoder)**
  VAE의 목표는 data와 같은 분포에서 sample을 뽑아 새로운 data를 생성해내는 것이다. 크게 Encoder와 Decoder로 구성되어 있다. 

  Encoder는 분포를 만드는 역할을 한다. 어떤 분포를 가정하는지에 따라 모델의 표현이 달라진다. 가우시안 분포를 가정한다면 평균과 분산을 알면 분포를 표현할 수 있으므로 Encoder를 통해 평균과 분산을 구하게 된다. 

   분포를 찾았으면 sampling을 해야 새로운 data를 생성할 수 있다. 따라서 reparameterization trick을 이용해 sampling을 한다. 표준 정규 분포에서 무작위로 값을 뽑고, 분산을 곱하고 평균을 더한 것과 Encoder에서 찾은 정규분포에서 값을 뽑는 것은 동일하다. 따라서 앞에서 설명한 방식으로sampling해 latent vector(일반적으로 z로 표현한다.)를 구한다.

  이렇게 구한 latent vector와 decoder를 이용해 원하는 output을 생성한다. 

### **MusicVAE Structure**
![musicVAE](/img/MusicVAE.png)

위의 그림에 보듯 크게 세 가지 부분으로 나눌 수 있다. Encoder와 decoder의 역할을 VAE에서 설명한 것과 같다. MusicVAE는 여기에 conductor를 추가했다. RNN을 지날 수록 latent state의 영향이 사라져가는 문제가 있는데 이를 방지하기 위한 것이다. Conductor가 추가된 decoder를 hierarchical decoder라 부른다.
#### **Encoder**
Two-layer Bidirectional LSTM을 사용한다.

![LSTM](/img/LSTM.jpg)

  LSTM은 forget, input, output gate로 구성되어 있다. C는 cell state라고 부르며 상태 정보를 나타낸다. 상태 정보는 어떤 정보를 기억해 . Forget gate는 sigmoid를 통해 어떤 정보를 고려하지 않을 것인지 정한다. Input gate는 어떤 정보를 cell state에 저장할 것인지 결정하는 역할을 한다. Output gate는 구해진 cell state를 곱해 새로운 hidden vector를 생성한다.

  LSTM은 지금까지 주어진 것으로 다음을 예측하는 것이므로 이후의 값이 결과에 영향을 주는 경우의 성능을 높이기 위해 Bidirectional LSTM이 등장했다. 이는 LSTM을 정방향과 역방향으로 각각 학습시키는 방법이다. 정방향 layer와 역방향 layer가 따로 존재하며 각 layer의 output을 concat하여 출력한다.
#### **Hierarchical Decoder**
a.	Conductor
생성할 마디의 수만큼 LSTM을 통과한다. 처음에는 encoder에서 나온 latent vector를 통과시키고 이후 나오는 output을 다시 LSTM에 통과시키는 방식이다. LSTM에서 나온 마디 수만큼의 output들은 fully-connected layer와 tanh를 거쳐 docoder에 이용된다.
b.	Decoder
Unidirectional LSTM이용한다. 마디별로 존재하는 conductor output을 매 step마다 이전 시점의 hidden vector와 함께 사용한다. 이는 기존 decoder가 숨겨져 있는 state를 무시하도록 하는 문제점이 있었는데 이를 해결해준다.
<br>

### Drum data
과제를 수행하는 모델은 drum에 해당하는 값만 사용하고 적은 수의 마디를 생성하므로 모델의 크기를 MusicVAE코드에 비해 작게 설정했다.

## Loss
___
MusicVAE는 KL divergence의 최소화를 기반으로 posterior($p(z|x)$)를 찾는다. Encoder의 분포가 $z \sim q_{\lambda}(z|x)$를 따르고, decoder는 $z \sim p(z)$, $x\sim p(x|z)$를 따른다고 하자. 이떄 KL divergence는 아래와 같다.
$$KL(q_{\lambda}(z|x)||p(z|x)) = E[\log q_{\lambda}(z|x)-\log p(z|x)]$$
이 식을 베이즈 정리와 KL divergence는 항상 양수라는 점을 이용해 정리하면 아래와 같다.
$$E[\log p_{\theta}(x|z)]-KL(q_{\lambda}(z|x)||p(x)) \leq \log p(x)$$
이때의 좌변을 ELBO(Evidence Lower BOund)라고 하고, 이것을 gradient(즉 loss)대신 사용하게 된다.  

실제 사용시에는 ELBO의 두 항을 따로 고려하고, KL divergence항은 hyperparameter($\beta$)로 관리된다. Hit를 선택하는 문제는 분류이므로 가능도의 평균인 $E[\log p_{\theta}(x|z)]$항은 cross entropy로 나타낸다. 코드에서는 적용되지 않았지만 offset이나 velocity에 대해서도 구한다면, 회귀문제가 되어 제곱합 오류 함수로 나타내어 진다.


## Optimizer
___
Adam을 사용했다. Learning rate는 1e-3이며, cosine annealing method로 scheduling했다.

<br>
<br>

# 생성
원하는 sequence를 만들어서 넣으면 새로운 데이터를 얻을 수 있다.

# 개선사항
- 필인 데이터 중 실제 악기가 연주된 구간이 4마디 미만인 데이터가 412개가 존재한다(필인만 짧게 연주된 데이터). 빈 마디를 추가하여 이 데이터를 사용할 수 있을 것으로 보인다.
- 

<br><br><br>

참고자료
-	https://github.com/sjhan91/MusicVAE
-	Roberts, A., Engel, J., Raffel, C., Hawthorne, C., and Eck, D. A hierarchical latent vector model for learning longterm structure in music. arXiv preprint arXiv:1803.05428, 2018.
-	Jon Gillick, Adam Roberts, Jesse Engel, Douglas Eck, and David Bamman. Learning to Groove with Inverse Sequence Transformations. arXiv preprint arXiv:1905.06118, 2019.
-	https://magenta.tensorflow.org/groovae
-	https://taeu.github.io/paper/deeplearning-paper-vae/
-	https://towardsdatascience.com/from-a-lstm-cell-to-a-multilayer-lstm-network-with-pytorch-2899eb5696f3


