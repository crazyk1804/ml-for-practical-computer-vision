# TPU(텐서 처리 유닛) 사용법
 
 ## TPU 가속 노트북
 
 Google의 Vertex AI Platform에서 TPU 가속 노트북을 프로비저닝할 수 있다. 필요한 gcloud 명령어를 [create-tpu-deep-learning-vm.sh](https://raw.githubusercontent.com/GoogleCloudPlatform/training-data-analyst/master/courses/fast-and-lean-data-science/create-tpu-deep-learning-vm.sh) 스크립트에 모아뒀다.
 
 자세한 지침은 아래쪽에 있다.
 
 Cloud AI Platform 노트북은 TPU 및 TPU 포드에서 작동하며, 가장 큰 포드는 2048 코어를 갖춘 TPUv3-2048이다.
  
 또한 [Colaboratory](https://colab.sandbox.google.com/github/GoogleCloudPlatform/training-data-analyst/blob/master/courses/fast-and-lean-data-science/07_Keras_Flowers_TPU_xception_fine_tuned_best.ipynb)(TPU v2-8)와 [Kaggle](https://www.kaggle.com/mgornergoogle/five-flowers-with-keras-and-xception-on-tpu)(TPU v3-8)에서 무료로 TPU를 사용할 수 있다.
 
TPU 기초가 [이곳에 설명돼 있다](https://www.kaggle.com/docs/tpu).

## Cloud TPU 가속기로 노트북을 프로비저닝하는 자세한 지침

[create-tpu-deep-learning-vm.sh](https://raw.githubusercontent.com/GoogleCloudPlatform/training-data-analyst/master/courses/fast-and-lean-data-science/create-tpu-deep-learning-vm.sh) 스크립트를 사용해 TPU를 갖춘 Vertex AI Notebook VM을 한 번에 생성할 수 있다.
이 스크립트는 VM과 TPU에 있는 Tensorflow의 버전이 일치하는지 확인한다. 자세한 단계는 다음과 같다.

 * [Google Cloud 콘솔](https://console.cloud.google.com/)으로 이동해, 결제를 활성화한 새 프로젝트를 만든다.
 * 셸 명령을 입력할 수 있도록 Cloud Shell(오른쪽 상단 >_ 아이콘)을 연다.
 * [create-tpu-deep-learning-vm.sh](https://raw.githubusercontent.com/GoogleCloudPlatform/training-data-analyst/master/courses/fast-and-lean-data-science/create-tpu-deep-learning-vm.sh) 스크립트를 파일에 저장하고, chmod u+x로 실행 퍼미션을 설정한다.
 * `gcloud init`을 실행하여 프로젝트를 설정하고 TPU가 있는 기본 영역을 선택한다. [Google Cloud Console](https://console.cloud.google.com/) Compute Engine > TPU > TPU 노드 만들기에서 영역 및 TPU 유형 필드를 사용하여 여러 영역에서 TPU 가용성을 확인할 수 있다. 이 데모에서는 8코어 TPU 또는 32코어 TPU Pod를 사용할 수 있다. TPU v2와 v3 모두 작동한다.
  영역에 따라 선택할 수 있는 TPU 유형(v3-8, v2-32, v2-8, v3-32)이 다르므로 테스트 목적에 맞게 영역을 고른다.
 * TPU 및 VM 생성 스크립트를 실행한다.<br/>
 `./create-tpu-deep-learning-vm.sh choose-a-name --tpu-type v3-8`
 * Tensorflow 버전을 `--version=2.5.0`로 지정하거나 `--nightly`를 사용할 수 있다. 대부분의 Tensorflow 버전을 TPU에서 사용할 수 있지만, 특정 메이저.마이너 버전이 필요할 수도 있다. 예를 들어 2.3이나 2.4.2는 작동하지만 2.4는 작동하지 않는다.
 * 머신이 기동되면, [Google cloud 콘솔](https://console.cloud.google.com/) Vertex AI > Workbench로 가서 방금 생성한 VM의 OPEN JUPYTERLAB 열기를 클릭한다.
 * Jupyter에서, 터미널을 열고 이 저장소를 복제한다.<br/>
 `git clone https://github.com/ychoi-kr/practical-ml-vision-book-ko.git`

TPU로 훈련할 준비가 됐다. 3장과 4장의 모델들은 모두 TPU 훈련을 지원한다.

[cloud 콘솔](https://console.cloud.google.com/)에서 수동으로 TPU를 프로비전할 수도 있다.
Compute Engine > TPU > TPU 노드 만들기로 이동한다. 버전 선택 상자를 사용해 VM에 있는 것과 같은 버전의 Tensorflow 버전을 골라라.
스크립트는 동일한 작업을 수행하지만 명령줄에서 VM 및 TPU 생성을 위한 두 개의 gcloud 명령어를 사용한다.
VM은 기본적으로 Jupyter 노트북을 지원하고 TPU_NAME 환경 변수가 TPU를 가리키도록 설정되어 있으며, 최첨단 기술이 필요한 경우에는 스크립트를 실행할 때 `--nightly` 매개변수를 추가함으로써 tf-nightly로 업그레이드할 수 있다.
