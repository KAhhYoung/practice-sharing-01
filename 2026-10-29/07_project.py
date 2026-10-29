"""실습 07~08: 앞 실습을 한 파일로 연결하고 환경·학습·평가 결과를 저장합니다."""

from pathlib import Path
import csv
import json
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


'''CPU의 Python 실행 흐름과 GPU의 텐서 연산

    실행 흐름은 처음부터 계속 CPU에서 진행됩니다. GPU로 갔다가 CPU로 돌아오는 구조가 아닙니다.
    x = x.to("cuda")    # CPU가 GPU로 데이터 전송을 요청
    y = x * 2           # CPU가 GPU에 텐서 연산을 요청
    print("다음 작업")   # CPU에서 실행
    GPU는 요청받은 텐서 연산을 수행하고, CPU는 Python 코드를 계속 실행합니다.
    GPU 연산이 끝나도 x, y는 GPU에 그대로 남습니다.

    또한 CPU는 GPU 연산이 끝나기 전에 다음 코드를 실행할 수도 있습니다. 결과가 필요한 .item()이나 .cpu() 같은 지점에서 완료를 기다립니다.
'''



# 결과 저장용 outputs 폴더 생성 (이미 있으면 그대로 사용)
base = Path(__file__).resolve().parent
out = base / "outputs"
out.mkdir(exist_ok=True)

# CSV의 각 데이터 행을 딕셔너리로 읽어 리스트에 저장
with (base / "data" / "points.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

# split 값에 따라 학습용(train) 행과 검증용(valid) 행 분리
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]

# x1, x2를 실수로 변환해 입력 텐서 생성: shape=(샘플 수, 2)
X_train = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in train_rows])
X_valid = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in valid_rows])

# label을 정수 클래스 인덱스로 변환해 정답 텐서 생성: shape=(샘플 수,)
# csv.DictReader는 값을 문자열로 읽으므로 int() 로 변환 필요
y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.long)
y_valid = torch.tensor([int(r["label"]) for r in valid_rows], dtype=torch.long)

# test 40건은 모델에 전달하지 않습니다. 순서를 바꾸거나 다시 분할하지 않습니다.
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torch.Generator().manual_seed(43): DataLoader가 사용할 난수 생성기를 만들고, 시드(seed)를 43으로 설정
# shuffle=True일 때 데이터를 섞는 순서를 실행마다 재현할 수 있게 합니다.
# 매 epoch마다 동일한 순서로 섞인다는 뜻은 아닙니다. 생성기의 상태가 계속 바뀌므로 epoch별 순서는 달라지지만, 프로그램을 같은 조건으로 다시 실행하면 그 순서들이 재현됩니다.
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                    generator=torch.Generator().manual_seed(43), num_workers=0)

# DataLoader 자체는 GPU로 옮기지 않습니다.
# DataLoader는 데이터를 배치로 묶어 전달하는 객체이고, GPU로 옮기는 대상은 모델과 텐서입니다.
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 전체 학습/ 전체 검증 데이터를 평가하기 위해 모델과 같은 장치로 이동
# 아래의 xb, yb 이동은 배치별 학습을 위한 것임
Xt, yt = X_train.to(device), y_train.to(device)
Xv, yv = X_valid.to(device), y_valid.to(device)
history = []

# [주의] epoch 0에서 초기 상태 평가 후, 80 epoch 학습
print("epoch | 학습데이터 손실 | 검증데이터 손실 | 검증 정확도")
for epoch in range(81):
    if epoch > 0:
        model.train()

        # 학습 데이터를 미니배치 단위로 받아 모델 학습
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            
            '''TODO A07: 
                각 batch에서 backward와 step을 연결하세요.
                train 자료만 loader에 들어갑니다. valid로 backward하거나 학습하지 마세요.
            '''
            # 학습 데이터의 손실로 기울기를 계산하고 파라미터 갱신
            # 검증 데이터는 평가에만 사용
            loss.backward()
            optimizer.step()

    # epoch 0 및 이후 10 epoch마다 평가 수행
    # epoch=0은 학습 전입니다. 로그는 갱신이 끝난 동일 모델로 측정합니다.
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(Xt), yt).item()
            logits = model(Xv)
            valid_loss = loss_fn(logits, yv).item()
            accuracy = (logits.argmax(dim=1) == yv).float().mean().item()
        history.append([epoch, train_loss, valid_loss, accuracy])
        print(f"{epoch} | {round(train_loss, 6)} | {round(valid_loss, 6)} | {round(accuracy, 4)}")

with (out / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "valid_loss", "valid_accuracy"])
    writer.writerows(history)

model.eval()
with torch.no_grad():
    logits = model(Xv)
    '''TODO A08:
        logits의 클래스 축에서 가장 큰 점수의 인덱스를 구하세요.
        CSV로 저장하기 전에 CPU로 옮깁니다. y_valid의 순서는 그대로 유지하세요.
    '''
    predictions = logits.argmax(dim=1).to("cpu")

metrics = {"device": str(device), "torch": str(torch.__version__), "epochs": 80,
           "train_rows": len(train_rows), "valid_rows": len(valid_rows), "test_used": False,
           "initial_train_loss": history[0][1], "final_train_loss": history[-1][1],
           "final_valid_loss": history[-1][2], "final_valid_accuracy": history[-1][3]}


# 학습·평가 지표(metrics)를 JSON 문자열로 변환하여 UTF-8 파일로 저장
(out / "final_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

# 검증 예측 결과를 저장할 CSV 파일을 열고 열 이름(id, 정답, 예측값) 작성
with (out / "validation_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f);
    writer.writerow(["id", "actual", "predicted"])

    # 참고) 텐서도 csv.writer에 전달할 수 있지만, 텐서의 문자열 표현이 저장  / ex) writer.writerow([y])       # tensor(1)
    #      따라서 int(y), y.item() 으로 변환하여 전달
    # writer.writerow([1, 0, 1])  # 가능: 리스트 하나 전달
    # writer.writerow((1, 0, 1))  # 가능: 튜플 하나 전달
    # writer.writerow(1, 0, 1)    # 오류: 인자 3개 전달
    writer.writerows((r["id"], int(y), int(p)) for r, y, p in zip(valid_rows, y_valid, predictions))

# PyTorch 버전, CUDA 빌드 버전, CUDA 사용 가능 여부와 선택한 연산 장치를 텍스트 파일로 저장
# torch.version.cuda: 현재 설치된 PyTorch가 빌드될 때 사용한 CUDA 버전
(out / "gpu_check.txt").write_text(
    f"torch={torch.__version__}\ncuda_build={torch.version.cuda}\n"
    f"cuda_available={torch.cuda.is_available()}\nselected_device={device}\n", encoding="utf-8")
print("생성: outputs의 환경·학습·검증 결과")


# >>> A10
'''TODO A10: 실제 오답 행만 ID와 함께 수집하세요.
    추가 A10: valid의 실제값과 예측을 비교해 오답 목록을 만드세요.
    입력: valid_rows, y_valid, predictions는 동일한 40개 ID 순서입니다.
    반환 형식: errors=[(요청ID, 실제정답int, 예측int), ...].
    조건: 실제값과 예측이 다른 행만 포함하고 원래 valid 순서를 유지합니다.
    금지: test 자료 사용, 정답 변경, 오답이 없을 때 가상의 오답 추가.
    완료: 오류 행 수 = 40 - 맞힌 개수, 저장 파일은 헤더를 포함합니다.
''' 
errors = [(r["id"], int(y), int(p)) for r, y, p in zip(valid_rows, y_valid, predictions) if y != p]           # type: torch.int64(long)


# 검증 데이터의 오답 목록을 CSV로 저장 (오답이 없으면 열 이름만 저장)
with (out / "validation_errors.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "actual", "predicted"])
    writer.writerows(errors)
print("valid 오답 수:", len(errors))
