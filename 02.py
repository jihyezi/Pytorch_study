import torch
import torch.nn as nn
import torch.optim as optim

# 데이터 준비
w_true = 2.0
b_true = 1.0

X = torch.randn(100) * 10

# y = 2x + 1 + 노이즈
Y = w_true * X + b_true + torch.randn(100) * 2

# 모델 설계
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x.unsqueeze(-1)).squeeze(-1)

# 모델 객체 생성     
model = LinearRegression()

# 손실 함수 정의
loss_fn = nn.MSELoss()

# 옵티마이저 정의
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습 루프
num_epochs = 1000

for epoch in range(num_epochs):
    # 순전파: 예측값 계산
    y_pred = model(X)

    # 손실 계산
    loss = loss_fn(y_pred, Y)

    # 역전파: 기울기 계산
    optimizer.zero_grad()
    loss.backward()

    # 파라미터 업데이트
    optimizer.step()

    # 100 에포크마다 손실 출력
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n--- 최종 파라미터 ---")
print(f"학습된 w: {model.w.item():.4f}")
print(f"학습된 b: {model.b.item():.4f}")
print("---------------------")
print(f"실제 w: {w_true}")
print(f"실제 b: {b_true}")
