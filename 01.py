import torch

#스칼라 텐서 만들기
x = torch.tensor(5)
print(x)
print(x.dim()) # 텐서 차원
print(x.dtype) # 텐서의 자료형 (torch.float32)
print(x.shape) # 텐서의 크기 (torch.Size([2, 2]))
print(x.device) # 텐서가 위치한 장치 (CPU 또는 GPU)

# 벡터 만들기
x = torch.tensor([1,2,3])
print(x)
print(x.dim())
print(x.dtype) 
print(x.shape) 
print(x.device) 

# 행렬 만들기
x = torch.tensor([[1,2], [3,4]])
print(x)
print(x.dim())
print(x.dtype) 
print(x.shape) 
print(x.device) 

# 모든 원소가 0인 텐서
zeros_tensor = torch.zeros(2,3)
print(zeros_tensor)

#모든 원소가 1인 텐서
ones_tensor = torch.ones(3,2)
print(ones_tensor)

# 랜덤 텐서
random_tensor = torch.rand(2,2)
print(random_tensor)

x = torch.tensor([[1,2], [3,4]])
y = torch.tensor([[5,6], [7,8]])

print(x+y)
print(torch.add(x,y))

print(x*y)
print(torch.mul(x,y))

# 행렬 곱셈
C = torch.matmul(x, y)
print(C)

D = x @ y
print(D)

# 브로드캐스팅 예시: x의 모든 원소에 10을 더하기
scalar = 10
result = x + scalar
print(result)

x = torch.tensor([[10, 20, 30],
                  [40, 50, 60],
                  [70, 80, 90]])

# 특정 원소 접근
print(x[1,2])

# 슬라이싱
print(x[0, :])

# 여러 행, 여러 열 가져오기
print(x[0:2, 1:])

# 실습 문제
# 1
x = torch.full((2, 4), 7)
print(x)

# 2
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

sum = torch.add(a, b)
print(sum)

# 3
x = torch.tensor([[1, 2, 3], [4, 5, 6], [70, 80, 90]])

y = torch.tensor([x[1,1], x[2,1]])
print(y)

# 4
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[1, 0], [0, 1]])

C = torch.matmul(A,B)
print(C)