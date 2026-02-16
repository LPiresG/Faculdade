# Scale Dot-Product Attention

Este projeto implementa manualmente o mecanismo de **Scaled Dot-Product Attention**.
A implementação foi feita usando a biblioteca NumPy.

Esse trabalho coloca em codigo a formula da atenção:
Attention(Q, K, V) = softmax((Q · Kᵀ) / √dk) · V

Onde:
Q = Query
K = Key
V =Value
dk = dimenção das chaves

## Normalização (√dk):

Primeiro é calculado o produto escalar de Q pala a matriz transposta de K, em seguida divide o resultado pela raiz quadrada de dk, onde dk é a dimensão dos vetores de chave (Key).
Quando a dimensão 𝑑k é grande, os valores do produto escalar tendem a crescer muito.
Isso pode fazer com que o softmax gere valores extremamente altos ou muito próximos de zero.

Em resumo a normalização por √𝑑k foi aplicada dividindo o produto escalar Q · Kᵀ por √dk para evitar valores muito grandes e garantir estabilidade numérica no softmax.

## Exemplo uso e de input e o output esperado:

import numpy as np

Q = np.array([[1, 0], [0, 1]])
K = np.array([[1, 0], [0, 1]])
V = np.array([[1, 2], [3, 4]])

result = ScaleDotProductAttention()
attention, softmax_result = result.attentionFormula(Q, K, V)

print("Saída:", attention)
print("Softmax:", softmax_result)

# Output:

Saída:
[[1.6604769  2.6604769 ]
 [2.3395231  3.3395231 ]]

Resultado do softmax:
[[0.66976155 0.33023845]
 [0.33023845 0.66976155]]
