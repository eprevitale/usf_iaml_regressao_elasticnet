# Algoritmo de Regressão Elastic Net

Prática Profissional: Inteligência Artificial e Aprendizagem de Máquina

_Por Eduardo Silva, Lucas Ransato e Nathan Borges_

## Introdução

O objetivo deste trabalho é realizar um relatório acompanhado de uma aplicação em Python no Colab, ferramenta que por sua vez realiza treinamentos de modelos de Machine Learning e Deep Learning, bem como processamento e visualização de dados em tempo real, desenvolvimento e testes de códigos Python sem precisar instalar nada localmente, até ensino e aprendizado de programação e ciência de dados.
Entretanto, nossa aplicação será em cima de um algoritmo de aprendizado de máquina supervisionado definido como Regressão Elastic Net, utilizando um Dataset do repositório Kaggle. Ademais, realizaremos transformações de dados, bem como divisão de dados em treino e testes, assim como features e labels. E por fim, iremos escolher e apresentar os resultados das métricas definidas.

## Teoria: Algoritmo Elastic Net

### Técnica de Regularização Elastic Net

A Regularização Elastic Net é uma técnica de aprendizado de máquina que combina as penalizações L1 e L2, provenientes, respectivamente, da Lasso e da Ridge Regression. Essa abordagem é especialmente útil em cenários onde há alta multicolinearidade entre as variáveis independentes, ou seja, quando as variáveis estão altamente correlacionadas entre si. A Elastic Net não apenas ajuda a prevenir o overfitting, mas também realiza a seleção de variáveis, tornando-a uma ferramenta poderosa na análise de dados e modelagem preditiva.

### Como funciona?

A Elastic Net utiliza uma função de custo que inclui dois termos de penalização: um para a soma dos valores absolutos dos coeficientes (L1) e outro para a soma dos quadrados dos coeficientes (L2). A combinação desses dois termos permite que a Elastic Net mantenha a robustez da Ridge, que lida bem com a multicolinearidade, enquanto também realiza a seleção de variáveis como a Lasso. O parâmetro de mistura, que varia entre 0 e 1, controla a proporção entre as duas penalizações, permitindo uma flexibilidade na modelagem.

### Quando utilizar?

A Regularização Elastic Net é particularmente recomendada em situações onde o número de variáveis preditoras é maior que o número de observações, ou quando há uma grande quantidade de variáveis correlacionadas. Em tais casos, a Elastic Net pode fornecer um modelo mais estável e interpretável, ao mesmo tempo em que mantém a precisão preditiva. Além disso, é uma escolha ideal quando se deseja um equilíbrio entre a seleção de variáveis e a regularização, especialmente em conjuntos de dados complexos.

### Considerações finais

A Regularização Elastic Net é uma técnica essencial para analistas de dados e cientistas de dados que buscam construir modelos preditivos robustos e interpretáveis. Sua capacidade de lidar com a multicolinearidade e realizar a seleção de variáveis a torna uma ferramenta valiosa em um arsenal de técnicas de modelagem. Com a crescente complexidade dos conjuntos de dados, a Elastic Net se torna cada vez mais relevante na prática de análise de dados.

---

## Referências

ESTATÍSTICA FÁCIL. O que é: Regularização Elastic Net. Disponível em: https://estatisticafacil.org/glossario/o-que-e-regularizacao-elastic-net/. Acesso em: 17 mar. 2025.

VALLE, R. C. Regressão Linear Regularizada. Disponível em: https://www.ime.unicamp.br/~valle/Teaching/MS571/Aula%2005%20-%20Regressão%20Linear%20Regularizada.pdf. Acesso em: 17 mar. 2025.
