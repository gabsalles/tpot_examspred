## Testes de performance com o TPOT


O *TPOT* é uma biblioteca de AUTOML baseada principalmente no scikit-learn. Ele é treinado utilizando um algoritmo de programação genética, o que garante que diversos modelos com diversos parâmetros diferentes sejam treinados por diversas gerações até encontrar o modelo ótimo, isso é, aquele que seja menos complexo e possua a maior métrica de avaliação. Sua proposta é para que ele seja uma espécie de *assistente do cientista de dados*, automatizando processos repetitivos para que a performance analítica do profissional seja melhor aproveitada.
Um artigo de 2016 sobre como essa biblioteca funciona e sua aplicação na área da biomedicina pode ser encontrado nesse [link](https://link.springer.com/chapter/10.1007/978-3-319-31204-0_9).

Para essa análise foi utilizado um dataset disponibilizado no kaggle: [Dataset](https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics?resource=download)

O dataset nesse teste em questão tem o objetivo de prever a pontuação de alunos em determinadas provas a partir de seus atributos demográficos como genêro, educação dos pais, etc.
Como se trata apenas de um teste, as previsões serão apenas das pontuações no teste de matemática, nomeada na tabela como `math score`

Os atributos presentes no dataset são:
`gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course`

Para fins de comparação além do modelo generado pelo *TPOT*, 
também foi treinado um modelo utilizando `SGDRegressor`

![tpot logo](/tpot_logo.jpg)