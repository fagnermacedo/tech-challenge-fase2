# Tech Challenge — Fase 2 | POSTECH Data Analytics

---

## 1. Identificação

| Campo | Valor |
|---|---|
| Turma | 2DTATBB |
| Grupo | Não se Aplica |
| Data de entrega | A confirmar antes da submissão |

### Integrantes

| Nome completo | RM | E-mail |
|---|---|---|
| FAGNER DO ESPÍRITO SANTO SÁ | RM377821 | fagner.sa@bb.com.br |
| MARIA APARECIDA BANDEIRA DA ROCHA| RM377784 | cidarocha97@bb.com.br|
| KATIA DA SILVA SARMENTO| RM377748 |katia.sarmento@bb.com.br |
| MARCIA REGINA CORREA DA SILVA | RM377782 | marcia.rossa@bb.com.br|
| MARILENE RODRIGUES QUINTINO| RM378099 | marilene_quintino@yahoo.com.br |

---

## 2. Links da entrega

Estes três links são **obrigatórios** e devem ser idênticos aos do PDF de submissão.

| Item | Link |
|---|---|
| Repositório | https://github.com/fagnermacedo/tech-challenge-fase2 |
| Vídeo executivo (≤ 5 min) | https://github.com/fagnermacedo/tech-challenge-fase2/blob/main/docs/Apresentacao_Tech_Challenge_09_10_2026.mp4 |
| Apresentação | Pendente de publicação em https://github.com/fagnermacedo/tech-challenge-fase2/blob/main/docs/apresentacao_executiva.pdf` |

> ⚠️ Um repositório privado ou inacessível pode comprometer a avaliação da
> Dimensão 1. Antes de enviarmos a entrega, precisamos confirmar o acesso em
> uma janela anônima.

---

## 3. O problema

Neste projeto, analisamos o risco de inadimplência na concessão de crédito. Nosso
objetivo é avaliar como técnicas de Machine Learning podem apoiar a identificação
de clientes com maior risco, utilizando os dados cadastrais e o histórico de
pagamentos disponibilizados no Kaggle.

### Variável alvo

A variável `TARGET` não vem informada nos arquivos originais; nós a construímos a
partir do histórico mensal de pagamentos (`credit_record.csv`). Para cada cliente,
definimos `TARGET = 1` se encontramos pelo menos um registro com `STATUS` igual
a `2`, `3`, `4` ou `5`; caso contrário, definimos `TARGET = 0`.

Os Status são:

* `X` -> Não possui empréstimo no mês.
* `C` -> Empréstimo quitado no mês.
* `0` -> Atraso entre 1 a 29 dias.
* `1` -> Atraso entre 30 e 59 dias.
* `2` -> Atraso entre 60 e 89 dias.
* `3` -> Atraso entre 90 e 119 dias.
* `4` -> Atraso entre 120 e 149 dias.
* `5` -> Atraso superior a 150 dias ou dívida crítica.

* **Definição:** Classificamos os status `2` a `5` como inadimplência (`TARGET = 1`), pois representam atrasos a partir de 60 dias segundo as categorias da base. Classificamos os status `X`, `C`, `0` e `1` como `TARGET = 0`.
* **Contexto e limitação:** Escolhemos esse limiar como regra de negócio acadêmica, considerando a gravidade dos atrasos. Usamos a Resolução CMN nº 2.682/1999 apenas como referência histórica — ela foi substituída, a partir de 1º de janeiro de 2025, pela Resolução CMN nº 4.966/2021. Também observamos que a faixa `STATUS = 2` começa em 60 dias, enquanto a classificação histórica de risco D citada na Resolução 2.682 começava em 61 dias. Portanto, não reproduzimos exatamente o critério regulatório nem afirmamos conformidade regulatória.
* **Horizonte do rótulo:** Construímos o target com base no pior status observado em qualquer mês disponível. Assim, identificamos inadimplência grave em algum momento do histórico, mas não estimamos diretamente a probabilidade de default nos próximos 6 ou 12 meses. Como o período observado varia entre clientes, reconhecemos a possibilidade de viés de exposição.

### Dataset

| Campo | Valor |
|---|---|
| Fonte | [Kaggle - Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction/data) |
| Linhas × colunas | Aplicações: 438.557 × 18 / Histórico de Crédito: 1.048.575 × 3 |
| Período / versão | Versão 2 (Kaggle) |
| Licença de uso | CC0: Public Domain |

Descrição do histórico financeiro (`credit_record.csv`):

| Variável | Tipo | Descrição |
|---|---|---|
| `ID` | Numérico | Número de identificação único do cliente (chave de cruzamento) |
| `MONTHS_BALANCE` | Numérico | Mês do registro (0 é o mês atual, -1 é o mês anterior, -2 é dois meses atrás, etc.) |
| `STATUS` | Categórico | Status do pagamento: `C` (pago), `X` (sem empréstimo no mês), `0` (atraso 1-29 dias), `1` (atraso 30-59 dias), `2` (atraso 60-89 dias), `3` (atraso 90-119 dias), `4` (atraso 120-149 dias), `5` (atraso > 150 dias ou dívida baixada). |

Descrição das variáveis base (`application_record.csv`):

| Variável | Tipo | Descrição |
|---|---|---|
| `ID` | Numérico | Número de identificação único do cliente |
| `CODE_GENDER` | Categórico | Gênero do solicitante (M/F) |
| `FLAG_OWN_CAR` | Categórico | Indica posse de veículo (Y/N) |
| `FLAG_OWN_REALTY` | Categórico | Indica posse de propriedade/imóvel (Y/N) |
| `CNT_CHILDREN` | Numérico | Quantidade de filhos |
| `AMT_INCOME_TOTAL` | Numérico | Renda anual total |
| `NAME_INCOME_TYPE` | Categórico | Categoria de renda (Trabalhador, Servidor Público, Pensionista, etc.) |
| `NAME_EDUCATION_TYPE` | Categórico | Nível de escolaridade |
| `NAME_FAMILY_STATUS` | Categórico | Estado civil |
| `NAME_HOUSING_TYPE` | Categórico | Tipo de moradia (Própria, Alugada, etc.) |
| `DAYS_BIRTH` | Numérico | Idade em dias (contagem regressiva a partir do dia atual, em valores negativos) |
| `DAYS_EMPLOYED` | Numérico | Tempo de emprego atual em dias (valores negativos, ou positivos para desempregados/aposentados) |
| `FLAG_MOBIL` | Numérico | Possui celular (1 = Sim, 0 = Não) |
| `FLAG_WORK_PHONE` | Numérico | Possui telefone de trabalho (1 = Sim, 0 = Não) |
| `FLAG_PHONE` | Numérico | Possui telefone fixo (1 = Sim, 0 = Não) |
| `FLAG_EMAIL` | Numérico | Possui e-mail (1 = Sim, 0 = Não) |
| `OCCUPATION_TYPE` | Categórico | Área de atuação profissional do cliente |
| `CNT_FAM_MEMBERS` | Numérico | Tamanho do grupo familiar |

---

## 4. Como reproduzir

```bash
git clone https://github.com/fagnermacedo/tech-challenge-fase2.git
cd tech-challenge-fase2

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
jupyter notebook
```

Baixamos os arquivos originais do dataset e os colocamos em `data/raw/` (os dados
**não** são versionados; consultamos `data/README.md` para obter as instruções).

Em seguida, executamos os notebooks nesta ordem:

| # | Notebook | O que faz |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | Análise exploratória |
| 2 | `notebooks/02_preprocessamento.ipynb` | Limpeza, escala e feature engineering |
| 3 | `notebooks/03_modelagem.ipynb` | Treino e comparação dos modelos |
| 4 | `notebooks/04_avaliacao.ipynb` | Métricas, importância de variáveis e conclusões |

**Semente fixa:** usamos `random_state = 42` nos pontos aleatórios documentados
dos notebooks de modelagem. Ao executarmos as etapas em ordem, com os arquivos
brutos e as dependências instalados, geramos os artefatos intermediários
necessários à avaliação. Os resultados podem variar se utilizarmos versões
diferentes das bibliotecas.

---

## 5. Resultados

| Modelo | Acurácia | Precisão | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Regressão Logística (baseline) | 0,61 | 0,02 | 0,46 | 0,04 | 0,5608 |
| Random Forest | 0,9606 | 0,1564 | 0,3027 | 0,2063 | 0,6681 |

Na tabela, apresentamos precisão, recall e F1 da classe `1` (inadimplência),
arredondados conforme a saída salva no Notebook 03. Também calculamos PR-AUC
de **0,1045** para o Random Forest no conjunto de teste. Como não calculamos essa
métrica para a Regressão Logística na saída correspondente, não a comparamos aqui.

**Modelo que selecionamos para a avaliação final:** Random Forest. No conjunto de
teste salvo, observamos F1 da classe 1 e ROC-AUC superiores aos da Regressão
Logística. Também identificamos um trade-off: a Regressão Logística apresentou
recall maior, mas precisão muito baixa; o Random Forest reduziu os falsos positivos,
mas também apresentou recall menor.

**Métricas que priorizamos:** recall e precisão da classe `1`, F1 e ROC-AUC, além
da matriz de confusão. Usamos o recall para medir a parcela de inadimplentes
identificados e a precisão para verificar quantos dos clientes sinalizados
realmente pertencem à classe inadimplente. Como observamos forte desbalanceamento
(1,69% de inadimplência no conjunto usado), não consideramos a acurácia isolada
suficiente. Também reconhecemos que o limiar de classificação afeta esse equilíbrio
e precisa ser avaliado antes de qualquer uso operacional.

---

## 6. Principais conclusões

1. Cruzamos os dados cadastrais e o histórico de crédito e obtivemos 36.457
   clientes para a modelagem; classificamos 1,69% na classe de inadimplência.
   Por isso, avaliamos métricas além da acurácia.
2. No teste, o Random Forest identificou 56 dos 185 inadimplentes (recall de
   30,27%) e sinalizou incorretamente 302 clientes da classe 0. Concluímos que
   o modelo deixou de identificar a maioria dos inadimplentes e não está pronto
   para ser usado como política automatizada de concessão.
3. Observamos que variáveis como idade, tempo de emprego e renda estão entre as
   mais importantes para o Random Forest. Interpretamos essa importância como
   contribuição relativa para as divisões do modelo, não como causalidade ou
   explicação individual das decisões.
4. Concluímos que os resultados podem apoiar uma análise exploratória de risco.
   Antes de qualquer decisão operacional, precisamos validar o target em uma
   janela temporal futura, avaliar limiares segundo os custos do negócio e
   realizar validações adicionais.

### Limitações e próximos passos

* Construímos um target retrospectivo, sem horizonte futuro fixo. Como próximo
  passo, precisamos definir uma janela de observação e outra de desempenho futuro.
* A classe de inadimplência representa apenas 1,69% da base de modelagem.
  Precisamos acompanhar o desempenho nessa classe e o equilíbrio entre falsos
  positivos e falsos negativos.
* Não comparamos o limiar padrão de classificação com limiares alternativos.
  Precisamos fazer essa análise usando validação, sem ajustar o limiar com base
  no conjunto de teste final.
* Reconhecemos que as importâncias do Random Forest não demonstram causalidade
  nem explicam individualmente as previsões. Em etapas futuras, podemos explorar
  métodos de explicabilidade e análises de viés.
* Antes da submissão, precisamos revisar as interpretações e saídas salvas dos
  notebooks para confirmar que os textos correspondem aos resultados atuais.

---

## 7. Estrutura do repositório

```
.
├── data/          dados brutos (raw) e tratados (processed) — não versionados
├── notebooks/     análise em ordem numerada
└── docs/          apresentação executiva
```

Documentamos os detalhes e as convenções em [`ESTRUTURA.md`](ESTRUTURA.md).
Antes de enviarmos o projeto, consultamos o [`CHECKLIST.md`](CHECKLIST.md).

---

## 8. Tecnologias

Utilizamos Python 3.12.8 (versão registrada nos metadados dos notebooks) e as
bibliotecas listadas em [`requirements.txt`](requirements.txt): pandas 2.2.3,
NumPy 2.1.3, scikit-learn 1.5.2, Matplotlib 3.9.2, Seaborn 0.13.2,
Jupyter 1.1.1 e joblib 1.4.2.
