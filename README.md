# Tech Challenge — Fase 2 | POSTECH Data Analytics

> **INSTRUÇÕES:** este README é um template. Substitua **todos** os blocos marcados com
> `<!-- PREENCHER -->` e apague as linhas de instrução antes de submeter.
> O README vale **3 pontos** na Dimensão 1 da rúbrica.

---

## 1. Identificação

| Campo | Valor |
|---|---|
| Turma | 2DTATBB |
| Grupo | <!-- PREENCHER: ex. Grupo 07 --> |
| Data de entrega | <!-- PREENCHER: DD/MM/AAAA --> |

### Integrantes

| Nome completo | RM | E-mail |
|---|---|---|
| FAGNER DO ESPÍRITO SANTO SÁ | RM377821 | fagner.sa@bb.com.br |
| MARIA APARECIDA BANDEIRA DA ROCHA| | cidarocha97@bb.com.br|
| KATIA DA SILVA SARMENTO| |katia.sarmento@bb.com.br |
| MARCIA REGINA CORREA DA SILVA | | marcia.rossa@bb.com.br|
| MARILENE RODRIGUES QUINTINO| | marilene_quintino@yahoo.com.br |

---

## 2. Links da entrega

Estes três links são **obrigatórios** e devem ser idênticos aos do PDF de submissão.

| Item | Link |
|---|---|
| Repositório | https://github.com/fagnermacedo/tech-challenge-fase2 |
| Vídeo executivo (≤ 5 min) | <!-- PREENCHER: YouTube não listado / Drive com acesso liberado --> |
| Apresentação | <!-- PREENCHER: link do arquivo em `docs/` ou Drive --> |

> ⚠️ Repositório privado ou inacessível **zera** toda a Dimensão 1 da rúbrica.
> Confira o acesso em uma janela anônima antes de enviar.

---

## 3. O problema

A concessão de crédito é uma operação crítica, onde o equilíbrio entre aprovar bons clientes e evitar a inadimplência pode determinar a rentabilidade do negócio. O objetivo desse trabalho é sugerir uma forma de minimizar o risco de inadimplência de novos solicitantes de cartão de crédito utilizando Machine Learning. Para isso, vamos utilizar datasets disponíveis no site Kaggle, que possuem dados de clientes que irão auxiliar na análise de crédito.

### Variável alvo

A variável alvo TARGET não vem informada no dataset e será construída através das análises que serão feitas nas bases de dados. A base de dados `credit_record.csv` contém o histórico financeiro dos clientes e fornece o `STATUS` mensal de pagamento. 

Os Status são:

* `X` -> Não possui empréstimo no mês.
* `C` -> Empréstimo quitado no mês.
* `0` -> Atraso entre 1 a 29 dias.
* `1` -> Atraso entre 30 e 59 dias.
* `2` -> Atraso entre 60 e 89 dias.
* `3` -> Atraso entre 90 e 119 dias.
* `4` -> Atraso entre 120 e 149 dias.
* `5` -> Atraso superior a 150 dias ou dívida crítica.

*   **Definição:** Iremos adotar, para definirmos atraso, valores iguais ou superiores a 60 dias (status `2`, `3`, `4` e `5`) classificando o cliente como "Mau Pagador" (1). Clientes com pagamentos em dia ou com atrasos menores que 60 dias (status `X`, `C`, `0`, `1`) serão classificados como "Bons Pagadores" (0).
*   **Justificativa:** Atrasos curtos podem ocorrer por diversos motivos, como demora no processamento das informações ou esquecimento. Mas o não registro do pagamento, superior a 60 dias, sinaliza problema real na capacidade de quitação dos débitos do cartão. Essa será a definição de alvo adotada no trabalho.

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
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

Baixe o dataset e coloque o arquivo bruto em `data/raw/` (os dados **não** são versionados —
veja `data/README.md`).

Depois execute os notebooks nesta ordem:

| # | Notebook | O que faz |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | Análise exploratória |
| 2 | `notebooks/02_preprocessamento.ipynb` | Limpeza, escala e feature engineering |
| 3 | `notebooks/03_modelagem.ipynb` | Treino e comparação dos modelos |
| 4 | `notebooks/04_avaliacao.ipynb` | Métricas, importância de variáveis e conclusões |

**Semente fixa:** `RANDOM_STATE = 42`, declarada na primeira célula de cada notebook.
Rodar os notebooks na ordem acima, a partir de um ambiente limpo, deve reproduzir
exatamente os números da seção 5.

---

## 5. Resultados

| Modelo | Acurácia | Precisão | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| <!-- PREENCHER --> | | | | | |
| | | | | | |

**Modelo escolhido:** <!-- PREENCHER --> — <!-- PREENCHER: por quê. -->

**Métricas priorizadas:** <!-- PREENCHER: justifique a escolha considerando o
     desbalanceamento de classes e o custo de cada tipo de erro no contexto do negócio. -->

---

## 6. Principais conclusões

<!-- PREENCHER: 3 a 5 conclusões em linguagem de negócio.
     Inclua quais variáveis mais influenciam o resultado e o que isso significa
     na prática para quem vai usar o modelo. -->

1.
2.
3.

### Limitações e próximos passos

<!-- PREENCHER -->

---

## 7. Estrutura do repositório

```
.
├── data/          dados brutos (raw) e tratados (processed) — não versionados
├── notebooks/     análise em ordem numerada
└── docs/          apresentação executiva
```

Detalhes e convenções em [`ESTRUTURA.md`](ESTRUTURA.md).
Antes de enviar, percorra o [`CHECKLIST.md`](CHECKLIST.md).

---

## 8. Tecnologias

<!-- PREENCHER: Python 3.11, pandas, scikit-learn, ... -->
