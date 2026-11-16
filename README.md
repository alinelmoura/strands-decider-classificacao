# Classificação hierárquica de comentários de clientes com um modelo de decisão local

Experimento para classificar comentários de clientes por **assunto (macro → subcategoria)** e **sentimento** usando o [Strands Decider](https://github.com/strands-labs/strands-decider), um modelo pequeno (2B) que roda localmente, sem API, e devolve uma **distribuição de probabilidades** sobre as opções em vez de texto livre.

A pergunta por trás do experimento: um modelo pequeno e barato consegue fazer parte do trabalho de uma LLM numa taxonomia grande e, principalmente, **sabe quando está errando**? Se a probabilidade separar acerto de erro, dá para montar um sistema híbrido, em que o modelo local resolve os casos fáceis e a LLM fica só com os difíceis.

## Como funciona

| Etapa | O que faz |
|---|---|
| 1. Mascaramento | Remove dados pessoais por regex antes de qualquer coisa (CPF, cartão, telefone, e-mail, conta, chave Pix, CEP, @usuário, nome após "meu nome é"...) e lista o que sobrou de suspeito para revisão manual |
| 2. Macro | *"Sobre o que o cliente está falando?"* entre as macrocategorias, cada uma com descrição e fronteiras |
| 3. Sub | A mesma pergunta entre as subcategorias da macro escolhida. Se a 2ª macro tiver probabilidade ≥ 0,25, olha as subs dela também |
| 4. Sentimento | *"Qual o sentimento do cliente em relação a {tema}?"*, condicionado à subcategoria escolhida |
| 5. Avaliação | Acurácia e kappa por nível, sub em comum (multirrótulo), top-k, calibração e matriz de confusão |

**Decisões de projeto**

- **Hierárquica em vez de plana**: o Decider lê no máximo 4.096 tokens por chamada (texto + pergunta + opções). Com ~130 subcategorias com descrição, a pergunta plana passaria de 8 mil tokens. O notebook estima o orçamento de tokens e escolhe a estratégia sozinho (`ESTRATEGIA = "auto"`).
- **Probabilidade conjunta**: a probabilidade de cada sub é P(macro) × P(sub | macro), para dar para comparar subs de macros diferentes e aceitar uma 2ª categoria.
- **Sentimento em chamada separada**: perguntas da mesma chamada são avaliadas de forma independente. Para o sentimento depender do tema, ele precisa vir depois.
- **Viés do gold**: quando parte dos rótulos é feita validando uma sugestão de LLM, há viés de ancoragem. O notebook separa as métricas por tipo de rotulação.

## Resultados

Teste com **40 comentários reais de clientes de uma fintech** (redes sociais, lojas de apps e pesquisa de satisfação), numa taxonomia de **37 macros e 131 subcategorias**, com o modelo de 2B rodando em CPU. Os dados não estão no repositório por serem confidenciais.

| Métrica | Resultado |
|---|---|
| Macro correta (1ª categoria) | 40% |
| Macro correta no top 3 | 65% |
| Subcategoria exata | 32,5% |
| Pelo menos uma sub em comum com o gold | 40% |
| Sentimento correto | 75% |
| Sentimento correto, quando acertou a sub | 94% |

**O que aprendi**

- **Sentimento funciona; assunto fino, não.** Com 37 macros, o modelo de 2B erra a macro na maioria dos casos. Mas a macro certa está no top 3 em 65% deles, o que serve para pré-filtrar as opções antes de mandar para uma LLM.
- **Ele se refugia na categoria "não classificável"**: escolheu essa macro em 27,5% dos casos, contra 5% no gold.
- **A confiança é baixa, mas informativa**: nenhuma previsão passou de 0,4 de probabilidade. Ainda assim, na faixa 0,2–0,4 ele acertou a sub em 56% dos casos, contra 26% abaixo de 0,2. Ou seja, há sinal para um limiar do híbrido, mas o volume que ficaria com o modelo local é pequeno.
- **Custo**: na CPU, cerca de 2 minutos por comentário, já que a estratégia hierárquica faz de 3 a 4 chamadas.

Próximos passos: comparar com uma LLM nos mesmos comentários, testar o modelo maior (`strands-decider-E4B-gemma4`) e reescrever as descrições das subcategorias com pior acurácia.

## Como rodar

O repositório traz um exemplo **fictício** em `dados/`: 16 comentários inventados de e-commerce, com dados pessoais falsos para testar o mascaramento, e uma taxonomia de 7 macros e 15 subs. Com esse exemplo pequeno, o notebook escolhe a estratégia plana. Para ver o caminho hierárquico, use `ESTRATEGIA = "hierarquica"`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# em outro terminal, com o venv ativo, suba o servidor do Decider
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --device cpu --port 8000
```

Depois, abra `classificacao_decider.ipynb` e rode as células em ordem. As saídas vão para `saida_decider/`.

**Para usar seus dados**, troque `ARQ_GOLD` e `ARQ_TAXONOMIA` na célula de configuração e ajuste os nomes de coluna:

- **Gold**: 1 linha por `(comment_id, ordem)`, com o texto, a macro, a sub e o sentimento.
- **Taxonomia**: uma aba de subcategorias (macro, sub, descrição, exemplos) e uma aba de macros (nome e descrição).

## Tecnologias

Python · pandas · Strands Decider (Qwen 3.5 2B) · requests · Jupyter

## Autora

Aline Moura · [LinkedIn](https://www.linkedin.com/in/aline-nataly-lima-de-moura-25048022b/)
