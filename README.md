# AlexNet Oxford Flowers
Execução da AlexNet com Keras usando o conjunto de dados oxford flower. Experimentos conduzidos por Ana Clara, Gabriel Luna e Lincoln Costa

Os experimentos foram realizados com um notebook com CPU Intel Core i5 de 2.4GHz, memória de 16GB no sistema operacional Windows 10 Home.

| Grupo | Parâmetros fixos | Parâmetros variáveis |
|---------|---|--------|
| 4 | Activation function: Relu. / Learning ratio: 0.0001. / Epochs: 100. | Optimizer: SGD, Adam. / Learning ratio: 0.0005, 0.001, 0.002. |

## Utilização do Keras-Prov

O objetivo dessa atividade é apresentar vantagens e limitações da utilização do Keras-Prov. Conforme descrito pelas autoras, o Keras-Prov foi inicialmente desenvolvido para que fosse utilizado no Linux, então inicialmente já podemos elencar isso como uma limitação, visto que uma utilização no Windows seria essencial. Prosseguimos com a configuração até que obtivéssemos sucesso em uma execução inicial e relembramos que **esses foram os passos para o computador e SO descritos acima, é possível que máquinas com outras configurações cheguem em resultados diferentes.**

## Passo 1: clone/download do repositório do Keras-Prov.

Acesse o [repositório do Keras-Prov](https://github.com/dbpina/keras-prov) e faça um `git clone` ou download do mesmo.

## Passo 2: extração das pastas (dependências) necessárias.

Copie as pastas `keras, DfAnalyzer e dfa-lib-python` para a raiz do projeto onde você pretende utilizar o Keras-Prov. É possível observar essa estrutura em nosso repositório, onde as três pastas também foram inseridas.

**É importante atentar-se a possíveis atualizações no repositório base do Keras-Prov, visto que elas são importantes para o seu bom funcionamento.**

## Passo 3: instalação do MonetDB.

Acesse a [página de downloads do MonetDB para windows](https://www.monetdb.org/downloads/Windows/) e selecione qual versão deseja baixar. Durante esta documentação baixamos a versão contida na pasta `Oct2020-SP5/`. Durante a instalação do MonetDB, **selecionamos a opção Complete ao invés da Basic**. Em uma primeira tentativa com a versão Basic não conseguimos inicializar o Server, então recomendamos a intalação completa.

## Passo 4: configuração do MonetDB.

### MonetDB Server

### MonetDB Client

## Passo 4: execução de scripts para configuração do ambiente MonetDB.

## Passo 5: alinhamento de versões do Python e Tensorflow

## Passo 6: configuração para utilização do Keras-Prov na sua rede

## Passo 7: execução do .py com a sua rede

## Passo 8: realização de consultas pós-execução