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

O MonetDB criará dois aplicativos no sistema operacional: MonetDB SQL Server e MonetDB SQL Client. O primeiro, como o próprio nome explica, é o servidor que será consumido pelas outras aplicações. O segundo, por sua vez, serve para execução de consultas via SQL; é uma CLI do MonetDB.

### MonetDB Server

Para utilização do MonetDB na versão atual do Keras-Prov foi necessário realizar uma alteração no arquivo `M5server.bat`. Ele é encontrado em `C:\Program Files\MonetDB\MonetDB5`. A alteração consiste em renomear o banco de dados que é criado pelo MonetDB, para isso basta alterar os parâmetros `demo` para `dataflow_analyzer` nas linhas 25 e 26. **Na pasta `monetdb` do nosso repositório existe um exemplo de como deve ficar a versão final desse arquivo.** Após alterar o arquivo, execute a aplicação MonetDB SQL Server e deixe rodando.

### MonetDB Client

Como mencionado anteriormente, o Client é a CLI e ao executá-lo é possível utilizar consultas SQL via linha de comando, como: `select * from otrainingmodel;`. **É importante a inserção do ponto-e-vírgula no final da consulta para funcionamento da mesma.** Nesse momento as consultas ainda não funcionarão pois o ambiente não foi inicializado.

## Passo 4: execução de scripts para configuração do ambiente MonetDB.

Com a aplicação MonetDB SQL Server em execução, execute também o MonetDB SQL Client e execute os seguintes comandos:

```
\<RAIZDOSEUPROJETO\DfAnalyzer\monetdb\sql\create-schema.sql
\<RAIZDOSEUPROJETO\DfAnalyzer\monetdb\sql\database-script.sql
```

Eles são responsáveis por criar o schema e as tabelas utilizadas no Keras-Prov. Após executá-los, algumas mensagens de sucesso devem ser apresentadas no CLI, e então você pode fechar o MonetDB SQL Client.

## Passo 5: alinhamento de versões do Python e Tensorflow

Existem alguns conflitos com versões mais novas do Python e do Tensorflow/Keras. Para que o ambiente funcionasse normalmente, realizamos um downgrade para a versão `3.7.7` do Python. Além disso, listamos todas as libs instaladas utilizando o `pip list` e desinstalamos todas as libs envolvendo tensorflow e keras com o comando `pip uninstall NOMEDALIB`. Na sequência, instalamos a versão 2.2.0 do tensorflow com o comando `pip install tensorflow==2.2.0`. Futuramente, foi preciso reinstalar algumas outras libs, mas isso será de acordo com o arquivo `.py` que contém a sua rede.

Na sequência, como apresentado na documentação do Keras-Prov, precisamos realizar algumas instalações baseadas nas pastas que extraímos no início. Os comandos para essas instalações consideram que você está na raiz do seu projeto e são:

```
cd dfa-lib-python
python setup.py install
```

```
cd keras
python setup.py install
```

```
cd DfAnalyzer
./start-dfanalyzer.sh
```

## Passo 6: configuração para utilização do Keras-Prov na sua rede

Conforme explicado no repositório do Keras-Prov, é necessário inserir o seguinte trecho de código para funcionamento da proveniência:

```
hyps = {"OPTIMIZER_NAME": True,
    "LEARNING_RATE": True,
    "DECAY": True,
    "MOMENTUM": True,
    "NUM_EPOCHS": True,
    "BATCH_SIZE": True,
    "NUM_LAYERS": True}

model.provenance(dataflow_tag='keras-alexnet-df', adaptation=True, hyps = hyps)
```

## Passo 7: execução do .py com a sua rede

Pronto! Agora você poderá executar o seu arquivo `.py` e a proveniência começará a ser registrada. Para garantir que tudo está funcionando, acesse o endereço `localhost:22000` no seu navegador, e observe que um novo registro deve ter sido criado com o nome `keras-alexnet-df` (caso você não tenha realizado alterações no trecho de código do Passo 6).

## Passo 8: realização de consultas pós-execução

Existem duas formas de realiar consultas no MonetDB: utilizando o MonetDB SQL Client, como apresentamos no Passo 4, ou utilizando o DfAnalyzer, acessando o endereço exibido no Passo 7. Para a segunda opção, clique no registro criado (provavelmente `keras-alexnet-df`) e selecione, por exemplo, a caixinha com o nome `otrainingmodel`. Selecione os atributos que deseja que sejam exibidos, como `elapsed_time`, `loss` e `accuracy`, clique em `Save changes` e em `Run Query`.