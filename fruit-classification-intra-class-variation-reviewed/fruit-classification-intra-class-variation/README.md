# Fruit Classification with Intra-Class Variation

Projeto de visão computacional em Python para **comparar duas imagens de frutas** e estimar se elas pertencem à mesma classe visual conhecida ou a classes diferentes. O exemplo inicial considera banana Prata e banana d'água.

> **Protótipo de pesquisa:** o repositório contém uma implementação inicial, não um classificador validado. É necessário obter imagens representativas, conferir os rótulos, treinar o modelo e avaliá-lo em imagens independentes antes de confiar nos resultados.

## Objetivo

O sistema usa uma rede do tipo **Siamese** baseada em EfficientNet-B0. A mesma rede transforma cada imagem em um vetor de características (*embedding*); a distância entre os vetores é usada para estimar se o par pertence à mesma classe.

- `train.py`: treina o modelo usando pares de imagens rotuladas.
- `app.py`: aplicativo web Streamlit para comparar duas imagens.
- `evaluate.py`: avalia pares de imagens em um conjunto de teste separado.
- `fruit_pairwise_training.ipynb`: notebook inicial para treinar no Google Colab.
- `docs/`: documentação sobre arquitetura, dataset e avaliação.

## Tecnologias

- Python 3.10 ou 3.11
- PyTorch / Torchvision
- Pillow
- Streamlit
- scikit-learn e NumPy
- Git / GitHub
- Google Colab opcional para treinamento em nuvem

Não é necessário software pago. Uma GPU pode acelerar o treinamento. O notebook do Colab é uma alternativa para quem não tem GPU local.

## Estrutura do projeto

```text
fruit-classification-intra-class-variation/
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
├── .gitignore
├── README.md
├── fruit_pairwise_training.ipynb
├── docs/
│   ├── architecture.md
│   ├── dataset.md
│   └── evaluation.md
└── data/
    └── fruits/
```

O dataset e o modelo treinado não são incluídos no Git. O checkpoint padrão é salvo em `models/siamese_fruit.pth`.

## 1. Instalação local

```bash
python -m venv .venv
```

Ative o ambiente virtual:

- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Instale as dependências:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se houver problemas na instalação do PyTorch, consulte o seletor oficial: https://pytorch.org/get-started/locally/.

## 2. Preparar o dataset

Organize imagens em uma pasta por classe, usando rótulos confirmados:

```text
data/fruits/
├── banana_prata/
│   ├── imagem_001.jpg
│   └── imagem_002.jpg
└── banana_dagua/
    ├── imagem_001.jpg
    └── imagem_002.jpg
```

Cada classe precisa de pelo menos duas imagens para que o script consiga formar pares positivos. Na prática, são necessárias muitas imagens variadas por classe para um experimento significativo.

Inclua frutas físicas diferentes, ângulos, iluminação, fundos, câmeras e estágios de maturação variados. Evite colocar fotos quase idênticas da mesma fruta em divisões diferentes: isso pode inflar as métricas. Consulte [`docs/dataset.md`](docs/dataset.md).

## 3. Treinar o modelo

```bash
python train.py --data-dir data/fruits --output models/siamese_fruit.pth --epochs 10
```

Na primeira execução, o Torchvision pode baixar pesos pré-treinados da EfficientNet-B0, então será necessária uma conexão com a internet. Veja as opções disponíveis com `python train.py --help`.

O script usa uma divisão aleatória por imagem para treinamento e validação. Para resultados científicos confiáveis, faça a separação por fruta física ou sessão de captura e reserve um conjunto de teste independente.

## 4. Executar o aplicativo

Depois que o treinamento criar o checkpoint:

```bash
streamlit run app.py
```

Abra o endereço local mostrado no terminal e envie duas imagens. O aplicativo apresenta a distância entre os vetores e estima se pertencem à mesma classe aprendida.

**Importante:** a distância não é uma probabilidade. O resultado não comprova a identidade biológica ou a variedade da fruta; depende dos rótulos, da diversidade do dataset e da validação experimental.

## 5. Avaliar em um conjunto de teste

Prepare uma pasta de teste com as mesmas classes do treinamento, mas contendo imagens independentes:

```text
data/test/
├── banana_prata/
└── banana_dagua/
```

Execute:

```bash
python evaluate.py --data-dir data/test --checkpoint models/siamese_fruit.pth
```

O script mede se pares da mesma classe e de classes diferentes são separados pelo limite salvo no checkpoint. Por padrão, avalia até 10.000 pares; use `--max-pairs 0` para avaliar todos os pares. Para uma avaliação científica, as imagens de teste devem estar separadas por espécime ou sessão de captura e não podem ter sido usadas no treino nem para escolher o limite.

Consulte [`docs/evaluation.md`](docs/evaluation.md) e registre quantidade de frutas físicas, quantidade de imagens, como os rótulos foram confirmados, métricas e erros observados.

## Treinamento no Google Colab

Abra `fruit_pairwise_training.ipynb` no Google Colab. O notebook está preparado para clonar este repositório, instalar as dependências, ler o dataset no Google Drive, treinar o modelo e copiar o checkpoint para o Drive. Ajuste `DATA_DIR` para o caminho real das imagens rotuladas.

## Limitações e próximos passos

- O repositório não inclui dataset nem modelo treinado.
- Ainda não há evidência de que o sistema diferencie banana Prata de banana d'água com precisão.
- Os resultados podem variar com maturação, iluminação, fundo, câmera e qualidade dos rótulos.
- O sistema só compara as classes conhecidas no treinamento; ainda não possui um mecanismo validado para rejeitar classes desconhecidas.
- Próximas etapas: obter dados rotulados, treinar o baseline, avaliar em teste independente, inspecionar erros e comparar arquiteturas.

## Licença e uso

Defina uma licença no repositório antes de permitir reutilização por terceiros. Verifique também as condições de licença de qualquer dataset usado.
