## Resumo
Neste repositório encontra-se os scripts para a reprodução do experimento que foi realizado para a contagem de microesferas.

## Requisitos
Para que seja possível executar o experimento é necessário:

* Python - 3.6
* OpenCV - 2.4.9
* Keras  - 2.0.9
* Tensorflow - 1.3.0

Além disso, é preciso compilar a biblioteca MGranul (https://github.com/joeljunior05/MGranul). 

O jeito mais fácil para obter todos os requisitos é utilizando o DockerFile que se encontra no repositório de MGranul. Nele  estão contidos todos os requisitos necessários. Para obter uma imagem com MGranul já compilado execute no seu terminal:

```
docker pull joeljunior/msgranul
```

Para o treino é utilizado dois arquivos .npy. Esse arquivos não estão na pasta ainda. Para baixar, execute o comando:

```
./init.sh
```


## Organização

Ele está dividido em três scripts principais:
* train_unet.py - ao executar esse script o modelo que encontra-se disponível na pasta **model**, será treinado utilizando as imagens da pasta **trainning/inputs** como entrada e **tranning/masks** como resultado esperado; 
* segment_w_unet.py - utiliza o modelo já treinado (disponível na pasta **trained_model**), para predizer as imagens em **prediction/inputs**. Ele coloca as imagens geradas na pasta **prediction/outputs_unet**;
* find_microspheres - roda granulometria por correlação nas imagens **prediction/outputs_unet** para identificar onde estão localizadas as microesferas;

## Exemplo

```
$ python find_microspheres.py

imagem & real & VP & FP & FN & T \\
1 & 157 & 155 & 1 & 2 & 0.9810126582278481 \\
2 & 177 & 173 & 7 & 4 & 0.9402173913043478 \\
3 & 150 & 139 & 11 & 11 & 0.8633540372670807 \\
4 & 176 & 166 & 7 & 10 & 0.907103825136612 \\
5 & 145 & 142 & 2 & 3 & 0.9659863945578231 \\
6 & 138 & 135 & 4 & 3 & 0.9507042253521126 \\
7 & 138 & 132 & 12 & 6 & 0.88 \\
8 & 117 & 115 & 2 & 2 & 0.9663865546218487 \\
9 & 142 & 138 & 7 & 4 & 0.9261744966442953 \\
10 & 128 & 124 & 2 & 4 & 0.9538461538461539 \\
ACC: 0.9865263016191902
T: 0.9334785736958123
EX: 1468 AC: 1474 
VP: 1419 FP: 55 FN: 49
```
