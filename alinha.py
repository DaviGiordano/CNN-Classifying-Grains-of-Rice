"""
Aluno: Davi Giordano Valério 11805273

Esse script alinha imagens de arroz no formato .jpg e escreve as imagens
alinhadas em formato .png, no mesmo local de origem
"""


import cv2 
import numpy as np
from numpy.lib import type_check
import matplotlib.pyplot as plt

def EncontraCentroOrientacao(img_path, print_imgs):
  img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

  # Calcula os momentos da imagem
  M = cv2.moments(binary)

  # Calcula centro
  cX = int(M["m10"] / M["m00"])
  cY = int(M["m01"] / M["m00"])

  # Calcula angulo de orientacao
  angle = 0.5 * np.arctan2(2*M["mu11"],(M["mu20"]-M["mu02"]))

  # Le a imagem colorida
  # img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  img_color = cv2.imread(img_path, cv2.IMREAD_COLOR) 
  img_clean = img_color.copy()

  cv2.circle(img_color, (cX, cY), 1, (0, 0, 255), -1)

  # Desenha a linha do angulo
  length = 75
  startX = cX - length * np.cos(angle)
  endX = cX + length * np.cos(angle)
  startY = cY - length * np.sin(angle)
  endY = cY + length * np.sin(angle)
  cv2.line(img_color, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 1)

  # Imprime imagem
  if print_imgs:
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.show()

  rows, cols = img.shape

  # Matriz de rotação
  M_rotate = cv2.getRotationMatrix2D((cX, cY), np.degrees(angle), 1)
  M_rotate = np.vstack([M_rotate, [0, 0, 1]])  # extend to a 3x3 matrix

  # Matriz de translação
  dx = cols//2 - cX
  dy = rows//2 - cY
  M_translate = np.float32([[1, 0, dx], [0, 1, dy], [0, 0, 1]])  # 3x3 matrix

  # Combina rotação com translação
  M_affine = np.matmul(M_translate, M_rotate)

  # Slice da matriz para inserir no warpAffine
  M_affine = M_affine[:2, :]

  # Aplica a rotação e translação
  img_transformed = cv2.warpAffine(img_clean, M_affine, (cols, rows))

  img_transformed_path = img_path[0:-3] + 'png'
  cv2.imwrite(img_transformed_path, img_transformed)  

  # Mostra o resultado final
  if print_imgs:
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()

def orientation_tests():
  basmati = './orientacao/Basmati1.jpg'
  jasmine = './orientacao/Jasmine2.jpg'
  ipsala = './orientacao/Ipsala1.jpg'
  karacadag = './orientacao/Karacadag1.jpg'
  arborio = './orientacao/Arborio1.jpg'

  paths = [basmati, jasmine, ipsala, karacadag, arborio]
  for path in paths:
    EncontraCentroOrientacao(path,1)

import os
print('##\n Esse script aplica a funcao EncontraCentroOrientacao em todos os arquivos .jpg encontrados no diretorio especificado,\n inclusive os arquivos em sub-diretorios\n##')
# orientation_tests()

root_dir = input('Insira o diretorio raiz:')
# Caminha pelo diretório
for dir_name, _, file_list in os.walk(root_dir):
  print(dir_name)
  for filename in file_list:
    if filename.endswith('.jpg'):
      # Constrói o caminho completo
      full_path = os.path.join(dir_name, filename)
      print(full_path)
      # Aplica a função
      EncontraCentroOrientacao(full_path,0)