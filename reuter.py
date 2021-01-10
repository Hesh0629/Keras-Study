{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "reuter.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "machine_shape": "hm",
      "authorship_tag": "ABX9TyNrIeHK6i1UQxNIqwAyy3tC",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/HeSH-0629/Keras-Study/blob/main/reuter.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "73GCuFv2jX_w"
      },
      "source": [
        "2021.01.10 (D+125)\r\n",
        "\r\n",
        "> 로이터 기사 카테고리 분류 (단일 레이블 다중 분류)  \r\n",
        "> categorical_crossentropy 이용  "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "q8TVrqnOjAbT"
      },
      "source": [
        "\r\n",
        "**데이터 전처리 부분**  \r\n",
        "주의 : 로이터 데이터셋은 np배열로 안주어지고 리스트 배열로 주어짐 따라서 to_categorical 사용 X"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Z8iymMaYb_GO",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2df00a58-664e-4647-80c5-f977d27442da"
      },
      "source": [
        "import numpy as np\r\n",
        "import keras\r\n",
        "from keras.datasets import reuters\r\n",
        "from keras.utils.np_utils import to_categorical\r\n",
        "(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000) #단어 개수는 가장 많이 쓰이는 단어 10,000개로 제한\r\n",
        "def vectorize_seq(seq,dimension=10000): #텍스트 처리를 위한 원 핫 인코딩 (각 레이블의 인덱스 자리는 1이고 나머지는 0)\r\n",
        "  results = np.zeros((len(seq),dimension))\r\n",
        "  for i, seq in enumerate(seq):\r\n",
        "    results[i,seq]=1.\r\n",
        "  return results\r\n",
        "x_train = vectorize_seq(train_data)\r\n",
        "x_test = vectorize_seq(test_data)\r\n",
        "#data와 달리 labels는 리스트 배열이 아니므로 to_categorical을 활용해서 빠르게 원-핫 인코딩 가능\r\n",
        "one_hot_train_labels = to_categorical(train_labels)\r\n",
        "one_hot_test_labels = to_categorical(test_labels)"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/datasets/reuters.py:148: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray\n",
            "  x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])\n",
            "/usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/datasets/reuters.py:149: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray\n",
            "  x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8YBs38QqjSPL"
      },
      "source": [
        "**신경망 구성 및 훈련 과정**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2Ida-gfKi4w4"
      },
      "source": [
        "from keras import models\r\n",
        "from keras import layers\r\n",
        "\r\n",
        "model = models.Sequential()\r\n",
        "model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))\r\n",
        "model.add(layers.Dense(64, activation='relu'))\r\n",
        "model.add(layers.Dense(46, activation='softmax')) #softmax층은 46개의 클래스에 대한 확률을 출력, 더 나아가 마지막 출력이 46이므로 중간층 히든 유닛은 46보다 커야 손실X\r\n",
        "model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) #categorical_crossentropy는 두 확률 부포의 사이의 거리를 측정\r\n",
        "x_val = x_train[:1000]\r\n",
        "x_train = x_train[1000:]\r\n",
        "y_val = one_hot_train_labels[:1000]\r\n",
        "y_train = one_hot_train_labels[1000:]\r\n",
        "history = model.fit(x_train,\r\n",
        "                    y_train,\r\n",
        "                    epochs=20,\r\n",
        "                    batch_size=512,\r\n",
        "                    validation_data=(x_val, y_val))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ojZVjH8Ui-sF"
      },
      "source": [
        "import matplotlib.pyplot as plt\r\n",
        "loss = history.history['loss']\r\n",
        "val_loss = history.history['val_loss']\r\n",
        "\r\n",
        "epochs = range(1, len(loss) + 1)\r\n",
        "\r\n",
        "plt.plot(epochs, loss, 'bo', label='Training loss')\r\n",
        "plt.plot(epochs, val_loss, 'b', label='Validation loss')\r\n",
        "plt.title('Training and validation loss')\r\n",
        "plt.xlabel('Epochs')\r\n",
        "plt.ylabel('Loss')\r\n",
        "plt.legend() #범례 그리기\r\n",
        "\r\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 278
        },
        "id": "s_SPr3v7lN4z",
        "outputId": "a216efac-2cd5-4b4c-a2d8-5453ee555b4d"
      },
      "source": [
        "plt.clf()\r\n",
        "\r\n",
        "acc = history.history['accuracy']\r\n",
        "val_acc=history.history['val_accuracy']\r\n",
        "\r\n",
        "plt.plot(epochs,acc,'bo',label='Training acc')\r\n",
        "plt.plot(epochs,val_acc,'b',label='Validation acc')\r\n",
        "plt.xlabel('Epochs')\r\n",
        "plt.ylabel('Accuracy')\r\n",
        "plt.legend()\r\n",
        "\r\n",
        "plt.show()"
      ],
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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xx57bK99XnvtNZ599lm2bNnC4YcfzuTJk/e69v7vf/87r776KocccggnnngiL774IiUlJVx++eX85S9/YcCAAZSWlqaMqWfPnjz99NN07NiRN954g9LSUsrLy3niiSf44x//yEsvvUSnTp348MMPAZgwYQLXX3895513Htu3b2fXrl0t/0WloUQgIpHo1y80B6Va3tIuuOACCgsLAdi8eTMXX3wxb7zxBmZGdXV1yn3OOeccOnToQIcOHejZsycffPABffr02WObESNG7F42ZMgQKioqKCoq4tBDD919nX5paSllZWV7Hb+6upopU6awdOlSCgsLef311wF45plnuOSSS+jUqRMA3bp1Y8uWLaxbt47zzjsPCDeFZZOahkQkEtOmQaKs261Tp7C8pXXu3Hn3+//8z/9k9OjRrFixgnnz5qW9pr5Dhw673xcWFlJTU7NP26Tz05/+lIMPPphly5ZRXl7eaGd2nCJNBGY2xsxWm9kaM7s+xfqJZrbBzJYmpm9GGY+IZM+ECVBWBv37g1l4LSvb947iTG3evJnevXsD8Ktf/arFj3/44Yfz1ltvUVFRAcDs2bPTxtGrVy8KCgqYMWMGO3fuBODMM8/kl7/8JVsTHSgffvghXbp0oU+fPsyZMweATz/9dPf6bIgsEZhZIXAvcBZwJFBqZkem2HS2uw9JTA9GFY+IZN+ECVBRAbt2hdeokwDAddddxw033MDQoUObdAafqf3335/77ruPMWPGMHz4cLp06cIBBxyw13bf+ta3mD59OoMHD+a1117bXWsZM2YMY8eOpaSkhCFDhnDHHXcAMGPGDO666y6OOeYYRo4cyfvvv9/isadjtb3gLX5gsxOAm939S4n5GwDc/b+TtpkIlLj7lEyPW1JS4uXl5S0crYhkYtWqVRxxxBFxhxG7qqoqioqKcHeuvPJKBg4cyDXXXBN3WLul+n8ys8XunvL62SibhnoD7ybNVyaW1fcVM1tuZr8zs76pDmRmk8ys3MzKN2zYEEWsIiIZe+CBBxgyZAhf+MIX2Lx5M5dffnncITVL3FcNzQN+6+6fmtnlwHTgtPobuXsZUAahRpDdEEVE9nTNNdfkVA2guaKsEawDks/w+ySW7ebuG93908Tsg8DwCOMREZEUokwEi4CBZjbAzNoDFwJzkzcws15Js2OBVRHGIyIiKUTWNOTuNWY2BXgSKAR+4e6vmtktQLm7zwWuMrOxQA3wITAxqnhERCS1SPsI3H0+ML/eshuT3t8A3BBlDCIi0jDdWSwircbo0aN58skn91h25513Mnny5LT7jBo1itpLzs8++2w2bdq01zY333zz7uv505kzZw4rV67cPX/jjTfyzDPPNCX8nKVEICKtRmlpKbNmzdpj2axZs9IO/Fbf/PnzOfDAA/fps+sngltuuYUzzjhjn46Va+K+fFREWqmrr4bEowFazJAhcOed6dePHz+e73//++zYsYP27dtTUVHBe++9x8knn8zkyZNZtGgR27ZtY/z48fzgBz/Ya//i4mLKy8vp0aMH06ZNY/r06fTs2ZO+ffsyfHi4aPGBBx6grKyMHTt28PnPf54ZM2awdOlS5s6dy/PPP88Pf/hDHnvsMW699VbOPfdcxo8fz4IFC7j22mupqanh2GOP5f7776dDhw4UFxdz8cUXM2/ePKqrq3n00UcZNGjQHjFVVFTw9a9/nU8++QSAe+65Z/fDcW677TYefvhhCgoKOOuss/jRj37EmjVruOKKK9iwYQOFhYU8+uijfO5zn2vW964agYi0Gt26dWPEiBE88cQTQKgNfPWrX8XMmDZtGuXl5Sxfvpznn3+e5cuXpz3O4sWLmTVrFkuXLmX+/PksWrRo97rzzz+fRYsWsWzZMo444ggeeughRo4cydixY7n99ttZunTpHgXv9u3bmThxIrNnz+aVV16hpqaG+++/f/f6Hj16sGTJEiZPnpyy+al2uOolS5Ywe/bs3U9RSx6uetmyZVx33XVAGK76yiuvZNmyZSxcuJBevXrtdcymUo1ARPZJQ2fuUaptHho3bhyzZs3ioYceAuCRRx6hrKyMmpoa1q9fz8qVKznmmGNSHuOFF17gvPPO2z0U9NixY3evW7FiBd///vfZtGkTVVVVfOlLX2owntWrVzNgwAAOO+wwAC6++GLuvfderr76aiAkFoDhw4fz+9//fq/9c2G46ryoEbT0c1NFJD7jxo1jwYIFLFmyhK1btzJ8+HDefvtt7rjjDhYsWMDy5cs555xz0g4/3ZiJEydyzz338Morr3DTTTft83Fq1Q5lnW4Y61wYrrrNJ4La56a+8w641z03VclApHUqKipi9OjRXHrppbs7iT/++GM6d+7MAQccwAcffLC76SidU045hTlz5rBt2za2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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3x2jeHIroXxz"
      },
      "source": [
        "원핫 인코딩이 귀찮다면 정수 텐서로 바로변환  \r\n",
        "손실함수는 **'sparse_categorical_crossentropy'**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SbJOTJo6oSzg"
      },
      "source": [
        "y_train = np.array(train_labels)\r\n",
        "y_test = np.array(test_labels)\r\n",
        "model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}