import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt


def carregar_imagem(imagem):
    return cv2.imread(imagem, cv2.IMREAD_GRAYSCALE)


def filtro_gaussiano(imagem, kernel_size=5):
    return cv2.GaussianBlur(imagem, (kernel_size, kernel_size), 0)


def filtro_media(imagem, kernel_size=5):
    return cv2.blur(imagem, (kernel_size, kernel_size))


def filtro_mediana(imagem, kernel_size=5):
    return cv2.medianBlur(imagem, kernel_size)


def filtro_sobel(imagem):
    sobel_x = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    return np.uint8(sobel)


def ruido_sal_pimenta(imagem, salt_prob=0.01, pepper_prob=0.01):
    ruido = np.copy(imagem)
    pixels = imagem.size
    num_sal = int(salt_prob * pixels)
    num_pimenta = int(pepper_prob * pixels)

    coordenadas = [np.random.randint(0, i, num_sal) for i in imagem.shape]
    ruido[coordenadas[0], coordenadas[1]] = 255

    coordenadas = [np.random.randint(0, i, num_pimenta) for i in imagem.shape]
    ruido[coordenadas[0], coordenadas[1]] = 0
    return ruido


def ruido_gaussiano(imagem, mean=0, sigma=25):
    ruido_gaussiano = np.random.normal(mean, sigma, imagem.shape)
    ruido_imagem = imagem + ruido_gaussiano
    ruido_imagem = np.clip(ruido_imagem, 0, 255).astype(np.uint8)
    return ruido_imagem


def transformada_fourier(imagem):
    dft = cv2.dft(np.float32(imagem), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectral = 20 * \
        np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    return np.uint8(magnitude_spectral)


def transformada_wavelet(imagem, wavelet='haar'):
    coeffs2 = pywt.dwt2(imagem, wavelet)
    LL, (LH, HL, HH) = coeffs2

    LL = np.uint8(255 * (LL - LL.min()) / (LL.max() - LL.min()))
    LH = np.uint8(255 * (LH - LH.min()) / (LH.max() - LH.min()))
    HL = np.uint8(255 * (HL - HL.min()) / (HL.max() - HL.min()))
    HH = np.uint8(255 * (HH - HH.min()) / (HH.max() - HH.min()))

    return LL, LH, HL, HH


def imprimir_imagens(imagens, titulos, cols=4):
    linhas = (len(imagens) + cols - 1) // cols
    plt.figure(figsize=(15, 4 * linhas))
    for i, (img, titulo) in enumerate(zip(imagens, titulos)):
        plt.subplot(linhas, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(titulo, fontsize=10)
        plt.axis('off')
        plt.subplots_adjust(hspace=0.4)
    plt.show()


def display_menu():
    print("Selecione uma opção:")
    print("1 - Visualizar todos os filtros e transformações aplicados de forma separada")
    print("2 - Escolher um filtro ou ruído específico para aplicar")
    print("3 - Aplicar uma combinação de filtros para tratamento na imagem")
    print("0 - Sair")


def escolher_filtro(imagem):
    print("Escolha um filtro ou ruído:")
    print("1 - Filtro Gaussiano")
    print("2 - Filtro de Média")
    print("3 - Filtro de Mediana")
    print("4 - Filtro de Sobel")
    print("5 - Ruído Sal e Pimenta")
    print("6 - Ruído Gaussiano")
    print("7 - Transformada Fourier")
    print("8 - Transformada Wavelet")

    escolha = int(input("Digite o número da opção: "))

    match escolha:
        case 1:
            imagem_filtrada = filtro_gaussiano(imagem)
        case 2:
            imagem_filtrada = filtro_media(imagem)
        case 3:
            imagem_filtrada = filtro_mediana(imagem)
        case 4:
            imagem_filtrada = filtro_sobel(imagem)
        case 5:
            imagem_filtrada = ruido_sal_pimenta(imagem)
        case 6:
            imagem_filtrada = ruido_gaussiano(imagem)
        case 7:
            imagem_filtrada = transformada_fourier(imagem)
        case 8:
            LL, LH, HL, HH = transformada_wavelet(imagem)
            imprimir_imagens([LL, LH, HL, HH], ["Wavelet LL",
                                                "Wavelet LH", "Wavelet HL", "Wavelet HH"])
            return
        case _:
            print("Opção inválida.")
            return

    if escolha != 8:
        imprimir_imagens([imagem, imagem_filtrada], [
                         "Original", "Filtro Aplicado"])


def filtros_combinados(imagem):
    imagem_combinada = filtro_gaussiano(imagem)
    imagem_combinada = filtro_mediana(imagem_combinada)
    imprimir_imagens([imagem, imagem_combinada], [
                     "Original", "Combinação de Filtros (Suavização)"])

    imagem_combinada_sobel = filtro_sobel(imagem)
    imagem_combinada_sobel = filtro_gaussiano(imagem_combinada_sobel)
    imagem_combinada_sobel = filtro_mediana(imagem_combinada_sobel)
    imprimir_imagens([imagem, imagem_combinada_sobel], [
                     "Original", "Combinação de Filtros (Sobel + Suavização)"])


def ver_todos_filtros(imagem):
    gaussiano_filtrado = filtro_gaussiano(imagem)
    media_filtrado = filtro_media(imagem)
    mediana_filtrado = filtro_mediana(imagem)
    sobel_filtrado = filtro_sobel(imagem)
    sal_pimenta = ruido_sal_pimenta(imagem)
    gaussiano = ruido_gaussiano(imagem)
    fourier = transformada_fourier(imagem)
    LL, LH, HL, HH = transformada_wavelet(imagem)

    imagens = [imagem, gaussiano_filtrado, media_filtrado, mediana_filtrado,
               sobel_filtrado, sal_pimenta, gaussiano, fourier, LL, LH, HL, HH]
    titulos = [
        "Original", "Filtro Gaussiano", "Filtro Média", "Filtro Mediana", "Filtro Sobel",
        "Ruído Sal & Pimenta", "Ruído Gaussiano", "Transformada Fourier",
        "Wavelet LL", "Wavelet LH", "Wavelet HL", "Wavelet HH"
    ]
    imprimir_imagens(imagens, titulos)


def main():
    imagem = "./images/albert-einstein.webp"
    imagem_original = carregar_imagem(imagem)

    while True:
        display_menu()
        opcao = int(input("Digite o número da opção: "))

        match opcao:
            case 1:
                ver_todos_filtros(imagem_original)
            case 2:
                escolher_filtro(imagem_original)
            case 3:
                filtros_combinados(imagem_original)
            case 0:
                print("Encerrando...")
                break
            case _:
                print("Opção inválida, tente novamente.")


main()
