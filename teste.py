import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Função para carregar a imagem


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

# Filtros


def apply_gaussian_filter(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_mean_filter(image, kernel_size=5):
    return cv2.blur(image, (kernel_size, kernel_size))


def apply_median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)


def apply_sobel_filter(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    return np.uint8(sobel)

# Ruídos


def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy = np.copy(image)
    total_pixels = image.size
    num_salt = int(salt_prob * total_pixels)
    num_pepper = int(pepper_prob * total_pixels)

    # Adiciona "sal" (pixels brancos)
    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy[coords[0], coords[1]] = 255

    # Adiciona "pimenta" (pixels pretos)
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy


def add_gaussian_noise(image, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

# Transformadas


def apply_fourier_transform(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * \
        np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    return np.uint8(magnitude_spectrum)


def apply_wavelet_transform(image, wavelet='haar'):
    coeffs2 = pywt.dwt2(image, wavelet)
    LL, (LH, HL, HH) = coeffs2

    # Normalização para melhorar visualização
    LL = np.uint8(255 * (LL - LL.min()) / (LL.max() - LL.min()))
    LH = np.uint8(255 * (LH - LH.min()) / (LH.max() - LH.min()))
    HL = np.uint8(255 * (HL - HL.min()) / (HL.max() - HL.min()))
    HH = np.uint8(255 * (HH - HH.min()) / (HH.max() - HH.min()))

    return LL, LH, HL, HH

# Exibir as imagens para comparação


def show_images(images, titles, cols=4):
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(15, 4 * rows))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title, fontsize=10)
        plt.axis('off')
        plt.subplots_adjust(hspace=0.4)
    plt.show()

# Opções do menu


def display_menu():
    print("Selecione uma opção:")
    print("1 - Visualizar todos os filtros e transformações aplicados de forma separada")
    print("2 - Escolher um filtro ou ruído específico para aplicar")
    print("3 - Aplicar uma combinação de filtros para tratamento na imagem")
    print("0 - Sair")


def choose_filter(image):
    print("Escolha um filtro ou ruído:")
    print("1 - Filtro Gaussiano")
    print("2 - Filtro de Média")
    print("3 - Filtro de Mediana")
    print("4 - Filtro de Sobel")
    print("5 - Ruído Sal e Pimenta")
    print("6 - Ruído Gaussiano")
    print("7 - Transformada Fourier")
    print("8 - Transformada Wavelet")

    choice = int(input("Digite o número da opção: "))
    match choice:
        case 1:
            filtered_image = apply_gaussian_filter(image)
        case 2:
            filtered_image = apply_mean_filter(image)
        case 3:
            filtered_image = apply_median_filter(image)
        case 4:
            filtered_image = apply_sobel_filter(image)
        case 5:
            filtered_image = add_salt_and_pepper_noise(image)
        case 6:
            filtered_image = add_gaussian_noise(image)
        case 7:
            filtered_image = apply_fourier_transform(image)
        case 8:
            LL, LH, HL, HH = apply_wavelet_transform(image)
            show_images([LL, LH, HL, HH], ["Wavelet LL",
                        "Wavelet LH", "Wavelet HL", "Wavelet HH"])
            return
        case _:
            print("Opção inválida.")
            return

    if choice != 8:
        show_images([image, filtered_image], ["Original", "Filtro Aplicado"])


def apply_combined_filters(image):
    # Alternativa 1: Sem o filtro Sobel
    combined_image = apply_gaussian_filter(image)
    combined_image = apply_median_filter(combined_image)
    show_images([image, combined_image], ["Original",
                "Combinação de Filtros (Suavização)"])

    # Alternativa 2: Ajustando a ordem dos filtros
    combined_image_sobel = apply_sobel_filter(image)
    combined_image_sobel = apply_gaussian_filter(combined_image_sobel)
    combined_image_sobel = apply_median_filter(combined_image_sobel)
    show_images([image, combined_image_sobel], ["Original",
                "Combinação de Filtros (Sobel + Suavização)"])


# Funções principais associadas ao menu
def view_all_filters(image):
    gaussian_filtered = apply_gaussian_filter(image)
    mean_filtered = apply_mean_filter(image)
    median_filtered = apply_median_filter(image)
    sobel_filtered = apply_sobel_filter(image)
    salt_pepper_noisy = add_salt_and_pepper_noise(image)
    gaussian_noisy = add_gaussian_noise(image)
    fourier_transformed = apply_fourier_transform(image)
    LL, LH, HL, HH = apply_wavelet_transform(image)

    images = [
        image, gaussian_filtered, mean_filtered, median_filtered, sobel_filtered,
        salt_pepper_noisy, gaussian_noisy, fourier_transformed, LL, LH, HL, HH
    ]
    titles = [
        "Original", "Filtro Gaussiano", "Filtro Média", "Filtro Mediana", "Filtro Sobel",
        "Ruído Sal & Pimenta", "Ruído Gaussiano", "Transformada Fourier",
        "Wavelet LL", "Wavelet LH", "Wavelet HL", "Wavelet HH"
    ]
    show_images(images, titles)

# Função principal com `match-case`


def main():
    image_path = "./images/albert-einstein.webp"  # Exemplo de caminho
    original_image = load_image(image_path)

    while True:
        display_menu()
        option = int(input("Digite o número da opção: "))

        match option:
            case 1:
                view_all_filters(original_image)
            case 2:
                choose_filter(original_image)
            case 3:
                apply_combined_filters(original_image)
            case 0:
                print("Encerrando...")
                break
            case _:
                print("Opção inválida, tente novamente.")


# Executa a função principal com menu
main()
